# evaluation/evaluate.py
"""
Evaluate a trained OfficeBot RL agent.

Usage:
    python evaluation/evaluate.py --algo ppo --seed 42 [--episodes 50]

Saves: logs/metrics/{algo}/seed{seed}/eval_results.json
"""

import argparse
import json
import os
import time

import numpy as np
import yaml

from environment.env import OfficeBotEnv
from agents.agent_factory import load_agent


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a trained OfficeBot agent.")
    p.add_argument("--algo", required=True, choices=["ppo", "dqn", "a2c"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--model", default="best", choices=["best", "final"])
    p.add_argument("--env-config", default="configs/env_config.yaml")
    return p.parse_args()


def _load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


def run_episodes(algo, seed, n_episodes, model_type="best",
                 env_config_path="configs/env_config.yaml"):
    """Load model + env, run n_episodes with deterministic=True."""
    mpath = (f"models/{algo}/seed{seed}/best/best_model.zip"
             if model_type == "best"
             else f"models/{algo}/seed{seed}/final_model.zip")
    if not os.path.exists(mpath):
        raise FileNotFoundError(f"Model not found: {mpath}")

    print(f"Loading {model_type} model: {mpath}")
    model = load_agent(algo, mpath, env=None)

    env_cfg = _load_yaml(env_config_path)
    env = OfficeBotEnv(config=env_cfg, seed=seed + 2000, eval_mode=True)

    print(f"Running {n_episodes} episodes  (algo={algo.upper()}, seed={seed}) ...")
    episodes, t0 = [], time.time()

    for ep in range(n_episodes):
        obs, info = env.reset()
        done, total_r, steps = False, 0.0, 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_r += reward
            steps += 1
            done = terminated or truncated

        c = info.get("completed_tasks", 0)
        f = info.get("failed_tasks", 0)
        episodes.append({
            "total_reward":    total_r,
            "steps":           steps,
            "completed_tasks": c,
            "failed_tasks":    f,
            "total_tasks":     c + f,
            "collision_count": info.get("collision_count", 0),
            "battery":         info.get("battery", 0.0),
            "success":         info.get("termination_reason", "") == "all_tasks_done",
        })
        if (ep + 1) % 10 == 0:
            print(f"  {ep+1}/{n_episodes}")

    env.close()
    return _compute_metrics(episodes, algo, seed, model_type, time.time() - t0), episodes


def _compute_metrics(eps, algo, seed, model_type, elapsed):
    rewards   = np.array([e["total_reward"]    for e in eps])
    steps_arr = np.array([e["steps"]           for e in eps])
    compl     = np.array([e["completed_tasks"] for e in eps])
    failed    = np.array([e["failed_tasks"]    for e in eps])
    total_t   = np.array([e["total_tasks"]     for e in eps])
    colls     = np.array([e["collision_count"] for e in eps])
    batt      = np.array([e["battery"]         for e in eps])
    succ      = np.array([e["success"]         for e in eps], dtype=float)

    mean_len = float(np.mean(steps_arr))
    with np.errstate(divide="ignore", invalid="ignore"):
        tcr  = np.where(total_t > 0, compl / total_t, 0.0)
        beff = np.where(1.0 - batt > 0.01, compl / (1.0 - batt), 0.0)

    return {
        "algorithm":              algo,
        "seed":                   seed,
        "model_type":             model_type,
        "n_eval_episodes":        len(eps),
        "eval_time_seconds":      elapsed,
        # 7 primary metrics
        "mean_episode_reward":    float(np.mean(rewards)),
        "std_episode_reward":     float(np.std(rewards)),
        "success_rate":           float(np.mean(succ)),
        "task_completion_rate":   float(np.mean(tcr)),
        "mean_collision_count":   float(np.mean(colls)),
        "mean_battery_remaining": float(np.mean(batt)),
        "mean_episode_length":    mean_len,
        # extra metrics
        "collision_rate":         float(np.mean(colls)) / mean_len if mean_len > 0 else 0.0,
        "battery_efficiency":     float(np.mean(beff)),
        "mean_completed_tasks":   float(np.mean(compl)),
        "mean_failed_tasks":      float(np.mean(failed)),
    }


def save_results(metrics, algo, seed):
    out_dir = f"logs/metrics/{algo}/seed{seed}"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "eval_results.json")
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved: {out_path}")


def print_metrics(m):
    bar = "=" * 60
    print(f"\n{bar}")
    print(f"  RESULTS - {m['algorithm'].upper()}  (seed={m['seed']})")
    print(bar)
    print(f"  Mean Episode Reward    : {m['mean_episode_reward']:>8.3f}"
          f"  +/- {m['std_episode_reward']:.3f}")
    print(f"  Success Rate           : {m['success_rate']:>8.1%}")
    print(f"  Task Completion Rate   : {m['task_completion_rate']:>8.1%}")
    print(f"  Mean Collisions        : {m['mean_collision_count']:>8.2f}")
    print(f"  Mean Battery Remaining : {m['mean_battery_remaining']:>8.1%}")
    print(f"  Mean Episode Length    : {m['mean_episode_length']:>8.1f} steps")
    print(bar + "\n")


def main():
    args = parse_args()
    try:
        metrics, _ = run_episodes(
            algo=args.algo,
            seed=args.seed,
            n_episodes=args.episodes,
            model_type=args.model,
            env_config_path=args.env_config,
        )
        print_metrics(metrics)
        save_results(metrics, args.algo, args.seed)
        print("Evaluation complete.\n")
    except FileNotFoundError as exc:
        print(f"\nError: {exc}")
        return 1
    except Exception:
        import traceback
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    exit(main())
