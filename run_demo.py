#!/usr/bin/env python
# run_demo.py
"""
Live demo: watch a trained OfficeBot agent run in a pygame window.

Usage:
    python run_demo.py --algo ppo --seed 42 --episodes 2
    python run_demo.py --algo dqn --seed 42 --episodes 3 --speed 15
    python run_demo.py --algo ppo --seed 42 --episodes 1 --headless
"""

import argparse
import sys
import os
import yaml

from environment.env import OfficeBotEnv
from agents.agent_factory import load_agent


# ── ANSI colour helpers ──────────────────────────────────────────────────────
GREEN  = '\033[92m'
YELLOW = '\033[93m'
RED    = '\033[91m'
CYAN   = '\033[96m'
RESET  = '\033[0m'
BOLD   = '\033[1m'


def parse_args():
    p = argparse.ArgumentParser(description='OfficeBot RL live demo')
    p.add_argument('--algo',     required=True, choices=['ppo', 'dqn', 'a2c'])
    p.add_argument('--seed',     type=int, default=42)
    p.add_argument('--episodes', type=int, default=2)
    p.add_argument('--speed',    type=int, default=10,
                   help='Render FPS (default 10). Higher = faster playback.')
    p.add_argument('--model',    default='best', choices=['best', 'final'],
                   help='Which checkpoint to load (default: best)')
    p.add_argument('--env-config', default='configs/env_config.yaml')
    p.add_argument('--headless', action='store_true',
                   help='Run without a display window (useful for CI / SSH)')
    return p.parse_args()


def format_status(step, reward_acc, info, algo, episode, total_episodes):
    """Build the per-step status line."""
    battery_pct  = info.get('battery', 0.0) * 100
    completed    = info.get('completed_tasks', 0)
    collisions   = info.get('collision_count', 0)
    task_desc    = info.get('current_task') or 'None (all done)'
    carrying     = info.get('carrying_item') or '—'

    # Colour battery by level
    if battery_pct > 50:
        bat_col = GREEN
    elif battery_pct > 20:
        bat_col = YELLOW
    else:
        bat_col = RED

    reward_col = GREEN if reward_acc >= 0 else RED

    return (
        f"\r{BOLD}Ep {episode}/{total_episodes}{RESET}  "
        f"Step {step:04d}  "
        f"Reward {reward_col}{reward_acc:+8.2f}{RESET}  "
        f"Battery {bat_col}{battery_pct:5.1f}%{RESET}  "
        f"Carrying: {CYAN}{carrying:<8}{RESET}  "
        f"Task: {YELLOW}{task_desc[:40]:<40}{RESET}  "
        f"Done: {completed}  Collisions: {collisions}"
    )


def run_demo():
    args = parse_args()

    # ── resolve model path ────────────────────────────────────────────────
    if args.model == 'best':
        model_path = f'models/{args.algo}/seed{args.seed}/best/best_model.zip'
    else:
        model_path = f'models/{args.algo}/seed{args.seed}/final_model.zip'

    if not os.path.exists(model_path):
        print(f'\n{RED}Error:{RESET} Model not found: {model_path}')
        print(f'Run training first:  python -m training.train --algo {args.algo} --seed {args.seed} ...')
        return 1

    # ── load env config ───────────────────────────────────────────────────
    with open(args.env_config) as f:
        env_cfg = yaml.safe_load(f)

    env_cfg['tile_size'] = 32           # ensure tile size is set

    render_mode = 'human' if not args.headless else 'rgb_array'
    env_cfg['render_fps'] = args.speed  # pass speed through config (backup)

    # Override render FPS in metadata via env after creation
    env = OfficeBotEnv(
        config=env_cfg,
        seed=args.seed + 9999,
        render_mode=render_mode,
        eval_mode=True,
    )
    env.metadata['render_fps'] = args.speed

    # ── load model ────────────────────────────────────────────────────────
    print(f'\n{BOLD}OfficeBot RL Demo{RESET}  —  {args.algo.upper()}  seed={args.seed}')
    print(f'Model : {model_path}')
    print(f'Mode  : {"headless (rgb_array)" if args.headless else "window (human)"}')
    print(f'Speed : {args.speed} FPS')
    print('-' * 70)

    model = load_agent(args.algo, model_path, env=None)

    total_rewards = []

    for ep in range(1, args.episodes + 1):
        obs, info = env.reset()
        env.render()

        done        = False
        step        = 0
        reward_acc  = 0.0
        ep_start    = True

        print()  # blank line before episode

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            reward_acc += reward
            step       += 1
            done        = terminated or truncated

            # Print status line (overwrite in place)
            status = format_status(step, reward_acc, info, args.algo, ep, args.episodes)
            print(status, end='', flush=True)

            env.render()

            # Check if window was closed (render returns None on close)
            if render_mode == 'human' and env.window is None:
                print(f'\n{YELLOW}Window closed — stopping demo.{RESET}')
                env.close()
                return 0

        total_rewards.append(reward_acc)
        reason = info.get('termination_reason', 'unknown')
        print(f'\n{BOLD}Episode {ep} done{RESET}  |  '
              f'Reward: {reward_acc:+.2f}  |  '
              f'Steps: {step}  |  '
              f'Reason: {reason}  |  '
              f'Tasks completed: {info.get("completed_tasks", 0)}')
        print('-' * 70)

    env.close()

    # ── summary ───────────────────────────────────────────────────────────
    import numpy as np
    print(f'\n{BOLD}=== Demo Summary ==={RESET}')
    print(f'Algorithm  : {args.algo.upper()}  (seed {args.seed})')
    print(f'Episodes   : {args.episodes}')
    print(f'Mean reward: {np.mean(total_rewards):+.2f}  +/-{np.std(total_rewards):.2f}')
    print()
    return 0


if __name__ == '__main__':
    sys.exit(run_demo())
