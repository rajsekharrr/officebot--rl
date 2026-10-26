#!/usr/bin/env python3
"""Detailed battery behavior evaluation."""

import yaml
import numpy as np
from stable_baselines3 import PPO
from environment.env import OfficeBotEnv
from environment.tasks import TaskType

def evaluate_battery_behavior(n_episodes=10):
    """Evaluate agent with detailed battery tracking."""

    # Load config
    with open('configs/env_config.yaml', 'r') as f:
        env_config = yaml.safe_load(f)

    # Create environment
    env = OfficeBotEnv(env_config, seed=42, eval_mode=True)

    # Load model
    model_path = "models/ppo/seed42/best/best_model.zip"
    model = PPO.load(model_path)
    print(f"Loaded model from: {model_path}\n")

    results = []

    for ep in range(n_episodes):
        obs, info = env.reset()
        episode_data = {
            'episode': ep + 1,
            'battery_history': [info['battery']],
            'rewards': [],
            'actions': [],
            'positions': [env.agent_pos],
            'tasks_completed': 0,
            'recharge_task_injected': False,
            'visited_charger': False,
            'recharge_count': 0,
        }

        done = False
        step = 0

        while not done and step < 500:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            episode_data['battery_history'].append(info['battery'])
            episode_data['rewards'].append(reward)
            episode_data['actions'].append(action)
            episode_data['positions'].append(env.agent_pos)

            # Check if at charging station
            if env.agent_pos == env.charging_pos:
                episode_data['visited_charger'] = True
                if action == 5:  # INTERACT
                    episode_data['recharge_count'] += 1

            # Check if recharge task was injected
            if env.current_task and env.current_task.task_type == TaskType.BATTERY_RECHARGE:
                episode_data['recharge_task_injected'] = True

            done = terminated or truncated
            step += 1

        episode_data['total_steps'] = step
        episode_data['final_battery'] = info['battery']
        episode_data['min_battery'] = min(episode_data['battery_history'])
        episode_data['max_battery'] = max(episode_data['battery_history'])
        episode_data['total_reward'] = sum(episode_data['rewards'])
        episode_data['tasks_completed'] = info['completed_tasks']
        episode_data['termination_reason'] = info.get('termination_reason')

        results.append(episode_data)

        # Print episode summary
        print(f"Episode {ep+1}:")
        print(f"  Steps: {episode_data['total_steps']}")
        print(f"  Total Reward: {episode_data['total_reward']:.2f}")
        print(f"  Battery: {episode_data['min_battery']:.3f} -> {episode_data['final_battery']:.3f} (min: {episode_data['min_battery']:.3f})")
        print(f"  Visited Charger: {episode_data['visited_charger']}")
        print(f"  Recharge Count: {episode_data['recharge_count']}")
        print(f"  Recharge Task Injected: {episode_data['recharge_task_injected']}")
        print(f"  Tasks Completed: {episode_data['tasks_completed']}")
        print(f"  Termination: {episode_data['termination_reason']}")
        print()

    # Overall statistics
    print("=" * 70)
    print("OVERALL STATISTICS")
    print("=" * 70)

    total_episodes = len(results)
    visited_charger = sum(1 for r in results if r['visited_charger'])
    recharged = sum(1 for r in results if r['recharge_count'] > 0)
    recharge_injected = sum(1 for r in results if r['recharge_task_injected'])
    battery_died = sum(1 for r in results if r['termination_reason'] == 'battery_dead')

    avg_min_battery = np.mean([r['min_battery'] for r in results])
    avg_final_battery = np.mean([r['final_battery'] for r in results])
    avg_recharge_count = np.mean([r['recharge_count'] for r in results])
    avg_reward = np.mean([r['total_reward'] for r in results])
    avg_tasks = np.mean([r['tasks_completed'] for r in results])

    print(f"Episodes evaluated: {total_episodes}")
    print(f"Visited charging station: {visited_charger}/{total_episodes} ({100*visited_charger/total_episodes:.1f}%)")
    print(f"Performed recharge: {recharged}/{total_episodes} ({100*recharged/total_episodes:.1f}%)")
    print(f"Recharge task injected: {recharge_injected}/{total_episodes} ({100*recharge_injected/total_episodes:.1f}%)")
    print(f"Battery died: {battery_died}/{total_episodes} ({100*battery_died/total_episodes:.1f}%)")
    print()
    print(f"Average min battery: {avg_min_battery:.3f}")
    print(f"Average final battery: {avg_final_battery:.3f}")
    print(f"Average recharge count: {avg_recharge_count:.2f}")
    print(f"Average reward: {avg_reward:.2f}")
    print(f"Average tasks completed: {avg_tasks:.2f}")
    print()

    # Check if agent learned to recharge
    if avg_min_battery < 0.1 and battery_died == 0:
        print("[SUCCESS] Agent learned to recharge before battery dies!")
    elif battery_died == 0 and avg_final_battery > 0.5:
        print("[PARTIAL] Agent survives but may not be actively managing battery")
    else:
        print("[FAILURE] Agent did not learn proper battery management")

    return results

if __name__ == '__main__':
    results = evaluate_battery_behavior(n_episodes=10)
