#!/usr/bin/env python3
"""Test battery system implementation."""

import yaml
import numpy as np
from environment.env import OfficeBotEnv

def test_battery_drain():
    """Test that battery drains correctly."""
    print("=" * 70)
    print("TEST 1: Battery Drain")
    print("=" * 70)

    with open('configs/env_config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    env = OfficeBotEnv(config, seed=42)
    obs, info = env.reset()

    initial_battery = info['battery']
    print(f"Initial battery: {initial_battery:.3f}")

    # Test step drain (action 0 = TURN_LEFT, no move)
    obs, reward, terminated, truncated, info = env.step(0)
    battery_after_turn = info['battery']
    drain_per_step = initial_battery - battery_after_turn
    print(f"After TURN_LEFT: {battery_after_turn:.3f} (drain: {drain_per_step:.5f})")
    print(f"Expected drain: {config['battery_drain_per_step']:.5f}")
    assert abs(drain_per_step - config['battery_drain_per_step']) < 1e-6, "Step drain incorrect!"

    # Test move drain (action 2 = MOVE_FORWARD)
    battery_before = battery_after_turn
    obs, reward, terminated, truncated, info = env.step(2)
    battery_after_move = info['battery']
    drain_per_move = battery_before - battery_after_move
    expected_drain = config['battery_drain_per_step'] + config['battery_drain_per_move']
    print(f"After MOVE_FORWARD: {battery_after_move:.3f} (drain: {drain_per_move:.5f})")
    print(f"Expected drain: {expected_drain:.5f}")
    assert abs(drain_per_move - expected_drain) < 1e-6, "Move drain incorrect!"

    print("[PASS] Battery drain working correctly\n")

def test_battery_recharge():
    """Test that battery recharges at charging station."""
    print("=" * 70)
    print("TEST 2: Battery Recharge")
    print("=" * 70)

    with open('configs/env_config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    env = OfficeBotEnv(config, seed=42)
    obs, info = env.reset()

    # Manually set battery low and position at charging station
    env.battery = 0.3
    env.agent_pos = env.charging_pos

    print(f"Battery before recharge: {env.battery:.3f}")
    print(f"Agent at charging station: {env.agent_pos}")

    # Use INTERACT action (5) at charging station
    obs, reward, terminated, truncated, info = env.step(5)

    battery_after = info['battery']
    expected_gain = config['charge_rate']
    expected_battery = 0.3 - config['battery_drain_per_step'] + expected_gain

    print(f"Battery after INTERACT: {battery_after:.3f}")
    print(f"Expected battery: ~{expected_battery:.3f}")
    print(f"Reward received: {reward:.2f}")

    # Check that battery increased
    assert battery_after > 0.3 - config['battery_drain_per_step'], "Battery didn't recharge!"

    # Check reward structure
    expected_reward = (
        config['rewards']['step_penalty'] +
        expected_gain * config['rewards']['battery_efficiency_coef']
    )
    print(f"Expected reward: ~{expected_reward:.2f}")

    print("[PASS] Battery recharge working correctly\n")

def test_battery_critical_injection():
    """Test that recharge task is injected when battery is critical."""
    print("=" * 70)
    print("TEST 3: Battery Critical Task Injection")
    print("=" * 70)

    with open('configs/env_config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    env = OfficeBotEnv(config, seed=42)
    obs, info = env.reset()

    initial_task = env.current_task
    print(f"Initial task: {initial_task.task_type if initial_task else 'None'}")
    print(f"Initial queue length: {len(env.task_queue)}")

    # Set battery below critical threshold
    env.battery = 0.04
    print(f"\nBattery set to critical: {env.battery:.3f}")
    print(f"Critical threshold: {config['battery_critical_threshold']:.3f}")

    # Take a step to trigger the check
    obs, reward, terminated, truncated, info = env.step(0)

    print(f"\nCurrent task after step: {env.current_task.task_type if env.current_task else 'None'}")
    print(f"Queue length: {len(env.task_queue)}")

    # Check that current task is now BATTERY_RECHARGE
    from environment.tasks import TaskType
    assert env.current_task is not None, "No current task!"
    assert env.current_task.task_type == TaskType.BATTERY_RECHARGE, "Recharge task not injected!"

    # Check that previous task was pushed back
    if initial_task:
        assert env.task_queue[0] == initial_task, "Previous task not pushed back to queue!"

    print("[PASS] Battery critical task injection working correctly\n")

def test_battery_depletion_terminates():
    """Test that battery depletion ends the episode."""
    print("=" * 70)
    print("TEST 4: Battery Depletion Termination")
    print("=" * 70)

    with open('configs/env_config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    env = OfficeBotEnv(config, seed=42)
    obs, info = env.reset()

    # Set battery very low
    env.battery = 0.001
    print(f"Battery set to: {env.battery:.5f}")

    # Take a step that will deplete battery
    obs, reward, terminated, truncated, info = env.step(0)

    print(f"Battery after step: {info['battery']:.5f}")
    print(f"Terminated: {terminated}")
    print(f"Termination reason: {info.get('termination_reason')}")
    print(f"Reward: {reward:.2f}")

    assert terminated, "Episode not terminated!"
    assert info['termination_reason'] == 'battery_dead', "Wrong termination reason!"
    assert info['battery'] == 0.0, "Battery not at 0!"

    # Check penalty was applied
    expected_penalty = config['rewards']['battery_depleted']
    print(f"Expected penalty in reward: {expected_penalty:.2f}")

    print("[PASS] Battery depletion termination working correctly\n")

def test_full_recharge_cycle():
    """Test a full recharge cycle with task completion."""
    print("=" * 70)
    print("TEST 5: Full Recharge Cycle")
    print("=" * 70)

    with open('configs/env_config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    env = OfficeBotEnv(config, seed=42)
    obs, info = env.reset()

    # Trigger critical battery and recharge task
    env.battery = 0.04
    env.agent_pos = env.charging_pos

    print(f"Starting battery: {env.battery:.3f}")
    print(f"Agent at charging station: {env.agent_pos}")

    # Step to trigger recharge task injection
    obs, reward, terminated, truncated, info = env.step(0)

    from environment.tasks import TaskType
    print(f"\nCurrent task: {env.current_task.task_type if env.current_task else 'None'}")
    assert env.current_task.task_type == TaskType.BATTERY_RECHARGE

    # Now recharge by using INTERACT multiple times
    recharge_count = 0
    max_recharges = 20

    while env.battery < 0.8 and recharge_count < max_recharges:
        obs, reward, terminated, truncated, info = env.step(5)  # INTERACT
        recharge_count += 1
        print(f"Recharge {recharge_count}: battery={info['battery']:.3f}, reward={reward:.2f}")

        if terminated:
            print("Episode terminated unexpectedly!")
            break

    print(f"\nFinal battery: {info['battery']:.3f}")
    print(f"Task completed: {env.current_task is None or env.current_task.task_type != TaskType.BATTERY_RECHARGE}")
    print(f"Completed tasks: {info['completed_tasks']}")

    # Task should be completed once battery >= 0.8
    assert info['battery'] >= 0.8, "Battery not recharged to threshold!"

    print("[PASS] Full recharge cycle working correctly\n")

if __name__ == '__main__':
    test_battery_drain()
    test_battery_recharge()
    test_battery_critical_injection()
    test_battery_depletion_terminates()
    test_full_recharge_cycle()

    print("=" * 70)
    print("ALL BATTERY TESTS PASSED!")
    print("=" * 70)
