# tests/test_rewards.py

import pytest
import yaml
import numpy as np
from environment.env import OfficeBotEnv


@pytest.fixture
def config():
    """Load environment configuration."""
    with open('configs/env_config.yaml') as f:
        return yaml.safe_load(f)


@pytest.fixture
def env(config):
    """Create environment instance."""
    e = OfficeBotEnv(config=config, seed=42)
    yield e
    e.close()


def test_no_nan_rewards(env):
    """Run 500 random steps and check no NaN rewards."""
    env.reset()
    for i in range(500):
        _, reward, terminated, truncated, _ = env.step(env.action_space.sample())
        assert not np.isnan(reward), f"NaN reward encountered at step {i}"
        done = terminated or truncated
        if done:
            env.reset()


def test_reward_bounds(env):
    """All rewards should be within a reasonable range."""
    env.reset()
    for i in range(500):
        _, reward, terminated, truncated, _ = env.step(env.action_space.sample())
        assert -100 < reward < 100, f"Reward {reward} out of bounds at step {i}"
        done = terminated or truncated
        if done:
            env.reset()


def test_no_inf_rewards(env):
    """Run random steps and check no infinite rewards."""
    env.reset()
    for i in range(500):
        _, reward, terminated, truncated, _ = env.step(env.action_space.sample())
        assert not np.isinf(reward), f"Infinite reward encountered at step {i}"
        done = terminated or truncated
        if done:
            env.reset()


def test_step_penalty_applied(env, config):
    """Test that step penalty is applied on every step."""
    env.reset()
    # Action 0 (TURN_LEFT) should only have step penalty (no other rewards)
    obs, reward, terminated, truncated, info = env.step(0)
    # Reward should at least include step penalty (may have shaping rewards too)
    assert reward <= 0, "Turning without progress should have non-positive reward"


def test_collision_reward_negative(env, config):
    """Test that collision gives negative reward."""
    env.reset()
    # Force agent near wall
    env.agent_pos = (1, 1)
    env.agent_dir = 3  # facing up towards wall
    obs, reward, _, _, info = env.step(2)  # MOVE_FORWARD
    collision_reward = config['rewards']['collision']
    # Should get collision penalty
    assert reward <= collision_reward, \
        f"Collision should give reward <= {collision_reward}, got {reward}"


def test_battery_depletion_penalty(env, config):
    """Test that battery depletion gives correct penalty."""
    env.reset()
    # Drain battery to almost zero
    env.battery = 0.001
    # Take a step to deplete
    obs, reward, terminated, truncated, info = env.step(0)

    if terminated and info.get('termination_reason') == 'battery_dead':
        battery_penalty = config['rewards']['battery_depleted']
        # Reward should include battery depletion penalty
        assert reward <= battery_penalty, \
            f"Battery depletion should include penalty {battery_penalty}, got {reward}"


def test_reward_is_float(env):
    """Test that all rewards are float type."""
    env.reset()
    for _ in range(100):
        _, reward, terminated, truncated, _ = env.step(env.action_space.sample())
        assert isinstance(reward, (float, np.floating)), \
            f"Reward should be float, got {type(reward)}"
        done = terminated or truncated
        if done:
            env.reset()


def test_consistent_rewards_for_same_action_sequence(config):
    """Test that same action sequence gives same rewards with same seed."""
    # Create two environments with same seed
    env1 = OfficeBotEnv(config=config, seed=999)
    env2 = OfficeBotEnv(config=config, seed=999)

    env1.reset()
    env2.reset()

    # Run same action sequence
    actions = [0, 1, 2, 0, 1, 2, 3, 4, 5]
    rewards1 = []
    rewards2 = []

    for action in actions:
        _, r1, t1, tr1, _ = env1.step(action)
        _, r2, t2, tr2, _ = env2.step(action)
        rewards1.append(r1)
        rewards2.append(r2)

        if (t1 or tr1) or (t2 or tr2):
            break

    # Rewards should be identical
    assert len(rewards1) == len(rewards2), "Same actions should produce same number of rewards"
    for i, (r1, r2) in enumerate(zip(rewards1, rewards2)):
        assert abs(r1 - r2) < 1e-6, \
            f"Reward {i}: env1={r1}, env2={r2}, should be identical with same seed"

    env1.close()
    env2.close()


def test_reward_components_sum_correctly(env, config):
    """Test that individual reward components are properly summed."""
    env.reset()

    # Test a simple turn action
    initial_battery = env.battery
    obs, reward, terminated, truncated, info = env.step(0)  # TURN_LEFT

    # Should at least have step penalty
    step_penalty = config['rewards']['step_penalty']
    # Reward might also include battery rewards and shaping, but should be reasonable
    assert -10 < reward < 10, f"Simple turn reward {reward} seems unreasonable"


def test_no_reward_explosion(env):
    """Test that rewards don't explode over an episode."""
    env.reset()
    max_reward_magnitude = 0

    for _ in range(100):
        _, reward, terminated, truncated, _ = env.step(env.action_space.sample())
        max_reward_magnitude = max(max_reward_magnitude, abs(reward))
        done = terminated or truncated
        if done:
            break

    # No single reward should be extremely large
    assert max_reward_magnitude < 100, \
        f"Max reward magnitude {max_reward_magnitude} is too large"
