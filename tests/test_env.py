# tests/test_env.py

import pytest
import yaml
import numpy as np
from gymnasium.utils.env_checker import check_env
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


def test_env_creation(env):
    """Test that environment can be created."""
    assert env is not None


def test_gymnasium_api_compliance(env):
    """Check that environment passes SB3/Gymnasium checks."""
    check_env(env, warn=True)


def test_observation_space(env):
    """Test observation space shape and bounds."""
    obs, info = env.reset()
    assert obs.shape == (53,), f"Expected obs shape (53,), got {obs.shape}"
    assert obs.dtype == np.float32, f"Expected dtype float32, got {obs.dtype}"
    assert np.all(obs >= 0.0), "Observation values should be >= 0.0"
    assert np.all(obs <= 1.0), "Observation values should be <= 1.0"


def test_action_space(env):
    """Test action space has correct number of actions."""
    assert env.action_space.n == 6, f"Expected 6 actions, got {env.action_space.n}"


def test_reset_returns_valid_obs(env):
    """Test that reset returns valid observation."""
    obs, info = env.reset()
    assert env.observation_space.contains(obs), "Reset observation not in observation space"


def test_step_returns_5_tuple(env):
    """Test that step returns proper 5-tuple."""
    env.reset()
    result = env.step(0)
    assert len(result) == 5, f"Expected 5-tuple from step, got {len(result)}"
    obs, reward, terminated, truncated, info = result
    assert isinstance(reward, (float, np.floating)), f"Reward should be float, got {type(reward)}"
    assert isinstance(terminated, bool), f"Terminated should be bool, got {type(terminated)}"
    assert isinstance(truncated, bool), f"Truncated should be bool, got {type(truncated)}"
    assert isinstance(info, dict), f"Info should be dict, got {type(info)}"


def test_episode_terminates(env):
    """Test that episodes terminate within reasonable time."""
    env.reset()
    done = False
    steps = 0
    while not done and steps < 1000:
        obs, rew, terminated, truncated, info = env.step(env.action_space.sample())
        done = terminated or truncated
        steps += 1
    assert done, "Episode should terminate within 1000 steps"


def test_collision_penalty(env, config):
    """Test that moving into a wall gives negative reward."""
    env.reset()
    # Force agent to face a wall and move forward
    env.agent_pos = (1, 1)
    env.agent_dir = 3  # facing up (towards wall at row 0)
    obs, reward, _, _, info = env.step(2)  # MOVE_FORWARD into wall
    assert reward <= config['rewards']['collision'], \
        f"Collision reward {reward} should be <= {config['rewards']['collision']}"
    assert info['collision_count'] >= 1, "Collision count should increase"


def test_battery_drains(env):
    """Test that battery drains over time."""
    env.reset()
    initial_battery = env.battery
    for _ in range(10):
        env.step(2)  # MOVE_FORWARD
    assert env.battery < initial_battery, "Battery should drain after multiple moves"


def test_task_queue_non_empty(env):
    """Test that environment starts with tasks."""
    env.reset()
    assert env.current_task is not None, "Environment should start with a current task"
