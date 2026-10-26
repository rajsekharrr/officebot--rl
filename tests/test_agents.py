# tests/test_agents.py

import pytest
import yaml
import tempfile
import os
from stable_baselines3 import PPO, DQN, A2C
from stable_baselines3.common.monitor import Monitor
from environment.env import OfficeBotEnv
from agents.agent_factory import create_agent, load_agent, ALGO_MAP


@pytest.fixture
def env_config():
    """Load environment configuration."""
    with open('configs/env_config.yaml') as f:
        return yaml.safe_load(f)


@pytest.fixture
def ppo_config():
    """Load PPO configuration."""
    with open('configs/ppo_config.yaml') as f:
        return yaml.safe_load(f)


@pytest.fixture
def dqn_config():
    """Load DQN configuration."""
    with open('configs/dqn_config.yaml') as f:
        return yaml.safe_load(f)


@pytest.fixture
def a2c_config():
    """Load A2C configuration."""
    with open('configs/a2c_config.yaml') as f:
        return yaml.safe_load(f)


@pytest.fixture
def test_env(env_config):
    """Create test environment."""
    env = Monitor(OfficeBotEnv(config=env_config, seed=42))
    yield env
    env.close()


def test_algo_map_contains_all_algorithms():
    """Test that ALGO_MAP contains all supported algorithms."""
    assert 'ppo' in ALGO_MAP, "ALGO_MAP should contain 'ppo'"
    assert 'dqn' in ALGO_MAP, "ALGO_MAP should contain 'dqn'"
    assert 'a2c' in ALGO_MAP, "ALGO_MAP should contain 'a2c'"


def test_algo_map_classes_correct():
    """Test that ALGO_MAP maps to correct SB3 classes."""
    assert ALGO_MAP['ppo'] == PPO, "PPO should map to SB3 PPO class"
    assert ALGO_MAP['dqn'] == DQN, "DQN should map to SB3 DQN class"
    assert ALGO_MAP['a2c'] == A2C, "A2C should map to SB3 A2C class"


def test_create_ppo_agent(test_env, ppo_config):
    """Test that create_agent creates a PPO model correctly."""
    model = create_agent('ppo', test_env, ppo_config, seed=42)
    assert model is not None, "Model should be created"
    assert isinstance(model, PPO), "Model should be PPO instance"


def test_create_dqn_agent(test_env, dqn_config):
    """Test that create_agent creates a DQN model correctly."""
    model = create_agent('dqn', test_env, dqn_config, seed=42)
    assert model is not None, "Model should be created"
    assert isinstance(model, DQN), "Model should be DQN instance"


def test_create_a2c_agent(test_env, a2c_config):
    """Test that create_agent creates an A2C model correctly."""
    model = create_agent('a2c', test_env, a2c_config, seed=42)
    assert model is not None, "Model should be created"
    assert isinstance(model, A2C), "Model should be A2C instance"


def test_create_agent_invalid_algo(test_env, ppo_config):
    """Test that create_agent raises error for invalid algorithm."""
    with pytest.raises(ValueError, match="Unknown algorithm"):
        create_agent('invalid_algo', test_env, ppo_config, seed=42)


def test_agent_can_predict(test_env, ppo_config):
    """Test that created agent can make predictions."""
    model = create_agent('ppo', test_env, ppo_config, seed=42)
    obs, _ = test_env.reset()
    action, _ = model.predict(obs, deterministic=True)
    assert action in range(6), f"Action {action} should be in range [0, 6)"


def test_model_save_and_load_ppo(test_env, ppo_config):
    """Test that PPO model can be saved and loaded."""
    # Create and train model briefly
    model = create_agent('ppo', test_env, ppo_config, seed=42)
    model.learn(total_timesteps=100)

    # Save model
    with tempfile.TemporaryDirectory() as tmp_dir:
        save_path = os.path.join(tmp_dir, 'test_ppo_model')
        model.save(save_path)

        # Load model
        loaded_model = load_agent('ppo', save_path, env=test_env)
        assert loaded_model is not None, "Loaded model should not be None"
        assert isinstance(loaded_model, PPO), "Loaded model should be PPO instance"

        # Test prediction
        obs, _ = test_env.reset()
        action, _ = loaded_model.predict(obs, deterministic=True)
        assert action in range(6), f"Loaded model action {action} should be valid"


def test_model_save_and_load_dqn(test_env, dqn_config):
    """Test that DQN model can be saved and loaded."""
    # Create and train model briefly
    model = create_agent('dqn', test_env, dqn_config, seed=42)
    model.learn(total_timesteps=100)

    # Save model
    with tempfile.TemporaryDirectory() as tmp_dir:
        save_path = os.path.join(tmp_dir, 'test_dqn_model')
        model.save(save_path)

        # Load model
        loaded_model = load_agent('dqn', save_path, env=test_env)
        assert loaded_model is not None, "Loaded model should not be None"
        assert isinstance(loaded_model, DQN), "Loaded model should be DQN instance"

        # Test prediction
        obs, _ = test_env.reset()
        action, _ = loaded_model.predict(obs, deterministic=True)
        assert action in range(6), f"Loaded model action {action} should be valid"


def test_model_save_and_load_a2c(test_env, a2c_config):
    """Test that A2C model can be saved and loaded."""
    # Create and train model briefly
    model = create_agent('a2c', test_env, a2c_config, seed=42)
    model.learn(total_timesteps=100)

    # Save model
    with tempfile.TemporaryDirectory() as tmp_dir:
        save_path = os.path.join(tmp_dir, 'test_a2c_model')
        model.save(save_path)

        # Load model
        loaded_model = load_agent('a2c', save_path, env=test_env)
        assert loaded_model is not None, "Loaded model should not be None"
        assert isinstance(loaded_model, A2C), "Loaded model should be A2C instance"

        # Test prediction
        obs, _ = test_env.reset()
        action, _ = loaded_model.predict(obs, deterministic=True)
        assert action in range(6), f"Loaded model action {action} should be valid"


def test_load_agent_invalid_algo():
    """Test that load_agent raises error for invalid algorithm."""
    with pytest.raises(ValueError, match="Unknown algorithm"):
        load_agent('invalid_algo', 'fake_path.zip')


def test_model_has_correct_seed(test_env, ppo_config):
    """Test that created model has correct seed."""
    model = create_agent('ppo', test_env, ppo_config, seed=999)
    assert model.seed == 999, f"Model seed should be 999, got {model.seed}"


def test_model_has_tensorboard_log(test_env, ppo_config):
    """Test that model has tensorboard logging configured."""
    tensorboard_dir = "./test_tb_logs"
    model = create_agent('ppo', test_env, ppo_config, seed=42, tensorboard_log=tensorboard_dir)
    assert model.tensorboard_log == tensorboard_dir, \
        f"Model tensorboard_log should be {tensorboard_dir}"


def test_loaded_model_predictions_deterministic(test_env, ppo_config):
    """Test that loaded model gives same predictions as original."""
    # Create and train model
    model = create_agent('ppo', test_env, ppo_config, seed=42)
    model.learn(total_timesteps=500)

    # Get prediction from original model
    obs, _ = test_env.reset(seed=42)
    action1, _ = model.predict(obs, deterministic=True)

    # Save and load model
    with tempfile.TemporaryDirectory() as tmp_dir:
        save_path = os.path.join(tmp_dir, 'test_model')
        model.save(save_path)
        loaded_model = load_agent('ppo', save_path, env=test_env)

        # Get prediction from loaded model
        obs, _ = test_env.reset(seed=42)
        action2, _ = loaded_model.predict(obs, deterministic=True)

        assert action1 == action2, \
            f"Original and loaded model should give same prediction: {action1} vs {action2}"
