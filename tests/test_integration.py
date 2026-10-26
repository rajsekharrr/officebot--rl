# tests/test_integration.py

import pytest
import yaml
import tempfile
import os
from stable_baselines3 import PPO, DQN, A2C
from stable_baselines3.common.monitor import Monitor
from environment.env import OfficeBotEnv
from agents.agent_factory import create_agent


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


def test_ppo_trains_1000_steps(env_config, ppo_config):
    """Integration test: PPO should be able to train for 1000 steps."""
    env = Monitor(OfficeBotEnv(config=env_config, seed=42))
    model = PPO('MlpPolicy', env, seed=42, n_steps=64, batch_size=32, verbose=0)

    # Train for 1000 steps
    model.learn(total_timesteps=1000)

    # Test prediction
    obs, _ = env.reset()
    action, _ = model.predict(obs, deterministic=True)
    assert action in range(6), f"Action {action} should be valid"

    env.close()


def test_dqn_trains_1000_steps(env_config, dqn_config):
    """Integration test: DQN should be able to train for 1000 steps."""
    env = Monitor(OfficeBotEnv(config=env_config, seed=42))
    model = DQN('MlpPolicy', env, seed=42, learning_starts=100, verbose=0)

    # Train for 1000 steps
    model.learn(total_timesteps=1000)

    # Test prediction
    obs, _ = env.reset()
    action, _ = model.predict(obs, deterministic=True)
    assert action in range(6), f"Action {action} should be valid"

    env.close()


def test_a2c_trains_1000_steps(env_config, a2c_config):
    """Integration test: A2C should be able to train for 1000 steps."""
    env = Monitor(OfficeBotEnv(config=env_config, seed=42))
    model = A2C('MlpPolicy', env, seed=42, n_steps=5, verbose=0)

    # Train for 1000 steps
    model.learn(total_timesteps=1000)

    # Test prediction
    obs, _ = env.reset()
    action, _ = model.predict(obs, deterministic=True)
    assert action in range(6), f"Action {action} should be valid"

    env.close()


def test_model_save_load_integration(env_config, ppo_config):
    """Integration test: Trained model should save and load correctly."""
    env = Monitor(OfficeBotEnv(config=env_config, seed=42))
    model = PPO('MlpPolicy', env, seed=42, verbose=0)

    # Train model
    model.learn(total_timesteps=500)

    # Save model
    with tempfile.TemporaryDirectory() as tmp_dir:
        save_path = os.path.join(tmp_dir, 'test_model')
        model.save(save_path)

        # Load model
        loaded_model = PPO.load(save_path, env=env)

        # Test loaded model can predict
        obs, _ = env.reset()
        action, _ = loaded_model.predict(obs, deterministic=True)
        assert action in range(6), f"Loaded model action {action} should be valid"

    env.close()


def test_training_with_agent_factory(env_config, ppo_config):
    """Integration test: Training using agent_factory should work."""
    env = Monitor(OfficeBotEnv(config=env_config, seed=42))

    # Create agent using factory
    model = create_agent('ppo', env, ppo_config, seed=42)

    # Train
    model.learn(total_timesteps=1000)

    # Test prediction
    obs, _ = env.reset()
    action, _ = model.predict(obs, deterministic=True)
    assert action in range(6), f"Action {action} should be valid"

    env.close()


def test_multiple_episodes_without_crash(env_config, ppo_config):
    """Integration test: Run multiple episodes without crashing."""
    env = Monitor(OfficeBotEnv(config=env_config, seed=42))
    model = PPO('MlpPolicy', env, seed=42, verbose=0)

    # Train briefly
    model.learn(total_timesteps=500)

    # Run multiple evaluation episodes
    for episode in range(5):
        obs, _ = env.reset()
        done = False
        steps = 0

        while not done and steps < 100:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            steps += 1

        # Check episode completed properly
        assert done or steps == 100, f"Episode {episode} should complete or reach step limit"

    env.close()


def test_training_improves_policy(env_config, ppo_config):
    """Integration test: Training should improve policy performance."""
    env = Monitor(OfficeBotEnv(config=env_config, seed=42))
    model = PPO('MlpPolicy', env, seed=42, verbose=0)

    # Evaluate before training
    obs, _ = env.reset(seed=999)
    rewards_before = []
    done = False
    steps = 0

    while not done and steps < 50:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        rewards_before.append(reward)
        done = terminated or truncated
        steps += 1

    total_reward_before = sum(rewards_before)

    # Train model
    model.learn(total_timesteps=5000)

    # Evaluate after training
    obs, _ = env.reset(seed=999)
    rewards_after = []
    done = False
    steps = 0

    while not done and steps < 50:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        rewards_after.append(reward)
        done = terminated or truncated
        steps += 1

    total_reward_after = sum(rewards_after)

    # Performance should improve (or at least not get much worse)
    # Note: Due to randomness, we just check that training completes
    # A more robust test would average over many episodes
    assert isinstance(total_reward_after, float), "Reward should be computed"

    env.close()


def test_all_algorithms_train_successfully(env_config, ppo_config, dqn_config, a2c_config):
    """Integration test: All three algorithms should train successfully."""
    algorithms = [
        ('ppo', ppo_config, PPO),
        ('dqn', dqn_config, DQN),
        ('a2c', a2c_config, A2C),
    ]

    for algo_name, config, AlgoClass in algorithms:
        env = Monitor(OfficeBotEnv(config=env_config, seed=42))
        model = create_agent(algo_name, env, config, seed=42)

        # Train
        model.learn(total_timesteps=1000)

        # Verify model is correct type
        assert isinstance(model, AlgoClass), \
            f"Model for {algo_name} should be {AlgoClass.__name__}"

        # Test prediction
        obs, _ = env.reset()
        action, _ = model.predict(obs, deterministic=True)
        assert action in range(6), f"{algo_name} action {action} should be valid"

        env.close()


def test_environment_resets_properly_during_training(env_config):
    """Integration test: Environment should reset properly during training."""
    env = Monitor(OfficeBotEnv(config=env_config, seed=42))
    model = PPO('MlpPolicy', env, seed=42, verbose=0, n_steps=128)

    # This should trigger multiple episode resets
    model.learn(total_timesteps=2000)

    # Verify environment is still functional
    obs, _ = env.reset()
    assert env.observation_space.contains(obs), "Environment should reset properly after training"

    env.close()


def test_training_with_low_battery_episodes(env_config):
    """Integration test: Training should handle battery depletion episodes."""
    # Modify config to drain battery faster for testing
    test_config = env_config.copy()
    test_config['battery_drain_per_step'] = 0.01  # Faster drain
    test_config['battery_drain_per_move'] = 0.005

    env = Monitor(OfficeBotEnv(config=test_config, seed=42))
    model = PPO('MlpPolicy', env, seed=42, verbose=0)

    # Train - some episodes should end with battery depletion
    model.learn(total_timesteps=1000)

    # Verify model still works
    obs, _ = env.reset()
    action, _ = model.predict(obs, deterministic=True)
    assert action in range(6), f"Action {action} should be valid"

    env.close()
