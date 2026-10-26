# tests/test_smoke.py
"""
Smoke tests — fast sanity checks that run in seconds.

Run:  pytest tests/test_smoke.py -v
"""

import os
import yaml
import numpy as np
import pytest

# ── helpers ──────────────────────────────────────────────────────────────────

CONFIG_PATH = os.path.join(os.path.dirname(__file__), '..', 'configs', 'env_config.yaml')


def _load_env():
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    from environment.env import OfficeBotEnv
    return OfficeBotEnv(config=cfg, seed=0)


# ── tests ─────────────────────────────────────────────────────────────────────

def test_obs_shape():
    """Observation vector must be (53,) float32."""
    env = _load_env()
    obs, _ = env.reset()
    assert obs.shape == (53,), f"Expected (53,), got {obs.shape}"
    assert obs.dtype == np.float32
    env.close()


def test_random_steps_no_crash():
    """20 random steps must complete without raising an exception."""
    env = _load_env()
    env.reset()
    for _ in range(20):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        assert obs.shape == (53,)
        assert isinstance(reward, float)
        assert isinstance(info, dict)
        if terminated or truncated:
            env.reset()
    env.close()


def test_ppo_model_predict():
    """Load the best PPO model and get a valid action for one observation."""
    from agents.agent_factory import load_agent
    model_path = 'models/ppo/seed42/best/best_model.zip'
    if not os.path.exists(model_path):
        pytest.skip(f'PPO model not found at {model_path} — run training first')
    model = load_agent('ppo', model_path)
    env = _load_env()
    obs, _ = env.reset()
    action, _states = model.predict(obs, deterministic=True)
    assert int(action) in range(env.action_space.n), \
        f"Action {action} out of range [0, {env.action_space.n})"
    env.close()


def test_env_reset_reproducible():
    """Two resets with the same env give obs in valid range."""
    env = _load_env()
    obs1, _ = env.reset()
    obs2, _ = env.reset()
    assert np.all(obs1 >= 0.0) and np.all(obs1 <= 1.0), "Obs out of [0,1]"
    assert np.all(obs2 >= 0.0) and np.all(obs2 <= 1.0), "Obs out of [0,1]"
    env.close()


def test_rgb_array_render():
    """rgb_array render mode must return a valid H×W×3 uint8 array."""
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    from environment.env import OfficeBotEnv
    env = OfficeBotEnv(config=cfg, seed=0, render_mode='rgb_array')
    env.reset()
    img = env.render()
    assert img is not None, "render() returned None in rgb_array mode"
    assert img.ndim == 3 and img.shape[2] == 3, f"Bad image shape: {img.shape}"
    assert img.dtype == np.uint8
    env.close()
