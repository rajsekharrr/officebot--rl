# agents/agent_factory.py

import yaml
from stable_baselines3 import PPO, DQN, A2C

ALGO_MAP = {
    'ppo': PPO,
    'dqn': DQN,
    'a2c': A2C,
}


def create_agent(algo_name: str, env, config: dict, seed: int = 42,
                 tensorboard_log: str = None):
    """
    Instantiate an SB3 model from a config dict.

    Args:
        algo_name: 'ppo', 'dqn', or 'a2c'
        env: Gymnasium environment instance (wrapped in Monitor)
        config: loaded YAML config dict
        seed: random seed
        tensorboard_log: directory for TensorBoard logs

    Returns:
        Instantiated SB3 model, not yet trained.
    """
    if algo_name not in ALGO_MAP:
        raise ValueError(f"Unknown algorithm: {algo_name}. Choose from {list(ALGO_MAP.keys())}")

    AlgoClass = ALGO_MAP[algo_name]
    model_kwargs = config.get('model_kwargs', {}).copy()

    # Handle net_arch: SB3 expects net_arch as a list of ints
    if 'policy_kwargs' in model_kwargs:
        pk = model_kwargs['policy_kwargs'].copy()
        if 'net_arch' in pk:
            # Convert net_arch to list of ints (in case it's loaded as strings or floats)
            pk['net_arch'] = [int(x) for x in pk['net_arch']]
        model_kwargs['policy_kwargs'] = pk

    model = AlgoClass(
        policy='MlpPolicy',
        env=env,
        seed=seed,
        verbose=1,
        tensorboard_log=tensorboard_log,
        **model_kwargs,
    )
    return model


def load_agent(algo_name: str, model_path: str, env=None):
    """
    Load a previously saved SB3 model from disk.

    Args:
        algo_name: 'ppo', 'dqn', or 'a2c'
        model_path: path to the .zip model file (with or without .zip extension)
        env: optional environment to set on the loaded model

    Returns:
        Loaded SB3 model
    """
    if algo_name not in ALGO_MAP:
        raise ValueError(f"Unknown algorithm: {algo_name}. Choose from {list(ALGO_MAP.keys())}")

    AlgoClass = ALGO_MAP[algo_name]

    # SB3 load automatically adds .zip if not present
    if env is not None:
        model = AlgoClass.load(model_path, env=env)
    else:
        model = AlgoClass.load(model_path)

    return model


def load_config(config_path: str) -> dict:
    """
    Load a YAML configuration file.

    Args:
        config_path: path to YAML config file

    Returns:
        Dictionary containing configuration
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config
