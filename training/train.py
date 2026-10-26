# training/train.py

import argparse
import os
import time
import yaml
import numpy as np
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import (
    EvalCallback, CheckpointCallback, CallbackList
)

from environment.env import OfficeBotEnv
from agents.agent_factory import create_agent, load_agent
from training.callbacks import MetricLoggerCallback


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train OfficeBot RL agent')
    parser.add_argument('--algo', type=str, required=True,
                        choices=['ppo', 'dqn', 'a2c'],
                        help='Algorithm to use: ppo, dqn, or a2c')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to algorithm config YAML file')
    parser.add_argument('--timesteps', type=int, default=None,
                        help='Total training timesteps (overrides config)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to existing model .zip to resume training from')
    parser.add_argument('--env-config', type=str, default='configs/env_config.yaml',
                        help='Path to environment config YAML file')
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    """Main training function."""
    args = parse_args()

    # Load configurations
    print(f"\n{'='*60}")
    print(f"OfficeBot RL Training - {args.algo.upper()}")
    print(f"{'='*60}\n")

    print(f"Loading configurations...")
    env_config = load_config(args.env_config)
    algo_config = load_config(args.config)

    print(f"  - Environment config: {args.env_config}")
    print(f"  - Algorithm config: {args.config}")
    print(f"  - Algorithm: {args.algo.upper()}")
    print(f"  - Seed: {args.seed}")

    # Set seeds for reproducibility
    np.random.seed(args.seed)

    # Create training environment (wrapped in Monitor for SB3)
    print(f"\nCreating training environment...")
    train_env = Monitor(OfficeBotEnv(config=env_config, seed=args.seed))
    print(f"  - Training environment created with seed {args.seed}")

    # Create separate evaluation environment (fixed seed, eval mode)
    print(f"Creating evaluation environment...")
    eval_env = Monitor(OfficeBotEnv(
        config=env_config,
        seed=args.seed + 1000,
        eval_mode=True
    ))
    print(f"  - Evaluation environment created with seed {args.seed + 1000}")

    # Determine total timesteps
    total_timesteps = args.timesteps if args.timesteps is not None else algo_config['total_timesteps']
    print(f"\nTraining configuration:")
    print(f"  - Total timesteps: {total_timesteps:,}")
    print(f"  - Eval frequency: {algo_config['eval_freq']:,}")
    print(f"  - Checkpoint frequency: {algo_config['checkpoint_freq']:,}")
    print(f"  - Eval episodes: {algo_config['n_eval_episodes']}")

    # Setup directories
    best_model_dir = f"models/{args.algo}/seed{args.seed}/best/"
    checkpoint_dir = f"models/{args.algo}/seed{args.seed}/checkpoints/"
    log_dir = f"logs/metrics/{args.algo}/seed{args.seed}/"
    tensorboard_dir = f"logs/tensorboard/{args.algo}_seed{args.seed}"

    os.makedirs(best_model_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    print(f"\nOutput directories:")
    print(f"  - Best model: {best_model_dir}")
    print(f"  - Checkpoints: {checkpoint_dir}")
    print(f"  - Metrics: {log_dir}")
    print(f"  - TensorBoard: {tensorboard_dir}")

    # Instantiate or resume model
    if args.resume:
        print(f"\nResuming training from: {args.resume}")
        model = load_agent(args.algo, args.resume, env=train_env)
        print(f"  - Model loaded successfully")
    else:
        print(f"\nInstantiating {args.algo.upper()} model...")
        model = create_agent(
            algo_name=args.algo,
            env=train_env,
            config=algo_config,
            seed=args.seed,
            tensorboard_log=tensorboard_dir,
        )
        print(f"  - Model created successfully")
        print(f"  - Policy: {model.policy.__class__.__name__}")
        print(f"  - Learning rate: {model.learning_rate}")

    # Setup callbacks
    print(f"\nSetting up callbacks...")

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=best_model_dir,
        log_path=log_dir,
        eval_freq=algo_config['eval_freq'],
        n_eval_episodes=algo_config['n_eval_episodes'],
        deterministic=True,
        render=False,
        verbose=1,
    )
    print(f"  - EvalCallback: eval every {algo_config['eval_freq']:,} steps")

    checkpoint_callback = CheckpointCallback(
        save_freq=algo_config['checkpoint_freq'],
        save_path=checkpoint_dir,
        name_prefix=f"officebot_{args.algo}",
        save_replay_buffer=False,
        save_vecnormalize=False,
        verbose=1,
    )
    print(f"  - CheckpointCallback: save every {algo_config['checkpoint_freq']:,} steps")

    metric_callback = MetricLoggerCallback(
        log_dir=log_dir,
        log_freq=5000,
        verbose=0,
    )
    print(f"  - MetricLoggerCallback: log every 5000 steps")

    callbacks = CallbackList([eval_callback, checkpoint_callback, metric_callback])

    # Train
    print(f"\n{'='*60}")
    print(f"Starting training...")
    print(f"{'='*60}\n")

    start_time = time.time()

    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            progress_bar=True,
            reset_num_timesteps=False if args.resume else True,
        )
    except KeyboardInterrupt:
        print(f"\n\nTraining interrupted by user.")

    end_time = time.time()
    training_time = end_time - start_time

    # Save final model
    final_model_path = f"models/{args.algo}/seed{args.seed}/final_model"
    model.save(final_model_path)

    print(f"\n{'='*60}")
    print(f"Training Complete!")
    print(f"{'='*60}")
    print(f"  - Total training time: {training_time/60:.2f} minutes")
    print(f"  - Final model saved to: {final_model_path}.zip")
    print(f"  - Best model saved in: {best_model_dir}")
    print(f"  - View logs with: tensorboard --logdir {tensorboard_dir}")
    print(f"{'='*60}\n")

    # Close environments
    train_env.close()
    eval_env.close()


if __name__ == '__main__':
    main()
