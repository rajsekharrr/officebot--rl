# training/callbacks.py

import os
import json
from stable_baselines3.common.callbacks import BaseCallback


class MetricLoggerCallback(BaseCallback):
    """Periodically saves a JSON snapshot of key training metrics."""

    def __init__(self, log_dir: str, log_freq: int = 5000, verbose: int = 0):
        super().__init__(verbose)
        self.log_dir = log_dir
        self.log_freq = log_freq
        self.metrics_history = []

    def _on_step(self) -> bool:
        """
        This method is called after each step in the environment.

        Returns:
            bool: If the callback returns False, training is aborted early.
        """
        if self.n_calls % self.log_freq == 0:
            # Get info from the last completed episode
            info_list = self.locals.get('infos', [{}])
            if info_list:
                info = info_list[0]
                snapshot = {
                    'timestep': self.num_timesteps,
                    'n_calls': self.n_calls,
                    'battery': info.get('battery', None),
                    'completed_tasks': info.get('completed_tasks', None),
                    'collision_count': info.get('collision_count', None),
                    'failed_tasks': info.get('failed_tasks', None),
                    'step_count': info.get('step_count', None),
                    'battery_low': info.get('battery_low', None),
                    'task_queue_length': info.get('task_queue_length', None),
                }
                self.metrics_history.append(snapshot)
                self._save_metrics()

                if self.verbose > 0:
                    print(f"[MetricLogger] Step {self.num_timesteps}: "
                          f"Battery={info.get('battery', 'N/A'):.2f}, "
                          f"Completed={info.get('completed_tasks', 'N/A')}, "
                          f"Collisions={info.get('collision_count', 'N/A')}")

        return True

    def _save_metrics(self):
        """Save metrics history to JSON file."""
        os.makedirs(self.log_dir, exist_ok=True)
        metrics_path = os.path.join(self.log_dir, 'training_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)

        if self.verbose > 0:
            print(f"[MetricLogger] Saved metrics to {metrics_path}")

    def _on_training_end(self) -> None:
        """Called at the end of training."""
        self._save_metrics()
        if self.verbose > 0:
            print(f"[MetricLogger] Training ended. Total snapshots: {len(self.metrics_history)}")
