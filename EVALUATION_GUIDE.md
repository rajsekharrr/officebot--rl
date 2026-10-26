# OfficeBot RL - Evaluation and Visualization Guide

This guide covers the evaluation and visualization scripts for analyzing trained OfficeBot RL models.

## Overview

The evaluation pipeline consists of three main components:

1. **`evaluation/evaluate.py`** - Evaluate a single trained model
2. **`evaluation/compare.py`** - Compare multiple algorithms across seeds
3. **`visualization/plot_results.py`** - Generate publication-quality plots

---

## 1. Evaluating a Trained Model

### Basic Usage

```bash
python evaluation/evaluate.py --algo ppo --seed 42 --episodes 50
```

### Command-Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--algo` | Algorithm (ppo, dqn, a2c) | Required |
| `--seed` | Seed used during training | 42 |
| `--episodes` | Number of evaluation episodes | 50 |
| `--model` | Model type (best, final) | best |
| `--render` | Render episodes visually | False |
| `--env-config` | Environment config path | configs/env_config.yaml |

### Examples

```bash
# Evaluate best PPO model (seed 42) with 10 episodes
python evaluation/evaluate.py --algo ppo --seed 42 --episodes 10

# Evaluate final DQN model with rendering
python evaluation/evaluate.py --algo dqn --seed 100 --episodes 5 --model final --render

# Evaluate A2C with custom episode count
python evaluation/evaluate.py --algo a2c --seed 200 --episodes 100 --model best
```

### Output

The script produces:

1. **Terminal output**: Formatted table with all metrics
2. **JSON metrics**: `logs/metrics/{algo}/seed{seed}/evaluation_metrics.json`
3. **Raw results**: `logs/metrics/{algo}/seed{seed}/evaluation_results.json`

### Metrics Computed (Section 11 of Spec)

- `mean_episode_reward` - Average total reward per episode
- `std_episode_reward` - Standard deviation of rewards
- `success_rate` - Fraction of episodes where all tasks completed
- `task_completion_rate` - Average fraction of tasks completed
- `mean_episode_length` - Average number of steps per episode
- `mean_collision_count` - Average collisions per episode
- `collision_rate` - Collisions per step
- `mean_battery_remaining` - Average battery at episode end
- `battery_efficiency` - Tasks completed per battery used
- `mean_completed_tasks` - Average tasks completed
- `mean_failed_tasks` - Average tasks failed

---

## 2. Comparing Algorithms

### Basic Usage

```bash
python evaluation/compare.py
```

### What It Does

Loads evaluation results for **PPO, DQN, and A2C** across seeds **42, 123, 999** and computes:
- Mean ± standard deviation for each metric
- Min/max values across seeds
- Number of seeds evaluated per algorithm

### Output

1. **Terminal output**: Formatted comparison table and summary statistics
2. **CSV table**: `results/comparison_table.csv`
3. **JSON summary**: `results/experiment_summary.json`

### Example Output

```
================================================================================
ALGORITHM COMPARISON - Mean ± Std across seeds
================================================================================

+------------------------+---------------+---------------+---------------+
| Metric                 | A2C           | DQN           | PPO           |
+========================+===============+===============+===============+
| Mean Episode Reward    | -34.50 ± 0.15 | -35.20 ± 0.32 | -34.70 ± 0.10 |
| Success Rate           | 0.00% ± 0.00% | 0.00% ± 0.00% | 0.00% ± 0.00% |
| Task Completion Rate   | 0.00% ± 0.00% | 0.00% ± 0.00% | 0.00% ± 0.00% |
...
```

---

## 3. Generating Plots

### Basic Usage

```bash
# Generate all plots for all algorithms
python visualization/plot_results.py --algo all

# Generate plots for PPO only
python visualization/plot_results.py --algo ppo
```

### Command-Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--algo` | Algorithm to plot (ppo, dqn, a2c, all) | all |
| `--seeds` | Seeds to include | 42 123 999 |
| `--output-dir` | Output directory for plots | results/plots |

### Examples

```bash
# Generate all plots with default seeds
python visualization/plot_results.py --algo all

# Generate PPO plots with custom seeds
python visualization/plot_results.py --algo ppo --seeds 42 99 777

# Generate all plots with custom output directory
python visualization/plot_results.py --algo all --output-dir outputs/figures
```

### Generated Plots (Section 11.4 of Spec)

When `--algo all` is used, the following plots are generated:

| Filename | Description |
|----------|-------------|
| `training_curves_ppo.png` | PPO training curve (mean ± std across seeds) |
| `training_curves_dqn.png` | DQN training curve |
| `training_curves_a2c.png` | A2C training curve |
| `training_curves_all.png` | All algorithms on one plot |
| `comparison_bar_chart.png` | 4-panel comparison of key metrics |
| `task_completion_rate.png` | Task completion rate by algorithm |
| `collision_rate.png` | Collision rate by algorithm |
| `battery_efficiency.png` | Battery efficiency by algorithm |

When `--algo <specific>` is used, only that algorithm's training curve is generated.

---

## Complete Workflow Example

### Step 1: Train models with multiple seeds

```bash
# Train PPO with 3 seeds
python training/train.py --algo ppo --seed 42 --config configs/ppo_config.yaml
python training/train.py --algo ppo --seed 123 --config configs/ppo_config.yaml
python training/train.py --algo ppo --seed 999 --config configs/ppo_config.yaml

# Train DQN with 3 seeds
python training/train.py --algo dqn --seed 42 --config configs/dqn_config.yaml
python training/train.py --algo dqn --seed 123 --config configs/dqn_config.yaml
python training/train.py --algo dqn --seed 999 --config configs/dqn_config.yaml

# Train A2C with 3 seeds
python training/train.py --algo a2c --seed 42 --config configs/a2c_config.yaml
python training/train.py --algo a2c --seed 123 --config configs/a2c_config.yaml
python training/train.py --algo a2c --seed 999 --config configs/a2c_config.yaml
```

### Step 2: Evaluate all models

```bash
# Evaluate all PPO models
python evaluation/evaluate.py --algo ppo --seed 42 --episodes 50
python evaluation/evaluate.py --algo ppo --seed 123 --episodes 50
python evaluation/evaluate.py --algo ppo --seed 999 --episodes 50

# Evaluate all DQN models
python evaluation/evaluate.py --algo dqn --seed 42 --episodes 50
python evaluation/evaluate.py --algo dqn --seed 123 --episodes 50
python evaluation/evaluate.py --algo dqn --seed 999 --episodes 50

# Evaluate all A2C models
python evaluation/evaluate.py --algo a2c --seed 42 --episodes 50
python evaluation/evaluate.py --algo a2c --seed 123 --episodes 50
python evaluation/evaluate.py --algo a2c --seed 999 --episodes 50
```

### Step 3: Compare algorithms

```bash
python evaluation/compare.py
```

### Step 4: Generate all plots

```bash
python visualization/plot_results.py --algo all
```

### Step 5: Review results

- Check `results/comparison_table.csv` for metrics comparison
- Check `results/experiment_summary.json` for detailed statistics
- View plots in `results/plots/` directory

---

## Directory Structure After Evaluation

```
officebot/
├── logs/
│   └── metrics/
│       ├── ppo/
│       │   ├── seed42/
│       │   │   ├── evaluation_metrics.json
│       │   │   ├── evaluation_results.json
│       │   │   └── evaluations.npz
│       │   ├── seed123/...
│       │   └── seed999/...
│       ├── dqn/...
│       └── a2c/...
├── results/
│   ├── comparison_table.csv
│   ├── experiment_summary.json
│   └── plots/
│       ├── training_curves_ppo.png
│       ├── training_curves_dqn.png
│       ├── training_curves_a2c.png
│       ├── training_curves_all.png
│       ├── comparison_bar_chart.png
│       ├── task_completion_rate.png
│       ├── collision_rate.png
│       └── battery_efficiency.png
```

---

## Troubleshooting

### "Model not found" error

Make sure you've trained a model first:
```bash
python training/train.py --algo ppo --seed 42 --config configs/ppo_config.yaml
```

### "No metrics found" warning

This is expected if you haven't evaluated a model for that seed yet. Run:
```bash
python evaluation/evaluate.py --algo <algo> --seed <seed> --episodes 50
```

### Missing training curves

Training curves come from the `evaluations.npz` file created during training by `EvalCallback`. If this file doesn't exist, the training curve plot will be skipped.

### Import errors

Make sure all dependencies are installed:
```bash
pip install -r requirements.txt
```

And run scripts with PYTHONPATH set:
```bash
cd D:/officebot
PYTHONPATH=. python evaluation/evaluate.py --algo ppo --seed 42 --episodes 10
```

---

## Dependencies

- `numpy` - Array operations
- `matplotlib` - Plotting
- `pandas` - Data tables
- `tabulate` - Terminal table formatting
- `pyyaml` - Config file loading
- `stable-baselines3` - Model loading
- `gymnasium` - Environment interface

All dependencies are listed in `requirements.txt`.
