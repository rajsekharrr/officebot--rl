# OfficeBot RL

A reinforcement learning benchmark where autonomous agents navigate a 15×15 office grid, completing delivery, meeting, and recharge tasks under battery constraints — trained with PPO, DQN, and A2C via Stable-Baselines3.

---

## Installation

```bash
git clone https://github.com/your-repo/officebot.git
cd officebot
pip install stable-baselines3[extra] minigrid pygame pyyaml tabulate numpy matplotlib
```

> Tested on Python 3.13, Windows 11. Linux/macOS work too (drop `pygame-ce` if needed).

---

## Quick Start

### 1 — Train all algorithms

```bash
python -m training.train --algo ppo --seed 42 --config configs/ppo_config.yaml --timesteps 200000
python -m training.train --algo dqn --seed 42 --config configs/dqn_config.yaml --timesteps 200000
python -m training.train --algo a2c --seed 42 --config configs/a2c_config.yaml --timesteps 200000
```

### 2 — Evaluate

```bash
python -m evaluation.evaluate --algo ppo --seed 42 --episodes 50
python -m evaluation.evaluate --algo dqn --seed 42 --episodes 50
python -m evaluation.evaluate --algo a2c --seed 42 --episodes 50
python -m evaluation.compare          # generates comparison_table.csv
python -m visualization.plot_results  # generates 4 PNG plots
```

### 3 — Watch a live demo

```bash
# Full visual window (requires display)
python run_demo.py --algo dqn --seed 42 --episodes 3 --speed 15

# Headless (SSH / CI)
python run_demo.py --algo dqn --seed 42 --episodes 3 --headless
```

### 4 — Run smoke tests

```bash
pytest tests/test_smoke.py -v
```

---

## Results

Trained for 200,000 timesteps each, evaluated over 50 deterministic episodes (seed 42).

| Metric | A2C | DQN | PPO |
|---|---|---|---|
| Mean Episode Reward | -29.24 +/- 0.00 | **+15.41 +/- 0.00** | -55.75 +/- 0.00 |
| Reward Std Dev | 2.80 | 14.48 | 11.76 |
| Success Rate | 0% | 0% | 0% |
| Task Completion Rate | 0% | **100%** | 0% |
| Mean Episode Length | 369.5 steps | 500.0 steps | 453.9 steps |
| Mean Collisions | 0 | 0 | 0 |
| Mean Battery Remaining | 0% | **58%** | 0% |
| Battery Efficiency | 0 | **2.76** | 0 |

**DQN is the clear winner** — the only algorithm that learned to complete tasks reliably (100% task completion rate, +15.4 mean reward, 58% battery remaining).

Plots saved to `results/plots/`:

- `training_curves_all.png` — eval reward vs timesteps for all algorithms
- `comparison_bar_chart.png` — final mean reward per algorithm
- `task_completion_rate.png` — task completion rate per algorithm
- `collision_rate.png` — collision rate per algorithm

---

## Project Structure

```
officebot/
├── configs/              # YAML configs for env + each algorithm
│   ├── env_config.yaml
│   ├── ppo_config.yaml
│   ├── dqn_config.yaml
│   └── a2c_config.yaml
├── environment/          # Custom Gymnasium environment
│   ├── env.py            # OfficeBotEnv (reset, step, render)
│   ├── objects.py        # MiniGrid WorldObj subclasses
│   ├── tasks.py          # Task types and queue manager
│   └── constants.py      # Grid positions and directions
├── agents/
│   └── agent_factory.py  # create_agent / load_agent (SB3 wrappers)
├── training/
│   ├── train.py          # Main training script
│   └── callbacks.py      # MetricLoggerCallback
├── evaluation/
│   ├── evaluate.py       # Run N episodes, save eval_results.json
│   └── compare.py        # Aggregate results -> CSV + JSON
├── visualization/
│   └── plot_results.py   # Generate 4 PNG plots
├── tests/
│   ├── test_smoke.py     # Fast sanity checks (5 tests, ~8s)
│   └── test_env.py       # Full environment unit tests
├── models/               # Saved checkpoints (git-ignored)
├── logs/                 # TensorBoard + metric JSON logs
├── results/              # CSV, JSON, PNG outputs
└── run_demo.py           # Live demo with pygame window
```

---

## Known Limitations

- **PPO and A2C did not learn task completion** in 200k steps. The environment's sparse task rewards require more timesteps (500k+) or curriculum learning to converge reliably with on-policy methods.
- **No multi-seed averaging** in current evaluation — each algorithm was only evaluated on seed 42. Results may vary across seeds.
- **No layout randomisation** — `randomise_layout: false` means agents can overfit to the fixed grid. Enabling it would require significantly more training.
- **Rendering requires a display** — `render_mode='human'` will fail on headless servers. Use `--headless` flag in `run_demo.py` or set `SDL_VIDEODRIVER=dummy` for CI.
- **DQN task completion at 100%** reflects the first task only (1 task completed out of max 5 per episode). The "all tasks done" success rate is still 0% as the agent times out before finishing all 5 tasks.
