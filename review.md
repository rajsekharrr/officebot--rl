# OfficeBot RL Project Review

## Professor-ready explanation

**Project:** OfficeBot RL - A Multi-Task Autonomous Office Assistant  
**Technology:** Python, Gymnasium, MiniGrid rendering, Stable-Baselines3, PyTorch, NumPy, YAML, Pygame, TensorBoard, pytest  
**Environment:** A custom 15 x 15 office grid  
**Algorithms:** PPO, DQN, and A2C  

> **Important evidence note:** This repository contains several earlier review documents whose
> numbers and descriptions are not fully synchronized with the current source code. This review
> uses the current implementation in `environment/`, `training/`, `evaluation/`, and `configs/`
> as the source of truth, and labels older or conflicting results where necessary.

---

## 1. What the project is

### Problem being solved

Office work contains repetitive navigation and delivery jobs: collect an item, carry it to the
right desk, attend a meeting, and recharge before the robot's battery is empty. The project
models this as a reinforcement-learning problem rather than writing a fixed route or a
hand-coded task planner.

The agent repeatedly interacts with an environment:

1. It receives an observation of the office and current task.
2. It chooses one action.
3. The environment changes state and returns a reward.
4. The agent learns a policy that should maximize long-term cumulative reward.

The academic goal is to implement the same office problem for three deep-RL algorithms and
compare their behavior under the same grid, reward function, task generation, and evaluation
metrics.

### What OfficeBot is

OfficeBot is the simulated autonomous office assistant. It is represented by an agent marker in
a MiniGrid-style 2-D office. It has a position, a facing direction, a battery, an optional
carried item, and a queue of tasks.

It is not a physical robot and it does not use camera images. The policy receives a normalized
53-number vector produced by the custom environment. The policy is an MLP neural network, not a
manually programmed route.

### Tasks OfficeBot learns

The task manager creates five tasks per normal episode by randomly selecting from these types:

1. **Coffee delivery:** navigate to the yellow coffee machine, pick up coffee, then deliver it
   to a randomly selected desk.
2. **Document delivery:** navigate to the blue printer, pick up a document, then deliver it to
   a randomly selected desk.
3. **Meeting visit:** navigate to the meeting-room check-in point and interact there.
4. **Battery recharge:** go to the green charging station and interact until sufficiently
   charged. This task is dynamically inserted when battery is critically low.
5. **Obstacle avoidance:** this is not a separate queue item; it is a requirement of every
   navigation task because walls and objects block movement and collisions are penalized.

Coffee and document tasks are two-stage tasks. The pickup must occur before delivery. Meeting and
recharge tasks are single-stage interaction tasks.

---

## 2. The office environment

### Grid and coordinate system

The environment is a fixed **15 x 15** grid. Positions are stored as `(row, column)`. The outer
border is made of walls. The robot starts at `(13, 4)`, and its initial direction is `0`, which
means facing right/east.

The fixed important positions are:

| Object | Position | Purpose |
|---|---:|---|
| Coffee machine | `(2, 6)` | Pickup location for coffee |
| Printer | `(7, 2)` | Pickup location for documents |
| Charging station | `(11, 2)` | Battery recharge location |
| Meeting room check-in | `(3, 11)` | Target for meeting visits |
| Meeting table | `(11, 12)` | Visual meeting-room decoration |
| Desk 1 | `(2, 2)` | Possible delivery destination |
| Desk 2 | `(7, 10)` | Possible delivery destination |
| Desk 3 | `(11, 10)` | Possible delivery destination |
| Robot start | `(13, 4)` | Initial robot location |

These constants are defined in `environment/constants.py`.

### Walls, rooms, and corridors

`OfficeBotEnv._build_grid()` in `environment/env.py` constructs the layout:

- A complete outer wall surrounds the 15 x 15 grid.
- A horizontal wall runs across row 5 except for corridor openings at columns 2 and 6.
- A second horizontal wall runs across row 9 except for openings at columns 2, 6, and 12.
- Vertical walls divide the upper office area at column 4 and column 8 for rows 1 through 4.
- The openings connect the lower, middle, and upper sections.
- Desks and machines are placed as fixed obstacles or interactive locations.

The result is an office with separated areas joined by narrow corridors. It is deliberately more
interesting than an empty grid: a direct Manhattan-distance route may be blocked by a wall, so
the agent needs to turn and use corridor openings.

### What is shown visually

The renderer uses MiniGrid's grid renderer and Pygame. `run_demo.py` asks for `human` rendering,
while `--headless` uses `rgb_array`.

| Visual mark | Meaning |
|---|---|
| **Yellow circle** | Coffee machine; pickup point for coffee deliveries |
| **Blue square** | Printer; pickup point for document deliveries |
| **Green circle with white center** | Charging station; the robot can stand there and recharge |
| **Grey squares/rectangles** | Desks; delivery destinations and blocking objects |
| **Red arrow** | OfficeBot's current position and facing direction; the arrow is supplied by MiniGrid |
| **Purple square/rectangle** | Meeting table; visual decoration in the meeting area |
| Dark/solid wall tiles | Impassable outer or internal office walls |

The environment also prints a terminal status line during the demo containing episode number,
step, cumulative reward, battery percentage, carried item, current task, completed task count,
and collision count.

### Battery system

The values are in `configs/env_config.yaml`:

- Initial battery: `1.0` (displayed as 100%).
- Every environment step drains `0.002`.
- A `MOVE_FORWARD` action drains an additional `0.001`.
- Battery is clamped to the range `[0, 1]`.
- At the charging station, action `INTERACT` adds `0.05`, capped at `1.0`.
- Recharge also receives a proportional efficiency reward.
- When battery drops below `0.05`, a recharge task is injected and the current task is pushed
  back into the queue.
- A recharge task is completed when the agent is at the station, battery is at least `0.8`, and
  it chooses `INTERACT`.
- If battery reaches zero, the episode terminates immediately with reason `battery_dead` and a
  `-20` depletion penalty.

The battery creates a resource-management problem: the agent must make progress while preserving
enough energy to reach the station. A known limitation is that recharge rewards can be easier to
exploit than task completion rewards, so reward design needs further tuning.

---

## 3. How reinforcement learning is used

### Markov decision process

The project is an MDP:

- **State/observation:** the 53-element vector returned by `_get_obs()`.
- **Action:** one of six discrete integer actions.
- **Transition:** the environment applies the action, updates position/task/battery, and returns
  the next observation.
- **Reward:** a scalar score for progress, efficiency, collisions, task completion, and battery.
- **Policy:** the neural network's mapping from observation to action.

### Observation space: what the agent sees

The Gymnasium observation space is:

```text
Box(low=0.0, high=1.0, shape=(53,), dtype=float32)
```

The vector contains:

| Indices | Contents |
|---|---|
| `0-1` | Robot row and column, normalized by grid size |
| `2` | Facing direction, normalized from 0-3 to 0-1 |
| `3` | Battery level |
| `4` | Carried item: none `0.0`, coffee `0.5`, document `1.0` |
| `5-9` | Current task type one-hot encoding |
| `10-11` | Current task target position |
| `12-13` | Current task pickup position, when applicable |
| `14` | Number of queued tasks normalized by maximum tasks |
| `15-34` | Five local cells, each encoded with four values: empty/out-of-bounds, wall, object, or floor |
| `35-46` | Fixed coordinates for coffee machine, printer, charger, and three desks |
| `47-52` | Reserved zero-padded extension slots |

All coordinates are normalized. The policy therefore receives symbolic state features rather than
raw pixels. The local view consists of the cells forward one, forward two, left, right, and
behind relative to the robot's current direction.

### Action space: what the agent can do

```text
Discrete(6)
```

| Action | Name | Effect |
|---:|---|---|
| `0` | `TURN_LEFT` | Rotate 90 degrees counter-clockwise |
| `1` | `TURN_RIGHT` | Rotate 90 degrees clockwise |
| `2` | `MOVE_FORWARD` | Attempt to move one cell forward |
| `3` | `PICKUP` | Pick up coffee/document when adjacent to the required pickup point |
| `4` | `DELIVER` | Deliver a carried item when adjacent to its target desk |
| `5` | `INTERACT` | Recharge at the charging station or complete a meeting visit |

An object can be adjacent rather than occupying the same cell for pickup/delivery. Desks and most
machines cannot be overlapped, while the charging station can be stood on.

### Reward function

The current reward configuration is:

| Reward component | Value | When it occurs |
|---|---:|---|
| Step penalty | `-0.1` | Every step |
| Move closer | `+0.5` | Manhattan distance to current subtarget decreases |
| Move away | `-0.3` | Manhattan distance increases |
| Collision | `-5.0` | Forward movement hits a wall/blocked object |
| Pickup success | `+10.0` | Correct pickup succeeds |
| Delivery success | `+50.0` | Correct carried item delivered |
| Meeting completion | `+40.0` | Meeting interaction succeeds |
| Battery efficiency | `0.5 * battery gained` | Recharge gains battery |
| Full-battery bonus | `0.0` | Disabled in current config |
| Battery depleted | `-20.0` | Battery reaches zero |
| All tasks bonus | `+100.0` | Entire task queue is complete |
| Timeout penalty | `-10.0` | Episode reaches maximum length |

The actual implementation first applies battery drain, then executes the selected action, then
adds distance shaping, checks for battery-critical task injection, increments the step count, and
checks termination.

Distance shaping currently uses Manhattan distance, not a wall-aware shortest path. This explains
why a learned policy can oscillate or move in square patterns: a geometrically closer action is
not always a reachable action through the office corridors.

### Episodes and termination

Normal episodes allow at most `500` steps and begin with `5` randomly generated tasks. An episode
ends when:

1. All tasks are complete: `termination_reason = all_tasks_done`;
2. Battery reaches zero: `termination_reason = battery_dead`; or
3. The step count reaches 500: `truncated = True`, `termination_reason = timeout`.

The environment returns the Gymnasium five-tuple:

```text
observation, reward, terminated, truncated, info
```

The `info` dictionary contains battery, current task, completed/failed tasks, collision count,
step count, termination reason, battery-low flag, and queue length.

---

## 4. The three algorithms

All three algorithms are created through `agents/agent_factory.py`, use Stable-Baselines3, use an
`MlpPolicy`, and receive the same environment interface. The training script chooses the algorithm
from the command line and loads its YAML hyperparameters.

### PPO: Proximal Policy Optimization

PPO is an on-policy actor-critic method. It collects a rollout from the current policy, estimates
advantages with Generalized Advantage Estimation, and updates the policy and value function.
Its clipped objective prevents a single update from changing the policy too aggressively:

```text
L_clip = E[min(r * A, clip(r, 1-epsilon, 1+epsilon) * A)]
```

where `r` is the new/old action probability ratio and `epsilon` is `0.2` in the current config.

PPO is appropriate here because the action set is discrete, rewards can be delayed until a
pickup/delivery is complete, and stable updates are useful for a navigation policy. Current PPO
settings include learning rate `0.0003`, rollout length `1024`, batch size `64`, `10` epochs,
`gamma=0.99`, `gae_lambda=0.95`, and two 256-unit hidden layers.

### DQN: Deep Q-Network

DQN is an off-policy value-based method. It learns a Q-value for each of the six possible
actions. It stores transitions in a replay buffer, samples random minibatches, and uses a separate
target network to stabilize temporal-difference targets:

```text
target = reward + gamma * max_a Q_target(next_state, a)
```

DQN is a natural comparison because OfficeBot has a discrete action space. Experience replay lets
it reuse earlier navigation experiences, while epsilon-greedy exploration starts fully random and
decays to mostly exploitation. The current config uses a 100,000-transition buffer, 5,000
learning-start steps, batch size 64, training every 4 steps, target update interval 500, and
epsilon decay over 30% of training.

### A2C: Advantage Actor-Critic

A2C is an on-policy actor-critic method that collects short rollouts and synchronously updates:

- The actor improves the probability of actions with positive advantage.
- The critic estimates the value of the current observation.

A2C is included as a fast, lower-memory actor-critic baseline. It has learning rate `0.001`,
rollout length `16`, `gamma=0.99`, `gae_lambda=1.0`, entropy coefficient `0.05`, and the same
256-256 MLP shape.

### How they are compared

`training/train.py` gives each algorithm:

- The same 15 x 15 environment;
- The same task-generation and battery rules;
- A separate evaluation environment seeded with `training_seed + 1000`;
- The same evaluation frequency and number of evaluation episodes from its YAML file;
- Deterministic evaluation actions;
- The same output categories: best model, final model, checkpoints, metrics, and TensorBoard logs.

`evaluation/evaluate.py` measures mean reward, reward standard deviation, all-task success rate,
task completion rate, episode length, collisions, remaining battery, battery efficiency, and
mean completed/failed tasks. `evaluation/compare.py` aggregates available JSON results by
algorithm and seed into CSV and JSON files.

---

## 5. Code structure

### Important folders

```text
officebot/
|-- agents/          Stable-Baselines3 model factory and loading
|-- configs/         Environment and algorithm YAML files
|-- environment/     OfficeBot Gymnasium environment, objects, tasks, constants
|-- evaluation/      Single-model evaluation and algorithm comparison
|-- logs/             Training logs, TensorBoard event files, metric JSON/NPZ files
|-- models/           Saved Stable-Baselines3 zip checkpoints
|-- results/          Comparison tables, summaries, and generated plots
|-- scripts/          Batch and post-training helper scripts
|-- tests/            Environment, task, reward, agent, and integration tests
|-- training/        Main training entry point and metric callback
|-- visualization/   Plot-generation script
|-- run_demo.py      Pygame visual demonstration
```

### Environment files

- `environment/env.py`: defines `OfficeBotEnv`, including reset, step, battery updates, task
  execution, observations, reward shaping, termination, grid construction, and Pygame/MiniGrid
  rendering.
- `environment/constants.py`: fixed positions, grid size, start position, and direction vectors.
- `environment/objects.py`: MiniGrid `WorldObj` subclasses and custom rendering:
  `CoffeeMachine`, `Printer`, `ChargingStation`, `Desk`, and `MeetingTable`.
- `environment/tasks.py`: task data class, task types, templates, and randomized queue manager.
- `environment/__init__.py`: package marker.

### Agent and training files

- `agents/agent_factory.py`: maps strings `ppo`, `dqn`, and `a2c` to SB3 classes; creates models
  from YAML and loads saved models.
- `training/train.py`: command-line entry point. It loads YAML, seeds NumPy, creates monitored
  training/evaluation environments, creates or resumes a model, registers callbacks, trains,
  saves `final_model.zip`, and closes environments.
- `training/callbacks.py`: `MetricLoggerCallback`, which writes periodic battery, task, collision,
  and queue snapshots to `training_metrics.json`.
- `training/__init__.py`: package marker.

### Configuration files

- `configs/env_config.yaml`: grid size, task count, maximum steps, battery values, reward values,
  rendering tile size, and seed defaults.
- `configs/ppo_config.yaml`: PPO rollout and optimization parameters.
- `configs/dqn_config.yaml`: replay buffer, exploration, target network, and optimization values.
- `configs/a2c_config.yaml`: short rollout and actor-critic optimization values.

The training CLI can override YAML `total_timesteps` with `--timesteps`. This means experiments
can be reproduced or shortened without editing Python.

### Evaluation and visualization

- `evaluation/evaluate.py`: loads `best_model.zip` or `final_model.zip`, runs deterministic
  episodes, computes metrics, and writes `logs/metrics/{algo}/seed{seed}/eval_results.json`.
- `evaluation/compare.py`: discovers evaluation JSON files and produces
  `results/comparison_table.csv` and `results/experiment_summary.json`.
- `visualization/plot_results.py`: reads `evaluations.npz` and evaluation JSON files and creates
  reward curves and metric bar charts in `results/plots/`.
- `evaluation/__init__.py` and `visualization/__init__.py`: package markers.

### Demo and utility files

- `run_demo.py`: parses algorithm/seed/model/episode/speed arguments, loads the model, opens the
  Pygame window, prints live status, and runs deterministic episodes.
- `evaluate_battery_behavior.py`: detailed PPO battery/action/position trace and recharge
  analysis.
- `scripts/train_all.bat`: Windows batch helper for multiple training runs.
- `scripts/post_training_pipeline.sh`: shell helper for evaluation/comparison/plot generation.
- `quick_demo_commands.sh`: quick command reference.
- `requirements.txt`: Python dependencies for SB3, MiniGrid, Gymnasium/Pygame support, YAML,
  NumPy, plotting, and testing.

### Tests

- `tests/test_env.py`: environment creation, Gymnasium API, spaces, reset/step, collision,
  battery, and task queue behavior.
- `tests/test_tasks.py`: task count, types, positions, IDs, descriptions, initial state, and
  seed variation.
- `tests/test_rewards.py`: reward type/bounds, penalties, deterministic sequences, and no NaN or
  infinite reward behavior.
- `tests/test_agents.py`: algorithm mapping, model creation, prediction, save/load, seed, and
  TensorBoard configuration.
- `tests/test_integration.py`: short training runs, model persistence, multiple episodes, policy
  improvement smoke checks, and low-battery training.
- `tests/test_smoke.py`: quick sanity checks.
- `tests/__init__.py`: test package marker.

### What the `.zip` model files are

Stable-Baselines3 serializes a trained model as a ZIP archive. It contains the policy network
parameters, optimizer/model metadata, spaces, and algorithm-specific state. It is not a source
code archive.

The project uses:

```text
models/{algo}/seed{seed}/final_model.zip
models/{algo}/seed{seed}/best/best_model.zip
models/{algo}/seed{seed}/checkpoints/officebot_{algo}_{steps}_steps.zip
```

`best_model.zip` is written by `EvalCallback` when an evaluation score improves. `final_model.zip`
is saved after training ends. The demo and evaluator load one of these files and call
`model.predict(observation, deterministic=True)`.

---

## 6. What has been completed

### Fully implemented components

The core implementation is present and wired end to end:

- Custom Gymnasium environment with valid reset/step/render/close behavior.
- Fixed 15 x 15 office grid and MiniGrid-compatible objects.
- Multi-stage coffee and document delivery.
- Meeting visit interaction.
- Automatic battery recharge task insertion.
- Battery drain, recharge, depletion termination, and battery telemetry.
- Six-action discrete control interface.
- PPO, DQN, and A2C model factory.
- YAML-controlled hyperparameters.
- Training CLI with resume support.
- Evaluation callback and best-model saving.
- Checkpoint saving.
- Periodic metric JSON logging.
- TensorBoard logging integration.
- Deterministic evaluator and comparison pipeline.
- Plot generation pipeline.
- Pygame visual demo and headless demo mode.
- Automated tests for environment, tasks, rewards, agents, and integration.

The repository's `TEST_SUITE_RESULTS.md` records **56 tests passed** with no failures. The
recorded warnings are Gymnasium checker deprecation/render-mode warnings and are not test
failures.

### Evidence from current result files

The checked-in `results/comparison_table.csv` and `results/experiment_summary.json` contain one
available evaluation record for each algorithm:

| Metric | A2C | DQN | PPO |
|---|---:|---:|---:|
| Mean episode reward | `-29.242` | `+15.409` | `-55.754` |
| Reward standard deviation | `2.805` | `14.480` | `11.756` |
| All-task success rate | `0%` | `0%` | `0%` |
| Task completion rate | `0%` | `100%` | `0%` |
| Mean episode length | `369.46` | `500.00` | `453.86` |
| Mean collisions | `0` | `0` | `0` |
| Mean battery remaining | `0%` | `58.01%` | `0%` |
| Battery efficiency | `0` | `2.760` | `0` |
| Mean completed tasks | `0` | `1` | `0` |

The DQN task-completion figure must be explained carefully: the README states that the 100%
completion figure reflects completion of the first task, not completion of all five tasks. Thus
the all-task success rate remains 0%.

The current PPO training log contains evaluation milestones such as:

```text
75,000 steps:  mean evaluation reward 1,985.22
100,000 steps: mean evaluation reward 4,750.66
150,000 steps: mean evaluation reward 4,888.16
300,000 steps: mean evaluation reward 4,487.06
450,000 steps: mean evaluation reward 4,888.11
500,000 steps: mean evaluation reward 4,888.01
```

These large rewards are not proof that all five tasks were completed. The older
`TRAINING_RESULTS_SUMMARY.md` and `BATTERY_SYSTEM_REVIEW.md` correctly identify the likely
reward-hacking explanation: an agent can earn positive recharge-related reward by camping at the
charger, while failing to finish the complete task queue. The more conservative current
comparison table is therefore important when presenting task success.

The repository also contains older notes claiming PPO improved from `-60.1` to `+473` at around
30,000 steps. These are useful as historical training observations, but they conflict with the
current 500,000-step log and should not be presented as the final benchmark without checking the
associated seed/model.

### TensorBoard evidence

`training/train.py` passes `tensorboard_log=logs/tensorboard/{algo}_seed{seed}` to Stable-Baselines3.
TensorBoard event files are present under directories such as:

```text
logs/tensorboard/ppo_seed42/
logs/tensorboard/ppo_seed777/
logs/tensorboard/ppo_seed99/
logs/tensorboard/dqn_seed42/
logs/tensorboard/a2c_seed42/
```

TensorBoard can show SB3 scalar groups such as episode reward/length, training losses, learning
rate, entropy, policy loss, value loss, and approximate KL depending on algorithm and run. The
metric callback separately records battery, completed tasks, collisions, failed tasks, and queue
length every 5,000 callback calls in `training_metrics.json`.

The correct interpretation is: TensorBoard demonstrates that training ran and provides learning
curves/loss diagnostics; it does not by itself prove that the agent completed every task.

---

## 7. What is still remaining

### Technical limitations

1. **PPO behavior is not reliably task-complete.** The visual PPO demo can loop or move in a
   square pattern and eventually run out of battery. This is consistent with limited local
   observations, randomized tasks, Manhattan shaping that ignores walls, and reward exploitation.
2. **The reward function needs redesign.** Recharge progress can be easier to exploit than
   navigation and delivery. A recharge reward should be conditional on meaningful battery
   recovery, and task completion should dominate survival/camping rewards.
3. **Navigation shaping should be wall-aware.** A shortest-path distance through free cells or a
   potential-field/path planner would be more informative than raw Manhattan distance.
4. **Observation coverage is limited.** The agent sees a compact local view plus coordinates, not
   a full occupancy map. A larger map encoding or recurrent policy could improve route planning.
5. **More robust evaluation is needed.** Current comparison files contain one seed per algorithm.
   Multiple seeds, confidence intervals, and a clearly fixed evaluation protocol should be used
   for a final scientific comparison.
6. **Demo/evaluation seeds differ from training evaluation seeds.** Training evaluates at
   `seed + 1000`, the evaluator uses `seed + 2000`, and the demo uses `seed + 9999`. This tests
   generalization but can make a demo look worse than the checkpoint's reported evaluation.
7. **Documentation is stale in places.** Some documents describe old reward values, old callback
   frequencies, or components as not implemented even though they now exist. This file is the
   consolidated current explanation.
8. **Layout randomization is not implemented.** `randomise_layout: false`; fixed layouts make
   overfitting possible.
9. **Task failure accounting is minimal.** The `failed` field exists, but the normal environment
   primarily terminates on timeout/battery depletion rather than marking individual tasks as
   failed.

### Recommended next steps

1. Fix reward design and add ablation tests for task reward versus recharge reward.
2. Add wall-aware shortest-path shaping and a repeated-position/loop penalty.
3. Add deterministic demo-task mode so a professor can reproduce the same route.
4. Train PPO and A2C again after reward changes; do not compare old and new rewards directly.
5. Evaluate every algorithm over at least three independent seeds and report mean plus standard
   deviation.
6. Add a task-by-task success table, not just aggregate completion.
7. Update the older review documents and README after the new benchmark is complete.
8. Keep the existing 56-test suite and add regression tests for reward hacking, wall-aware paths,
   recharge cooldowns, and deterministic task sequences.

---

## 8. How to run the project

Run these commands from the repository root, for example `D:\officebot` in PowerShell.

### Install dependencies

```powershell
python -m pip install -r requirements.txt
```

The README also lists the equivalent direct installation:

```powershell
python -m pip install "stable-baselines3[extra]" minigrid pygame pyyaml tabulate numpy matplotlib
```

### Start PPO training

This is the exact PPO training command:

```powershell
python -m training.train --algo ppo --seed 42 --config configs/ppo_config.yaml --timesteps 100000
```

For the configuration default of 200,000 steps, omit the override:

```powershell
python -m training.train --algo ppo --seed 42 --config configs/ppo_config.yaml
```

For a complete long experiment, use the requested timesteps explicitly; do not confuse this with
the short demo training:

```powershell
python -m training.train --algo ppo --seed 42 --config configs/ppo_config.yaml --timesteps 500000
```

Expected output includes the algorithm/seed, environment setup, progress bar, periodic
evaluation, checkpoint messages, and final paths such as:

```text
models/ppo/seed42/final_model.zip
models/ppo/seed42/best/best_model.zip
models/ppo/seed42/checkpoints/
logs/metrics/ppo/seed42/
logs/tensorboard/ppo_seed42/
```

To compare the other algorithms, change `--algo` and use the matching YAML:

```powershell
python -m training.train --algo dqn --seed 42 --config configs/dqn_config.yaml --timesteps 200000
python -m training.train --algo a2c --seed 42 --config configs/a2c_config.yaml --timesteps 200000
```

### Open TensorBoard

```powershell
tensorboard --logdir logs/tensorboard
```

Open this address in a browser:

```text
http://localhost:6006
```

To inspect only PPO seed 42:

```powershell
tensorboard --logdir logs/tensorboard/ppo_seed42
```

Expected view: scalar charts for rollout episode reward/length and algorithm training values. The
curves should be interpreted together with evaluation JSON and task-completion metrics.

### Evaluate a saved model

```powershell
python -m evaluation.evaluate --algo ppo --seed 42 --episodes 50 --model best
```

This loads `models/ppo/seed42/best/best_model.zip`, runs deterministic episodes, prints metrics,
and writes `logs/metrics/ppo/seed42/eval_results.json`.

### Compare algorithms

```powershell
python -m evaluation.compare
```

Expected outputs:

```text
results/comparison_table.csv
results/experiment_summary.json
```

### Generate plots

```powershell
python -m visualization.plot_results
```

Expected outputs are under `results/plots/`, including `training_curves_all.png`,
`comparison_bar_chart.png`, `task_completion_rate.png`, and `collision_rate.png`.

### Open the visual PPO demo

Use the existing seed-42 best model:

```powershell
python run_demo.py --algo ppo --seed 42 --model best --episodes 2 --speed 1
```

If that model is not present, use the existing seed-99 model:

```powershell
python run_demo.py --algo ppo --seed 99 --model best --episodes 2 --speed 1
```

Expected behavior:

- A Pygame window titled `OfficeBot RL` opens.
- The grid shows walls and the colored office objects.
- The red arrow shows OfficeBot position and direction.
- The terminal continuously prints Step, Reward, Battery, and Task values.
- `--speed 1` means one rendered frame per second; it does not improve the policy.
- Close the window or wait for the requested episodes to end.

For a machine without a display:

```powershell
python run_demo.py --algo ppo --seed 42 --model best --episodes 2 --headless
```

### Run tests

```powershell
python -m pytest tests/ -v
```

Expected result from the repository's recorded test report: 56 passed, with only harmless
Gymnasium checker warnings.

---

## 9. Ten likely professor questions and ideal answers

### 1. Why is this an RL problem rather than a normal path-planning problem?

**Ideal answer:** The robot must choose sequential actions while balancing delayed task rewards,
obstacles, randomized task targets, and a battery resource. The desired behavior is learned from
interaction and reward rather than written as a fixed route. A classical planner could solve a
known layout, but this project specifically studies policy learning and algorithm comparison.

### 2. What exactly does the agent observe?

**Ideal answer:** It receives a 53-dimensional normalized vector: position, direction, battery,
carried item, task type, target and pickup coordinates, queue length, five nearby relative cells,
and fixed coordinates for important objects. It receives symbolic state features, not raw images.

### 3. Why are the actions discrete?

**Ideal answer:** OfficeBot's choices are naturally discrete: turn left, turn right, move forward,
pick up, deliver, or interact. `Discrete(6)` also lets DQN be compared directly with PPO and A2C.

### 4. How does PPO prevent unstable learning?

**Ideal answer:** PPO compares the new policy with the old policy using a probability ratio and
clips that ratio to a range such as 0.8 to 1.2. This prevents a single minibatch update from
changing the policy too far and destroying useful behavior. It also uses a critic and GAE to
estimate advantages.

### 5. Why include DQN if PPO and A2C are actor-critic methods?

**Ideal answer:** DQN provides a value-based, off-policy baseline. It is especially appropriate
for a discrete action space and reuses experience through replay, while PPO and A2C learn
on-policy actor-critic policies. Comparing them shows how algorithm design affects this task.

### 6. What is the difference between `best_model.zip` and `final_model.zip`?

**Ideal answer:** `final_model.zip` is saved after the requested training timesteps. `best_model.zip`
is saved by `EvalCallback` whenever deterministic evaluation reaches a new best score. The demo
normally uses the best checkpoint, while both can be selected for evaluation.

### 7. When does an episode finish?

**Ideal answer:** It terminates when all queued tasks are complete or the battery reaches zero. It
is truncated after 500 steps if neither condition occurs. The `info` dictionary records the
termination reason.

### 8. What does the battery contribute scientifically?

**Ideal answer:** Battery adds a resource constraint. Movement consumes energy, the robot must
reach and interact with a charger, and critically low battery dynamically inserts a recharge task.
This tests whether the policy can trade immediate task progress against survival. It also exposed
a reward-hacking issue, which is an important experimental finding.

### 9. Why can PPO move in a square instead of completing a task?

**Ideal answer:** The model has a small local observation, the shaping reward uses Manhattan
distance without checking walls, tasks are randomized, and the reward can favor easy local or
recharge behavior. A loop can therefore produce locally plausible rewards without solving the
global route. This is a policy/reward-design limitation, not a Pygame speed problem.

### 10. What would you improve next?

**Ideal answer:** I would use wall-aware shortest-path shaping, add a larger map or recurrent
observation, penalize repeated positions, redesign recharge rewards to prevent camping, use
deterministic demo tasks, and evaluate all algorithms over multiple seeds with task-by-task
success statistics. I would then retrain and compare the new results against the same metrics.

---

## Closing presentation statement

> “OfficeBot RL is a custom Gymnasium reinforcement-learning benchmark for a simulated office
> assistant. The agent navigates a 15 x 15 grid, performs coffee and document deliveries, visits
> meetings, and manages a limited battery. I implemented the environment, task queue, reward
> system, PPO/DQN/A2C training pipeline, checkpointing, TensorBoard logging, evaluation, plots,
> and a Pygame demo. The main engineering components are working and tested. The current results
> also reveal an important limitation: high training reward does not automatically mean complete
> multi-task success, because the reward and observation design still allow recharge or looping
> strategies. That gives a clear next research step: improve reward shaping and navigation
> representation, then rerun a multi-seed comparison.”

