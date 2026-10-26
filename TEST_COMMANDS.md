# OfficeBot Training Test Commands

## 1. Run Full Training Test (10,000 timesteps)

```bash
python -m training.train --algo ppo --seed 42 --config configs/ppo_config.yaml --timesteps 10000
```

## 2. Run Training with Evaluation (30,000 timesteps to trigger eval)

```bash
python -m training.train --algo ppo --seed 99 --config configs/ppo_config.yaml --timesteps 30000
```

## 3. Verify Model Checkpoint

```bash
# Check final model
ls -lh models/ppo/seed99/final_model.zip

# Check best model from evaluation
ls -lh models/ppo/seed99/best/best_model.zip
```

## 4. Verify TensorBoard Logs

```bash
# List TensorBoard event files
find logs/tensorboard/ppo_seed99 -name "*.tfevents.*"

# Launch TensorBoard viewer
tensorboard --logdir logs/tensorboard/ppo_seed99
```

## 5. Check Training Metrics

```bash
# View JSON metrics
cat logs/metrics/ppo/seed99/training_metrics.json | python -m json.tool
```

## 6. Test Other Algorithms

```bash
# Test DQN
python -m training.train --algo dqn --seed 100 --config configs/dqn_config.yaml --timesteps 10000

# Test A2C
python -m training.train --algo a2c --seed 200 --config configs/a2c_config.yaml --timesteps 10000
```

## 7. Test Resume Functionality

```bash
# Initial training
python -m training.train --algo ppo --seed 77 --config configs/ppo_config.yaml --timesteps 5000

# Resume from checkpoint
python -m training.train --algo ppo --seed 77 --config configs/ppo_config.yaml --timesteps 10000 --resume models/ppo/seed77/final_model.zip
```

## Expected Results

### Training Output:
- ✅ Model saved to: `models/{algo}/seed{X}/final_model.zip`
- ✅ Best model (if eval triggered): `models/{algo}/seed{X}/best/best_model.zip`
- ✅ Metrics logged: `logs/metrics/{algo}/seed{X}/training_metrics.json`
- ✅ TensorBoard logs: `logs/tensorboard/{algo}_seed{X}/`

### Evaluation Results (at 25,000 timesteps):
- ✅ **Mean Reward: -35.00**
- ✅ Episode Length: 200.00
- ✅ "New best mean reward!" message appears

### Performance Metrics Observed:
- Episode reward improved from **-60.1** (start) to **473** (30k steps)
- Episode length increased from **199** to **336** steps
- Agent learned to complete tasks and gain positive rewards
