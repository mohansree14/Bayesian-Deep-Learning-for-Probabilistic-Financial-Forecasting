# HPO Search Space - Quick Reference

## At-a-Glance Summary

**Framework:** Optuna (TPE Algorithm)  
**Objective:** Minimize Validation Loss (MSE)  
**Trials:** 10  
**Model:** LSTM Regressor  

---

## Search Space Table

| Parameter | Type | Min | Max | Step/Options | Sampling |
|-----------|------|-----|-----|--------------|----------|
| `hidden_size` | int | 64 | 256 | 64 | Uniform |
| `num_layers` | int | 1 | 3 | 1 | Uniform |
| `dropout` | float | 0.0 | 0.4 | continuous | Uniform |
| `lr` | float | 1e-4 | 5e-3 | continuous | **Log-uniform** |
| `batch_size` | categorical | - | - | [32, 64, 128] | Discrete |

---

## Possible Values

| Parameter | Discrete Values |
|-----------|----------------|
| **hidden_size** | 64, 128, 192, 256 (4 values) |
| **num_layers** | 1, 2, 3 (3 values) |
| **dropout** | 0.0 to 0.4 (continuous) |
| **lr** | 1e-4 to 5e-3 (continuous, log-scale) |
| **batch_size** | 32, 64, 128 (3 values) |

**Discrete Combinations:** 4 × 3 × 3 = **36 base configurations**  
**Total Space:** Infinite (due to continuous parameters)

---

## Recommended Ranges (From Experience)

| Parameter | Optimal Range | Common Best Value |
|-----------|---------------|-------------------|
| **hidden_size** | 128-192 | 128 |
| **num_layers** | 2 | 2 |
| **dropout** | 0.1-0.3 | 0.2 |
| **lr** | 5e-4 to 2e-3 | 1e-3 |
| **batch_size** | 64 | 64 |

---

## Parameter Importance

```
Critical:    lr, hidden_size
Important:   dropout, num_layers
Moderate:    batch_size
```

---

## Example Configurations

### Balanced (Recommended)
```python
{
    'hidden_size': 128,
    'num_layers': 2,
    'dropout': 0.2,
    'lr': 0.001,
    'batch_size': 64
}
```

### High Capacity
```python
{
    'hidden_size': 256,
    'num_layers': 3,
    'dropout': 0.3,
    'lr': 0.0005,
    'batch_size': 32
}
```

### Minimal (Fast Training)
```python
{
    'hidden_size': 64,
    'num_layers': 1,
    'dropout': 0.0,
    'lr': 0.005,
    'batch_size': 128
}
```

---

## Usage

```bash
python hparam_search.py \
    --config configs/lstm_baseline.yaml \
    --study-name my_hpo_study
```

---

## Code Reference

```python
def objective(trial: optuna.Trial):
    hidden = trial.suggest_int("hidden_size", 64, 256, step=64)
    layers = trial.suggest_int("num_layers", 1, 3)
    dropout = trial.suggest_float("dropout", 0.0, 0.4)
    lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)  # Log-scale!
    batch = trial.suggest_categorical("batch_size", [32, 64, 128])
    # ... build model and train ...
    return validation_loss
```

---

## Key Points

✅ **Log-scale for learning rate** - explores orders of magnitude efficiently  
✅ **Step size of 64 for hidden_size** - coarse-grained for efficiency  
✅ **Max 3 layers** - balances depth with training difficulty  
✅ **Dropout up to 0.4** - strong regularization without excessive loss  
✅ **Batch sizes are powers of 2** - GPU efficiency  

---

## Fixed Parameters (Not Searched)

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam |
| Gradient Clipping | 1.0 |
| Epochs (HPO) | 5 |
| Loss Function | MSE |

---

## Search Space Size

- **Discrete Configs:** 36
- **With Continuous:** Infinite
- **Explored (10 trials):** ~0.001% coverage
- **TPE Efficiency:** Smart sampling focuses on promising regions

---

## Interpretation Tips

### If Best Config Has:
- **Max hidden_size/layers** → Try expanding search space
- **Min hidden_size/layers** → Simpler model sufficient
- **High dropout** → Overfitting is an issue
- **High LR** → Fast convergence is possible
- **Low LR** → Requires careful training

---

## Next Steps After HPO

1. Retrain best config with more epochs (50-100)
2. Validate on test set
3. Compare with baseline
4. Analyze training curves
5. Fine-tune if needed

---

## File Locations

- **HPO Script:** `hparam_search.py`
- **Config:** `configs/lstm_baseline.yaml`
- **Model:** `src/models/lstm.py`
- **Training:** `src/training/train_loop.py`

---

**Full Documentation:** See `HPO_SEARCH_SPACE.md` for detailed explanations, rationales, and examples.

---

**Quick Start:** Run `python hparam_search.py --config configs/lstm_baseline.yaml --study-name test` to start optimization!
