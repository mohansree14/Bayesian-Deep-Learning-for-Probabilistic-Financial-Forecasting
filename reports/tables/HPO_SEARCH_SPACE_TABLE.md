# Optuna HPO Search Space - Detailed Tables

## Table 1: Complete Search Space Definition

| Hyperparameter | Data Type | Sampling Method | Minimum | Maximum | Step Size | Choices | Distribution | Search Space Size |
|----------------|-----------|-----------------|---------|---------|-----------|---------|--------------|-------------------|
| **hidden_size** | Integer | `suggest_int()` | 64 | 256 | 64 | [64, 128, 192, 256] | Uniform | 4 discrete values |
| **num_layers** | Integer | `suggest_int()` | 1 | 3 | 1 | [1, 2, 3] | Uniform | 3 discrete values |
| **dropout** | Float | `suggest_float()` | 0.0 | 0.4 | N/A | Continuous | Uniform | Infinite |
| **lr** (Learning Rate) | Float | `suggest_float(log=True)` | 1.0×10⁻⁴ | 5.0×10⁻³ | N/A | Continuous | **Log-Uniform** | Infinite |
| **batch_size** | Categorical | `suggest_categorical()` | N/A | N/A | N/A | [32, 64, 128] | Discrete | 3 discrete values |

---

## Table 2: Parameter Ranges and Impact

| Parameter | Range | Typical Optimal | Impact on Performance | Impact on Training Time | Impact on Memory |
|-----------|-------|-----------------|----------------------|------------------------|------------------|
| **hidden_size** | 64-256 | 128-192 | ⭐⭐⭐⭐⭐ Very High | ⭐⭐⭐⭐ High | ⭐⭐⭐⭐⭐ Very High |
| **num_layers** | 1-3 | 2 | ⭐⭐⭐⭐ High | ⭐⭐⭐⭐ High | ⭐⭐⭐⭐ High |
| **dropout** | 0.0-0.4 | 0.1-0.3 | ⭐⭐⭐ Medium | ⭐ Low | ⭐ Low |
| **lr** | 1e-4 to 5e-3 | 5e-4 to 2e-3 | ⭐⭐⭐⭐⭐ Very High | ⭐⭐ Medium | ⭐ Low |
| **batch_size** | {32, 64, 128} | 64 | ⭐⭐ Low | ⭐⭐⭐ Medium | ⭐⭐⭐ Medium |

**Legend:**  
⭐ Low | ⭐⭐ Low-Medium | ⭐⭐⭐ Medium | ⭐⭐⭐⭐ High | ⭐⭐⭐⭐⭐ Very High

---

## Table 3: Parameter Trade-offs

| Parameter | Low Value Effect | Medium Value Effect | High Value Effect |
|-----------|------------------|---------------------|-------------------|
| **hidden_size** | **64:** Fast, underfitting risk | **128-192:** Balanced | **256:** Slow, overfitting risk |
| **num_layers** | **1:** Simple, limited capacity | **2:** Good balance ✓ | **3:** Deep, hard to train |
| **dropout** | **0.0-0.1:** Minimal regularization | **0.1-0.3:** Balanced ✓ | **0.3-0.4:** Strong regularization |
| **lr** | **1e-4:** Slow, stable | **5e-4 to 2e-3:** Balanced ✓ | **5e-3:** Fast, unstable |
| **batch_size** | **32:** Noisy, generalizes well | **64:** Balanced ✓ | **128:** Stable, may overfit |

---

## Table 4: Search Space Cardinality

| Component | Calculation | Result |
|-----------|-------------|--------|
| **Discrete Parameters** | hidden_size × num_layers × batch_size | 4 × 3 × 3 = **36 combinations** |
| **Continuous Parameters** | dropout × lr | **Infinite** |
| **Total Theoretical Space** | Discrete × Continuous | **Infinite** |
| **Explored in Study** | Number of trials | **10 configurations** |
| **Coverage** | 10 / ∞ | **~0% (TPE guides exploration)** |

---

## Table 5: Example Configurations with Expected Outcomes

| Config Name | hidden_size | num_layers | dropout | lr | batch_size | Est. Params | Training Speed | Expected Performance | Use Case |
|-------------|-------------|------------|---------|-----|------------|-------------|----------------|---------------------|----------|
| **Minimal** | 64 | 1 | 0.0 | 5e-3 | 128 | ~50K | ⚡⚡⚡⚡ Very Fast | 🎯🎯 Fair | Quick baseline, debugging |
| **Balanced** | 128 | 2 | 0.2 | 1e-3 | 64 | ~200K | ⚡⚡⚡ Fast | 🎯🎯🎯🎯 Good | **Recommended start** |
| **High Capacity** | 256 | 3 | 0.3 | 5e-4 | 32 | ~800K | ⚡ Slow | 🎯🎯🎯🎯🎯 Best | Max performance |
| **Conservative** | 64 | 1 | 0.4 | 1e-4 | 128 | ~50K | ⚡⚡⚡⚡ Very Fast | 🎯🎯🎯 Good | Strong regularization |
| **Aggressive** | 256 | 2 | 0.1 | 5e-3 | 128 | ~500K | ⚡⚡ Medium | 🎯🎯🎯 Varies | Experimental |

**Legends:**  
⚡ = Speed | 🎯 = Performance

---

## Table 6: Parameter Sampling Details

| Parameter | Optuna Method Call | Python Equivalent | Notes |
|-----------|-------------------|-------------------|-------|
| **hidden_size** | `trial.suggest_int("hidden_size", 64, 256, step=64)` | `random.choice([64, 128, 192, 256])` | Discrete steps |
| **num_layers** | `trial.suggest_int("num_layers", 1, 3)` | `random.randint(1, 3)` | Inclusive range |
| **dropout** | `trial.suggest_float("dropout", 0.0, 0.4)` | `random.uniform(0.0, 0.4)` | Continuous uniform |
| **lr** | `trial.suggest_float("lr", 1e-4, 5e-3, log=True)` | `10^random.uniform(-4, log10(5e-3))` | **Log-scale** |
| **batch_size** | `trial.suggest_categorical("batch_size", [32,64,128])` | `random.choice([32, 64, 128])` | Explicit choices |

---

## Table 7: Optimization Settings

| Setting | Value | Explanation |
|---------|-------|-------------|
| **Study Direction** | Minimize | Lower validation loss is better |
| **Objective Metric** | Validation Loss (MSE) | Final validation loss after 5 epochs |
| **Number of Trials** | 10 | Computational budget limitation |
| **Sampler** | TPE (Tree-structured Parzen Estimator) | Bayesian optimization (default Optuna) |
| **Pruner** | None | All trials run to completion |
| **Timeout** | None | No time limit per trial |
| **Parallel Jobs** | 1 | Sequential execution |

---

## Table 8: Fixed (Non-Optimized) Parameters

| Parameter | Fixed Value | Rationale |
|-----------|-------------|-----------|
| **Optimizer** | Adam | Adaptive learning rates; industry standard |
| **Gradient Clipping** | 1.0 | Prevents exploding gradients in RNNs |
| **Loss Function** | MSE (Mean Squared Error) | Standard for regression tasks |
| **Epochs (HPO)** | 5 | Quick evaluation for hyperparameter search |
| **Epochs (Final)** | Variable (with early stopping) | Full training after HPO |
| **Device** | CPU | Accessibility; GPU would enable more trials |
| **Random Seed** | 42 | Reproducibility |
| **Input Dimension** | From data | Determined by feature engineering |
| **Sequence Length** | From data | Determined by preprocessing |

---

## Table 9: Parameter Priorities for Tuning

| Priority | Parameters | Reason |
|----------|-----------|--------|
| **1 - Critical** | Learning Rate (`lr`) | Most sensitive; wrong value prevents convergence |
| **2 - Critical** | Hidden Size (`hidden_size`) | Determines model capacity; direct performance impact |
| **3 - Important** | Dropout (`dropout`) | Key for generalization and preventing overfitting |
| **4 - Important** | Number of Layers (`num_layers`) | Affects architectural depth and capacity |
| **5 - Moderate** | Batch Size (`batch_size`) | Affects training dynamics more than final performance |

**Tuning Strategy:** Focus first on learning rate and hidden size, then refine dropout and layers.

---

## Table 10: Interpretation Guide

| Observation | Interpretation | Action |
|-------------|----------------|--------|
| Best `hidden_size` = 256 | Model benefits from high capacity | Consider expanding to 512 in future |
| Best `hidden_size` = 64 | Simpler model sufficient | Data may not need complex patterns |
| Best `dropout` > 0.3 | Strong overfitting tendency | Add more regularization or data |
| Best `dropout` < 0.1 | Overfitting not an issue | Can reduce dropout or add complexity |
| Best `lr` > 2e-3 | Fast convergence possible | Model/data allows aggressive learning |
| Best `lr` < 5e-4 | Requires careful training | Sensitive to learning rate |
| Best `num_layers` = 3 | Benefits from depth | Consider even deeper architectures |
| Best `num_layers` = 1 | Shallow network sufficient | Task may be simpler than expected |
| Best `batch_size` = 32 | Noisy gradients beneficial | Small batches improve generalization |
| Best `batch_size` = 128 | Stable training preferred | Large batches work well for this task |

---

## Table 11: Search Space Comparison with Other Approaches

| Method | hidden_size | num_layers | dropout | lr | batch_size | Total Configs | Trials Needed |
|--------|-------------|------------|---------|-----|------------|---------------|---------------|
| **Optuna (Current)** | 4 values | 3 values | Continuous | Continuous | 3 values | Infinite | 10 (smart) |
| **Grid Search** | 4 values | 3 values | 5 values | 5 values | 3 values | 900 | 900 (exhaustive) |
| **Random Search** | 4 values | 3 values | Continuous | Continuous | 3 values | Infinite | 50-100 (random) |
| **Manual Tuning** | Custom | Custom | Custom | Custom | Custom | N/A | 5-20 (expert) |

**Efficiency Comparison:**
- **Optuna TPE:** Best performance with 10 trials (Bayesian guidance)
- **Grid Search:** Exhaustive but computationally expensive (900 trials)
- **Random Search:** Better than grid for continuous params (50-100 trials)
- **Manual:** Requires domain expertise (5-20 trials)

---

## Table 12: Expected Parameter Distributions (TPE Behavior)

| Parameter | Early Trials (1-3) | Mid Trials (4-7) | Late Trials (8-10) |
|-----------|-------------------|------------------|-------------------|
| **hidden_size** | Uniform exploration | Focus on promising values | Refinement around optimum |
| **num_layers** | Try all 3 values | Favor better-performing | Mostly optimal value |
| **dropout** | Wide range sampling | Narrow to sweet spot | Fine-tune around best |
| **lr** | Log-scale sampling | Concentrate on good range | Optimize final value |
| **batch_size** | Test all 3 | Prefer better choice | Stick with best |

**TPE Learning:** Algorithm becomes smarter over trials, focusing on promising regions.

---

## Usage Examples

### Basic HPO Run
```bash
python hparam_search.py \
    --config configs/lstm_baseline.yaml \
    --study-name lstm_hpo_001
```

### With Custom Config
```bash
python hparam_search.py \
    --config my_custom_config.yaml \
    --study-name custom_study
```

### Expected Output
```
[I 2025-10-14 20:00:00] Trial 0 finished with value: 0.0234
[I 2025-10-14 20:02:15] Trial 1 finished with value: 0.0198
...
[I 2025-10-14 20:20:00] Trial 9 finished with value: 0.0156

Best trial: {'hidden_size': 128, 'num_layers': 2, 'dropout': 0.22, 'lr': 0.00089, 'batch_size': 64}
```

---

## Key Insights

### Why Log-Uniform for Learning Rate?
Learning rates span multiple orders of magnitude. Log-uniform sampling ensures:
- Equal exploration across scales (1e-4, 1e-3, 1e-2)
- Prevents bias toward higher values
- Matches how learning rates affect training (multiplicative effect)

### Why Step=64 for Hidden Size?
- Coarse-grained search reduces search space
- Powers of 2 and multiples align with hardware efficiency
- 64-unit increments provide meaningful capacity differences
- Allows exploring 4 distinct architecture sizes efficiently

### Why Max Dropout=0.4?
- Higher dropout (>0.5) can cause severe underfitting
- 0.4 provides strong regularization without excessive information loss
- Financial time series often need moderate dropout (0.1-0.3)
- Allows algorithm to discover if strong regularization is needed

---

**Document Type:** Reference Tables  
**Last Updated:** October 14, 2025  
**Related:** `HPO_SEARCH_SPACE.md` (full documentation), `HPO_QUICK_REFERENCE.md` (quick guide)
