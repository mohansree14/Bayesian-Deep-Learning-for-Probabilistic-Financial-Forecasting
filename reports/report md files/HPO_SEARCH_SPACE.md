# Hyperparameter Optimization (HPO) Search Space

## Executive Summary

This document provides comprehensive documentation of the hyperparameter search space used in the Optuna-based hyperparameter optimization study for the LSTM regression model. The search space is designed to balance model capacity, regularization, and computational efficiency while exploring a wide range of configurations.

**Optimization Framework:** Optuna (Tree-structured Parzen Estimator - TPE)  
**Objective:** Minimize validation loss (MSE)  
**Number of Trials:** 10  
**Model Architecture:** LSTM Regressor  

---

## Table 1: Complete Hyperparameter Search Space

| Hyperparameter | Type | Search Method | Min Value | Max Value | Step/Options | Default | Description |
|----------------|------|---------------|-----------|-----------|--------------|---------|-------------|
| **hidden_size** | Integer | Uniform | 64 | 256 | 64 | N/A | Number of hidden units in each LSTM layer |
| **num_layers** | Integer | Uniform | 1 | 3 | 1 | N/A | Number of stacked LSTM layers |
| **dropout** | Float | Uniform | 0.0 | 0.4 | Continuous | N/A | Dropout probability for regularization |
| **lr** (Learning Rate) | Float | Log-uniform | 1e-4 | 5e-3 | Continuous | N/A | Adam optimizer learning rate |
| **batch_size** | Categorical | Discrete Choice | N/A | N/A | [32, 64, 128] | N/A | Training batch size |

---

## Table 2: Hyperparameter Details and Rationale

### 1. Hidden Size

| Attribute | Value |
|-----------|-------|
| **Parameter Name** | `hidden_size` |
| **Type** | Integer |
| **Sampling Method** | `trial.suggest_int()` |
| **Range** | [64, 256] |
| **Step Size** | 64 |
| **Possible Values** | 64, 128, 192, 256 |
| **Search Space Size** | 4 discrete values |

**Rationale:**
- **Lower Bound (64):** Sufficient capacity for learning temporal patterns without excessive parameters
- **Upper Bound (256):** Balances model capacity with computational cost and overfitting risk
- **Step Size (64):** Coarse-grained search for efficiency; provides 4 distinct architecture sizes
- **Impact:** Directly affects model capacity and number of trainable parameters

**Trade-offs:**
- **Small values (64-128):** Faster training, less overfitting, may underfit complex patterns
- **Large values (192-256):** Better representational capacity, slower training, higher overfitting risk

---

### 2. Number of Layers

| Attribute | Value |
|-----------|-------|
| **Parameter Name** | `num_layers` |
| **Type** | Integer |
| **Sampling Method** | `trial.suggest_int()` |
| **Range** | [1, 3] |
| **Step Size** | 1 |
| **Possible Values** | 1, 2, 3 |
| **Search Space Size** | 3 discrete values |

**Rationale:**
- **Lower Bound (1):** Single LSTM layer (baseline architecture)
- **Upper Bound (3):** Balances depth benefits with diminishing returns and training difficulty
- **Step Size (1):** Each layer adds significant capacity; incremental steps appropriate
- **Impact:** Affects model depth, hierarchical feature learning capability

**Trade-offs:**
- **1 layer:** Simple, fast, may lack hierarchical feature extraction
- **2 layers:** Good balance between capacity and training stability (common choice)
- **3 layers:** Maximum depth, better hierarchical learning, slower and harder to train

---

### 3. Dropout Probability

| Attribute | Value |
|-----------|-------|
| **Parameter Name** | `dropout` |
| **Type** | Float |
| **Sampling Method** | `trial.suggest_float()` |
| **Range** | [0.0, 0.4] |
| **Distribution** | Uniform |
| **Search Space Size** | Continuous (infinite) |

**Rationale:**
- **Lower Bound (0.0):** No dropout (no regularization)
- **Upper Bound (0.4):** Strong regularization without excessive information loss
- **Distribution:** Uniform sampling allows equal exploration across range
- **Impact:** Regularization strength; controls overfitting

**Trade-offs:**
- **Low values (0.0-0.1):** Minimal regularization, higher overfitting risk, better training performance
- **Medium values (0.1-0.25):** Balanced regularization, typical sweet spot
- **High values (0.25-0.4):** Strong regularization, may cause underfitting

**Common Optimal Range:** 0.1-0.3 for financial time series

---

### 4. Learning Rate

| Attribute | Value |
|-----------|-------|
| **Parameter Name** | `lr` |
| **Type** | Float |
| **Sampling Method** | `trial.suggest_float()` with `log=True` |
| **Range** | [1e-4, 5e-3] |
| **Distribution** | Log-uniform |
| **Search Space Size** | Continuous (infinite) |

**Rationale:**
- **Lower Bound (1e-4 = 0.0001):** Conservative learning rate for stable convergence
- **Upper Bound (5e-3 = 0.005):** Aggressive but manageable learning rate
- **Log-scale:** Learning rate is scale-sensitive; log-uniform explores orders of magnitude efficiently
- **Impact:** Training speed, convergence stability, final performance

**Trade-offs:**
- **Low values (1e-4 to 5e-4):** Slow but stable convergence, less likely to diverge
- **Medium values (5e-4 to 2e-3):** Good balance, typical Adam default is 1e-3
- **High values (2e-3 to 5e-3):** Fast convergence, risk of instability or overshooting

**Why Log-scale?**
- Learning rates span multiple orders of magnitude
- Small changes at low values are as important as large changes at high values
- Example: 1e-4 to 2e-4 (2× increase) vs 4e-3 to 5e-3 (1.25× increase)

---

### 5. Batch Size

| Attribute | Value |
|-----------|-------|
| **Parameter Name** | `batch_size` |
| **Type** | Categorical |
| **Sampling Method** | `trial.suggest_categorical()` |
| **Choices** | [32, 64, 128] |
| **Search Space Size** | 3 discrete values |

**Rationale:**
- **32:** Small batches, noisy gradients, better generalization, slower training
- **64:** Balanced choice, good trade-off between stability and efficiency
- **128:** Large batches, stable gradients, faster training per epoch, may generalize less well
- **Impact:** Training dynamics, memory usage, generalization

**Trade-offs:**
- **32:** More gradient updates per epoch, noisy but regularizing, 3× more iterations
- **64:** Standard choice, good balance (often optimal)
- **128:** Fewer but more stable gradient updates, 1.5× faster per epoch than 32

**Why These Values?**
- Power of 2 (GPU efficiency)
- Cover typical range for time series tasks
- Limited by memory constraints and sequence length

---

## Table 3: Search Space Cardinality

| Component | Size | Type |
|-----------|------|------|
| **hidden_size** | 4 | Discrete |
| **num_layers** | 3 | Discrete |
| **dropout** | ∞ | Continuous |
| **lr** | ∞ | Continuous |
| **batch_size** | 3 | Discrete |
| **Discrete Combinations** | 4 × 3 × 3 = 36 | Finite |
| **Total Search Space** | ∞ (due to continuous params) | Infinite |
| **Effective Trials** | 10 | Limited by computational budget |

**Note:** While the search space is technically infinite due to continuous parameters, the 10 trials explore a representative sample using Optuna's TPE algorithm.

---

## Table 4: Parameter Importance and Sensitivity

| Parameter | Impact on Performance | Computational Cost | Optimization Priority | Typical Optimal Range |
|-----------|----------------------|-------------------|----------------------|----------------------|
| **Learning Rate** | **High** | Low | **Critical** | 5e-4 to 2e-3 |
| **Hidden Size** | **High** | **High** | **Critical** | 128-192 |
| **Dropout** | **Medium** | Low | **Important** | 0.1-0.3 |
| **Num Layers** | **Medium** | **High** | **Important** | 2 |
| **Batch Size** | **Low** | Medium | Moderate | 64 |

**Priority Ranking:**
1. **Learning Rate:** Most sensitive; wrong value can prevent convergence
2. **Hidden Size:** Determines model capacity; critical for performance
3. **Dropout:** Regularization; important for generalization
4. **Num Layers:** Architectural depth; moderate impact
5. **Batch Size:** Least critical; affects training dynamics more than final performance

---

## Table 5: Example Configurations

| Config | Hidden | Layers | Dropout | LR | Batch | Description | Est. Parameters |
|--------|--------|--------|---------|-----|-------|-------------|-----------------|
| **Minimal** | 64 | 1 | 0.0 | 5e-3 | 128 | Fastest training, baseline | ~50K |
| **Balanced** | 128 | 2 | 0.2 | 1e-3 | 64 | Good trade-off, recommended | ~200K |
| **High Capacity** | 256 | 3 | 0.3 | 5e-4 | 32 | Max capacity, slower | ~800K |
| **Conservative** | 64 | 1 | 0.4 | 1e-4 | 128 | Strong regularization | ~50K |
| **Aggressive** | 256 | 2 | 0.1 | 5e-3 | 128 | Fast learning, risky | ~500K |

**Note:** Parameter counts are approximate and depend on input dimensionality.

---

## Optimization Strategy

### Optuna TPE Algorithm

**Tree-structured Parzen Estimator (TPE)** is used for Bayesian optimization:

1. **Exploration vs Exploitation:** Balances trying new regions vs refining promising areas
2. **Sequential Model-Based Optimization:** Uses previous trials to inform next suggestions
3. **Non-parametric:** No assumptions about search space structure
4. **Efficient:** Finds good configurations with fewer trials than random/grid search

### Optimization Process

```
For each trial (1 to 10):
    1. TPE suggests hyperparameter configuration
    2. Build LSTM model with suggested parameters
    3. Train for 5 epochs on training set
    4. Evaluate on validation set
    5. Record validation loss
    6. TPE updates internal model
```

**Objective:** Minimize final validation loss after 5 epochs

---

## Constraints and Considerations

### 1. **Computational Constraints**

| Constraint | Value | Rationale |
|------------|-------|-----------|
| **Max Trials** | 10 | Limited computational budget |
| **Epochs per Trial** | 5 | Quick evaluation; full training uses early stopping |
| **Device** | CPU | Accessibility; GPU would allow more trials |
| **Gradient Clipping** | 1.0 | Prevents exploding gradients (fixed) |

### 2. **Fixed Parameters**

The following parameters are **not** searched:

| Parameter | Fixed Value | Rationale |
|-----------|-------------|-----------|
| **Optimizer** | Adam | Adaptive learning rate; good default |
| **Gradient Clip** | 1.0 | Prevents training instability |
| **Epochs** | 5 (HPO) / Variable (full training) | Quick HPO evaluation |
| **Loss Function** | MSE | Standard for regression |
| **Sequence Length** | From data preprocessing | Task-dependent |
| **Input Dimension** | From feature engineering | Data-dependent |

### 3. **Search Space Design Principles**

1. **Completeness:** Cover architecturally distinct configurations
2. **Efficiency:** Limit search to computationally feasible range
3. **Practicality:** Focus on ranges known to work well for financial time series
4. **Balance:** Mix discrete (architecture) and continuous (optimization) parameters

---

## Usage Example

### Running HPO Study

```bash
# Using configuration file
python hparam_search.py \
    --config configs/lstm_baseline.yaml \
    --study-name lstm_hpo_study_001

# Output: Best trial parameters printed to console
```

### Configuration File Example

```yaml
# configs/lstm_baseline.yaml
data:
  tickers: ["AAPL"]
  data_dir: "data/processed"

training:
  epochs: 5  # For HPO; full training uses more
  
seed: 42
```

### Expected Output

```
Best trial: {
    'hidden_size': 128,
    'num_layers': 2,
    'dropout': 0.22,
    'lr': 0.00089,
    'batch_size': 64
}
```

---

## Interpretation Guidelines

### How to Interpret Results

1. **Hidden Size & Layers:**
   - If best config has max values → model may benefit from larger architecture
   - If best config has min values → simpler model sufficient

2. **Dropout:**
   - High dropout (>0.3) → model prone to overfitting
   - Low dropout (<0.1) → regularization not critical

3. **Learning Rate:**
   - High LR (>2e-3) → fast convergence possible
   - Low LR (<5e-4) → requires careful, slow training

4. **Batch Size:**
   - Small (32) → noisy gradients beneficial
   - Large (128) → stable training preferred

### Validation

After HPO, validate best configuration:
1. Retrain with full epochs (e.g., 50-100)
2. Test on held-out test set
3. Compare with baseline configurations
4. Analyze training curves for overfitting

---

## Extensions and Future Work

### Potential Search Space Expansions

1. **Additional Architecture Parameters:**
   - Bidirectional LSTM (True/False)
   - Layer normalization (True/False)
   - Recurrent dropout vs standard dropout

2. **Optimizer Variations:**
   - Choice between Adam, AdamW, SGD
   - Momentum values
   - Weight decay (L2 regularization)

3. **Advanced Regularization:**
   - Gradient noise
   - Layer-wise learning rates
   - Warmup schedules

4. **Data Augmentation:**
   - Sequence length
   - Feature scaling methods
   - Time series jittering

### Scaling HPO

For production systems:
- Increase trials to 50-100
- Use distributed optimization (Optuna supports this)
- Consider multi-objective optimization (loss + training time)
- Add pruning for early trial termination

---

## References

1. **Optuna Documentation:** [https://optuna.org/](https://optuna.org/)
2. **TPE Algorithm:** Bergstra et al. (2011) "Algorithms for Hyper-Parameter Optimization"
3. **LSTM Architecture:** Hochreiter & Schmidhuber (1997) "Long Short-Term Memory"
4. **Hyperparameter Tuning Best Practices:** Goodfellow et al. (2016) "Deep Learning"

---

## Related Documentation

- **Project Report:** `reports/PROJECT_ANALYSIS_REPORT.md`
- **Training Pipeline:** `src/training/train_loop.py`
- **Model Architecture:** `src/models/lstm.py`
- **Configuration Files:** `configs/lstm_baseline.yaml`
- **HPO Script:** `hparam_search.py`

---

**Document Version:** 1.0  
**Last Updated:** October 14, 2025  
**Author:** ML Project Documentation  
**Purpose:** Comprehensive reference for Optuna HPO search space configuration
