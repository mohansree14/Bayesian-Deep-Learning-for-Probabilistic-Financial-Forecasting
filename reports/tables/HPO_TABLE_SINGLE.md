# Hyperparameter Optimization Search Space

## Table: Optuna HPO Search Space Configuration

| Hyperparameter | Type | Minimum Value | Maximum Value | Step Size | Possible Values | Sampling Distribution | Total Configurations |
|---|---|---|---|---|---|---|---|
| **Hidden Size** | Integer | 64 | 256 | 64 | 64, 128, 192, 256 | Uniform | 4 |
| **Number of Layers** | Integer | 1 | 3 | 1 | 1, 2, 3 | Uniform | 3 |
| **Dropout** | Float | 0.0 | 0.4 | Continuous | 0.0–0.4 | Uniform | ∞ |
| **Learning Rate** | Float | 0.0001 | 0.005 | Continuous | 1×10⁻⁴ to 5×10⁻³ | **Log-Uniform** | ∞ |
| **Batch Size** | Categorical | — | — | — | 32, 64, 128 | Discrete Choice | 3 |

**Optimization Details:**
- **Framework:** Optuna v3.x
- **Algorithm:** Tree-structured Parzen Estimator (TPE)
- **Objective:** Minimize Validation Loss (MSE)
- **Number of Trials:** 10
- **Model:** LSTM Regressor

**Search Space Size:**
- Discrete combinations: 4 × 3 × 3 = 36
- Total search space: Infinite (continuous parameters)
- Coverage: 10 trials with intelligent Bayesian sampling

---

**Note:** Learning rate uses log-uniform distribution to efficiently explore orders of magnitude (10⁻⁴, 10⁻³, 10⁻²).
