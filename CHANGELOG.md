# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2025-10-14

### Added
- **Model Architectures**
  - LSTM Baseline implementation
  - MC Dropout LSTM for uncertainty quantification
  - Bayesian Neural Network (Pyro-based)
  - Transformer architecture for time series

- **Data Pipeline**
  - Yahoo Finance data fetching (`scripts/fetch_data.py`)
  - Technical indicators calculation (21 indicators)
  - Data preprocessing and normalization
  - Train/validation/test splitting

- **Training Infrastructure**
  - Modular training loop with early stopping
  - Gradient clipping for RNN stability
  - Model checkpointing
  - Configuration-based experiments

- **Evaluation & Analysis**
  - Comprehensive metrics (RMSE, MAE, R², MAPE)
  - Backtesting system with strategy evaluation
  - Uncertainty-aware position sizing
  - Performance visualization

- **Hyperparameter Optimization**
  - Optuna integration for HPO
  - TPE algorithm for efficient search
  - 5-parameter search space (hidden_size, num_layers, dropout, lr, batch_size)

- **Documentation**
  - Comprehensive README.md
  - PROJECT_ANALYSIS_REPORT.md (detailed technical analysis)
  - QUICK_REFERENCE.md (at-a-glance guide)
  - FIGURES_DOCUMENTATION.md (all visualizations documented)
  - HPO_SEARCH_SPACE.md (hyperparameter details)
  - TECHNICAL_INDICATORS_TABLE.md (all 21 indicators with formulas)
  - CONTRIBUTING.md (contribution guidelines)

- **Visualizations**
  - Figure 6: AAPL Price History (2015-2024)
  - Figure 7: Feature Correlation Heatmap
  - Figure 8: Training/Validation Loss Curves
  - Figure 9: Uncertainty Bands (COVID-19 demonstration)
  - Figure 10: Cumulative Returns Comparison
  - Professional table images for reports

- **Web Application**
  - Streamlit interactive dashboard
  - Model prediction interface
  - Visualization panels
  - Configuration options

- **Testing**
  - Unit tests for models (`test_models.py`)
  - Unit tests for preprocessing (`test_preprocess.py`)
  - Unit tests for metrics (`test_metrics.py`)
  - Project launcher with test execution

- **Utilities**
  - Figure generation scripts
  - Table image generation
  - Configuration management
  - Results logging

### Project Organization
- Organized code into professional structure
- Created `utils/` for utility scripts
- Established `reports/` for documentation
- Set up `configs/` for YAML configurations
- Implemented `src/` modular architecture

### Configuration Files
- `configs/lstm_baseline.yaml`
- `configs/mc_dropout_lstm.yaml`
- `configs/bnn_vi.yaml`
- `configs/transformer_baseline.yaml`
- `configs/app.yaml`

### Scripts
- `train.py` - Main training script
- `evaluate.py` - Model evaluation
- `backtest.py` - Strategy backtesting
- `hparam_search.py` - Hyperparameter optimization
- `run_project.py` - Project launcher (tests + app)
- `utils/generate_figures.py` - Documentation figures
- `utils/generate_hpo_table.py` - HPO table image
- `utils/generate_indicators_table.py` - Indicators table image

## [0.1.0] - Initial Development

### Added
- Basic project structure
- Initial model implementations
- Data fetching capabilities
- Simple training scripts

---

## Version History

- **v1.0.0** (2025-10-14): Production-ready release with comprehensive documentation
- **v0.1.0** (Development): Initial implementation phase
