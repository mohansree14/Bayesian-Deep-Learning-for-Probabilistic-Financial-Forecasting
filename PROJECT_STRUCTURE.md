# Project Structure

This document provides a comprehensive overview of the project organization.

## Directory Tree

```
ML-Intern/
├── app/                              # Web Application
│   └── streamlit_app.py             # Main Streamlit dashboard
│
├── aws/                              # AWS Deployment
│   └── README.md                    # AWS deployment guide
│
├── configs/                          # Configuration Files
│   ├── baseline_lstm.yaml           # LSTM baseline configuration
│   ├── bnn_pyro.yaml               # Bayesian NN configuration
│   ├── mc_dropout.yaml             # MC Dropout configuration
│   ├── model_config.yaml           # General model config
│   ├── train_config.yaml           # Training configuration
│   └── transformer.yaml            # Transformer configuration
│
├── data/                             # Data Storage
│   └── raw/                         # Raw downloaded data
│
├── experiments/                      # Experiment Tracking
│   └── README.md                    # Experiment documentation
│
├── reports/                          # Documentation & Reports
│   ├── figures/                     # Generated figures & plots
│   │   ├── figure_6_*.png/pdf      # AAPL price visualization
│   │   ├── figure_7_*.png/pdf      # Correlation heatmap
│   │   ├── figure_8_*.png/pdf      # Training/validation loss
│   │   ├── figure_9_*.png/pdf      # Uncertainty bands
│   │   ├── figure_10_*.png/pdf     # Cumulative returns
│   │   ├── hpo_table_*.png/pdf     # HPO search space table
│   │   └── technical_indicators_*.png/pdf  # Indicators table
│   │
│   ├── tables/                      # Documentation tables
│   │   ├── HPO_SEARCH_SPACE_TABLE.md
│   │   ├── HPO_TABLE_SINGLE.md
│   │   ├── TECHNICAL_INDICATORS_TABLE.md
│   │   └── README.md
│   │
│   ├── FIGURES_DOCUMENTATION.md     # Detailed figure descriptions
│   ├── HPO_QUICK_REFERENCE.md       # Quick HPO guide
│   ├── HPO_SEARCH_SPACE.md          # Complete HPO documentation
│   ├── PROJECT_ANALYSIS_REPORT.md   # Project analysis
│   ├── QUICK_REFERENCE.md           # Quick project reference
│   ├── README.md                    # Reports index
│   ├── final_report.md              # Final project report
│   ├── intern_plan.md               # Internship plan
│   └── slides.md                    # Presentation slides
│
├── results/                          # Training Results
│   └── README.md                    # Results documentation
│
├── scripts/                          # Utility Scripts
│   ├── fetch_data.py                # Data fetching script
│   └── make_dataset.py              # Dataset creation script
│
├── src/                              # Source Code
│   ├── data/                        # Data Processing
│   │   ├── indicators.py            # Technical indicators (21 indicators)
│   │   └── preprocess.py            # Data preprocessing
│   │
│   ├── evaluation/                  # Model Evaluation
│   │   ├── backtesting.py           # Backtesting framework
│   │   └── metrics.py               # Evaluation metrics
│   │
│   ├── models/                      # Model Implementations
│   │   ├── bnn_vi_pyro.py          # Bayesian Neural Network (Pyro)
│   │   ├── lstm.py                  # LSTM baseline
│   │   ├── mc_dropout_lstm.py       # MC Dropout LSTM
│   │   └── transformer.py           # Transformer model
│   │
│   ├── training/                    # Training Pipeline
│   │   └── train_loop.py            # Training loop implementation
│   │
│   └── utils/                       # Utilities
│       ├── config.py                # Configuration management
│       ├── logging_mlflow.py        # MLflow logging
│       ├── results_logger.py        # Results logging
│       ├── seeding.py               # Random seed management
│       └── visualization.py         # Visualization utilities
│
├── tests/                            # Unit Tests
│   ├── test_metrics.py              # Metrics tests
│   ├── test_models.py               # Model tests
│   └── test_preprocess.py           # Preprocessing tests
│
├── utils/                            # Project Utilities
│   ├── generate_figures.py          # Generate all documentation figures
│   ├── generate_hpo_table.py        # Generate HPO table image
│   └── generate_indicators_table.py # Generate indicators table image
│
├── .editorconfig                     # Editor configuration
├── .gitignore                        # Git ignore rules
├── CHANGELOG.md                      # Version history
├── CONTRIBUTING.md                   # Contribution guidelines
├── Dockerfile                        # Docker configuration
├── LICENSE                           # MIT License
├── MANIFEST.in                       # Package manifest
├── PROJECT_STRUCTURE.md              # This file
├── README.md                         # Main project README
├── backtest.py                       # Backtesting script
├── evaluate.py                       # Model evaluation script
├── hparam_search.py                  # Hyperparameter search
├── requirements.txt                  # Python dependencies
├── run_project.py                    # Main project runner
├── run_streamlit.bat                 # Windows batch launcher
├── run_streamlit.ps1                 # PowerShell launcher
├── setup.py                          # Package setup
├── train.py                          # Training script
├── train_all_tickers.py             # Multi-ticker training
└── train_bnn.py                      # BNN training script
```

## Key Components

### 1. Application Layer (`app/`)
- **streamlit_app.py**: Interactive web dashboard for model deployment and visualization

### 2. Configuration Management (`configs/`)
- YAML-based configuration for all models and training settings
- Supports multiple model architectures (LSTM, Transformer, BNN, MC Dropout)

### 3. Source Code (`src/`)

#### Data Processing (`src/data/`)
- **indicators.py**: 21 technical indicators (SMA, EMA, RSI, MACD, Bollinger Bands, etc.)
- **preprocess.py**: Data cleaning, normalization, and feature engineering

#### Model Implementations (`src/models/`)
- **lstm.py**: Baseline LSTM model
- **mc_dropout_lstm.py**: Monte Carlo Dropout for uncertainty quantification
- **bnn_vi_pyro.py**: Bayesian Neural Network with Pyro (full probabilistic)
- **transformer.py**: Transformer-based architecture

#### Evaluation (`src/evaluation/`)
- **metrics.py**: RMSE, MAE, MAPE, Directional Accuracy, Sharpe Ratio
- **backtesting.py**: Strategy backtesting with uncertainty-aware position sizing

### 4. Documentation (`reports/`)

#### Figures (`reports/figures/`)
- All publication-quality visualizations (PNG + PDF)
- 300 DPI resolution for reports and publications

#### Tables (`reports/tables/`)
- HPO search space documentation
- Technical indicators reference
- Markdown and LaTeX-ready formats

#### Reports
- Comprehensive project analysis
- HPO study documentation
- Quick reference guides

### 5. Utilities (`utils/`)
- Figure generation scripts
- Table generation scripts
- Professional visualization tools

### 6. Training & Evaluation Scripts (Root Level)
- **train.py**: Single model training
- **train_all_tickers.py**: Multi-ticker training
- **train_bnn.py**: Bayesian model training
- **evaluate.py**: Model evaluation
- **backtest.py**: Strategy backtesting
- **hparam_search.py**: Optuna-based hyperparameter optimization

## File Organization Principles

### Configuration Files
- All YAML configs in `configs/`
- Environment-specific configs separate
- Clear naming conventions

### Source Code
- Modular design with clear separation of concerns
- Data, models, training, evaluation in separate modules
- Shared utilities in `src/utils/`

### Documentation
- All documentation in `reports/`
- Figures and tables separated
- README files in each major directory

### Scripts
- Data scripts in `scripts/`
- Visualization scripts in `utils/`
- Training scripts at root level for easy access

### Testing
- All tests in `tests/` directory
- Test file naming: `test_*.py`
- Mirrors source code structure

## Quick Navigation

| Need | Location |
|------|----------|
| Run the app | `python run_project.py` or `run_streamlit.bat` |
| Train a model | `python train.py` |
| Evaluate models | `python evaluate.py` |
| Run backtests | `python backtest.py` |
| HPO study | `python hparam_search.py` |
| Generate figures | `python utils/generate_figures.py` |
| View documentation | `reports/README.md` |
| Configuration | `configs/` |
| Results | `results/` |

## Dependencies

### Core Dependencies
- **PyTorch**: Deep learning framework
- **Pyro**: Probabilistic programming (BNN)
- **Optuna**: Hyperparameter optimization
- **Streamlit**: Web application framework
- **yfinance**: Financial data fetching
- **pandas/numpy**: Data manipulation
- **matplotlib/seaborn**: Visualization
- **scikit-learn**: Preprocessing and metrics

### Development Dependencies
- **pytest**: Unit testing
- **black**: Code formatting
- **flake8**: Linting
- **mypy**: Type checking

## Development Workflow

1. **Setup**: Install dependencies with `pip install -r requirements.txt`
2. **Configuration**: Edit YAML files in `configs/`
3. **Training**: Run training scripts (`train.py`, etc.)
4. **Evaluation**: Use `evaluate.py` or `backtest.py`
5. **Visualization**: Generate figures with `utils/generate_figures.py`
6. **Deployment**: Run Streamlit app with `run_project.py`

## Versioning

This project follows [Semantic Versioning](https://semver.org/):
- **Major**: Breaking changes
- **Minor**: New features (backward compatible)
- **Patch**: Bug fixes

See [CHANGELOG.md](CHANGELOG.md) for version history.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on:
- Code style and standards
- Testing requirements
- Pull request process
- Development workflow

**Author**: Mohansree Vijayakumar  
**Email**: mohansreesk14@gmail.com

---

**Last Updated**: 2025-10-14  
**Version**: 1.0.0
