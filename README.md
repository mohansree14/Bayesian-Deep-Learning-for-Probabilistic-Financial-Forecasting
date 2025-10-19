# Financial Time Series Forecasting with Uncertainty Quantification

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-ee4c2c.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.0+-FF4B4B.svg)](https://streamlit.io/)
[![Live Demo](https://img.shields.io/badge/🚀%20Live%20Demo-Streamlit%20App-FF4B4B?style=for-the-badge)](https://bayesian-financial-forecasting.streamlit.app/)

A comprehensive machine learning project for stock price prediction using LSTM, MC Dropout, Bayesian Neural Networks, and Transformers with probabilistic uncertainty estimation.

## 🚀 Live Demo

**Try the interactive web application:** [https://bayesian-financial-forecasting.streamlit.app/](https://bayesian-financial-forecasting.streamlit.app/)

Explore real-time stock price predictions with uncertainty quantification for 8 major stocks (AAPL, MSFT, GOOGL, TSLA, NVDA, AMZN, META, NFLX) using our deployed Streamlit application.

**Author**: Mohansree Vijayakumar  
**Email**: mohansreesk14@gmail.com

---

## 📋 Table of Contents

- [Live Demo](#-live-demo)
- [Project Overview](#-project-overview)
- [Quick Start](#-quick-start)
- [Project Structure](#-project-structure)
- [Model Architectures](#️-model-architectures)
- [Technical Indicators](#-technical-indicators-21-total)
- [Documentation](#-documentation)
- [Results](#-results)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🎯 Project Overview

This project implements state-of-the-art deep learning models for financial time series forecasting with a focus on **uncertainty quantification**. The system provides not just point predictions, but confidence intervals that widen during volatile market periods, enabling better risk-aware trading decisions.

### Key Features

- ✅ **Multiple Model Architectures**: LSTM, MC Dropout LSTM, Bayesian Neural Networks (Pyro), Transformers
- ✅ **Uncertainty Quantification**: Probabilistic forecasting with confidence intervals
- ✅ **21 Technical Indicators**: Comprehensive feature engineering
- ✅ **Hyperparameter Optimization**: Optuna-based automated tuning
- ✅ **Backtesting System**: Strategy evaluation with uncertainty-aware position sizing
- ✅ **Interactive Dashboard**: Streamlit web application
- ✅ **Professional Documentation**: Detailed reports and visualizations

## 📁 Project Structure

```
ML-Intern/
├── app/                          # Streamlit web application
│   └── streamlit_app.py
├── configs/                      # Model and experiment configurations
│   ├── lstm_baseline.yaml
│   ├── mc_dropout.yaml
│   ├── bnn_vi.yaml
│   └── transformer_baseline.yaml
├── data/                         # Data storage (gitignored)
│   ├── raw/
│   └── processed/
├── reports/                      # Documentation and analysis
│   ├── figures/                  # Generated visualizations
│   ├── tables/                   # Reference tables
│   ├── PROJECT_ANALYSIS_REPORT.md
│   ├── QUICK_REFERENCE.md
│   ├── FIGURES_DOCUMENTATION.md
│   └── HPO_SEARCH_SPACE.md
├── results/                      # Experiment results (gitignored)
├── scripts/                      # Data preparation scripts
│   ├── fetch_data.py
│   └── make_dataset.py
├── src/                          # Source code
│   ├── data/                     # Data processing
│   │   ├── indicators.py
│   │   └── preprocess.py
│   ├── evaluation/               # Model evaluation
│   │   ├── backtesting.py
│   │   └── metrics.py
│   ├── models/                   # Model implementations
│   │   ├── lstm.py
│   │   ├── mc_dropout_lstm.py
│   │   ├── bnn_vi_pyro.py
│   │   └── transformer.py
│   ├── training/                 # Training pipeline
│   │   └── train_loop.py
│   └── utils/                    # Utilities
│       ├── config.py
│       ├── logging_mlflow.py
│       └── visualization.py
├── tests/                        # Unit tests
│   ├── test_metrics.py
│   ├── test_models.py
│   └── test_preprocess.py
├── utils/                        # Utility scripts
│   ├── generate_figures.py       # Generate documentation figures
│   ├── generate_hpo_table.py     # Generate HPO table image
│   └── generate_indicators_table.py  # Generate indicators table
├── train.py                      # Main training script
├── evaluate.py                   # Model evaluation script
├── backtest.py                   # Backtesting script
├── hparam_search.py             # Hyperparameter optimization
├── run_project.py               # Project launcher (tests + app)
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- pip package manager

### Installation

1. Clone or download this repository
2. Install dependencies:

```bash
pip install -r requirements.txt
```

### Running the Project

**Option 1: Run Everything (Tests + Web App)**
```bash
python run_project.py
```

**Option 2: Run Web Application Only**
```bash
streamlit run app/streamlit_app.py
```
or
```bash
.\run_streamlit.bat  # Windows batch file
.\run_streamlit.ps1  # PowerShell script
```
**Option 3: Access the Live Streamlit App**
Visit our hosted Streamlit app:
[https://bayesian-financial-forecasting.streamlit.app/](https://bayesian-financial-forecasting.streamlit.app/)

## 📊 Usage Examples

### 1. Train a Model

```bash
# LSTM Baseline
python train.py --config configs/lstm_baseline.yaml

# MC Dropout LSTM
python train.py --config configs/mc_dropout_lstm.yaml

# Bayesian Neural Network
python train_bnn.py --config configs/bnn_vi.yaml

# Transformer
python train.py --config configs/transformer_baseline.yaml
```

### 2. Hyperparameter Optimization

```bash
python hparam_search.py --config configs/lstm_baseline.yaml --study-name my_study
```

### 3. Evaluate Model

```bash
python evaluate.py --config configs/lstm_baseline.yaml --checkpoint results/experiment_XXXXX/best_model.pt
```

### 4. Run Backtesting

```bash
python backtest.py --config configs/mc_dropout.yaml --checkpoint results/experiment_XXXXX/best_model.pt
```

### 5. Generate Documentation Figures

```bash
# Generate all figures (6, 7, 8, 9, 10)
python utils/generate_figures.py

# Generate HPO table
python utils/generate_hpo_table.py

# Generate technical indicators table
python utils/generate_indicators_table.py
```

## 🏗️ Model Architectures

### 1. LSTM Baseline
- Standard LSTM for time series forecasting
- Dropout regularization
- Configuration: `configs/lstm_baseline.yaml`

### 2. MC Dropout LSTM
- Monte Carlo Dropout for uncertainty estimation
- Multiple forward passes at inference
- Provides prediction mean and variance
- Configuration: `configs/mc_dropout_lstm.yaml`

### 3. Bayesian Neural Network (Pyro)
- Full Bayesian treatment with Pyro
- Variational inference
- Posterior distribution over weights
- Configuration: `configs/bnn_vi.yaml`

### 4. Transformer
- Attention-based architecture
- Multi-head self-attention
- Positional encoding for sequences
- Configuration: `configs/transformer_baseline.yaml`

## 📈 Technical Indicators (21 Total)

The system uses 21 technical indicators across 6 categories:

- **Price Features** (5): Open, High, Low, Close, Adj Close
- **Volume** (1): Trading volume
- **Returns** (2): Daily return, Log return
- **Trend** (4): SMA(10), SMA(20), EMA(12), EMA(26)
- **Momentum** (6): RSI, MACD, MACD Signal, MACD Histogram, Stochastic %K, %D
- **Volatility** (3): Bollinger Bands (Upper, Middle, Lower)

See `reports/tables/TECHNICAL_INDICATORS_TABLE.md` for detailed formulas.

## 🔬 Hyperparameter Search Space

Optuna-based optimization with the following search space:

| Parameter | Type | Range | Distribution |
|---|---|---|---|
| Hidden Size | Integer | 64-256 (step 64) | Uniform |
| Num Layers | Integer | 1-3 | Uniform |
| Dropout | Float | 0.0-0.4 | Uniform |
| Learning Rate | Float | 1e-4 to 5e-3 | Log-Uniform |
| Batch Size | Categorical | [32, 64, 128] | Discrete |

See `reports/HPO_SEARCH_SPACE.md` for complete details.

## 📊 Key Results

### Performance Highlights

- **Validation Loss**: ~0.009 MSE (LSTM Baseline)
- **Uncertainty Calibration**: Bands widen 1.48x during COVID-19 crash
- **Risk-Adjusted Returns**: 
  - Uncertainty-aware strategy: 22.7% lower volatility
  - 22.4% smaller maximum drawdown
  - Sharpe ratio improvement: 0.88 vs 0.78

### Visualizations

All figures available in `reports/figures/`:
- **Figure 6**: AAPL Price History (2015-2024)
- **Figure 7**: Feature Correlation Heatmap
- **Figure 8**: Training/Validation Loss Curves
- **Figure 9**: Uncertainty Bands (COVID-19 volatility demonstration)
- **Figure 10**: Cumulative Returns Comparison

## 📚 Documentation

### Comprehensive Reports
- **[PROJECT_ANALYSIS_REPORT.md](reports/PROJECT_ANALYSIS_REPORT.md)**: Complete project analysis
- **[FIGURES_DOCUMENTATION.md](reports/FIGURES_DOCUMENTATION.md)**: Detailed figure documentation
- **[HPO_SEARCH_SPACE.md](reports/HPO_SEARCH_SPACE.md)**: Hyperparameter optimization details
- **[TECHNICAL_INDICATORS_TABLE.md](reports/tables/TECHNICAL_INDICATORS_TABLE.md)**: All 21 indicators with formulas

### Quick References
- **[QUICK_REFERENCE.md](reports/QUICK_REFERENCE.md)**: At-a-glance project guide
- **[HPO_QUICK_REFERENCE.md](reports/HPO_QUICK_REFERENCE.md)**: HPO quick guide

## 🧪 Testing

Run unit tests:

```bash
# All tests
python -m pytest tests/

# Specific test file
python -m pytest tests/test_models.py

# With coverage
python -m pytest tests/ --cov=src
```

Or use the project launcher:
```bash
python run_project.py  # Runs tests first, then launches app
```

## 🛠️ Configuration

All experiments use YAML configuration files in `configs/`:

```yaml
# Example: configs/lstm_baseline.yaml
data:
  tickers: ["AAPL"]
  data_dir: "data/processed"
  train_ratio: 0.7
  valid_ratio: 0.15

model:
  type: "lstm"
  hidden_size: 128
  num_layers: 2
  dropout: 0.1

training:
  epochs: 50
  batch_size: 64
  learning_rate: 0.001
  early_stopping_patience: 10

seed: 42
```

## 📦 Dependencies

Core libraries:
- **PyTorch**: Deep learning framework
- **Pandas/NumPy**: Data manipulation
- **yfinance**: Financial data fetching
- **Streamlit**: Web dashboard
- **Optuna**: Hyperparameter optimization
- **Pyro**: Bayesian deep learning
- **Matplotlib/Seaborn**: Visualization
- **MLflow**: Experiment tracking (optional)

See `requirements.txt` for complete list.

## 🎯 Key Innovations

1. **Uncertainty Quantification**: Not just predictions, but confidence intervals
2. **Volatility-Aware Trading**: Dynamic position sizing based on uncertainty
3. **Comprehensive Technical Analysis**: 21 engineered features
4. **Multiple Architectures**: Compare LSTM, Bayesian, Transformer approaches
5. **Professional Documentation**: Publication-ready reports and figures

## 📝 Citation

If you use this project in your research or work, please cite:

```
Financial Time Series Forecasting with Uncertainty Quantification
ML Internship Project, 2025
GitHub: [Your Repository URL]
```

## 🤝 Contributing

This is an educational project. Suggestions and improvements are welcome!

## 📄 License

This project is for educational and research purposes.

## 🙏 Acknowledgments

- **Data Source**: Yahoo Finance (yfinance library)
- **Frameworks**: PyTorch, Pyro, Streamlit, Optuna
- **Inspiration**: Modern deep learning research in finance and uncertainty quantification

## 📞 Contact

For questions or collaboration opportunities, please reach out via GitHub issues.

---

**Last Updated**: October 14, 2025  
**Version**: 1.0  
**Status**: Production-Ready ✅
