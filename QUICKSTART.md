# Quick Start Guide

This is a quick reference guide for common tasks in the Financial Time Series Forecasting project.

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/financial-ts-forecasting.git
cd financial-ts-forecasting

# Install dependencies
pip install -r requirements.txt

# Or install as a package
pip install -e .
```

## Running the Application

### Option 1: Python Script
```bash
python run_project.py
```

### Option 2: Windows Batch File
```bash
run_streamlit.bat
```

### Option 3: PowerShell Script
```bash
.\run_streamlit.ps1
```

The application will open in your browser at `http://localhost:8501`

## Training Models

### Single Model Training
```bash
python train.py
```

### Train All Tickers
```bash
python train_all_tickers.py
```

### Train Bayesian Neural Network
```bash
python train_bnn.py
```

## Model Evaluation

### Evaluate Trained Models
```bash
python evaluate.py
```

### Run Backtesting
```bash
python backtest.py
```

## Hyperparameter Optimization

```bash
python hparam_search.py
```

This will run an Optuna study to find optimal hyperparameters.

## Generate Documentation Figures

```bash
# Generate all figures
python utils/generate_figures.py

# Generate HPO table
python utils/generate_hpo_table.py

# Generate technical indicators table
python utils/generate_indicators_table.py
```

## Configuration

All configuration files are in the `configs/` directory:

- `baseline_lstm.yaml` - LSTM baseline configuration
- `mc_dropout.yaml` - MC Dropout configuration
- `bnn_pyro.yaml` - Bayesian Neural Network configuration
- `transformer.yaml` - Transformer configuration
- `train_config.yaml` - Training parameters

Edit these files to customize model behavior.

## Data

### Fetch New Data
```bash
python scripts/fetch_data.py
```

### Prepare Dataset
```bash
python scripts/make_dataset.py
```

Data is stored in the `data/` directory.

## Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ -v --cov=src --cov-report=html

# Run specific test file
python -m pytest tests/test_models.py -v
```

## Code Quality

### Format Code
```bash
black src/ tests/ utils/ --line-length=100
isort src/ tests/ utils/ --profile black
```

### Lint Code
```bash
flake8 src/ tests/ --max-line-length=100
mypy src/ --ignore-missing-imports
```

## Docker

### Build Docker Image
```bash
docker build -t financial-ts-forecasting .
```

### Run Docker Container
```bash
docker run -p 8501:8501 financial-ts-forecasting
```

## Project Structure

```
ML-Intern/
├── app/                  # Streamlit application
├── configs/              # Configuration files
├── data/                 # Data storage
├── reports/              # Documentation & figures
├── scripts/              # Utility scripts
├── src/                  # Source code
├── tests/                # Unit tests
├── utils/                # Project utilities
├── train.py              # Training script
├── evaluate.py           # Evaluation script
└── run_project.py        # Main runner
```

## Common Issues

### Issue: Missing Dependencies
**Solution**: Run `pip install -r requirements.txt`

### Issue: CUDA Out of Memory
**Solution**: Reduce batch size in `configs/train_config.yaml`

### Issue: Data Not Found
**Solution**: Run `python scripts/fetch_data.py` to download data

### Issue: Streamlit Port Already in Use
**Solution**: Kill the process or change port:
```bash
streamlit run app/streamlit_app.py --server.port 8502
```

## Documentation

- **Full README**: [README.md](README.md)
- **Project Structure**: [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)
- **Contributing Guide**: [CONTRIBUTING.md](CONTRIBUTING.md)
- **Change Log**: [CHANGELOG.md](CHANGELOG.md)
- **Reports**: [reports/README.md](reports/README.md)
- **HPO Documentation**: [reports/HPO_SEARCH_SPACE.md](reports/HPO_SEARCH_SPACE.md)
- **Figures Documentation**: [reports/FIGURES_DOCUMENTATION.md](reports/FIGURES_DOCUMENTATION.md)

## Support

For issues, questions, or contributions:

**Author**: Mohansree Vijayakumar  
**Email**: mohansreesk14@gmail.com

1. Check the [documentation](reports/README.md)
2. Review [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines
3. Open an issue on GitHub

---

**Version**: 1.0.0  
**Last Updated**: 2025-10-14
| Run app | `python run_project.py` |
| Train model | `python train.py` |
| Evaluate | `python evaluate.py` |
| Backtest | `python backtest.py` |
| HPO | `python hparam_search.py` |
| Test | `pytest tests/ -v` |
| Format | `black src/ --line-length=100` |
| Lint | `flake8 src/ --max-line-length=100` |

---

**Version**: 1.0.0  
**Last Updated**: 2025-10-14
