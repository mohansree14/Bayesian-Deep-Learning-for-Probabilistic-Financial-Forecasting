# ML Results Management System

This directory contains all outputs, logs, and visualizations from ML experiments.

## 📁 Directory Structure

```
results/
├── logs/                          # Detailed experiment logs
├── plots/                         # Generated visualizations
│   ├── training/                  # Training progress plots
│   ├── forecasts/                 # Forecast vs actual plots
│   ├── backtesting/              # Backtesting results
│   └── metrics/                  # Performance metrics
├── models/                        # Saved model checkpoints
├── data/                         # Processed datasets
├── reports/                      # Generated reports
└── experiment_YYYYMMDD_HHMMSS/   # Individual experiment folders
    ├── config.json              # Experiment configuration
    ├── training_history.json    # Training metrics
    ├── evaluation_metrics.json  # Model performance
    ├── backtest_results.json    # Trading performance
    ├── *.png                    # Generated visualizations
    └── experiment_summary.json  # Complete experiment summary
```

## 🚀 Quick Start

### Generate Complete Results
```bash
# Run full pipeline with visualizations
python generate_results.py --config configs/lstm_baseline.yaml

# Skip specific steps
python generate_results.py --config configs/lstm_baseline.yaml --skip-training
```

### View Results
```bash
# List all experiments
python view_results.py --list

# Show latest experiment details
python view_results.py --latest

# Show specific experiment
python view_results.py --experiment experiment_20250928_222146

# Open plots folder
python view_results.py --open-plots

# Open logs folder
python view_results.py --open-logs
```

## 📊 Generated Visualizations

### Training Plots
- **Training History**: Loss curves for training and validation
- **Model Architecture**: Parameter count and structure info

### Evaluation Plots
- **Forecast vs Actual**: Time series predictions vs ground truth
- **Residual Analysis**: 
  - Residuals vs Predicted values
  - Residual distribution histogram
  - Q-Q plot for normality
  - Residuals over time
- **Metrics Comparison**: Bar chart of performance metrics

### Backtesting Plots
- **Forecast with Uncertainty**: Predictions with confidence intervals
- **Backtesting Results**:
  - Cumulative returns
  - Daily returns
  - Drawdown analysis
  - Returns distribution

## 📈 Performance Metrics

### Model Performance
- **RMSE**: Root Mean Square Error
- **MAE**: Mean Absolute Error
- **MAPE**: Mean Absolute Percentage Error
- **R²**: Coefficient of determination

### Trading Performance
- **Sharpe Ratio**: Risk-adjusted returns
- **VaR 95%**: Value at Risk (95% confidence)
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Win Rate**: Percentage of profitable trades

## 🔧 Configuration

All experiments are configured via YAML files in the `configs/` directory:

```yaml
# Example: configs/lstm_baseline.yaml
seed: 42
data:
  tickers: [AAPL]
  window: 64
  horizon: 1
  target_col: close
training:
  batch_size: 64
  lr: 0.001
  epochs: 5
model:
  type: lstm
  hidden_size: 128
  num_layers: 2
```

## 📝 Logging

### Experiment Logs
- **Detailed logs**: `logs/experiment_YYYYMMDD_HHMMSS.log`
- **General logs**: `logs/general.log`
- **Error tracking**: Comprehensive error logging with context

### Log Levels
- **INFO**: General information about experiment progress
- **DEBUG**: Detailed debugging information
- **ERROR**: Error messages with full tracebacks
- **WARNING**: Warning messages for potential issues

## 🎯 Best Practices

### Experiment Organization
1. **Unique IDs**: Each experiment gets a unique timestamp-based ID
2. **Complete Tracking**: All inputs, outputs, and intermediate results saved
3. **Reproducibility**: Full configuration and random seeds preserved
4. **Version Control**: Model checkpoints with metadata

### Visualization Standards
1. **Professional Style**: Consistent color scheme and formatting
2. **High Resolution**: 300 DPI for publication quality
3. **Clear Labels**: Descriptive titles and axis labels
4. **Multiple Formats**: Both PNG and data files saved

### Performance Monitoring
1. **Real-time Logging**: Progress tracked during training
2. **Comprehensive Metrics**: Multiple evaluation criteria
3. **Visual Validation**: Plots for quick assessment
4. **Error Handling**: Graceful failure with detailed logs

## 🔍 Troubleshooting

### Common Issues
1. **Missing Dependencies**: Ensure all packages installed via `requirements.txt`
2. **Path Issues**: Use `PYTHONPATH=.` when running scripts
3. **Memory Issues**: Reduce batch size or sequence length
4. **Visualization Errors**: Check matplotlib backend and dependencies

### Debug Commands
```bash
# Check experiment status
python view_results.py --latest

# View specific log file
python view_results.py --open logs/experiment_YYYYMMDD_HHMMSS.log

# Re-run specific step
python train.py --config configs/lstm_baseline.yaml
python evaluate.py --config configs/lstm_baseline.yaml
python backtest.py --config configs/lstm_baseline.yaml --model lstm
```

## 📊 Example Results

### Training Progress
- **Epoch 1**: Train Loss = 0.330, Valid Loss = 0.050
- **Epoch 5**: Train Loss = 0.013, Valid Loss = 0.009
- **Convergence**: Model shows good convergence with decreasing loss

### Model Performance
- **RMSE**: 0.287 (lower is better)
- **MAE**: 0.260 (lower is better)
- **MAPE**: 14.7% (lower is better)
- **R²**: 0.065 (higher is better, but low for financial data is common)

### Trading Performance
- **Sharpe Ratio**: -0.085 (negative indicates poor risk-adjusted returns)
- **VaR 95%**: 0.069 (6.9% daily loss at 95% confidence)

## 🎉 Success!

The results system provides:
- ✅ **Complete Experiment Tracking**: Every run is logged and saved
- ✅ **Professional Visualizations**: Publication-quality plots
- ✅ **Comprehensive Metrics**: Multiple performance indicators
- ✅ **Easy Navigation**: Simple commands to view and analyze results
- ✅ **Reproducibility**: Full configuration and data preservation
- ✅ **Error Handling**: Robust error tracking and recovery

Your ML experiments are now fully organized and professionally documented! 🚀

