# Uncertainty-Aware Bayesian Deep Learning for Financial Time Series Forecasting

## Abstract
We study uncertainty-aware forecasting for financial OHLCV series using deterministic baselines (LSTM, Transformer) and Bayesian methods (Monte Carlo Dropout, Pyro variational Bayesian heads). We evaluate accuracy (RMSE/MAE) and probabilistic quality (NLL, coverage, sharpness), and backtest uncertainty-aware trading rules (Sharpe, VaR).

## Data and Preprocessing
- Data: `yfinance` OHLCV for {TICKERS}, {START_DATE}–{END_DATE}
- Features: returns, SMA/EMA, RSI, MACD, Bollinger Bands, Stochastic Oscillator
- Scaling: StandardScaler fitted on train; sliding windows of size {WINDOW} with horizon 1

## Models
- LSTM and Transformer baselines for point forecasts
- MC Dropout LSTM for approximate Bayesian inference via dropout at test time
- Pyro variational Bayesian linear head on LSTM embeddings (AutoDiagonalNormal)

## Training and Logging
- Deterministic seeds; MLflow logging of parameters and metrics
- Optuna HPO for LSTM: hidden size, layers, dropout, learning rate, batch size

## Evaluation
- Accuracy: RMSE, MAE
- Probabilistic: Gaussian NLL from mean/std, calibration coverage and sharpness
- Backtesting: simple long/short entry using uncertainty thresholds; Sharpe and VaR

## Results (template)
- Baselines achieve RMSE ~X, MAE ~Y on {TICKER}
- MC Dropout improves NLL by ~Z% and yields better coverage at similar sharpness
- Pyro VI provides well-calibrated intervals with moderate compute cost

## Discussion
- Uncertainty helps avoid trades in high-risk periods (lower VaR tails) and improves risk-adjusted returns
- Limitations: stationarity assumptions, simplistic strategy, single-horizon targets

## Reproducibility
- `requirements.txt`, `Dockerfile`, deterministic seeds
- Scripts to fetch data, preprocess, train, evaluate, backtest

## Conclusion
Bayesian methods deliver calibrated uncertainty that can be exploited in trading decisions. Future work: multi-horizon probabilistic forecasting, volatility modeling, and regime detection.

## References
- Gal & Ghahramani (2016). Dropout as a Bayesian Approximation.
- Pyro documentation.

