# Slide Deck: Uncertainty-Aware Bayesian Deep Learning for Financial Time Series

1. Title & Motivation
- Problem: Forecast prices with quantified uncertainty
- Why uncertainty: risk management, decision thresholds

2. Data & Task
- OHLCV from yfinance; daily {START_DATE}–{END_DATE}
- Target: next-day close; horizon 1

3. Preprocessing
- Indicators: SMA/EMA, RSI, MACD, Bollinger, Stochastic
- Scaling on train; window length {WINDOW}

4. Baselines
- LSTM regressor (point)
- Transformer encoder (point)

5. Bayesian Methods
- MC Dropout LSTM (approximate Bayesian)
- Pyro VI: Bayesian linear head on LSTM embeddings

6. Training & Reproducibility
- Seeds; MLflow logging; configs
- Optuna for HPO

7. Probabilistic Evaluation
- NLL, calibration coverage, sharpness
- Prediction intervals

8. Backtesting
- Uncertainty-aware entry/exit
- Metrics: Sharpe, VaR

9. Results Snapshot (template)
- RMSE/MAE table; NLL; coverage
- MC Dropout vs baseline

10. Case Study Plots
- Price, mean prediction, and CI bands
- Signal overlay

11. Discussion
- Benefits, limitations, compute costs
- When uncertainty helps

12. Conclusions & Next Steps
- Summary; roadmap: multi-horizon, volatility modeling, regimes

