from __future__ import annotations
import numpy as np


def compute_returns(prices: np.ndarray) -> np.ndarray:
    return np.diff(prices) / prices[:-1]


def sharpe_ratio(returns: np.ndarray, risk_free: float = 0.0) -> float:
    excess = returns - risk_free
    std = np.std(excess, ddof=1)
    return float(np.mean(excess) / (std + 1e-12))


def value_at_risk(returns: np.ndarray, alpha: float = 0.95) -> float:
    return float(-np.quantile(returns, 1 - alpha))


def backtest_signals(prices: np.ndarray, mu: np.ndarray, sigma: np.ndarray, entry_z: float = 0.0):
    """Toy backtest: long if mean > last price by entry_z*sigma, short if below."""
    last = prices[:-1]
    mu = mu[:-1]
    sigma = sigma[:-1]
    signal = np.where(mu - last > entry_z * sigma, 1.0, np.where(last - mu > entry_z * sigma, -1.0, 0.0))
    rets = compute_returns(prices)
    strat = signal * rets
    return {
        "daily_returns": strat,
        "sharpe": sharpe_ratio(strat),
        "var95": value_at_risk(strat, 0.95),
    }

