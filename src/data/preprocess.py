from __future__ import annotations
"""Preprocessing: indicators, scaling, and windowing."""
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .indicators import sma, ema, rsi, macd, bollinger_bands, stochastic_oscillator


@dataclass
class WindowConfig:
    input_window: int
    forecast_horizon: int
    step: int = 1
    target_col: str = "close"


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["ret_1d"] = out["close"].pct_change()
    out["log_ret_1d"] = np.log(out["close"]).diff()
    out["sma_10"] = sma(out, 10)
    out["sma_20"] = sma(out, 20)
    out["ema_12"] = ema(out, 12)
    out["ema_26"] = ema(out, 26)
    out["rsi_14"] = rsi(out, 14)
    out = out.join(macd(out))
    out = out.join(bollinger_bands(out, 20, 2.0))
    out = out.join(stochastic_oscillator(out))
    out = out.dropna()
    return out


def fit_scalers(train_df: pd.DataFrame, feature_cols: List[str]) -> Dict[str, StandardScaler]:
    scaler_x = StandardScaler(with_mean=True, with_std=True)
    scaler_x.fit(train_df[feature_cols].values)
    return {"x": scaler_x}


def apply_scalers(df: pd.DataFrame, scalers: Dict[str, StandardScaler], feature_cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    out[feature_cols] = scalers["x"].transform(out[feature_cols].values)
    return out


def make_windowed_arrays(df: pd.DataFrame, feature_cols: List[str], cfg: WindowConfig) -> Tuple[np.ndarray, np.ndarray, List[pd.Timestamp]]:
    X_list: List[np.ndarray] = []
    y_list: List[np.ndarray] = []
    idx_list: List[pd.Timestamp] = []
    target = df[cfg.target_col].values
    feats = df[feature_cols].values
    T = len(df)
    for end in range(cfg.input_window, T - cfg.forecast_horizon + 1, cfg.step):
        start = end - cfg.input_window
        x = feats[start:end]
        y = target[end : end + cfg.forecast_horizon]
        if np.isnan(x).any() or np.isnan(y).any():
            continue
        X_list.append(x)
        y_list.append(y)
        idx_list.append(df.index[end - 1])
    X = np.stack(X_list, axis=0) if X_list else np.zeros((0, cfg.input_window, len(feature_cols)))
    y = np.stack(y_list, axis=0) if y_list else np.zeros((0, cfg.forecast_horizon))
    return X, y, idx_list

