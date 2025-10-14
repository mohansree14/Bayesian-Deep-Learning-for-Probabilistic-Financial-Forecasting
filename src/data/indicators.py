"""
Technical indicators for OHLCV DataFrames.
Expected columns: open, high, low, close, adj_close, volume. Index is DatetimeIndex.
"""
from __future__ import annotations
import pandas as pd


def sma(df: pd.DataFrame, window: int, price_col: str = "close") -> pd.Series:
    return df[price_col].rolling(window=window, min_periods=window).mean()


def ema(df: pd.DataFrame, span: int, price_col: str = "close") -> pd.Series:
    return df[price_col].ewm(span=span, adjust=False, min_periods=span).mean()


def rsi(df: pd.DataFrame, window: int = 14, price_col: str = "close") -> pd.Series:
    delta = df[price_col].diff()
    gain = (delta.where(delta > 0, 0.0)).rolling(window=window, min_periods=window).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(window=window, min_periods=window).mean()
    rs = gain / (loss.replace(0, 1e-12))
    return 100 - (100 / (1 + rs))


def macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9, price_col: str = "close") -> pd.DataFrame:
    ema_fast = ema(df, span=fast, price_col=price_col)
    ema_slow = ema(df, span=slow, price_col=price_col)
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    hist = macd_line - signal_line
    return pd.DataFrame({"macd": macd_line, "macd_signal": signal_line, "macd_hist": hist}, index=df.index)


def bollinger_bands(df: pd.DataFrame, window: int = 20, num_std: float = 2.0, price_col: str = "close") -> pd.DataFrame:
    m = sma(df, window=window, price_col=price_col)
    s = df[price_col].rolling(window=window, min_periods=window).std()
    upper = m + num_std * s
    lower = m - num_std * s
    return pd.DataFrame({"bb_mid": m, "bb_upper": upper, "bb_lower": lower}, index=df.index)


def stochastic_oscillator(df: pd.DataFrame, k_window: int = 14, d_window: int = 3) -> pd.DataFrame:
    low_min = df["low"].rolling(window=k_window, min_periods=k_window).min()
    high_max = df["high"].rolling(window=k_window, min_periods=k_window).max()
    k = 100 * (df["close"] - low_min) / (high_max - low_min).replace(0, 1e-12)
    d = k.rolling(window=d_window, min_periods=d_window).mean()
    return pd.DataFrame({"stoch_k": k, "stoch_d": d}, index=df.index)

