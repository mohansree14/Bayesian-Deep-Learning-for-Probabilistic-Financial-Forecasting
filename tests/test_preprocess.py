import numpy as np
import pandas as pd

from src.data.preprocess import add_technical_indicators, make_windowed_arrays, WindowConfig


def sample_df(n=200):
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    df = pd.DataFrame({
        "open": np.linspace(100, 120, n),
        "high": np.linspace(101, 121, n),
        "low": np.linspace(99, 119, n),
        "close": np.linspace(100, 120, n) + np.random.randn(n)*0.1,
        "adj_close": np.linspace(100, 120, n),
        "volume": np.random.randint(1e5, 2e5, n),
    }, index=idx)
    return df


def test_add_indicators():
    df = add_technical_indicators(sample_df())
    assert "rsi_14" in df.columns and "sma_10" in df.columns


def test_windowing_shapes():
    df = add_technical_indicators(sample_df())
    feat_cols = [c for c in df.columns if c not in ["ticker"]]
    X, y, idx = make_windowed_arrays(df, feat_cols, WindowConfig(32, 1))
    assert X.ndim == 3 and y.ndim == 2
    assert X.shape[1] == 32 and y.shape[1] == 1
    assert len(idx) == X.shape[0]

