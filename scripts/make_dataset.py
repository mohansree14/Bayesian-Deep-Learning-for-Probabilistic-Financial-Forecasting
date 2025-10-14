#!/usr/bin/env python
from __future__ import annotations
"""Build processed datasets with indicators, scaling, and windowing."""
import argparse
import json
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd

from src.data.preprocess import add_technical_indicators, fit_scalers, apply_scalers, make_windowed_arrays, WindowConfig


def load_one(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    elif path.suffix == ".csv":
        df = pd.read_csv(path, parse_dates=True, index_col=0)
    else:
        raise ValueError(f"Unsupported file: {path}")
    
    # Fix MultiIndex columns from yfinance
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    return df


def select_features(df: pd.DataFrame) -> List[str]:
    cols = [
        "open","high","low","close","adj_close","volume",
        "ret_1d","log_ret_1d",
        "sma_10","sma_20","ema_12","ema_26","rsi_14",
        "macd","macd_signal","macd_hist",
        "bb_mid","bb_upper","bb_lower",
        "stoch_k","stoch_d",
    ]
    return [c for c in cols if c in df.columns]


def split_df(df: pd.DataFrame, train: float, valid: float, test: float) -> Dict[str, pd.DataFrame]:
    n = len(df)
    i = int(n * train)
    j = int(n * (train + valid))
    return {"train": df.iloc[:i], "valid": df.iloc[i:j], "test": df.iloc[j:]}


def main(args: argparse.Namespace) -> None:
    in_dir = Path(args.input)
    out_root = Path(args.output)
    out_root.mkdir(parents=True, exist_ok=True)

    files: List[Path] = []
    for t in args.tickers:
        p = in_dir / f"{t}_{args.interval}.{args.format}"
        if not p.exists():
            alt = in_dir / f"{t}_{args.interval}.parquet"
            if alt.exists():
                p = alt
            else:
                raise SystemExit(f"Missing raw file for {t}")
        files.append(p)

    for fp in files:
        df = load_one(fp)
        if "ticker" not in df.columns:
            df["ticker"] = fp.stem.split("_")[0]
        df = df.sort_index()
        df = add_technical_indicators(df)
        feature_cols = select_features(df)

        parts = split_df(df, args.train_split, args.valid_split, args.test_split)
        scalers = fit_scalers(parts["train"], feature_cols)

        processed = {}
        for name, d in parts.items():
            d_scaled = apply_scalers(d, scalers, feature_cols)
            X, y, idx = make_windowed_arrays(d_scaled, feature_cols, WindowConfig(args.window, args.horizon, args.step, args.target_col))
            processed[name] = {"X": X, "y": y, "timestamps": [str(t) for t in idx]}

        ticker = df["ticker"].iloc[0]
        out_dir = out_root / ticker
        out_dir.mkdir(parents=True, exist_ok=True)
        for name, d in processed.items():
            np.save(out_dir / f"X_{name}.npy", d["X"])
            np.save(out_dir / f"y_{name}.npy", d["y"])
        meta = {
            "feature_cols": feature_cols,
            "window": args.window,
            "horizon": args.horizon,
            "step": args.step,
            "target_col": args.target_col,
            "interval": args.interval,
        }
        with open(out_dir / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)
        print(f"Wrote processed arrays for {ticker} -> {out_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="data/raw")
    p.add_argument("--output", default="data/processed")
    p.add_argument("--tickers", nargs="+", required=True)
    p.add_argument("--interval", default="1d")
    p.add_argument("--format", default="parquet", choices=["parquet", "csv"]) 
    p.add_argument("--window", type=int, default=64)
    p.add_argument("--horizon", type=int, default=1)
    p.add_argument("--step", type=int, default=1)
    p.add_argument("--target-col", default="close")
    p.add_argument("--train-split", type=float, default=0.7)
    p.add_argument("--valid-split", type=float, default=0.15)
    p.add_argument("--test-split", type=float, default=0.15)
    args = p.parse_args()
    main(args)

