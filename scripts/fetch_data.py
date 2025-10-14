#!/usr/bin/env python
from __future__ import annotations
"""Fetch OHLCV data with yfinance and save Parquet/CSV.

Example:
  python scripts/fetch_data.py --tickers AAPL MSFT --start 2015-01-01 --end 2024-12-31 --interval 1d --out data/raw --format parquet --verbose
"""
import argparse
import time
from pathlib import Path
from typing import List

import pandas as pd
import yfinance as yf


def fetch_one(ticker: str, start: str, end: str, interval: str) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end, interval=interval, auto_adjust=False, progress=False)
    if df.empty:
        raise ValueError(f"No data returned for {ticker}")
    df = df.rename(columns=str.lower).rename(columns={"adj close": "adj_close"})
    required = ["open", "high", "low", "close", "adj_close", "volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns for {ticker}: {missing}")
    df = df.sort_index()
    df["ticker"] = ticker
    return df


def save_df(df: pd.DataFrame, path: Path, fmt: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "parquet":
        df.to_parquet(path, index=True)
    elif fmt == "csv":
        df.to_csv(path, index=True)
    else:
        raise ValueError(f"Unsupported format: {fmt}")


def main(args: argparse.Namespace) -> None:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    failures: List[str] = []
    for ticker in args.tickers:
        attempt = 0
        while True:
            try:
                df = fetch_one(ticker, args.start, args.end, args.interval)
                outfile = out_dir / f"{ticker}_{args.interval}.{args.format}"
                save_df(df, outfile, args.format)
                if args.verbose:
                    print(f"Saved {ticker}: {len(df)} rows -> {outfile}")
                break
            except Exception as e:
                attempt += 1
                if attempt > args.retries:
                    print(f"[ERROR] {ticker} failed after {args.retries} retries: {e}")
                    failures.append(ticker)
                    break
                time.sleep(min(2.0 * attempt, 10.0))
    if failures:
        raise SystemExit(f"Failed tickers: {failures}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tickers", nargs="+", required=True)
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--interval", default="1d")
    p.add_argument("--out", default="data/raw")
    p.add_argument("--format", default="parquet", choices=["parquet", "csv"])
    p.add_argument("--retries", type=int, default=2)
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()
    main(args)

