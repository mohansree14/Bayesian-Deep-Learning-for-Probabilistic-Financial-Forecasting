from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import torch

from src.utils.config import load_yaml
from src.utils.results_logger import ResultsLogger
from src.utils.visualization import plot_backtesting_results, plot_forecasts
from src.evaluation.backtesting import backtest_signals
from src.models.lstm import LSTMRegressor
from src.models.mc_dropout_lstm import MCDropoutLSTM


def load_processed(data_dir: Path, ticker: str):
    tdir = data_dir / ticker
    X_test = np.load(tdir / "X_test.npy")
    y_test = np.load(tdir / "y_test.npy")
    with open(tdir / "meta.json", "r") as f:
        meta = json.load(f)
    return X_test, y_test, meta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", default="lstm", choices=["lstm", "mc_dropout_lstm"]) 
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    ticker = cfg["data"]["tickers"][0]

    # Initialize results logger
    results_logger = ResultsLogger()
    
    try:
        X_test, y_test, meta = load_processed(Path(cfg["data"]["data_dir"]), ticker)
        input_dim = X_test.shape[-1]
        ckpt = Path(cfg["training"]["ckpt_dir"]) / args.model / f"{ticker}.pt"

        if args.model == "lstm":
            model = LSTMRegressor(input_dim)
            state = torch.load(ckpt, map_location="cpu"); model.load_state_dict(state["model_state_dict"]) 
            model.eval()
            with torch.no_grad():
                mu = []
                for i in range(0, len(X_test), 256):
                    xb = torch.from_numpy(X_test[i:i+256]).float()
                    mu.append(model(xb).cpu().numpy())
                mu = np.concatenate(mu, 0)
                sigma = np.full_like(mu, np.std(mu))
        else:
            model = MCDropoutLSTM(input_dim)
            state = torch.load(ckpt, map_location="cpu"); model.load_state_dict(state["model_state_dict"]) 
            with torch.no_grad():
                S = []
                for i in range(0, len(X_test), 128):
                    xb = torch.from_numpy(X_test[i:i+128]).float()
                    S.append(model.mc_predict(xb, mc_samples=50).cpu().numpy())
                S = np.concatenate(S, 1) if len(S) > 1 else S[0]
                mu = S.mean(0); sigma = S.std(0)

        y_true = y_test[:, 0] if y_test.ndim == 2 else y_test
        prices = y_true  # interpret target as price-like for demo
        res = backtest_signals(prices, mu, sigma, entry_z=cfg.get("backtest", {}).get("entry_z", 0.5))
        
        # Log backtesting results
        results_logger.log_backtesting_results(res)
        
        # Generate backtesting visualizations
        # 1. Forecast with uncertainty
        plot_forecasts(
            y_true, mu, y_std=sigma,
            save_path=results_logger.experiment_dir / "forecast_with_uncertainty.png",
            title=f"Forecast with Uncertainty - {args.model.upper()} on {ticker}"
        )
        
        # 2. Backtesting results
        if "returns" in res and "cumulative_returns" in res and "drawdown" in res:
            plot_backtesting_results(
                res["returns"], res["cumulative_returns"], res["drawdown"],
                save_path=results_logger.experiment_dir / "backtesting_results.png",
                title=f"Backtesting Results - {args.model.upper()} on {ticker}"
            )
        
        # 3. Save predictions with uncertainty
        results_logger.save_predictions(y_true, mu, y_std=sigma, filename="backtest_predictions")
        
        print({"sharpe": res["sharpe"], "var95": res["var95"]})
        
    except Exception as e:
        results_logger.log_error(e, "backtesting")
        raise
    finally:
        results_logger.close()


if __name__ == "__main__":
    main()

