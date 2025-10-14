from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error

from src.utils.config import load_yaml
from src.utils.results_logger import ResultsLogger
from src.utils.visualization import (plot_forecasts, plot_residuals, 
                                   plot_metrics_comparison, create_summary_dashboard)
from src.models.lstm import LSTMRegressor
from src.models.transformer import TransformerRegressor


def load_processed_test(data_dir: Path, ticker: str):
    tdir = data_dir / ticker
    X_test = np.load(tdir / "X_test.npy")
    y_test = np.load(tdir / "y_test.npy")
    with open(tdir / "meta.json", "r") as f:
        meta = json.load(f)
    return X_test, y_test, meta


def build_model(model_cfg: dict, input_dim: int):
    if model_cfg["type"] == "lstm":
        return LSTMRegressor(input_dim=input_dim, hidden_size=model_cfg["hidden_size"], num_layers=model_cfg["num_layers"], dropout=model_cfg.get("dropout", 0.0))
    elif model_cfg["type"] == "transformer":
        return TransformerRegressor(input_dim=input_dim, d_model=model_cfg["d_model"], nhead=model_cfg["nhead"], num_layers=model_cfg["num_layers"], dim_feedforward=model_cfg["dim_feedforward"], dropout=model_cfg.get("dropout", 0.0))
    else:
        raise ValueError(f"Unknown model type: {model_cfg['type']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ticker", default=None)
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    ticker = args.ticker or cfg["data"]["tickers"][0]

    # Initialize results logger
    results_logger = ResultsLogger()
    
    try:
        X_test, y_test, meta = load_processed_test(Path(cfg["data"]["data_dir"]), ticker)
        input_dim = X_test.shape[-1]
        model = build_model(cfg["model"], input_dim)
        ckpt = Path(cfg["training"]["ckpt_dir"]) / cfg["model"]["type"] / f"{ticker}.pt"
        state = torch.load(ckpt, map_location="cpu")
        model.load_state_dict(state["model_state_dict"])
        model.eval()

        with torch.no_grad():
            preds = []
            for i in range(0, len(X_test), 256):
                xb = torch.from_numpy(X_test[i:i+256]).float()
                yb = model(xb).cpu().numpy()
                preds.append(yb)
            y_pred = np.concatenate(preds, axis=0)

        y_true = y_test[:, 0] if y_test.ndim == 2 else y_test
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        
        # Calculate additional metrics
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        r2 = 1 - (np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2))
        
        metrics = {
            "RMSE": float(rmse),
            "MAE": float(mae),
            "MAPE": float(mape),
            "R2": float(r2)
        }
        
        print(metrics)
        
        # Log evaluation results
        results_logger.log_evaluation_results(metrics)
        
        # Generate visualizations
        # 1. Forecast vs Actual
        plot_forecasts(
            y_true, y_pred,
            save_path=results_logger.experiment_dir / "forecast_vs_actual.png",
            title=f"Forecast vs Actual - {cfg['model']['type'].upper()} on {ticker}"
        )
        
        # 2. Residual analysis
        plot_residuals(
            y_true, y_pred,
            save_path=results_logger.experiment_dir / "residual_analysis.png",
            title=f"Residual Analysis - {cfg['model']['type'].upper()} on {ticker}"
        )
        
        # 3. Metrics comparison
        plot_metrics_comparison(
            metrics,
            save_path=results_logger.experiment_dir / "metrics_comparison.png",
            title=f"Performance Metrics - {cfg['model']['type'].upper()} on {ticker}"
        )
        
        # 4. Save predictions
        results_logger.save_predictions(y_true, y_pred, filename="evaluation_predictions")
        
        # 5. Create summary dashboard if training history is available
        if "training_history" in state:
            create_summary_dashboard(
                state["training_history"],
                metrics,
                y_true,
                y_pred,
                save_path=results_logger.experiment_dir / "summary_dashboard.png"
            )
        
    except Exception as e:
        results_logger.log_error(e, "evaluation")
        raise
    finally:
        results_logger.close()


if __name__ == "__main__":
    main()

