from __future__ import annotations
from pathlib import Path
import argparse
import json
import numpy as np
import torch

from src.utils.config import load_yaml
from src.utils.seeding import seed_everything
from src.utils.logging_mlflow import init_mlflow, mlflow_run
from src.utils.results_logger import ResultsLogger
from src.utils.visualization import plot_training_history, create_summary_dashboard
from src.training.train_loop import build_dataloaders, train_model
from src.models.lstm import LSTMRegressor
from src.models.transformer import TransformerRegressor
from src.models.mc_dropout_lstm import MCDropoutLSTM


def load_processed(data_dir: Path, ticker: str):
    tdir = data_dir / ticker
    X_train = np.load(tdir / "X_train.npy")
    y_train = np.load(tdir / "y_train.npy")
    X_valid = np.load(tdir / "X_valid.npy")
    y_valid = np.load(tdir / "y_valid.npy")
    with open(tdir / "meta.json", "r") as f:
        meta = json.load(f)
    return (X_train, y_train, X_valid, y_valid, meta)


def build_model(model_cfg: dict, input_dim: int):
    if model_cfg["type"] == "lstm":
        return LSTMRegressor(input_dim=input_dim, hidden_size=model_cfg["hidden_size"], num_layers=model_cfg["num_layers"], dropout=model_cfg.get("dropout", 0.0))
    elif model_cfg["type"] == "transformer":
        return TransformerRegressor(input_dim=input_dim, d_model=model_cfg["d_model"], nhead=model_cfg["nhead"], num_layers=model_cfg["num_layers"], dim_feedforward=model_cfg["dim_feedforward"], dropout=model_cfg.get("dropout", 0.0))
    elif model_cfg["type"] == "mc_dropout_lstm":
        return MCDropoutLSTM(input_dim=input_dim, hidden_size=model_cfg["hidden_size"], num_layers=model_cfg["num_layers"], dropout=model_cfg.get("dropout", 0.0))
    else:
        raise ValueError(f"Unknown model type: {model_cfg['type']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_yaml(args.config)

    # Initialize results logger
    results_logger = ResultsLogger()
    
    try:
        seed_everything(cfg.get("seed", 42))

        mlflow_cfg = cfg.get("mlflow", {"tracking_uri": "./mlruns", "experiment_name": "default"})
        init_mlflow(mlflow_cfg["tracking_uri"], mlflow_cfg["experiment_name"])

        data_cfg = cfg["data"]
        ticker = data_cfg["tickers"][0]
        
        # Log experiment start
        results_logger.log_experiment_start(cfg, cfg["model"]["type"], ticker)
        
        X_train, y_train, X_valid, y_valid, meta = load_processed(Path(data_cfg["data_dir"]), ticker)
        input_dim = X_train.shape[-1]

        # Log data summary
        data_info = {
            "ticker": ticker,
            "train_samples": len(X_train),
            "valid_samples": len(X_valid),
            "input_dim": input_dim,
            "sequence_length": X_train.shape[1],
            "target_shape": y_train.shape
        }
        results_logger.save_data_summary(data_info)

        model = build_model(cfg["model"], input_dim)
        
        # Log model info
        model_info = {
            "type": cfg["model"]["type"],
            "input_dim": input_dim,
            "parameters": sum(p.numel() for p in model.parameters()),
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)
        }
        results_logger.log_training_start(model_info)

        loaders = build_dataloaders(X_train, y_train, X_valid, y_valid, batch_size=cfg["training"]["batch_size"])
        device = "cuda" if torch.cuda.is_available() else "cpu"
        ckpt_dir = Path(cfg["training"]["ckpt_dir"]) / cfg["model"]["type"]
        ckpt_path = ckpt_dir / f"{ticker}.pt"

        with mlflow_run(run_name=f"{cfg['model']['type']}_{ticker}", params={"config": args.config}):
            history = train_model(
                model,
                loaders,
                epochs=cfg["training"]["epochs"],
                lr=cfg["training"]["lr"],
                grad_clip=cfg["training"].get("grad_clip", 0.0),
                device=device,
                ckpt_path=ckpt_path,
            )
            
            # Log training completion
            best_epoch = np.argmin(history["valid_loss"]) + 1
            best_loss = min(history["valid_loss"])
            results_logger.log_training_complete(history, best_epoch, best_loss)
            
            # Generate training plots
            plot_training_history(
                history, 
                save_path=results_logger.experiment_dir / "training_history.png",
                title=f"Training Progress - {cfg['model']['type'].upper()} on {ticker}"
            )
            
            # Save model
            results_logger.save_model(
                model, 
                f"{cfg['model']['type']}_{ticker}",
                additional_info={
                    "config": cfg,
                    "training_history": history,
                    "data_info": data_info
                }
            )
            
            print("History:", history)
            
    except Exception as e:
        results_logger.log_error(e, "training")
        raise
    finally:
        results_logger.close()


if __name__ == "__main__":
    main()

