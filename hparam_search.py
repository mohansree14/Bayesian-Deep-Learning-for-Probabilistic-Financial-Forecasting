from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import optuna
import torch

from src.utils.config import load_yaml
from src.utils.seeding import seed_everything
from src.training.train_loop import build_dataloaders, train_model
from src.models.lstm import LSTMRegressor


def load_processed(data_dir: Path, ticker: str):
    tdir = data_dir / ticker
    X_train = np.load(tdir / "X_train.npy"); y_train = np.load(tdir / "y_train.npy")
    X_valid = np.load(tdir / "X_valid.npy"); y_valid = np.load(tdir / "y_valid.npy")
    with open(tdir / "meta.json", "r") as f:
        meta = json.load(f)
    return (X_train, y_train, X_valid, y_valid, meta)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--study-name", required=True)
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    seed_everything(cfg.get("seed", 42))

    data_cfg = cfg["data"]
    ticker = data_cfg["tickers"][0]
    X_train, y_train, X_valid, y_valid, meta = load_processed(Path(data_cfg["data_dir"]), ticker)
    input_dim = X_train.shape[-1]

    def objective(trial: optuna.Trial):
        hidden = trial.suggest_int("hidden_size", 64, 256, step=64)
        layers = trial.suggest_int("num_layers", 1, 3)
        dropout = trial.suggest_float("dropout", 0.0, 0.4)
        lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
        batch = trial.suggest_categorical("batch_size", [32, 64, 128])

        model = LSTMRegressor(input_dim=input_dim, hidden_size=hidden, num_layers=layers, dropout=dropout)
        loaders = build_dataloaders(X_train, y_train, X_valid, y_valid, batch_size=batch)
        hist = train_model(model, loaders, epochs=cfg["training"].get("epochs", 5), lr=lr, grad_clip=1.0, device="cpu", ckpt_path=None)
        return hist["valid_loss"][-1]

    study = optuna.create_study(direction="minimize", study_name=args.study_name)
    study.optimize(objective, n_trials=10)
    print("Best trial:", study.best_trial.params)


if __name__ == "__main__":
    main()

