from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import pyro

from src.utils.config import load_yaml
from src.utils.seeding import seed_everything
from src.utils.logging_mlflow import init_mlflow, mlflow_run
from src.models.bnn_vi_pyro import PyroBNN


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
    args = parser.parse_args()
    cfg = load_yaml(args.config)

    seed_everything(cfg.get("seed", 42))
    pyro.clear_param_store()

    mlflow_cfg = cfg.get("mlflow", {"tracking_uri": "./mlruns", "experiment_name": "bnn_vi"})
    init_mlflow(mlflow_cfg["tracking_uri"], mlflow_cfg["experiment_name"])

    data_cfg = cfg["data"]
    ticker = data_cfg["tickers"][0]
    X_train, y_train, X_valid, y_valid, meta = load_processed(Path(data_cfg["data_dir"]), ticker)
    input_dim = X_train.shape[-1]

    model = PyroBNN(input_dim=input_dim, hidden_size=cfg["model"]["hidden_size"], lr=cfg["training"]["lr"])

    Xtr = torch.from_numpy(X_train).float()
    ytr = torch.from_numpy(y_train[:, 0]).float() if y_train.ndim == 2 else torch.from_numpy(y_train).float()
    Xva = torch.from_numpy(X_valid).float()
    yva = torch.from_numpy(y_valid[:, 0]).float() if y_valid.ndim == 2 else torch.from_numpy(y_valid).float()

    with mlflow_run(run_name=f"bnn_vi_{ticker}", params={"config": args.config}):
        for epoch in range(1, cfg["training"]["epochs"] + 1):
            loss = model.fit_epoch(Xtr, ytr, batch_size=cfg["training"]["batch_size"])
            with torch.no_grad():
                mu, sigma = model.predict(Xva, num_samples=50)
                val_mse = torch.mean((mu - yva) ** 2).item()
            print({"epoch": epoch, "elbo": loss, "val_mse": val_mse})


if __name__ == "__main__":
    main()

