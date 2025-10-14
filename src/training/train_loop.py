from __future__ import annotations
from pathlib import Path
from typing import Dict, Any
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


def build_dataloaders(X_train: np.ndarray, y_train: np.ndarray, X_valid: np.ndarray, y_valid: np.ndarray, batch_size: int) -> Dict[str, DataLoader]:
    train_ds = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float())
    valid_ds = TensorDataset(torch.from_numpy(X_valid).float(), torch.from_numpy(y_valid).float())
    return {
        "train": DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        "valid": DataLoader(valid_ds, batch_size=batch_size, shuffle=False),
    }


def train_model(model: nn.Module, loaders: Dict[str, DataLoader], epochs: int, lr: float, grad_clip: float, device: str, ckpt_path: Path | None = None) -> Dict[str, Any]:
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    best_val = float("inf")
    history = {"train_loss": [], "valid_loss": []}

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for xb, yb in tqdm(loaders["train"], desc=f"Epoch {epoch}"):
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            if yb.ndim == 2:  # horizon > 1 -> take first step for point baseline
                yb = yb[:, 0]
            loss = loss_fn(pred, yb)
            loss.backward()
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            train_losses.append(loss.item())
        history["train_loss"].append(float(np.mean(train_losses)))

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in loaders["valid"]:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                if yb.ndim == 2:
                    yb = yb[:, 0]
                val_loss = loss_fn(pred, yb)
                val_losses.append(val_loss.item())
        val_mean = float(np.mean(val_losses))
        history["valid_loss"].append(val_mean)

        if val_mean < best_val and ckpt_path is not None:
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model_state_dict": model.state_dict()}, ckpt_path)
            best_val = val_mean

    return history

