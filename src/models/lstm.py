from __future__ import annotations
import torch
from torch import nn


class LSTMRegressor(nn.Module):
    """Many-to-one LSTM regressor for sequence forecasting (point estimate)."""

    def __init__(self, input_dim: int, hidden_size: int = 128, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        yhat = self.head(last)
        return yhat.squeeze(-1)

