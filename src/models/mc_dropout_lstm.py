from __future__ import annotations
import torch
from torch import nn


class MCDropoutLSTM(nn.Module):
    """LSTM with dropout active at inference for MC sampling."""

    def __init__(self, input_dim: int, hidden_size: int = 128, num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        last = self.dropout(last)
        yhat = self.head(last)
        return yhat.squeeze(-1)

    def mc_predict(self, x: torch.Tensor, mc_samples: int = 50) -> torch.Tensor:
        self.train()  # enable dropout
        preds = []
        with torch.no_grad():
            for _ in range(mc_samples):
                preds.append(self(x))
        return torch.stack(preds, dim=0)  # [S, B]

