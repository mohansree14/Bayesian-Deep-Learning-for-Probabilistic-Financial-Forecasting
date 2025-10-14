from __future__ import annotations
import torch
from torch import nn


class TransformerRegressor(nn.Module):
    """Transformer encoder for sequence regression (predict next-step value)."""

    def __init__(self, input_dim: int, d_model: int = 128, nhead: int = 4, num_layers: int = 2, dim_feedforward: int = 256, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        h = self.encoder(h)
        last = h[:, -1, :]
        yhat = self.head(last)
        return yhat.squeeze(-1)

