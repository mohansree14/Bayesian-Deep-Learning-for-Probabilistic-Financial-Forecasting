from __future__ import annotations
"""Simple Pyro Bayesian MLP head for sequence features (pooled last token)."""
from typing import Tuple
import torch
from torch import nn
import pyro
import pyro.distributions as dist
from pyro.nn.module import PyroModule, PyroSample
from pyro.infer import SVI, Trace_ELBO
from pyro.infer.autoguide import AutoDiagonalNormal


class DeterministicBackbone(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int = 128):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_size, num_layers=1, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return out[:, -1, :]


class BayesianHead(PyroModule):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.fc = PyroModule[nn.Linear](hidden_size, 1)
        prior_scale = 0.1
        self.fc.weight = PyroSample(dist.Normal(0.0, prior_scale).expand(self.fc.weight.shape).to_event(2))
        self.fc.bias = PyroSample(dist.Normal(0.0, prior_scale).expand(self.fc.bias.shape).to_event(1))
        self.sigma = PyroSample(dist.LogNormal(-1.0, 0.5))

    def forward(self, h: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        mean = self.fc(h).squeeze(-1)
        sigma = self.sigma
        with pyro.plate("data", h.shape[0]):
            obs = pyro.sample("obs", dist.Normal(mean, sigma), obs=y)
        return mean


class PyroBNN:
    """Wraps deterministic backbone with Bayesian head and provides VI training."""

    def __init__(self, input_dim: int, hidden_size: int = 128, lr: float = 1e-3):
        self.backbone = DeterministicBackbone(input_dim, hidden_size)
        self.head = BayesianHead(hidden_size)
        self.guide = AutoDiagonalNormal(self.head)
        self.optimizer = pyro.optim.Adam({"lr": lr})
        self.svi = SVI(self.head, self.guide, self.optimizer, loss=Trace_ELBO())

    def fit_epoch(self, X: torch.Tensor, y: torch.Tensor, batch_size: int = 64) -> float:
        self.backbone.train()
        self.head.train()
        losses = []
        for i in range(0, len(X), batch_size):
            xb = X[i:i+batch_size]
            yb = y[i:i+batch_size]
            h = self.backbone(xb)
            loss = self.svi.step(h.detach(), yb)
            losses.append(loss / len(xb))
        return float(sum(losses) / len(losses))

    @torch.no_grad()
    def predict(self, X: torch.Tensor, num_samples: int = 100) -> Tuple[torch.Tensor, torch.Tensor]:
        self.backbone.eval()
        h = self.backbone(X)
        samples = []
        for _ in range(num_samples):
            mean = pyro.poutine.trace(self.guide).get_trace(h).call_module(self.head, (), {})  # unused
            pred = pyro.poutine.trace(pyro.poutine.replay(self.head, guide=self.guide)).get_trace(h).nodes["obs"]["fn"].loc
            samples.append(pred)
        S = torch.stack(samples, dim=0)  # [S, B]
        return S.mean(0), S.std(0)

