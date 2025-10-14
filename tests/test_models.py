import numpy as np
import torch

from src.models.lstm import LSTMRegressor
from src.models.transformer import TransformerRegressor
from src.models.mc_dropout_lstm import MCDropoutLSTM


def fake_batch(B=8, T=32, F=10):
    return torch.randn(B, T, F)


def test_lstm_forward():
    x = fake_batch()
    m = LSTMRegressor(input_dim=x.shape[-1])
    y = m(x)
    assert y.shape == (x.shape[0],)


def test_transformer_forward():
    x = fake_batch()
    m = TransformerRegressor(input_dim=x.shape[-1])
    y = m(x)
    assert y.shape == (x.shape[0],)


def test_mc_dropout_mc_predict():
    x = fake_batch()
    m = MCDropoutLSTM(input_dim=x.shape[-1])
    S = m.mc_predict(x, mc_samples=5)
    assert S.shape == (5, x.shape[0])

