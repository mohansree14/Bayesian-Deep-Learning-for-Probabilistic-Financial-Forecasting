from __future__ import annotations
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(mean_absolute_error(y_true, y_pred))


def gaussian_nll(y_true: np.ndarray, mu: np.ndarray, sigma: np.ndarray, eps: float = 1e-6) -> float:
    var = np.maximum(sigma ** 2, eps)
    return float(0.5 * np.mean(np.log(2 * np.pi * var) + (y_true - mu) ** 2 / var))


def calibration_curve(y_true: np.ndarray, samples: np.ndarray, quantiles=(0.05, 0.5, 0.95)):
    qs = np.quantile(samples, quantiles, axis=0)
    cover = ((y_true >= qs[0]) & (y_true <= qs[-1])).mean()
    sharpness = np.mean(qs[-1] - qs[0])
    return {"coverage": float(cover), "sharpness": float(sharpness)}

