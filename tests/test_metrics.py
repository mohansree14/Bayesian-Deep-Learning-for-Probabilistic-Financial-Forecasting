import numpy as np

from src.evaluation.metrics import rmse, mae, gaussian_nll, calibration_curve


def test_metrics_shapes():
    y = np.array([0.0, 1.0, 2.0, 3.0])
    p = np.array([0.1, 0.9, 2.1, 2.9])
    s = np.full_like(y, 0.5)
    assert rmse(y, p) >= 0
    assert mae(y, p) >= 0
    assert gaussian_nll(y, p, s) >= -10  # sanity bound


def test_calibration_curve():
    samples = np.random.normal(0, 1, size=(100, 50))
    y = np.zeros(50)
    cal = calibration_curve(y, samples)
    assert 0 <= cal["coverage"] <= 1

