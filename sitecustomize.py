"""
sitecustomize shim: if running inside Streamlit, monkeypatch the predict_point function
to call the external FastAPI service at localhost:8000 so the Streamlit app doesn't need changes.
Place this file at repo root so Python adds it automatically on startup when available in PYTHONPATH.
"""
import os
import json
import requests
import numpy as np
import sys
from pathlib import Path

# Ensure repo root and src are on sys.path so `import src` works when Streamlit imports app
_REPO_ROOT = Path(__file__).resolve().parent
_SRC_PATH = _REPO_ROOT / "src"
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_SRC_PATH) not in sys.path:
    sys.path.insert(0, str(_SRC_PATH))

API_URL = os.environ.get("PREDICT_SERVICE_URL", "http://127.0.0.1:8000/predict")

def _call_predict_api(model_type, ckpt_dir, ticker, X):
    try:
        payload = {"model": model_type, "ticker": ticker, "X": np.asarray(X).tolist(), "mc_samples": 50}
        resp = requests.post(API_URL, json=payload, timeout=5.0)
        resp.raise_for_status()
        j = resp.json()
        return np.array(j["pred_mean"]), np.array(j["pred_std"])
    except Exception:
        # On any failure, return None so caller can fallback to local prediction
        return None


# Try to monkeypatch the predict function used by streamlit_app if present
try:
    # Import the module where the Streamlit app defines predict_point
    import importlib
    mod = importlib.import_module('app.streamlit_app')
    if hasattr(mod, 'predict_point'):
        original = mod.predict_point

        def proxy_predict_point(model_type: str, ckpt_dir: Path, ticker: str, X):
            res = _call_predict_api(model_type, ckpt_dir, ticker, X)
            if res is None:
                return original(model_type, ckpt_dir, ticker, X)
            return res

        mod.predict_point = proxy_predict_point
except Exception:
    # If import fails, don't break application startup; this shim is optional
    pass
