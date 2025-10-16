from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from pathlib import Path
import numpy as np
import torch

app = FastAPI(title="Model Prediction Service")


class PredictRequest(BaseModel):
    model: str
    ticker: str
    X: List[List[List[float]]]  # n x seq_len x features
    mc_samples: Optional[int] = 50


class PredictResponse(BaseModel):
    pred_mean: List[float]
    pred_std: List[float]
    n: int


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    # basic validation
    X_arr = np.array(req.X)
    if X_arr.ndim != 3:
        raise HTTPException(status_code=400, detail="X must be shape (n, seq_len, features)")

    n, seq_len, features = X_arr.shape

    ckpt_dir = Path("experiments/checkpoints")
    ckpt_path = ckpt_dir / req.model / f"{req.ticker}.pt"
    if not ckpt_path.exists():
        raise HTTPException(status_code=404, detail=f"checkpoint not found: {ckpt_path}")

    try:
        # instantiate model locally and run inference here (server-side)
        # this mirrors the model classes used in the repo
        if req.model == "lstm":
            from src.models.lstm import LSTMRegressor as ModelClass
        elif req.model == "mc_dropout_lstm":
            from src.models.mc_dropout_lstm import MCDropoutLSTM as ModelClass
        else:
            # fallback to lstm
            from src.models.lstm import LSTMRegressor as ModelClass

        model = ModelClass(features)
        state = torch.load(ckpt_path, map_location="cpu")
        if isinstance(state, dict) and "model_state_dict" in state:
            model.load_state_dict(state["model_state_dict"])
        else:
            model.load_state_dict(state)
        model.eval()

        with torch.no_grad():
            xb = torch.from_numpy(X_arr).float()
            if req.model == "mc_dropout_lstm":
                preds = model.mc_predict(xb, mc_samples=req.mc_samples).cpu().numpy()
                mean = preds.mean(0)
                std = preds.std(0)
            else:
                pred = model(xb).cpu().numpy()
                mean = pred
                std = np.full_like(pred, fill_value=float(np.std(pred)))

        return PredictResponse(pred_mean=mean.tolist(), pred_std=std.tolist(), n=n)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
