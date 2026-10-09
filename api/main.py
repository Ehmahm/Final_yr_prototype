"""
FastAPI inference service for ELIA grid load forecasting.

Serves the ONNX-compiled student model via a synchronous /predict endpoint.
Model loads once at startup; endpoint runs in a thread pool to avoid blocking.
"""

import time
import numpy as np
import onnxruntime as ort
import joblib
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
state = {}

class WindowRequest(BaseModel):
    window: list[float] = Field(..., min_length=24, max_length=24)

class PredictionResponse(BaseModel):
    prediction_mw: float
    latency_ms: float

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading ONNX model and scalers...")
    state["session"]  = ort.InferenceSession(
        str(MODELS / "student.onnx"),
        providers=["CPUExecutionProvider"]
    )
    state["x_scaler"] = joblib.load(MODELS / "scaler.pkl")
    state["y_scaler"] = joblib.load(MODELS / "elia_y_scaler.pkl")
    print("Startup complete. Model and scalers loaded.")
    yield
    state.clear()
    print("Shutdown complete.")

app = FastAPI(
    title="ELIA Peak Load Forecaster",
    version="0.1.0",
    lifespan=lifespan
)

@app.get("/health")
def health():
    if "session" not in state:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy", "model_loaded": True}

@app.post("/predict", response_model=PredictionResponse)
def predict(request: WindowRequest):
    start = time.perf_counter()
    try:
        x_raw = np.array(request.window, dtype=np.float32).reshape(1, -1)
        x_scaled = state["x_scaler"].transform(x_raw).astype(np.float32)
        pred_scaled = state["session"].run(None, {"input": x_scaled})[0]
        pred_mw = float(
            state["y_scaler"].inverse_transform(pred_scaled.reshape(-1, 1)).flatten()[0]
        )
        latency_ms = (time.perf_counter() - start) * 1000
        return PredictionResponse(prediction_mw=pred_mw, latency_ms=latency_ms)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")