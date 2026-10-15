"""
ELIA Peak Load Forecaster - FastAPI service + live dashboard backend.

- /predict       : raw-MW 24h window -> next-hour forecast (ONNX student)
- /live          : everything the dashboard needs (forecast, actuals, metrics, system, health)
- /health        : service health
- /system        : CPU / memory
- /export.csv    : full prediction log
- /              : dashboard (api/dashboard.html)

Data source: polls the ELIA open-data API (ods001). If ELIA data is stale or the
API is unreachable, it falls back to REPLAY mode (test-set hours streamed one
per few seconds) so the dashboard always has something honest to show.
"""

import io
import csv
import time
import threading
from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import onnxruntime as ort
import joblib
import requests
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field

try:
    import psutil
except ImportError:  # pip install psutil
    psutil = None

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
DATA = ROOT / "data" / "processed"
DASHBOARD = Path(__file__).resolve().parent / "dashboard.html"

ELIA_URL = "https://opendata.elia.be/api/explore/v2.1/catalog/datasets/ods001/records"
POLL_SECONDS = 300          # live mode: how often to ask ELIA for new data
STALE_HOURS = 24             # newest ELIA hour older than this -> use replay mode
REPLAY_TICK_SECONDS = 3     # replay mode: one "hour" arrives every N seconds
BACKFILL_HOURS = 60         # hours ingested at startup so the chart is not empty
MAX_LOG_ROWS = 1000

state = {}
stop_event = threading.Event()
lock = threading.Lock()

buf = {}            # hourly actuals: {Timestamp(UTC): MW}
processed = set()   # hours already ingested
pred_log = []       # one dict per forecast
sys_hist = deque(maxlen=90)
info = {"mode": "starting", "started": time.time(), "last_fetch": None,
        "data_latest": None, "last_error": None}


# ----------------------------------------------------------------- model ----
def run_model(x_raw):
    """x_raw: 24 raw-MW values -> next-hour MW (numpy scaling, no sklearn)."""
    x = ((np.asarray(x_raw, dtype=np.float32) - state["x_min"]) / state["x_rng"])
    x = x.reshape(1, -1).astype(np.float32)
    p = state["session"].run(None, {"input": x})[0]
    return float(p.ravel()[0]) * state["y_rng"] + state["y_min"]


class WindowRequest(BaseModel):
    window: list[float] = Field(..., min_length=24, max_length=24)


class PredictionResponse(BaseModel):
    prediction_mw: float
    latency_ms: float


# ------------------------------------------------------------- ingestion ----
def iso(ts):
    return None if ts is None else pd.Timestamp(ts).isoformat()


def ingest_hour(ts, value):
    """Called once per completed hour, in chronological order."""
    ts = pd.Timestamp(ts)
    if ts in processed:
        return
    with lock:
        processed.add(ts)
        buf[ts] = float(value)
        for k in sorted(buf)[:-300]:
            del buf[k]

        # 1) resolve the forecast that was made for this hour
        for row in pred_log:
            if row["target_time"] == iso(ts) and row["actual"] is None:
                row["actual"] = float(value)
                row["error"] = row["predicted"] - row["actual"]
                row["abs_pct_err"] = abs(row["error"]) / row["actual"] * 100

        # 2) forecast the NEXT hour from the last 24 completed hours
        hours = [ts - pd.Timedelta(hours=k) for k in range(23, -1, -1)]
        if all(h in buf for h in hours):
            window = [buf[h] for h in hours]
            t0 = time.perf_counter()
            pred = run_model(window)
            lat = (time.perf_counter() - t0) * 1000
            pred_log.append({
                "made_at": datetime.now(timezone.utc).isoformat(),
                "based_on_hour": iso(ts),
                "target_time": iso(ts + pd.Timedelta(hours=1)),
                "horizon_h": 1,
                "predicted": pred,
                "actual": None,
                "error": None,
                "abs_pct_err": None,
                "model_ms": lat,
            })
            del pred_log[:-MAX_LOG_ROWS]


def fetch_elia_hourly():
    """Last ~100 hours of ELIA total load, as completed hourly means."""
    rows = []
    for offset in range(0, 400, 100):
        r = requests.get(ELIA_URL, timeout=20, params={
            "where": "totalload IS NOT NULL",
            "order_by": "datetime desc",
            "limit": 100, "offset": offset,
        })
        r.raise_for_status()
        rows += r.json().get("results", [])
    if not rows:
        raise RuntimeError("ELIA returned no rows")
    df = pd.DataFrame(rows)[["datetime", "totalload"]]
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    s = df.set_index("datetime")["totalload"].sort_index()
    g = s.resample("1h")
    hourly = g.mean()[g.count() == 4]       # only fully completed hours
    return hourly.dropna(), s.index.max()


def poller():
    # decide mode
    hourly = None
    try:
        hourly, newest = fetch_elia_hourly()
        info["last_fetch"] = datetime.now(timezone.utc).isoformat()
        info["data_latest"] = iso(newest)
        age_h = (pd.Timestamp.now(tz="UTC") - hourly.index.max()).total_seconds() / 3600
        if age_h > STALE_HOURS:
            info["last_error"] = f"ELIA data is {age_h:.0f} h old -> replay mode"
            hourly = None
    except Exception as e:
        info["last_error"] = f"ELIA fetch failed: {e} -> replay mode"
        hourly = None

    if hourly is not None:
        info["mode"] = "live"
        for ts, v in hourly.items():
            ingest_hour(ts, v)
        while not stop_event.wait(POLL_SECONDS):
            try:
                hourly, newest = fetch_elia_hourly()
                info["last_fetch"] = datetime.now(timezone.utc).isoformat()
                info["data_latest"] = iso(newest)
                info["last_error"] = None
                for ts, v in hourly.items():
                    ingest_hour(ts, v)
            except Exception as e:
                info["last_error"] = f"ELIA fetch failed: {e}"
        return

    # ---- replay mode: stream the held-out test hours ----
    info["mode"] = "replay"
    df = pd.read_csv(DATA / "elia_hourly_load.csv", parse_dates=["Datetime"])
    df["Datetime"] = pd.to_datetime(df["Datetime"], utc=True)
    series = df.set_index("Datetime")["totalload"]
    start = int(len(series) * 0.85)            # same 70/15/15 split -> test part
    test = series.iloc[start:]
    it = iter(test.items())
    for _ in range(BACKFILL_HOURS):
        ts, v = next(it)
        ingest_hour(ts, v)
    for ts, v in it:
        if stop_event.wait(REPLAY_TICK_SECONDS):
            return
        ingest_hour(ts, v)
        info["last_fetch"] = datetime.now(timezone.utc).isoformat()
        info["data_latest"] = iso(ts)


def sampler():
    if psutil is None:
        return
    proc = psutil.Process()
    psutil.cpu_percent(None)
    proc.cpu_percent(None)
    while not stop_event.wait(2):
        sys_hist.append({
            "t": time.time(),
            "cpu_system": psutil.cpu_percent(None),
            "cpu_process": proc.cpu_percent(None) / (psutil.cpu_count() or 1),
            "mem_process_mb": proc.memory_info().rss / 1024 / 1024,
            "mem_system_pct": psutil.virtual_memory().percent,
        })


# --------------------------------------------------------------- metrics ----
def metrics(rows):
    r = [x for x in rows if x["actual"] is not None]
    if not r:
        return None
    a = np.array([x["actual"] for x in r])
    p = np.array([x["predicted"] for x in r])
    e = p - a
    pk = a >= np.percentile(a, 90)
    under, over = e < 0, e > 0
    return {
        "n": len(r),
        "mape": float(np.mean(np.abs(e) / a) * 100),
        "mae": float(np.mean(np.abs(e))),
        "rmse": float(np.sqrt(np.mean(e ** 2))),
        "under_rate": float(under.mean() * 100),
        "over_rate": float(over.mean() * 100),
        "mean_under_mw": float(np.abs(e[under]).mean()) if under.any() else 0.0,
        "mean_over_mw": float(e[over].mean()) if over.any() else 0.0,
        "peak_n": int(pk.sum()),
        "peak_mape": float(np.mean(np.abs(e[pk]) / a[pk]) * 100),
        "peak_under_rate": float((e[pk] < 0).mean() * 100),
    }


def system_now():
    if psutil is None:
        return {"available": False, "hint": "pip install psutil"}
    last = sys_hist[-1] if sys_hist else {}
    return {"available": True, "current": last,
            "history": list(sys_hist)[-45:],
            "cpu_count": psutil.cpu_count()}


# ------------------------------------------------------------------- app ----
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading ONNX model and scalers...")
    state["session"] = ort.InferenceSession(
        str(MODELS / "student.onnx"), providers=["CPUExecutionProvider"])
    xs = joblib.load(MODELS / "elia_scaler.pkl")      # ELIA (MW), not scaler.pkl
    ys = joblib.load(MODELS / "elia_y_scaler.pkl")
    state["x_min"] = xs.data_min_.astype(np.float32)
    state["x_rng"] = (xs.data_max_ - xs.data_min_).astype(np.float32)
    state["y_min"] = float(ys.data_min_[0])
    state["y_rng"] = float(ys.data_max_[0] - ys.data_min_[0])
    state["scaler_range"] = [float(xs.data_min_[0]), float(xs.data_max_[0])]
    stop_event.clear()
    for fn in (poller, sampler):
        threading.Thread(target=fn, daemon=True).start()
    print("Startup complete.")
    yield
    stop_event.set()
    state.clear()
    print("Shutdown complete.")


app = FastAPI(title="ELIA Peak Load Forecaster", version="0.3.0", lifespan=lifespan)


@app.get("/", response_class=HTMLResponse)
def dashboard():
    if not DASHBOARD.exists():
        return HTMLResponse("<h3>api/dashboard.html not found</h3>", status_code=404)
    return HTMLResponse(DASHBOARD.read_text(encoding="utf-8"))


@app.get("/health")
def health():
    if "session" not in state:
        raise HTTPException(status_code=503, detail="Model not loaded")
    stale = None
    if info["mode"] == "live" and info["data_latest"]:
        stale = (pd.Timestamp.now(tz="UTC") - pd.Timestamp(info["data_latest"])
                 ).total_seconds() / 60
    return {
        "status": "healthy",
        "model_loaded": True,
        "scaler_range_mw": state["scaler_range"],
        "mode": info["mode"],
        "uptime_s": round(time.time() - info["started"]),
        "last_fetch": info["last_fetch"],
        "data_latest": info["data_latest"],
        "staleness_min": None if stale is None else round(stale, 1),
        "last_error": info["last_error"],
        "predictions_logged": len(pred_log),
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(request: WindowRequest):
    start = time.perf_counter()
    x_raw = np.array(request.window, dtype=np.float32)
    if np.any(x_raw <= 0) or np.median(x_raw) < 1000:
        raise HTTPException(
            status_code=422,
            detail="Window must be raw load in MW (expected thousands, e.g. 6000-14000).")
    try:
        x_scaled = ((x_raw - state["x_min"]) / state["x_rng"]).reshape(1, -1).astype(np.float32)
        pred_scaled = state["session"].run(None, {"input": x_scaled})[0]
        pred_mw = float(pred_scaled.ravel()[0]) * state["y_rng"] + state["y_min"]
        latency_ms = (time.perf_counter() - start) * 1000
        return PredictionResponse(prediction_mw=pred_mw, latency_ms=latency_ms)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")


@app.get("/system")
def system():
    return system_now()


@app.get("/live")
def live():
    with lock:
        rows = [dict(r) for r in pred_log]
    rows.sort(key=lambda r: r["target_time"])
    resolved = [r for r in rows if r["actual"] is not None]
    pending = [r for r in rows if r["actual"] is None]
    last24 = resolved[-24:]

    peak = None
    if last24:
        ra = max(last24, key=lambda r: r["actual"])
        rp = max(last24, key=lambda r: r["predicted"])
        peak = {
            "actual_time": ra["target_time"], "actual_mw": ra["actual"],
            "pred_time": rp["target_time"], "pred_mw": rp["predicted"],
            "next_forecast_is_peak": bool(
                pending and pending[-1]["predicted"] >= ra["actual"]),
        }

    return {
        "mode": info["mode"],
        "server_time": datetime.now(timezone.utc).isoformat(),
        "health": health(),
        "next": pending[-1] if pending else None,
        "last_resolved": resolved[-1] if resolved else None,
        "history": rows[-60:],
        "metrics_24h": metrics(resolved[-24:]),
        "metrics_7d": metrics(resolved[-168:]),
        "peak": peak,
        "horizons": {"1h": pending[-1] if pending else None,
                     "6h": None, "12h": None, "24h": None},
        "system": system_now(),
    }


@app.get("/export.csv")
def export_csv():
    with lock:
        rows = [dict(r) for r in pred_log]
    cols = ["made_at", "based_on_hour", "target_time", "horizon_h",
            "predicted", "actual", "error", "abs_pct_err", "model_ms"]
    out = io.StringIO()
    w = csv.DictWriter(out, fieldnames=cols)
    w.writeheader()
    for r in rows:
        w.writerow({c: r.get(c) for c in cols})
    return StreamingResponse(
        iter([out.getvalue()]), media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=prediction_log.csv"})