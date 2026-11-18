"""
ELIA Peak Load Forecaster - FastAPI service + live dashboard backend.
"""
import io
import csv
import time
import threading
from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
import platform
import sys
import socket
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
except ImportError:
    psutil = None

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
DATA = ROOT / "data" / "processed"
DASHBOARD = Path(__file__).resolve().parent / "dashboard.html"
sys.path.insert(0, str(ROOT / "src"))
from peak_data import build_query_matrix  # noqa: E402

ELIA_URL = "https://opendata.elia.be/api/explore/v2.1/catalog/datasets/ods001/records"
POLL_SECONDS = 300
STALE_HOURS = 24
REPLAY_TICK_SECONDS = 3
BACKFILL_HOURS = 60
MAX_LOG_ROWS = 1000
PEAK_HORIZONS = (1, 6, 12, 24)
HIST_HOURS = 48

state = {}
stop_event = threading.Event()
lock = threading.Lock()

buf = {}          # best-known hourly load per hour: actual if published, else ELIA nowcast
act_buf = {}      # only hours where ELIA has published the measured actual
pred_log = []
peak_log = []
sys_hist = deque(maxlen=90)
info = {"mode": "starting", "started": time.time(), "last_fetch": None,
        "data_latest": None, "last_error": None, "peak_model": None,
        "actual_latest": None, "nowcast_hours": 0}

# Live IoT Sensor Data
live_sensor = {"temp": 26.0, "light": 500.0, "voltage": 240.0, "humidity": 65.0}
sensor_events = {"readings": 0, "voltage_sags": 0, "sms_dispatched": 0}

# Every figure below is measured from the prediction log or the sensor stream.
DEMO_NOTE = ("Accuracy and event counts are computed from live data. Sensor telemetry is "
             "simulated by simulate/virtual_sensor.py and SMS dispatch only prints to console.")


def run_model(x_raw):
    x = ((np.asarray(x_raw, dtype=np.float32) - state["x_min"]) / state["x_rng"])
    x = x.reshape(1, -1).astype(np.float32)
    p = state["session"].run(None, {"input": x})[0]
    return float(p.ravel()[0]) * state["y_rng"] + state["y_min"]


def run_peak(window, based_on):
    """Peak MW expected over the next h hours, for each configured horizon."""
    if state.get("peak_session") is None:
        return []
    feats = build_query_matrix(window, based_on, horizons=PEAK_HORIZONS)
    out = state["peak_session"].run(None, {"raw_features": feats.astype(np.float32)})[0]
    return [float(v) for v in out.ravel()]


def resolve_pending_peaks(ts):
    """
    Score a peak alert only once every hour of its window has a published
    *actual*. Nowcasts are never used as ground truth, so reported accuracy
    stays comparable to the offline test numbers.
    """
    for row in peak_log:
        if row["actual"] is not None:
            continue
        start = pd.Timestamp(row["based_on_hour"]) + pd.Timedelta(hours=1)
        hours = [start + pd.Timedelta(hours=k) for k in range(row["horizon_h"])]
        if hours[-1] > ts:
            continue
        vals = [act_buf.get(h) for h in hours]
        if any(v is None for v in vals):
            continue
        actual = float(max(vals))
        row["actual"] = actual
        row["error"] = row["predicted"] - actual
        row["abs_pct_err"] = abs(row["error"]) / actual * 100
        row["actual_peak_hour"] = iso(hours[int(np.argmax(vals))])

class WindowRequest(BaseModel):
    window: list[float] = Field(..., min_length=24, max_length=24)


class PeakWindowRequest(BaseModel):
    window: list[float] = Field(..., min_length=HIST_HOURS, max_length=HIST_HOURS)
    last_hour_utc: datetime | None = Field(
        None, description="Timestamp of the final hour in the window; defaults to now.")


class PredictionResponse(BaseModel):
    prediction_mw: float
    latency_ms: float


class SensorReading(BaseModel):
    temp: float = Field(..., ge=-20, le=60)
    light: float = Field(..., ge=0, le=200_000)
    voltage: float = Field(..., ge=0, le=500)
    humidity: float = Field(..., ge=0, le=100)


class AlertRequest(BaseModel):
    phone: str = Field(..., min_length=9, max_length=20)
    message: str = Field(..., min_length=1, max_length=500)


def iso(ts):
    return None if ts is None else pd.Timestamp(ts).isoformat()

def ingest_hour(ts, value, is_actual=True):
    """
    Feed one hour into the buffer. `is_actual=False` means ELIA has not
    published the measurement yet and we are using its own nowcast, which is
    what lets forecasts update before the ~6 h reporting lag clears.
    Re-ingesting an hour as an actual upgrades the input but does not create a
    duplicate forecast - the next hour is computed from the corrected history.
    """
    ts = pd.Timestamp(ts)
    with lock:
        seen = ts in buf
        if seen and is_actual == (ts in act_buf):
            return
        buf[ts] = float(value)
        if is_actual:
            act_buf[ts] = float(value)
        for k in sorted(buf)[:-300]:
            del buf[k]
        for k in sorted(act_buf)[:-300]:
            del act_buf[k]

        if is_actual:
            for row in pred_log:
                if row["target_time"] == iso(ts) and row["actual"] is None:
                    row["actual"] = float(value)
                    row["error"] = row["predicted"] - row["actual"]
                    row["abs_pct_err"] = abs(row["error"]) / row["actual"] * 100
        resolve_pending_peaks(max(act_buf) if act_buf else ts)

        if seen:
            return

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
                "input_source": "actual" if ts in act_buf else "nowcast",
                "model_ms": lat,
            })
            del pred_log[:-MAX_LOG_ROWS]

        hist_hours = [ts - pd.Timedelta(hours=k) for k in range(HIST_HOURS - 1, -1, -1)]
        if all(h in buf for h in hist_hours):
            t0 = time.perf_counter()
            peaks = run_peak([buf[h] for h in hist_hours], ts)
            peak_ms = (time.perf_counter() - t0) * 1000
            for horizon, p in zip(PEAK_HORIZONS, peaks):
                peak_log.append({
                    "made_at": datetime.now(timezone.utc).isoformat(),
                    "based_on_hour": iso(ts),
                    "horizon_h": horizon,
                    "target_window_end": iso(ts + pd.Timedelta(hours=horizon)),
                    "predicted": p, "actual": None, "error": None,
                    "abs_pct_err": None, "actual_peak_hour": None,
                    "input_source": "actual" if ts in act_buf else "nowcast",
                    "model_ms": peak_ms,
                })
            del peak_log[:-MAX_LOG_ROWS]


def _hourly_from_rows(rows, col):
    df = pd.DataFrame(rows)[["datetime", col]]
    df = df.dropna()
    if df.empty:
        return pd.Series(dtype=float)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    s = df.set_index("datetime")[col].sort_index().astype(float)
    g = s.resample("1h")
    # only hours where all four quarter-hour slots exist are trustworthy
    return g.mean()[g.count() == 4].dropna()


def fetch_elia():
    """
    Return (actuals, nowcasts). Actuals arrive from ELIA with a ~6 h lag; the
    nowcast column is published for future hours immediately, so the service
    can forecast the coming hours instead of waiting for them.
    """
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
    actual = _hourly_from_rows(rows, "totalload")
    actual = actual.dropna()
    newest = pd.to_datetime([r["datetime"] for r in rows], utc=True).max()

    frows = []
    for offset in range(0, 300, 100):
        r = requests.get(ELIA_URL, timeout=20, params={
            "where": f"totalload IS NULL AND datetime >= '{newest.isoformat()}'"
                     f" AND mostrecentforecast IS NOT NULL",
            "order_by": "datetime asc",
            "limit": 100, "offset": offset,
        })
        r.raise_for_status()
        frows += r.json().get("results", [])
    nowcast = _hourly_from_rows(frows, "mostrecentforecast")
    nowcast = nowcast[nowcast.index > actual.index.max()] if len(actual) else nowcast
    return actual, nowcast, newest

def machine_info():
    import onnxruntime as _ort
    info = {
        "hostname": socket.gethostname(),
        "os": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or "unknown",
        "python": sys.version.split()[0],
        "onnxruntime": _ort.__version__,
    }
    if psutil:
        info["cpu_count"] = psutil.cpu_count()
        info["ram_gb"] = round(psutil.virtual_memory().total / 1024**3, 1)
    return info

def ingest_feed(actual, nowcast):
    """Actuals first, then nowcasts, so an hour is never recorded as an actual by accident."""
    for ts, v in actual.items():
        ingest_hour(ts, v, is_actual=True)
    for ts, v in nowcast.items():
        ingest_hour(ts, v, is_actual=False)
    info["actual_latest"] = iso(actual.index.max()) if len(actual) else None
    horizon_end = max([t for t in (actual.index.max() if len(actual) else None,
                                   nowcast.index.max() if len(nowcast) else None)
                       if t is not None])
    info["data_latest"] = iso(horizon_end)
    info["nowcast_hours"] = int(len(nowcast))


def poller():
    actual = nowcast = None
    try:
        actual, nowcast, newest = fetch_elia()
        info["last_fetch"] = datetime.now(timezone.utc).isoformat()
        age_h = (pd.Timestamp.now(tz="UTC") - actual.index.max()).total_seconds() / 3600
        if age_h > STALE_HOURS:
            info["last_error"] = f"ELIA actuals are {age_h:.0f} h old -> replay mode"
            actual = None
    except Exception as e:
        info["last_error"] = f"ELIA fetch failed: {e} -> replay mode"
        actual = None

    if actual is not None:
        info["mode"] = "live"
        ingest_feed(actual, nowcast)
        while not stop_event.wait(POLL_SECONDS):
            try:
                actual, nowcast, newest = fetch_elia()
                info["last_fetch"] = datetime.now(timezone.utc).isoformat()
                info["last_error"] = None
                ingest_feed(actual, nowcast)
            except Exception as e:
                info["last_error"] = f"ELIA fetch failed: {e}"
        return

    info["mode"] = "replay"
    df = pd.read_csv(DATA / "elia_hourly_load.csv", parse_dates=["Datetime"])
    df["Datetime"] = pd.to_datetime(df["Datetime"], utc=True)
    series = df.set_index("Datetime")["totalload"]
    start = int(len(series) * 0.85)
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


def impact_now():
    """Every value here is counted or computed at request time - nothing is hardcoded."""
    with lock:
        hourly = [dict(r) for r in pred_log]
        peaks = [dict(r) for r in peak_log]
    hourly_ok = [r for r in hourly if r["actual"] is not None]
    m = metrics(hourly_ok)
    peak_m = {h: metrics([r for r in peaks if r["horizon_h"] == h]) for h in PEAK_HORIZONS}
    resolved_peaks = [r for r in peaks if r["actual"] is not None]
    return {
        "hourly_mape_pct": None if m is None else round(m["mape"], 3),
        "hourly_accuracy_pct": None if m is None else round(100 - m["mape"], 3),
        "hours_forecast": len(hourly),
        "hours_resolved": len(hourly_ok),
        "peak_alerts": len(peaks),
        "peak_alerts_resolved": len(resolved_peaks),
        "peak_mape_pct": {h: None if v is None else round(v["mape"], 3)
                          for h, v in peak_m.items()},
        "peak_under_rate_pct": {h: None if v is None else round(v["under_rate"], 1)
                               for h, v in peak_m.items()},
        "sensor_readings": sensor_events["readings"],
        "voltage_sags": sensor_events["voltage_sags"],
        "sms_dispatched": sensor_events["sms_dispatched"],
        "note": DEMO_NOTE,
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading ONNX model and scalers...")
    state["session"] = ort.InferenceSession(
        str(MODELS / "student.onnx"), providers=["CPUExecutionProvider"])
    xs = joblib.load(MODELS / "elia_scaler.pkl")
    ys = joblib.load(MODELS / "elia_y_scaler.pkl")
    state["x_min"] = xs.data_min_.astype(np.float32)
    state["x_rng"] = (xs.data_max_ - xs.data_min_).astype(np.float32)
    state["y_min"] = float(ys.data_min_[0])
    state["y_rng"] = float(ys.data_max_[0] - ys.data_min_[0])
    state["scaler_range"] = [float(xs.data_min_[0]), float(xs.data_max_[0])]
    peak_path = MODELS / "student_peak.onnx"
    if peak_path.exists():
        state["peak_session"] = ort.InferenceSession(
            str(peak_path), providers=["CPUExecutionProvider"])
        info["peak_model"] = peak_path.name
    else:
        state["peak_session"] = None
        info["peak_model"] = None
        print(f"WARNING: {peak_path} missing - multi-horizon peaks disabled. "
              f"Run: python src/train_peak_student.py")
    stop_event.clear()
    for fn in (poller, sampler):
        threading.Thread(target=fn, daemon=True).start()
    print("Startup complete.")
    yield
    stop_event.set()
    state.clear()
    print("Shutdown complete.")

app = FastAPI(title="Elektron | Mukono District", version="0.4.0", lifespan=lifespan)

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
    if info["mode"] == "live" and info["actual_latest"]:
        stale = (pd.Timestamp.now(tz="UTC") - pd.Timestamp(info["actual_latest"])
                 ).total_seconds() / 60
    return {
        "status": "healthy",
        "model_loaded": True,
        "scaler_range_mw": state["scaler_range"],
        "multi_horizon_model": info["peak_model"],
        "mode": info["mode"],
        "uptime_s": round(time.time() - info["started"]),
        "last_fetch": info["last_fetch"],
        "data_latest": info["data_latest"],
        "actual_latest": info["actual_latest"],
        "nowcast_hours_ahead": info["nowcast_hours"],
        "staleness_min": None if stale is None else round(stale, 1),
        "last_error": info["last_error"],
        "predictions_logged": len(pred_log),
        "peak_alerts_logged": len(peak_log),
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
        pred_mw = run_model(x_raw)
        latency_ms = (time.perf_counter() - start) * 1000
        return PredictionResponse(prediction_mw=pred_mw, latency_ms=latency_ms)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")

@app.post("/predict_peak")
def predict_peak(request: PeakWindowRequest):
    """Peak MW expected over the next 1/6/12/24 h from a 48-hour history window."""
    if state.get("peak_session") is None:
        raise HTTPException(status_code=503, detail="Multi-horizon peak model not loaded")
    x_raw = np.array(request.window, dtype=np.float32)
    if np.any(x_raw <= 0) or np.median(x_raw) < 1000:
        raise HTTPException(
            status_code=422,
            detail="Window must be raw load in MW (expected thousands, e.g. 6000-14000).")
    anchor = pd.Timestamp(request.last_hour_utc or pd.Timestamp.now(tz="UTC")).floor("h")
    if anchor.tzinfo is None:
        anchor = anchor.tz_localize("UTC")
    start = time.perf_counter()
    peaks = run_peak(x_raw, anchor)
    latency_ms = (time.perf_counter() - start) * 1000
    return {
        "based_on_hour": iso(anchor),
        "latency_ms": round(latency_ms, 3),
        "peaks": [{"horizon_h": h,
                   "peak_mw": p,
                   "window_end": iso(anchor + pd.Timedelta(hours=h))}
                  for h, p in zip(PEAK_HORIZONS, peaks)],
    }


@app.get("/system")
def system():
    return {**system_now(), "machine": machine_info()}

@app.post("/sensor")
def receive_sensor_data(data: SensorReading):
    global live_sensor
    live_sensor = data.model_dump()
    sensor_events["readings"] += 1
    if data.voltage < 230:
        sensor_events["voltage_sags"] += 1
    return {"status": "ok", "sags_so_far": sensor_events["voltage_sags"]}


@app.post("/alert_operator")
def alert_operator(payload: AlertRequest):
    sensor_events["sms_dispatched"] += 1

    print(f"\n=========================================")
    print(f"SMS DISPATCHED (console demo, no gateway): {payload.phone}")
    print(f"MESSAGE: {payload.message}")
    print(f"=========================================\n")

    return {"status": "queued", "recipient": payload.phone}

@app.get("/live")
def live():
    with lock:
        rows = [dict(r) for r in pred_log]
        prows = [dict(r) for r in peak_log]
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

    latest_pending, resolved_peaks = {}, []
    for r in prows:
        if r["actual"] is None:
            latest_pending[r["horizon_h"]] = r
        else:
            resolved_peaks.append(r)
    resolved_peaks.sort(key=lambda r: r["target_window_end"])

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
        "horizons": {f"{h}h": latest_pending.get(h) for h in PEAK_HORIZONS},
        "peak_history": resolved_peaks[-40:],
        "peak_metrics": {f"{h}h": metrics(
            [r for r in resolved_peaks if r["horizon_h"] == h][-72:])
            for h in PEAK_HORIZONS},
        "system": system_now(),
        "sensor": live_sensor,
        "impact": impact_now()
    }


def csv_response(rows, cols, filename):
    out = io.StringIO()
    w = csv.DictWriter(out, fieldnames=cols)
    w.writeheader()
    for r in rows:
        w.writerow({c: r.get(c) for c in cols})
    return StreamingResponse(
        iter([out.getvalue()]), media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"})


@app.get("/export.csv")
def export_csv():
    with lock:
        rows = [dict(r) for r in pred_log]
    cols = ["made_at", "based_on_hour", "target_time", "horizon_h",
            "predicted", "actual", "error", "abs_pct_err", "model_ms"]
    return csv_response(rows, cols, "prediction_log.csv")


@app.get("/export_peaks.csv")
def export_peaks_csv():
    with lock:
        rows = [dict(r) for r in peak_log]
    cols = ["made_at", "based_on_hour", "horizon_h", "target_window_end",
            "predicted", "actual", "error", "abs_pct_err", "actual_peak_hour", "model_ms"]
    return csv_response(rows, cols, "peak_forecast_log.csv")