"""
CSV replay client: sends raw-MW 24h windows to /predict one at a time.
Reports model accuracy plus server-side vs HTTP round-trip latency.
"""

import time
import joblib
import requests
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"
MODELS = ROOT / "models"
API_URL = "http://127.0.0.1:8000/predict"
HEALTH_URL = "http://127.0.0.1:8000/health"

N_SAMPLES = 200
DELAY_SEC = 0.05


def main():
    x_scaler = joblib.load(MODELS / "elia_scaler.pkl")
    session = requests.Session()  # reuse one connection

    r = session.get(HEALTH_URL, timeout=5)
    if r.status_code != 200:
        raise RuntimeError(f"Server unhealthy: {r.text}")
    print("Server healthy.")

    X_test_scaled = np.load(DATA / "elia_X_test.npy").astype(np.float32)
    y_test = np.load(DATA / "elia_y_test.npy").astype(np.float32)  # raw MW
    n = min(N_SAMPLES, len(X_test_scaled))
    X_raw = x_scaler.inverse_transform(X_test_scaled[:n])

    preds, acts, rtt, server = [], [], [], []
    for i in range(n):
        t0 = time.perf_counter()
        resp = session.post(API_URL, json={"window": X_raw[i].tolist()}, timeout=5)
        rtt.append((time.perf_counter() - t0) * 1000)
        resp.raise_for_status()
        out = resp.json()
        preds.append(out["prediction_mw"])
        server.append(out["latency_ms"])
        acts.append(float(y_test[i]))
        time.sleep(DELAY_SEC)

    preds, acts = np.array(preds), np.array(acts)
    rtt, server = np.array(rtt), np.array(server)
    err = preds - acts

    print(f"\nWindows:         {n}")
    print(f"MAPE:            {np.mean(np.abs(err) / acts) * 100:.3f}%")
    print(f"MAE:             {np.mean(np.abs(err)):.1f} MW")
    print(f"Under-predict:   {np.mean(err < 0) * 100:.1f}% of hours")
    print(f"Server latency:  mean {server.mean():.2f} ms | p95 {np.percentile(server, 95):.2f} ms")
    print(f"HTTP round trip: mean {rtt.mean():.2f} ms | p95 {np.percentile(rtt, 95):.2f} ms")


if __name__ == "__main__":
    main()