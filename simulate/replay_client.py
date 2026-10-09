"""
CSV replay client — simulates a live sensor stream by sending 24-hour
windows to the FastAPI /predict endpoint one at a time.
"""

import time
import requests
import numpy as np
import pandas as pd
from pathlib import Path

API_URL = "http://127.0.0.1:8000/predict"
HEALTH_URL = "http://127.0.0.1:8000/health"
DATA = Path(__file__).resolve().parent.parent / "data" / "processed"

# Number of test windows to replay
N_SAMPLES = 200
DELAY_SEC = 0.05   # simulate ~20 predictions per second


def load_test_data():
    """Load ELIA test features (scaled) and raw MW targets."""
    X_test_scaled = np.load(DATA / "elia_X_test.npy").astype(np.float32)
    y_test_raw    = np.load(DATA / "y_test_raw.npy").astype(np.float32)
    return X_test_scaled, y_test_raw


def unscale_window(x_scaled_window, x_scaler):
    """Convert a scaled window back to raw MW for sending to the API."""
    return x_scaler.inverse_transform(x_scaled_window.reshape(1, -1)).flatten().tolist()


def main():
    import joblib
    x_scaler = joblib.load(
        Path(__file__).resolve().parent.parent / "models" / "scaler.pkl"
    )

    # Health check
    r = requests.get(HEALTH_URL, timeout=5)
    if r.status_code != 200:
        raise RuntimeError(f"Server unhealthy: {r.text}")
    print("Server healthy.")

    X_test, y_test = load_test_data()
    n = min(N_SAMPLES, len(X_test))

    predictions = []
    actuals = []
    latencies = []

    print(f"\nReplaying {n} windows...\n")
    print(f"{'i':>4} {'predicted':>12} {'actual':>12} {'abs err':>10} {'latency':>10}")
    print("-" * 55)

    for i in range(n):
        x_scaled = X_test[i]
        actual_mw = float(y_test[i])

        # Convert scaled window to raw MW for the API (realistic edge input)
        raw_window = unscale_window(x_scaled, x_scaler)

        try:
            resp = requests.post(
                API_URL,
                json={"window": raw_window},
                timeout=2.0
            )
            resp.raise_for_status()
            out = resp.json()
            pred = out["prediction_mw"]
            lat = out["latency_ms"]
        except requests.exceptions.RequestException as e:
            print(f"Request {i} failed: {e}")
            continue

        predictions.append(pred)
        actuals.append(actual_mw)
        latencies.append(lat)

        if i % 20 == 0 or i == n - 1:
            err = abs(pred - actual_mw)
            print(f"{i:>4} {pred:>12.2f} {actual_mw:>12.2f} {err:>10.2f} {lat:>9.2f}ms")

        time.sleep(DELAY_SEC)

    # --- Summary ---
    preds = np.array(predictions)
    acts  = np.array(actuals)
    lats  = np.array(latencies)

    mape = np.mean(np.abs((acts - preds) / np.clip(np.abs(acts), 1e-6, None))) * 100
    mae  = np.mean(np.abs(acts - preds))

    print("\n" + "=" * 55)
    print(f"Replayed:         {len(predictions)} windows")
    print(f"MAPE:             {mape:.4f}%")
    print(f"MAE:              {mae:.2f} MW")
    print(f"Mean latency:     {lats.mean():.3f} ms")
    print(f"P95 latency:      {np.percentile(lats, 95):.3f} ms")
    print(f"Max latency:      {lats.max():.3f} ms")
    print("=" * 55)


if __name__ == "__main__":
    main()