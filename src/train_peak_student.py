"""
Multi-horizon peak-load forecaster: teacher ensemble -> distilled student -> ONNX.

Run:  python src/train_peak_student.py

Outputs
  models/student_peak.onnx        self-contained (input normalisation and MW
                                  denormalisation are baked into the graph)
  data/processed/peak_forecast_results.json
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from lightgbm import LGBMRegressor
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import onnxruntime as ort  # noqa: E402

from peak_data import (HORIZONS, N_FEATURES, build_dataset, evaluate,  # noqa: E402
                       hourly_series)

DATA = ROOT / "data" / "processed"
MODELS = ROOT / "models"
SEED = 42
HIDDEN = (64, 32)
EPOCHS = 120
BATCH = 1024
LR = 3e-3
KD_HARD_WEIGHT = 0.2
# A peak alert that is low is the dangerous direction, so both teacher and
# student target the TAU-th percentile of the peak rather than its mean.
TAU = 0.85


def main():
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)

    df = pd.read_csv(DATA / "elia_hourly_load.csv", parse_dates=["Datetime"])
    s = hourly_series(df)

    ds = build_dataset(s)
    split = ds["split"]
    tr, va, te = split == "train", split == "val", split == "test"
    print(f"rows: train {tr.sum()} | val {va.sum()} | test {te.sum()} "
          f"| dropped (cross-boundary) {(split == '').sum()}")

    x_scaler = MinMaxScaler().fit(ds["X"][tr])
    y_scaler = MinMaxScaler().fit(ds["y"][tr].reshape(-1, 1))
    X = x_scaler.transform(ds["X"]).astype(np.float32)
    y = y_scaler.transform(ds["y"].reshape(-1, 1)).ravel().astype(np.float32)

    # ---------------------------------------------------------- teacher ----
    t0 = time.perf_counter()
    xgb = XGBRegressor(n_estimators=600, max_depth=9, learning_rate=0.05,
                       subsample=0.8, colsample_bytree=0.8, random_state=SEED, n_jobs=-1,
                       objective="reg:quantileerror", quantile_alpha=TAU)
    lgb = LGBMRegressor(n_estimators=600, learning_rate=0.05, num_leaves=63,
                        subsample=0.8, colsample_bytree=0.8, random_state=SEED, n_jobs=-1,
                        verbose=-1, objective="quantile", alpha=TAU)
    xgb.fit(X[tr], y[tr])
    lgb.fit(X[tr], y[tr])
    teacher_fit_s = time.perf_counter() - t0

    p_xgb, p_lgb = xgb.predict(X), lgb.predict(X)
    teacher = 0.5 * (p_xgb + p_lgb)

    # ---------------------------------------------------------- student ----
    class Student(nn.Module):
        def __init__(self):
            super().__init__()
            layers, din = [], N_FEATURES
            for dout in HIDDEN:
                layers += [nn.Linear(din, dout), nn.ReLU()]
                din = dout
            self.body = nn.Sequential(*layers, nn.Linear(din, 1))
            self.register_buffer("x_min", torch.tensor(x_scaler.data_min_, dtype=torch.float32))
            self.register_buffer("x_rng", torch.tensor(
                np.maximum(x_scaler.data_max_ - x_scaler.data_min_, 1e-6), dtype=torch.float32))
            self.register_buffer("y_min", torch.tensor(float(y_scaler.data_min_[0])))
            self.register_buffer("y_rng", torch.tensor(float(y_scaler.data_max_[0] - y_scaler.data_min_[0])))

        def forward(self, raw):
            x = (raw - self.x_min) / self.x_rng
            return self.body(x).squeeze(-1) * self.y_rng + self.y_min

    class Core(nn.Module):
        def __init__(self):
            super().__init__()
            layers, din = [], N_FEATURES
            for dout in HIDDEN:
                layers += [nn.Linear(din, dout), nn.ReLU()]
                din = dout
            self.net = nn.Sequential(*layers, nn.Linear(din, 1))

        def forward(self, x):
            return self.net(x).squeeze(-1)

    core = Core()
    n_params = sum(p.numel() for p in core.parameters())
    Xt = torch.from_numpy(X[tr])
    yt = torch.from_numpy(y[tr])
    Tt = torch.from_numpy(teacher[tr])
    Xv = torch.from_numpy(X[va])
    yv = torch.from_numpy(y[va])

    def scaled_to_raw(v):
        return v.detach().numpy() * float(y_scaler.data_max_[0] - y_scaler.data_min_[0]) \
            + float(y_scaler.data_min_[0])

    opt = torch.optim.Adam(core.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    loss_fn = nn.MSELoss()

    def pinball(pred, target, tau=TAU):
        e = target - pred
        return torch.mean(torch.where(e >= 0, tau * e, (tau - 1.0) * e))

    best, best_state, patience = np.inf, None, 0
    idx = np.arange(len(Xt))
    for epoch in range(EPOCHS):
        core.train()
        rng.shuffle(idx)
        for i in range(0, len(idx), BATCH):
            b = idx[i:i + BATCH]
            opt.zero_grad()
            pred = core(Xt[b])
            loss = loss_fn(pred, Tt[b]) + KD_HARD_WEIGHT * pinball(pred, yt[b])
            loss.backward()
            opt.step()
        core.eval()
        with torch.no_grad():
            vl = loss_fn(core(Xv), torch.from_numpy(teacher[va])).item()
        sched.step()
        if vl < best - 1e-6:
            best, best_state, patience = vl, {k: v.clone() for k, v in core.state_dict().items()}, 0
        else:
            patience += 1
            if patience >= 25:
                break
    core.load_state_dict(best_state)

    student = Student()
    student.body = core.net

    # ------------------------------------------------- raw-space evaluation -
    with torch.no_grad():
        raw_pred = scaled_to_raw(core(torch.from_numpy(X)))
    teacher_raw = teacher * float(y_scaler.data_max_[0] - y_scaler.data_min_[0]) \
        + float(y_scaler.data_min_[0])
    y_true = ds["y"]
    persist = ds["persistence"]

    results = {"dataset": {
        "rows_total": int(len(y_true)), "n_features": N_FEATURES,
        "horizons": list(HORIZONS), "target": "max(load[t+1 .. t+h]) MW",
        "train_rows": int(tr.sum()), "val_rows": int(va.sum()), "test_rows": int(te.sum()),
        "test_span": [str(ds["target_end"][te].min()), str(ds["target_end"][te].max())],
    }, "objective": {
        "tau": TAU, "teacher": "pinball/quantile", "student_hard_term": "pinball",
        "rationale": "under-predicting a peak is the costly error for an operator alert",
    }, "teacher": {"members": ["XGBoost", "LightGBM"], "fit_seconds": round(teacher_fit_s, 1),
                   "epochs_run": epoch + 1},
        "student": {"params": int(n_params), "hidden": list(HIDDEN),
                    "kd_hard_label_weight": KD_HARD_WEIGHT},
        "per_horizon_test": []}
    for h in HORIZONS:
        m = te & (ds["horizon_h"] == h)
        results["per_horizon_test"].append({
            "persistence": evaluate(y_true[m], persist[m], h),
            "teacher": evaluate(y_true[m], teacher_raw[m], h),
            "student": evaluate(y_true[m], raw_pred[m], h),
        })
        st = results["per_horizon_test"][-1]["student"]
        print(f"h={h:>2}h  student MAPE {st['MAPE_pct']:.2f}%  coverage {st['coverage_pct']:.1f}%  "
              f"bias {st['bias_MW']:+.0f} MW  |  teacher "
              f"{results['per_horizon_test'][-1]['teacher']['MAPE_pct']:.2f}%  "
              f"persistence {results['per_horizon_test'][-1]['persistence']['MAPE_pct']:.2f}%")

    # ------------------------------------------------------------- export --
    with open(DATA / "peak_forecast_results.json", "w") as f:
        json.dump(results, f, indent=2)
    MODELS.mkdir(exist_ok=True)
    onnx_path = MODELS / "student_peak.onnx"
    student.eval()
    dummy = torch.randn(1, N_FEATURES) * 8000 + 9000
    dummy[:, 48:54] = 0.5  # cyclic calendar + horizon columns are already ~unit scale
    torch.onnx.export(student, dummy, str(onnx_path), dynamo=False,
                      input_names=["raw_features"], output_names=["peak_mw"],
                      dynamic_axes={"raw_features": {0: "batch"}, "peak_mw": {0: "batch"}},
                      opset_version=17)
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    Xte_raw = ds["X"][te].astype(np.float32)
    with torch.no_grad():
        ref = student(torch.from_numpy(Xte_raw)).numpy()
    got = sess.run(None, {"raw_features": Xte_raw})[0]
    max_abs_dev = float(np.max(np.abs(ref - got)))
    lat = []
    for row in Xte_raw[:300]:
        t0 = time.perf_counter()
        sess.run(None, {"raw_features": row[None, :].astype(np.float32)})
        lat.append((time.perf_counter() - t0) * 1000)

    results["onnx"] = {
        "path": str(onnx_path.relative_to(ROOT)),
        "bytes": onnx_path.stat().st_size,
        "scaling_baked_in": True,
        "max_abs_dev_vs_pytorch_MW": max_abs_dev,
        "single_window_latency_ms_mean": float(np.mean(lat)),
        "single_window_latency_ms_p95": float(np.percentile(lat, 95)),
        "four_horizon_call_ms": round(np.mean(lat) * len(HORIZONS), 3),
    }
    with open(DATA / "peak_forecast_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results["onnx"], indent=2))
    print("wrote", onnx_path)


if __name__ == "__main__":
    main()
