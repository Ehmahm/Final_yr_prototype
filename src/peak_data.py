"""
Multi-horizon peak-load dataset.

For a window ending at hour t and a forecast horizon h, the target is
    peak_h(t) = max(load[t+1 .. t+h])
so h=1 is next-hour load and h=24 is the next-day peak demand.

Only information available at or before t is used: the trailing 48 hourly
loads and the calendar position of the target hour.
"""
import numpy as np
import pandas as pd

HIST_HOURS = 48
HORIZONS = (1, 6, 12, 24)
N_FEATURES = HIST_HOURS + 8
SPLIT_FRACTIONS = (0.70, 0.85)


def hourly_series(df, value_col="totalload", time_col="Datetime"):
    """Return a gap-filled, contiguous hourly series in UTC."""
    s = df.set_index(pd.to_datetime(df[time_col], utc=True))[value_col].sort_index()
    s = s[~s.index.duplicated(keep="last")]
    s = s.resample("1h").mean().interpolate(limit=3).dropna()
    return s


def _future_max(values, h):
    """future_max[i] = max(values[i+1 .. i+h]); NaN where incomplete."""
    rm = pd.Series(values).rolling(h).max().shift(-h).to_numpy()
    return rm


def _calendar(times, h):
    """Cyclic features of the target hour t+h, plus the horizon itself."""
    tgt = times + pd.Timedelta(hours=h)
    hour = tgt.hour.to_numpy() + tgt.minute.to_numpy() / 60.0
    dow = tgt.dayofweek.to_numpy()
    ang_h = 2 * np.pi * hour / 24.0
    ang_d = 2 * np.pi * dow / 7.0
    return np.column_stack([
        np.sin(ang_h), np.cos(ang_h), np.sin(ang_d), np.cos(ang_d),
        (dow >= 5).astype(float),
        np.full(hour.shape, float(h) / max(HORIZONS)),
    ])


def _build_one(values, times, h, hist, positions):
    # swv[k] covers values[k .. k+hist-1]; a window ending at i starts at i-hist+1
    windows = np.lib.stride_tricks.sliding_window_view(values, hist)[positions - hist + 1]
    feats = np.column_stack([
        windows,
        _calendar(times[positions], h),
        windows[:, -24:].max(axis=1),
        windows[:, -24:].mean(axis=1),
    ])
    return feats, _future_max(values, h)[positions], windows


def build_dataset(series, horizons=HORIZONS, hist=HIST_HOURS):
    """Stack every horizon into one training set, keyed by (t, h)."""
    values = series.to_numpy(dtype=np.float64)
    times = series.index
    n = len(values)
    parts = []
    for h in horizons:
        lo = hist - 1
        hi = n - 1 - h
        if hi <= lo:
            continue
        positions = np.arange(lo, hi + 1)
        feats, target, windows = _build_one(values, times, h, hist, positions)
        ok = np.isfinite(target)
        parts.append({
            "horizon_h": np.full(positions.size, h),
            "X": feats, "y": target, "valid": ok,
            "based_on": times[positions], "target_end": times[positions + h],
            "pos": positions, "persistence": windows[:, -24:].max(axis=1),
        })
    X = np.concatenate([p["X"][p["valid"]] for p in parts])
    y = np.concatenate([p["y"][p["valid"]] for p in parts])
    h_arr = np.concatenate([p["horizon_h"][p["valid"]] for p in parts])
    pos = np.concatenate([p["pos"][p["valid"]] for p in parts])
    tend = np.concatenate([p["target_end"][p["valid"]] for p in parts])
    persist = np.concatenate([p["persistence"][p["valid"]] for p in parts])
    split = _temporal_split(pos, h_arr, n, hist)
    keep = split != ""
    return {"X": X[keep], "y": y[keep], "horizon_h": h_arr[keep],
            "target_end": tend[keep], "persistence": persist[keep],
            "split": split[keep]}


def _temporal_split(positions, h_arr, n, hist):
    """
    Split by the position of the target hour, then drop rows whose input
    window reaches into another split - that would leak future load into
    training. Dropped rows get the label "".
    """
    edges = [int(n * f) for f in SPLIT_FRACTIONS]
    end_pos = positions + h_arr
    start_pos = positions - hist + 1

    def label(p):
        return np.where(p < edges[0], 0, np.where(p < edges[1], 1, 2))

    names = np.array(["train", "val", "test"])[label(end_pos)]
    names[label(start_pos) != label(end_pos)] = ""
    return names


def build_query_matrix(values, window_end_time, horizons=HORIZONS, hist=HIST_HOURS):
    """
    One feature row per horizon, in exactly the column order produced by
    build_dataset. `window_end_time` is the last observed hour t; the target
    hour is t+h. Used by the serving layer so training and inference cannot
    drift apart.
    """
    v = np.asarray(values, dtype=np.float64)[-hist:]
    if v.shape[0] != hist:
        raise ValueError(f"need {hist} hourly loads, got {v.shape[0]}")
    w24 = v[-24:]
    t = pd.Timestamp(window_end_time)
    rows = []
    for h in horizons:
        rows.append(np.concatenate(
            [v, _calendar(pd.DatetimeIndex([t]), h)[0], [w24.max(), w24.mean()]]))
    return np.asarray(rows, dtype=np.float64)


def evaluate(y_true, y_pred, horizon, y_persist=None):
    e = y_pred - y_true
    out = {
        "horizon_h": int(horizon),
        "n": int(y_true.size),
        "MAE_MW": float(np.mean(np.abs(e))),
        "RMSE_MW": float(np.sqrt(np.mean(e ** 2))),
        "MAPE_pct": float(np.mean(np.abs(e) / y_true) * 100),
        "bias_MW": float(np.mean(e)),
        "under_predict_pct": float(np.mean(e < 0) * 100),
        # For an alert, the useful question is "did it stay above the real peak?"
        "coverage_pct": float(np.mean(y_pred >= y_true) * 100),
        "mean_shortfall_MW": float(np.mean(np.clip(y_true - y_pred, 0, None))),
    }
    if y_persist is not None:
        mape_p = float(np.mean(np.abs(y_persist - y_true) / y_true) * 100)
        out["persistence_MAPE_pct"] = mape_p
        out["skill_vs_persistence_pct"] = (mape_p - out["MAPE_pct"]) / mape_p * 100
    peak_mask = y_true >= np.percentile(y_true, 90)
    if peak_mask.sum() > 0:
        out["top10pct_MAPE_pct"] = float(
            np.mean(np.abs(e[peak_mask]) / y_true[peak_mask]) * 100)
    return out
