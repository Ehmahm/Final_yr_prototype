import pandas as pd

def load_raw_data(path="data/household_power_consumption.txt"):
    """
    Loads the raw UCI minute-level dataset.
    """
    df = pd.read_csv(
        path,
        sep=";",
        low_memory=False,
        na_values=["?"],
    )
    df["datetime"] = pd.to_datetime(
        df["Date"] + " " + df["Time"], format="%d/%m/%Y %H:%M:%S"
    )
    df = df.drop(columns=["Date", "Time"])
    df = df.set_index("datetime")
    return df


def resample_hourly(df):
    """
    Resamples minute-level data to hourly averages,
    interpolating short gaps in missing data.
    """
    numeric_cols = df.select_dtypes(include="number").columns
    hourly = df[numeric_cols].resample("h").mean()
    hourly = hourly.interpolate(method="linear", limit=6)  # up to 6h gaps
    hourly = hourly.dropna()
    return hourly


def add_peak_load_target(df, horizon_hours=24, target_col="Global_active_power"):
    """
    Creates the forecasting target: the peak (max) load
    over the next `horizon_hours` window.
    """
    df = df.copy()
    df["peak_load_next_24h"] = (
        df[target_col]
        .shift(-1)
        .rolling(window=horizon_hours)
        .max()
        .shift(-(horizon_hours - 1))
    )
    df = df.dropna()
    return df

import numpy as np

def build_sequences(df, feature_cols, seq_len):
    """
    Converts a DataFrame into overlapping sequences of shape
    (num_windows, seq_len, num_features) for LSTM input.
    """
    data = df[feature_cols].values
    sequences = []
    for i in range(len(data) - seq_len + 1):
        sequences.append(data[i:i + seq_len])
    return np.array(sequences)


def resample_daily(hourly_df):
    """
    Aggregates hourly data to daily means, for monthly-scale patterns.
    """
    numeric_cols = hourly_df.select_dtypes(include="number").columns
    daily = hourly_df[numeric_cols].resample("D").mean()
    daily = daily.dropna()
    return daily

if __name__ == "__main__":
    raw = load_raw_data()
    print("Raw shape:", raw.shape)

    hourly = resample_hourly(raw)
    print("Hourly shape:", hourly.shape)

    final = add_peak_load_target(hourly)
    print("With target shape:", final.shape)
    print(final.head())