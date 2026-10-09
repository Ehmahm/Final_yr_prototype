import requests
import pandas as pd
import time
import os

def fetch_elia_data(start_date="2021-01-01", end_date="2026-06-30"):
    base_url = "https://opendata.elia.be/api/records/1.0/search/"
    all_records = []
    offset = 0
    limit = 100

    # First, get the total count so we can show progress
    count_params = {
        "dataset": "ods001",
        "rows": 0,
        "where": f"totalload IS NOT NULL AND datetime >= '{start_date}' AND datetime <= '{end_date}'",
    }
    count_resp = requests.get(base_url, params=count_params, timeout=30)
    count_resp.raise_for_status()
    total = count_resp.json().get("nhits", 0)
    print(f"Total records to fetch: {total}")
    print(f"Estimated calls: {(total // limit) + 1}")
    print(f"Estimated time: ~{((total // limit) + 1) * 0.5 / 60:.1f} minutes\n")

    while offset < total:
        params = {
            "dataset": "ods001",
            "rows": limit,
            "start": offset,
            "where": f"totalload IS NOT NULL AND datetime >= '{start_date}' AND datetime <= '{end_date}'",
            "order_by": "datetime",
            "select": "datetime, totalload, resolutioncode"
        }

        try:
            response = requests.get(base_url, params=params, timeout=30)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            print(f"\nRequest failed at offset {offset}: {e}")
            print("Retrying in 5 seconds...")
            time.sleep(5)
            continue

        data = response.json()
        records = data.get("records", [])
        if not records:
            break

        for rec in records:
            fields = rec.get("fields", {})
            all_records.append({
                "datetime": fields.get("datetime"),
                "totalload": fields.get("totalload"),
                "resolutioncode": fields.get("resolutioncode")
            })

        offset += limit
        print(f"Fetched {len(all_records)} / {total} records", end="\r")
        time.sleep(0.5)

    print(f"\n\nDownload complete: {len(all_records)} records")

    df = pd.DataFrame(all_records)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.set_index("datetime").sort_index()

    # Resample from 15-min to hourly
    df_hourly = df.resample("1h").mean()

    return df_hourly


if __name__ == "__main__":
    # Ensure output directory exists
    os.makedirs("data/raw", exist_ok=True)

    print("Fetching ELIA data (2021-01-01 to 2026-06-30)...\n")
    df = fetch_elia_data(start_date="2021-01-01", end_date="2026-06-30")

    output_path = "data/raw/elia_hourly_load.csv"
    df.to_csv(output_path)

    print(f"\nSaved {len(df)} hourly records to {output_path}")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    print(f"\nFirst 5 rows:")
    print(df.head())
    print(f"\nLast 5 rows:")
    print(df.tail())
    print(f"\nMissing values in totalload: {df['totalload'].isna().sum()}")