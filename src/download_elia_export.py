import requests
import pandas as pd
import os

def download_elia_export():
    base_url = "https://opendata.elia.be/api/explore/v2.1/catalog/datasets/ods001/exports/csv"
    
    params = {
        "use_labels": "true",
        "where": "totalload IS NOT NULL AND datetime >= '2021-01-01' AND datetime <= '2026-06-30'",
        "order_by": "datetime"
    }
    
    print("Downloading full ELIA dataset via export endpoint...")
    response = requests.get(base_url, params=params, timeout=120)
    response.raise_for_status()
    
    # Save raw CSV
    os.makedirs("data/raw", exist_ok=True)
    raw_path = "data/raw/elia_raw_export.csv"
    with open(raw_path, "wb") as f:
        f.write(response.content)
    
    print(f"Raw export saved to {raw_path}")
    print(f"File size: {len(response.content) / (1024*1024):.1f} MB")
    
    # Load and inspect
    df = pd.read_csv(raw_path)
    print(f"\nRaw shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(df.head())
    
    return df


if __name__ == "__main__":
    df = download_elia_export()
    print(f"\nTotal records: {len(df)}")