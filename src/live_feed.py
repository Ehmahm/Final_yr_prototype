import os
import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("EIA_API_KEY")

def get_live_demand(ba_code="PJM", hours=48):
    """
    Fetches the most recent hourly electricity demand for a given
    balancing authority from the EIA Open Data API v2.
    """
    url = "https://api.eia.gov/v2/electricity/rto/region-data/data/"
    params = {
        "api_key": API_KEY,
        "frequency": "hourly",
        "data[0]": "value",
        "facets[respondent][]": ba_code,
        "facets[type][]": "D",  # D = Demand
        "sort[0][column]": "period",
        "sort[0][direction]": "desc",
        "offset": 0,
        "length": hours,
    }

    response = requests.get(url, params=params)
    response.raise_for_status()
    data = response.json()["response"]["data"]

    df = pd.DataFrame(data)
    df["period"] = pd.to_datetime(df["period"])
    df = df.rename(columns={"value": "demand_mw"})
    df = df[["period", "demand_mw"]].sort_values("period").reset_index(drop=True)
    return df


if __name__ == "__main__":
    df = get_live_demand()
    print(df.head())
    print(f"\nFetched {len(df)} rows for PJM.")