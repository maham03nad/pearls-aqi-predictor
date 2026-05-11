"""
STEP 2: Backfill Historical Data
- Fetches past AQI data from OpenWeatherMap Air Pollution History API
- Engineers the same features as feature_pipeline.py
- Stores all historical rows in Hopsworks Feature Store (bulk insert)

Run once before training:
    python backfill.py --days 365
"""

import os
import argparse
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import hopsworks

OPENWEATHER_KEY = os.getenv("OPENWEATHER_KEY", "YOUR_OWM_KEY")
HOPSWORKS_KEY   = os.getenv("HOPSWORKS_API_KEY", "YOUR_HW_KEY")
CITY            = os.getenv("CITY", "karachi")
LAT             = float(os.getenv("LAT", "24.8607"))
LON             = float(os.getenv("LON", "67.0011"))


def fetch_historical_aqi(lat, lon, start_dt: datetime, end_dt: datetime) -> list:
    """
    OpenWeatherMap Air Pollution History API returns hourly data.
    Free tier supports history.
    """
    start_unix = int(start_dt.timestamp())
    end_unix   = int(end_dt.timestamp())
    url = (
        f"https://api.openweathermap.org/data/2.5/air_pollution/history"
        f"?lat={lat}&lon={lon}&start={start_unix}&end={end_unix}&appid={OPENWEATHER_KEY}"
    )
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    return resp.json().get("list", [])


def fetch_historical_weather(lat, lon, dt: datetime) -> dict:
    """
    OpenWeatherMap One Call API - historical (costs 1 call per day).
    Falls back to zeros if unavailable.
    """
    unix_ts = int(dt.timestamp())
    url = (
        f"https://api.openweathermap.org/data/2.5/onecall/timemachine"
        f"?lat={lat}&lon={lon}&dt={unix_ts}&appid={OPENWEATHER_KEY}&units=metric"
    )
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        hourly = resp.json().get("hourly", [{}])
        # pick the closest hour
        target_h = dt.hour
        entry = min(hourly, key=lambda x: abs(
            datetime.utcfromtimestamp(x["dt"]).hour - target_h
        ))
        return entry
    except Exception:
        return {"temp": 25, "humidity": 50, "pressure": 1013,
                "wind_speed": 0, "wind_deg": 0}


def build_features_from_history(aqi_entry: dict, dt: datetime) -> dict:
    """Build feature row from historical AQI API entry."""
    comp   = aqi_entry.get("components", {})
    aqi_v  = aqi_entry.get("main", {}).get("aqi", 0)

    # OWM returns AQI 1-5 scale; convert to 0-500 rough estimate
    aqi_map = {1: 25, 2: 75, 3: 125, 4: 175, 5: 300}
    aqi_val = aqi_map.get(aqi_v, 0)

    hour   = dt.hour
    day    = dt.weekday()
    month  = dt.month

    return {
        "timestamp":   dt.strftime("%Y-%m-%d %H:%M:%S"),
        "city":        CITY,
        "aqi":         float(aqi_val),
        "pm25":        float(comp.get("pm2_5", 0)),
        "pm10":        float(comp.get("pm10",  0)),
        "o3":          float(comp.get("o3",    0)),
        "no2":         float(comp.get("no2",   0)),
        "so2":         float(comp.get("so2",   0)),
        "co":          float(comp.get("co",    0)),
        "temp":        0.0,   # filled below if weather available
        "humidity":    0.0,
        "pressure":    1013.0,
        "wind_speed":  0.0,
        "wind_deg":    0.0,
        "hour":        int(hour),
        "day_of_week": int(day),
        "month":       int(month),
        "is_weekend":  int(day >= 5),
        "hour_sin":    float(np.sin(2 * np.pi * hour  / 24)),
        "hour_cos":    float(np.cos(2 * np.pi * hour  / 24)),
        "month_sin":   float(np.sin(2 * np.pi * month / 12)),
        "month_cos":   float(np.cos(2 * np.pi * month / 12)),
    }


def compute_aqi_change_rate(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived feature: how fast AQI is changing hour-over-hour."""
    df = df.sort_values("timestamp").copy()
    df["aqi_change_rate"] = df["aqi"].diff().fillna(0)
    df["aqi_rolling_6h"]  = df["aqi"].rolling(6,  min_periods=1).mean()
    df["aqi_rolling_24h"] = df["aqi"].rolling(24, min_periods=1).mean()
    # Target: AQI 3 hours in the future (what we want to predict)
    df["target_aqi_3h"]   = df["aqi"].shift(-3)
    df["target_aqi_24h"]  = df["aqi"].shift(-24)
    df["target_aqi_72h"]  = df["aqi"].shift(-72)
    return df


def run(days: int = 365):
    print(f"=== Backfill: last {days} days ===")
    end_dt   = datetime.utcnow()
    start_dt = end_dt - timedelta(days=days)

    print("Fetching historical AQI data...")
    raw_list = fetch_historical_aqi(LAT, LON, start_dt, end_dt)
    print(f"  Got {len(raw_list)} hourly records")

    rows = []
    for entry in raw_list:
        dt = datetime.utcfromtimestamp(entry["dt"])
        row = build_features_from_history(entry, dt)
        rows.append(row)

    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = compute_aqi_change_rate(df)
    df = df.dropna(subset=["target_aqi_3h"])   # drop last rows with no target
    print(f"  Feature df shape: {df.shape}")

    print("Storing in Hopsworks...")
    project = hopsworks.login(api_key_value=HOPSWORKS_KEY)
    fs = project.get_feature_store()

    fg = fs.get_or_create_feature_group(
        name="aqi_features",
        version=1,
        primary_key=["city", "timestamp"],
        description="Hourly AQI + weather features (with targets)",
        event_time="timestamp",
    )
    fg.insert(df, write_options={"wait_for_job": True})
    print(f"[✓] Inserted {len(df)} rows into feature store")
    print("=== Backfill complete ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=365,
                        help="How many past days to backfill (default 365)")
    args = parser.parse_args()
    run(days=args.days)