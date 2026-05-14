"""
Backfill Historical AQI + Weather Data
1. Fetches historical air pollution data from OpenWeather Air Pollution History API.
2. Fetches historical hourly weather from Open-Meteo Archive API.
3. Builds engineered features and AQI targets.
4. Stores rows in Hopsworks Feature Store.

Run:
    python backfill.py --days 365
"""

import os
import argparse
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone

import hopsworks

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

#  CONFIG 
OPENWEATHER_KEY = os.getenv("OPENWEATHER_KEY")
HOPSWORKS_KEY = os.getenv("HOPSWORKS_API_KEY")

CITY = os.getenv("CITY", "karachi")

def get_float_env(name: str, default: float) -> float:
    value = os.getenv(name)
    return float(value) if value not in (None, "") else default

LAT = get_float_env("LAT", 24.8607)
LON = get_float_env("LON", 67.0011)

HOPSWORKS_HOST = os.getenv("HOPSWORKS_HOST", "eu-west.cloud.hopsworks.ai")
HOPSWORKS_PROJECT = os.getenv("HOPSWORKS_PROJECT", "aqi_project_10pearls")
HOPSWORKS_PORT = int(os.getenv("HOPSWORKS_PORT", "443"))

AQI_MAP = {
    1: 25.0,    # Good
    2: 75.0,    # Moderate
    3: 125.0,   # Unhealthy for Sensitive Groups
    4: 175.0,   # Unhealthy
    5: 300.0,   # Very Unhealthy / Hazardous-like
}

#  FETCH AIR POLLUTION HISTORY 

def fetch_air_pollution_history(start_dt: datetime, end_dt: datetime) -> list:
    """
    Fetch historical hourly air pollution data from OpenWeather.
    """
    if not OPENWEATHER_KEY:
        raise ValueError("OPENWEATHER_KEY is missing.")

    start_unix = int(start_dt.timestamp())
    end_unix = int(end_dt.timestamp())

    url = (
        "https://api.openweathermap.org/data/2.5/air_pollution/history"
        f"?lat={LAT}&lon={LON}&start={start_unix}&end={end_unix}&appid={OPENWEATHER_KEY}"
    )

    print("Fetching historical AQI/pollution data...")
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()

    data = resp.json()
    records = data.get("list", [])

    if not records:
        raise ValueError(f"No air pollution records returned: {data}")

    print(f"Got {len(records)} hourly AQI/pollution records")
    return records

#FETCH HISTORICAL WEATHER 

def fetch_historical_weather_df(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch historical hourly weather from Open-Meteo Archive API.
    No API key needed.
    """
    print(f"Fetching historical weather from {start_date} to {end_date}...")

    url = "https://archive-api.open-meteo.com/v1/archive"

    params = {
        "latitude": LAT,
        "longitude": LON,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join([
            "temperature_2m",
            "relative_humidity_2m",
            "surface_pressure",
            "wind_speed_10m",
            "wind_direction_10m",
        ]),
        "timezone": "UTC",
    }

    resp = requests.get(url, params=params, timeout=60)
    resp.raise_for_status()

    data = resp.json()
    hourly = data.get("hourly", {})

    times = hourly.get("time", [])

    if not times:
        raise ValueError(f"No weather records returned: {data}")

    weather_df = pd.DataFrame({
        "timestamp": pd.to_datetime(times),
        "temp": hourly.get("temperature_2m", []),
        "humidity": hourly.get("relative_humidity_2m", []),
        "pressure": hourly.get("surface_pressure", []),
        "wind_speed": hourly.get("wind_speed_10m", []),
        "wind_deg": hourly.get("wind_direction_10m", []),
    })

    print(f"Got {len(weather_df)} hourly weather records")
    return weather_df

#FEATURE BUILDING

def build_pollution_rows(records: list) -> pd.DataFrame:
    rows = []

    for entry in records:
        dt = datetime.fromtimestamp(entry["dt"], tz=timezone.utc).replace(tzinfo=None)

        main = entry.get("main", {})
        components = entry.get("components", {})

        raw_aqi = int(main.get("aqi", 2))
        mapped_aqi = AQI_MAP.get(raw_aqi, 75.0)

        row = {
            "timestamp": dt,
            "city": CITY.capitalize(),

            "aqi": float(mapped_aqi),
            "pm25": float(components.get("pm2_5", 0.0)),
            "pm10": float(components.get("pm10", 0.0)),
            "o3": float(components.get("o3", 0.0)),
            "no2": float(components.get("no2", 0.0)),
            "so2": float(components.get("so2", 0.0)),
            "co": float(components.get("co", 0.0)),
        }

        rows.append(row)

    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    return df

def merge_weather(df: pd.DataFrame) -> pd.DataFrame:
    start_date = df["timestamp"].min().strftime("%Y-%m-%d")
    end_date = df["timestamp"].max().strftime("%Y-%m-%d")
    weather_df = fetch_historical_weather_df(start_date, end_date)

    df = df.sort_values("timestamp").reset_index(drop=True)
    weather_df = weather_df.sort_values("timestamp").reset_index(drop=True)
    df = pd.merge_asof(
        df,
        weather_df,
        on="timestamp",
        direction="nearest",
        tolerance=pd.Timedelta("1h"),
    )
    weather_cols = ["temp", "humidity", "pressure", "wind_speed", "wind_deg"]
    for col in weather_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Safe fallback values if any weather hour is missing
    df["temp"] = df["temp"].fillna(df["temp"].median()).fillna(30.0)
    df["humidity"] = df["humidity"].fillna(df["humidity"].median()).fillna(60.0)
    df["pressure"] = df["pressure"].fillna(df["pressure"].median()).fillna(1013.0)
    df["wind_speed"] = df["wind_speed"].fillna(df["wind_speed"].median()).fillna(2.0)
    df["wind_deg"] = df["wind_deg"].fillna(df["wind_deg"].median()).fillna(0.0)

    return df

def compute_features_and_targets(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["city", "timestamp"]).reset_index(drop=True)
    df["hour"] = df["timestamp"].dt.hour.astype(int)
    df["day_of_week"] = df["timestamp"].dt.weekday.astype(int)
    df["month"] = df["timestamp"].dt.month.astype(int)
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["aqi_change_rate"] = (
        df.groupby("city")["aqi"]
        .diff()
        .fillna(0.0)
    )
    df["aqi_rolling_6h"] = (
        df.groupby("city")["aqi"]
        .transform(lambda s: s.rolling(window=6, min_periods=1).mean())
    )
    df["aqi_rolling_24h"] = (
        df.groupby("city")["aqi"]
        .transform(lambda s: s.rolling(window=24, min_periods=1).mean())
    )
    # Targets for forecasting
    df["target_aqi_3h"] = df.groupby("city")["aqi"].shift(-3)
    df["target_aqi_24h"] = df.groupby("city")["aqi"].shift(-24)
    df["target_aqi_72h"] = df.groupby("city")["aqi"].shift(-72)

    # Drop final rows where future targets are not available
    df = df.dropna(subset=["target_aqi_3h", "target_aqi_24h", "target_aqi_72h"])
    numeric_cols = [
        "aqi", "pm25", "pm10", "o3", "no2", "so2", "co",
        "temp", "humidity", "pressure", "wind_speed", "wind_deg",
        "hour_sin", "hour_cos", "month_sin", "month_cos",
        "aqi_change_rate", "aqi_rolling_6h", "aqi_rolling_24h",
        "target_aqi_3h", "target_aqi_24h", "target_aqi_72h",
    ]
    int_cols = ["hour", "day_of_week", "month", "is_weekend"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)
    for col in int_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
    df["city"] = df["city"].astype(str)
    return df

# STORE IN HOPSWORKS 

def store_in_hopsworks(df: pd.DataFrame):
    if not HOPSWORKS_KEY:
        raise ValueError("HOPSWORKS_API_KEY is missing.")
    print(f"Storing in Hopsworks... DataFrame shape: {df.shape}")
    project = hopsworks.login(
        host=HOPSWORKS_HOST,
        port=HOPSWORKS_PORT,
        project=HOPSWORKS_PROJECT,
        api_key_value=HOPSWORKS_KEY,
    )
    fs = project.get_feature_store()
    fg = fs.get_or_create_feature_group(
        name="aqi_features",
        version=1,
        primary_key=["city", "timestamp"],
        description="Hourly AQI + weather features with forecasting targets",
        event_time="timestamp",
    )

    fg.insert(df, write_options={"wait_for_job": True})
    print(f"Inserted/updated {len(df)} rows into aqi_features")

# MAIN RUNNER 

def run(days: int = 365):
    end_dt = datetime.utcnow().replace(minute=0, second=0, microsecond=0)
    start_dt = end_dt - timedelta(days=days)
    print("=== Backfill Historical AQI + Weather Data ===")
    print(f"City: {CITY}")
    print(f"Latitude: {LAT}, Longitude: {LON}")
    print(f"Start: {start_dt}")
    print(f"End: {end_dt}")

    records = fetch_air_pollution_history(start_dt, end_dt)
    pollution_df = build_pollution_rows(records)
    print(f"Pollution df shape: {pollution_df.shape}")

    full_df = merge_weather(pollution_df)
    print(f"After weather merge shape: {full_df.shape}")
    final_df = compute_features_and_targets(full_df)
    print(f"Final feature df shape: {final_df.shape}")
    print("Weather sample:")
    print(final_df[["timestamp", "temp", "humidity", "pressure", "wind_speed", "wind_deg"]].head())

    store_in_hopsworks(final_df)
    print("=== Backfill complete ===")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=365)
    args = parser.parse_args()
    run(days=args.days)