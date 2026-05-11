"""
STEP 1: Feature Pipeline
- Fetches raw weather + AQI data from AQICN API
- Engineers features (time-based + derived)
- Stores features in Hopsworks Feature Store
"""

import os
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import hopsworks

# ─── CONFIG 
AQICN_TOKEN   = os.getenv("AQICN_TOKEN")        # https://aqicn.org/api/
OPENWEATHER_KEY = os.getenv("OPENWEATHER_KEY")       # https://openweathermap.org/api
HOPSWORKS_KEY  = os.getenv("HOPSWORKS_API_KEY")       # https://app.hopsworks.ai
CITY           = os.getenv("CITY", "karachi")                  
LAT            = float(os.getenv("LAT", "24.86146"))
LON            = float(os.getenv("LON", "67.00994"))

def fetch_aqi_data(city: str) -> dict:
    """Fetch current AQI data from AQICN."""
    url = f"https://api.waqi.info/feed/{city}/?token={AQICN_TOKEN}"
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    if data["status"] != "ok":
        raise ValueError(f"AQICN error: {data}")
    return data["data"]


def fetch_weather_data(lat: float, lon: float) -> dict:
    """Fetch current weather from OpenWeatherMap."""
    url = (
        f"https://api.openweathermap.org/data/2.5/weather"
        f"?lat={lat}&lon={lon}&appid={OPENWEATHER_KEY}&units=metric"
    )
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    return resp.json()


def engineer_features(aqi_data: dict, weather_data: dict) -> dict:
    """Compute model-ready features from raw API data."""
    now = datetime.utcnow()

    # Raw pollutants (may be missing → fill 0)
    iaqi = aqi_data.get("iaqi", {})
    pm25 = iaqi.get("pm25", {}).get("v", 0)
    pm10 = iaqi.get("pm10", {}).get("v", 0)
    o3   = iaqi.get("o3",   {}).get("v", 0)
    no2  = iaqi.get("no2",  {}).get("v", 0)
    so2  = iaqi.get("so2",  {}).get("v", 0)
    co   = iaqi.get("co",   {}).get("v", 0)
    aqi  = aqi_data.get("aqi", 0)

    # Weather
    temp     = weather_data["main"]["temp"]
    humidity = weather_data["main"]["humidity"]
    pressure = weather_data["main"]["pressure"]
    wind_speed  = weather_data["wind"]["speed"]
    wind_deg    = weather_data["wind"].get("deg", 0)

    # Time-based features
    hour  = now.hour
    day   = now.weekday()          # 0=Monday
    month = now.month
    is_weekend = int(day >= 5)

    # Cyclical encoding (so that model understand hour 23 is close to 0hr)
    hour_sin  = np.sin(2 * np.pi * hour  / 24)
    hour_cos  = np.cos(2 * np.pi * hour  / 24)
    month_sin = np.sin(2 * np.pi * month / 12)
    month_cos = np.cos(2 * np.pi * month / 12)

    return {
        "timestamp":   now.strftime("%Y-%m-%d %H:%M:%S"),
        "city":        CITY,
        "aqi":         float(aqi),
        "pm25":        float(pm25),
        "pm10":        float(pm10),
        "o3":          float(o3),
        "no2":         float(no2),
        "so2":         float(so2),
        "co":          float(co),
        "temp":        float(temp),
        "humidity":    float(humidity),
        "pressure":    float(pressure),
        "wind_speed":  float(wind_speed),
        "wind_deg":    float(wind_deg),
        "hour":        int(hour),
        "day_of_week": int(day),
        "month":       int(month),
        "is_weekend":  int(is_weekend),
        "hour_sin":    float(hour_sin),
        "hour_cos":    float(hour_cos),
        "month_sin":   float(month_sin),
        "month_cos":   float(month_cos),
    }


def store_in_hopsworks(features: dict):
    """Push one row of features to the Hopsworks Feature Store."""
    project = hopsworks.login(api_key_value=HOPSWORKS_KEY)
    fs = project.get_feature_store()

    df = pd.DataFrame([features])
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    fg = fs.get_or_create_feature_group(
        name="aqi_features",
        version=1,
        primary_key=["city", "timestamp"],
        description="Hourly AQI + weather features",
        event_time="timestamp",
    )
    fg.insert(df, write_options={"wait_for_job": False})
    print(f"[✓] Inserted 1 row at {features['timestamp']}")


def run():
    print("=== Feature Pipeline ===")
    print(f"Fetching data for {CITY}...")
    aqi_data     = fetch_aqi_data(CITY)
    weather_data = fetch_weather_data(LAT, LON)
    features     = engineer_features(aqi_data, weather_data)
    print(f"  AQI={features['aqi']}  PM2.5={features['pm25']}  Temp={features['temp']}°C")
    store_in_hopsworks(features)
    print("=== Done ===")


if __name__ == "__main__":
    run()