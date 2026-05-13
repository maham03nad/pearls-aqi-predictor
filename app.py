"""
Streamlit Dashboard
Runs directly on Streamlit Cloud without separate FastAPI backend.

Run locally:
    streamlit run app.py
"""

import os
import joblib
import requests
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime, timedelta

import hopsworks

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

#  CONFIG 

HOPSWORKS_KEY = os.getenv("HOPSWORKS_API_KEY")
OPENWEATHER_KEY = os.getenv("OPENWEATHER_KEY")
AQICN_TOKEN = os.getenv("AQICN_TOKEN")

CITY = os.getenv("CITY") or "karachi"


def get_float_env(name, default):
    value = os.getenv(name)
    return float(value) if value not in (None, "") else default

LAT = get_float_env("LAT", 24.8607)
LON = get_float_env("LON", 67.0011)

HOPSWORKS_HOST = os.getenv("HOPSWORKS_HOST") or "eu-west.cloud.hopsworks.ai"
HOPSWORKS_PROJECT = os.getenv("HOPSWORKS_PROJECT") or "aqi_project_10pearls"
HOPSWORKS_PORT = int(os.getenv("HOPSWORKS_PORT") or 443)

AQI_THRESHOLDS = [
    (50, "Good", "#00e400"),
    (100, "Moderate", "#ffff00"),
    (150, "Unhealthy for Sensitive Groups", "#ff7e00"),
    (200, "Unhealthy", "#ff0000"),
    (300, "Very Unhealthy", "#8f3f97"),
    (500, "Hazardous", "#7e0023"),
]

#  PAGE CONFIG 

st.set_page_config(
    page_title="AQI Predictor",
    page_icon="🌬️",
    layout="wide",
    initial_sidebar_state="collapsed",
)


#  CUSTOM CSS 

st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Inter:wght@300;400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        background-color: #0b0f19;
    }

    h1, h2, h3 {
        font-family: 'Space Mono', monospace;
    }

    .metric-card {
        background: #0f3460;
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        color: white;
    }

    .alert-banner {
        background: linear-gradient(90deg, #ff416c, #ff4b2b);
        color: white;
        padding: 12px 20px;
        border-radius: 10px;
        font-weight: 600;
        text-align: center;
        margin: 10px 0;
    }

    .small-note {
        opacity: 0.75;
        font-size: 13px;
    }
</style>
""",
    unsafe_allow_html=True,
)

#  MODEL LOADING

@st.cache_resource(show_spinner=False)
def load_model():
    """Load trained model from Hopsworks Model Registry."""

    if not HOPSWORKS_KEY:
        st.warning("Missing HOPSWORKS_API_KEY. Forecast will use fallback logic.")
        return None

    try:
        project = hopsworks.login(
            host=HOPSWORKS_HOST,
            port=HOPSWORKS_PORT,
            project=HOPSWORKS_PROJECT,
            api_key_value=HOPSWORKS_KEY,
        )

        mr = project.get_model_registry()
        hw_model = mr.get_model("aqi_predictor", version=1)
        model_dir = hw_model.download()

        for model_file in ["GradientBoost.pkl", "RandomForest.pkl", "Ridge.pkl"]:
            path = os.path.join(model_dir, model_file)
            if os.path.exists(path):
                return joblib.load(path)

        st.warning("No trained model file found in Hopsworks model registry.")
        return None

    except Exception as e:
        st.warning(f"Could not load model from Hopsworks: {e}")
        return None

# DATA FETCHING 

def get_aqi_category(aqi: float):
    for limit, label, color in AQI_THRESHOLDS:
        if aqi <= limit:
            return label, color
    return "Hazardous", "#7e0023"

def get_health_advice(category: str) -> str:
    advice = {
        "Good": "Air quality is satisfactory. Enjoy outdoor activities.",
        "Moderate": "Acceptable air quality. Unusually sensitive people should limit outdoor exertion.",
        "Unhealthy for Sensitive Groups": "Sensitive groups should reduce outdoor activity.",
        "Unhealthy": "Everyone should reduce prolonged outdoor exertion.",
        "Very Unhealthy": "Health alert — everyone should avoid outdoor activity.",
        "Hazardous": "Emergency conditions. Everyone should stay indoors.",
    }
    return advice.get(category, "Check local guidelines.")

@st.cache_data(ttl=300, show_spinner=False)
def fetch_current_data() -> dict:
    """Fetch current AQI and weather data directly."""

    if not OPENWEATHER_KEY:
        raise ValueError("Missing OPENWEATHER_KEY")

    # AQICN first
    try:
        if not AQICN_TOKEN:
            raise ValueError("Missing AQICN_TOKEN")

        aqi_url = f"https://api.waqi.info/feed/{CITY}/?token={AQICN_TOKEN}"
        aqi_resp = requests.get(aqi_url, timeout=10)
        aqi_resp.raise_for_status()
        aqi_json = aqi_resp.json()

        if aqi_json.get("status") != "ok":
            raise ValueError(f"AQICN error: {aqi_json}")

        iaqi = aqi_json["data"].get("iaqi", {})
        current_aqi = float(aqi_json["data"].get("aqi", 0))

        pm25 = float(iaqi.get("pm25", {}).get("v", 0))
        pm10 = float(iaqi.get("pm10", {}).get("v", 0))
        o3 = float(iaqi.get("o3", {}).get("v", 0))
        no2 = float(iaqi.get("no2", {}).get("v", 0))
        so2 = float(iaqi.get("so2", {}).get("v", 0))
        co = float(iaqi.get("co", {}).get("v", 0))

    except Exception:
        # OpenWeather fallback
        ap_url = (
            f"https://api.openweathermap.org/data/2.5/air_pollution"
            f"?lat={LAT}&lon={LON}&appid={OPENWEATHER_KEY}"
        )
        ap_resp = requests.get(ap_url, timeout=10)
        ap_resp.raise_for_status()
        ap_json = ap_resp.json()

        comp = ap_json["list"][0]["components"]
        aqi_raw = ap_json["list"][0]["main"]["aqi"]

        aqi_map = {1: 25, 2: 75, 3: 125, 4: 175, 5: 300}
        current_aqi = float(aqi_map.get(aqi_raw, 50))

        pm25 = float(comp.get("pm2_5", 0))
        pm10 = float(comp.get("pm10", 0))
        o3 = float(comp.get("o3", 0))
        no2 = float(comp.get("no2", 0))
        so2 = float(comp.get("so2", 0))
        co = float(comp.get("co", 0))

    # Weather
    w_url = (
        f"https://api.openweathermap.org/data/2.5/weather"
        f"?lat={LAT}&lon={LON}&appid={OPENWEATHER_KEY}&units=metric"
    )

    w_resp = requests.get(w_url, timeout=10)
    w_resp.raise_for_status()
    w_json = w_resp.json()

    temp = float(w_json["main"]["temp"])
    humidity = float(w_json["main"]["humidity"])
    pressure = float(w_json["main"]["pressure"])
    wind_speed = float(w_json["wind"]["speed"])
    wind_deg = float(w_json["wind"].get("deg", 0))

    now = datetime.utcnow()
    hour = now.hour
    day = now.weekday()
    month = now.month

    return {
        "current_aqi": current_aqi,
        "pm25": pm25,
        "pm10": pm10,
        "o3": o3,
        "no2": no2,
        "so2": so2,
        "co": co,
        "temp": temp,
        "humidity": humidity,
        "pressure": pressure,
        "wind_speed": wind_speed,
        "wind_deg": wind_deg,
        "hour": hour,
        "day_of_week": day,
        "month": month,
        "is_weekend": int(day >= 5),
        "hour_sin": float(np.sin(2 * np.pi * hour / 24)),
        "hour_cos": float(np.cos(2 * np.pi * hour / 24)),
        "month_sin": float(np.sin(2 * np.pi * month / 12)),
        "month_cos": float(np.cos(2 * np.pi * month / 12)),
        "aqi_change_rate": 0.0,
        "aqi_rolling_6h": current_aqi,
        "aqi_rolling_24h": current_aqi,
    }

def build_feature_vector(data: dict, future_time: datetime) -> list:
    fh = future_time.hour
    fday = future_time.weekday()
    fmon = future_time.month

    return [
        data["pm25"],
        data["pm10"],
        data["o3"],
        data["no2"],
        data["so2"],
        data["co"],
        data["temp"],
        data["humidity"],
        data["pressure"],
        data["wind_speed"],
        data["wind_deg"],
        fh,
        fday,
        fmon,
        int(fday >= 5),
        float(np.sin(2 * np.pi * fh / 24)),
        float(np.cos(2 * np.pi * fh / 24)),
        float(np.sin(2 * np.pi * fmon / 12)),
        float(np.cos(2 * np.pi * fmon / 12)),
        data["aqi_change_rate"],
        data["aqi_rolling_6h"],
        data["aqi_rolling_24h"],
    ]

def predict_72_hours(current_data: dict) -> list:
    model = load_model()
    now = datetime.utcnow()
    forecasts = []

    for h in range(1, 73):
        future_time = now + timedelta(hours=h)
        features = build_feature_vector(current_data, future_time)

        if model is not None:
            pred_aqi = float(model.predict([features])[0])
        else:
            noise = np.random.normal(0, 3)
            pred_aqi = current_data["current_aqi"] + noise

        pred_aqi = max(0.0, min(500.0, pred_aqi))
        label, color = get_aqi_category(pred_aqi)

        forecasts.append({
            "timestamp": future_time.strftime("%Y-%m-%d %H:%M"),
            "hour_from_now": h,
            "aqi": round(pred_aqi, 1),
            "category": label,
            "color": color,
        })

    return forecasts


@st.cache_data(ttl=3600, show_spinner=False)
def get_forecast_cached(current_data: dict):
    return predict_72_hours(current_data)

# CHARTS 

def make_gauge(aqi_value: float, color: str):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=aqi_value,
        domain={"x": [0, 1], "y": [0, 1]},
        title={"text": "AQI", "font": {"size": 20, "color": "white"}},
        number={"font": {"size": 48, "color": color}},
        gauge={
            "axis": {
                "range": [0, 500],
                "tickcolor": "white",
                "tickfont": {"color": "white"},
            },
            "bar": {"color": color, "thickness": 0.25},
            "bgcolor": "#1a1a2e",
            "bordercolor": "rgba(255,255,255,0.2)",
            "steps": [
                {"range": [0, 50], "color": "#004d00"},
                {"range": [50, 100], "color": "#4d4d00"},
                {"range": [100, 150], "color": "#7a3d00"},
                {"range": [150, 200], "color": "#660000"},
                {"range": [200, 300], "color": "#3b0050"},
                {"range": [300, 500], "color": "#2d0013"},
            ],
        },
    ))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=280,
        margin=dict(l=20, r=20, t=40, b=20),
        font={"color": "white"},
    )
    return fig


def make_pollutant_chart(pollutants: dict):
    labels = list(pollutants.keys())
    values = list(pollutants.values())
    colors = ["#00d4ff", "#ff6b6b", "#ffd93d", "#6bcb77", "#4d96ff", "#ff922b"]

    fig = go.Figure(go.Bar(
        x=labels,
        y=values,
        marker_color=colors,
        text=[f"{v:.1f}" for v in values],
        textposition="outside",
        textfont=dict(color="white"),
    ))

    fig.update_layout(
        title="Pollutant Levels",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(13,20,40,0.8)",
        font=dict(color="white"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
        height=300,
    )
    return fig

def make_forecast_chart(predictions: list):
    df = pd.DataFrame(predictions)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df["timestamp"],
        y=df["aqi"],
        mode="lines+markers",
        line=dict(color="#00d4ff", width=2.5),
        marker=dict(size=5, color=df["color"]),
        fill="tozeroy",
        fillcolor="rgba(0, 212, 255, 0.08)",
        hovertemplate="<b>%{x}</b><br>AQI: %{y}<extra></extra>",
    ))

    thresholds = [
        (50, "Good", "#00e400"),
        (100, "Moderate", "#ffff00"),
        (150, "USG", "#ff7e00"),
        (200, "Unhealthy", "#ff0000"),
    ]

    for val, label, color in thresholds:
        fig.add_hline(
            y=val,
            line_dash="dot",
            line_color=color,
            annotation_text=label,
            annotation_position="right",
            annotation_font_color=color,
        )

    fig.update_layout(
        title="72-Hour AQI Forecast",
        xaxis_title="Time",
        yaxis_title="AQI",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(13,20,40,0.8)",
        font=dict(color="white"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.05)", range=[0, 350]),
        height=380,
        hovermode="x unified",
    )
    return fig

def build_daily_summary(predictions: list):
    days_dict = {}

    for p in predictions:
        day_key = p["timestamp"][:10]
        days_dict.setdefault(day_key, []).append(p)

    daily = []

    for date, hours in days_dict.items():
        aqis = [h["aqi"] for h in hours]
        avg_aqi = round(float(np.mean(aqis)), 1)
        label, color = get_aqi_category(avg_aqi)

        daily.append({
            "date": date,
            "avg_aqi": avg_aqi,
            "max_aqi": round(max(aqis), 1),
            "min_aqi": round(min(aqis), 1),
            "category": label,
            "color": color,
            "advice": get_health_advice(label),
            "hourly": hours,
        })

    return daily

#  MAIN APP

def main():
    st.title("🌬️ AQI Predictor Dashboard")
    st.caption(f"Live air quality monitoring and 3-day forecast • Updated: {datetime.utcnow().strftime('%H:%M UTC')}")

    try:
        current = fetch_current_data()
    except Exception as e:
        st.error(f"Could not fetch current AQI/weather data: {e}")
        st.stop()

    predictions = get_forecast_cached(current)

    label, color = get_aqi_category(current["current_aqi"])
    advice = get_health_advice(label)

    if current["current_aqi"] > 150:
        st.markdown(
            f'<div class="alert-banner">🚨 AIR QUALITY ALERT — {label.upper()} — Avoid outdoor activities!</div>',
            unsafe_allow_html=True,
        )

    city_name = CITY.capitalize()

    col_gauge, col_metrics = st.columns([1, 2])

    with col_gauge:
        st.markdown(f"### {city_name}")
        st.markdown(
            f'<div style="color:{color}; font-size:22px; font-weight:700;">{label}</div>',
            unsafe_allow_html=True,
        )
        st.plotly_chart(make_gauge(current["current_aqi"], color), use_container_width=True)

    with col_metrics:
        st.markdown("### Current Conditions")
        m1, m2, m3, m4 = st.columns(4)

        m1.metric("🌡️ Temp", f"{current['temp']:.1f}°C")
        m2.metric("💧 Humidity", f"{current['humidity']:.0f}%")
        m3.metric("🌀 Pressure", f"{current['pressure']:.0f} hPa")
        m4.metric("💨 Wind", f"{current['wind_speed']:.1f} m/s")

        st.info(advice)

        pollutants = {
            "pm25": current["pm25"],
            "pm10": current["pm10"],
            "o3": current["o3"],
            "no2": current["no2"],
            "so2": current["so2"],
            "co": current["co"],
        }

        st.markdown("### Pollutant Breakdown")
        st.plotly_chart(make_pollutant_chart(pollutants), use_container_width=True)

    st.divider()

    st.plotly_chart(make_forecast_chart(predictions), use_container_width=True)

    daily = build_daily_summary(predictions)

    st.markdown("### 3-Day Summary")
    day_cols = st.columns(len(daily))

    for i, day in enumerate(daily):
        with day_cols[i]:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div style="font-size:13px; opacity:0.7">{day['date']}</div>
                    <div style="font-size:28px; font-weight:700; color:{day['color']}">{day['avg_aqi']}</div>
                    <div style="font-size:12px; color:{day['color']}">{day['category']}</div>
                    <div style="font-size:11px; margin-top:6px; opacity:0.6">
                        ↑{day['max_aqi']} ↓{day['min_aqi']}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.divider()

    with st.expander("📖 AQI Scale Reference"):
        scale_data = {
            "AQI Range": ["0–50", "51–100", "101–150", "151–200", "201–300", "301–500"],
            "Category": [
                "Good",
                "Moderate",
                "Unhealthy for Sensitive Groups",
                "Unhealthy",
                "Very Unhealthy",
                "Hazardous",
            ],
            "Health Impact": [
                "Air quality is satisfactory.",
                "Acceptable; some pollutants may affect very sensitive people.",
                "Members of sensitive groups may experience health effects.",
                "Everyone may begin to experience health effects.",
                "Health alert: everyone may experience serious effects.",
                "Health warnings of emergency conditions.",
            ],
        }

        st.dataframe(pd.DataFrame(scale_data), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.caption("Powered by AQICN + OpenWeatherMap + Hopsworks + Streamlit")

if __name__ == "__main__":
    main()