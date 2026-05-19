"""
Streamlit Dashboard
Runs directly on Streamlit Cloud without a separate FastAPI backend.

Run locally:
    streamlit run streamlit.app/app.py
"""

import os
import joblib
import requests
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import plotly.graph_objects as go
import shap
from datetime import datetime, timedelta

import hopsworks

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

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

FEATURE_COLS = [
    "pm25", "pm10", "o3", "no2", "so2", "co",
    "temp", "humidity", "pressure", "wind_speed", "wind_deg",
    "hour", "day_of_week", "month", "is_weekend",
    "hour_sin", "hour_cos", "month_sin", "month_cos",
    "aqi_change_rate", "aqi_rolling_6h", "aqi_rolling_24h",
]

# PAGE CONFIG
st.set_page_config(
    page_title="AQI Predictor",
    page_icon="🌬️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

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

    .very-unhealthy-banner {
        background: linear-gradient(90deg, #8f3f97, #5e2a84);
        color: white;
        padding: 14px 22px;
        border-radius: 10px;
        font-weight: 700;
        text-align: center;
        margin: 10px 0;
        border: 1px solid rgba(255,255,255,0.25);
    }

    .hazard-banner {
        background: linear-gradient(90deg, #7e0023, #b00020);
        color: white;
        padding: 16px 24px;
        border-radius: 12px;
        font-weight: 800;
        text-align: center;
        margin: 12px 0;
        border: 2px solid #ffccd5;
        box-shadow: 0 0 18px rgba(255, 0, 0, 0.35);
    }

    .small-note {
        opacity: 0.75;
        font-size: 13px;
    }
</style>
""",
    unsafe_allow_html=True,
)

# FILE HELPERS

def find_existing_file(possible_paths):
    """Return the first existing file path from a list of possible paths."""
    for path in possible_paths:
        if os.path.exists(path):
            return path
    return None

#LOADING MODEL

@st.cache_resource(show_spinner=False)
def load_model():
    """
    Load trained model.
    First tries local model files, then Hopsworks Model Registry.
    """

    local_model_paths = [
        "models",
        "streamlit.app/models",
        "../models",
    ]

    for folder in local_model_paths:
        for model_file in ["GradientBoost.pkl", "RandomForest.pkl", "Ridge.pkl"]:
            local_path = os.path.join(folder, model_file)
            if os.path.exists(local_path):
                try:
                    model = joblib.load(local_path)
                    return model
                except Exception:
                    pass

    if not HOPSWORKS_KEY:
        return None

    try:
        project = hopsworks.login(
            host=HOPSWORKS_HOST,
            port=HOPSWORKS_PORT,
            project=HOPSWORKS_PROJECT,
            api_key_value=HOPSWORKS_KEY,
        )

        model_registry = project.get_model_registry()
        hw_model = model_registry.get_model("aqi_predictor", version=1)
        model_dir = hw_model.download()

        for model_file in ["GradientBoost.pkl", "RandomForest.pkl", "Ridge.pkl"]:
            path = os.path.join(model_dir, model_file)
            if os.path.exists(path):
                model = joblib.load(path)
                return model

        return None

    except Exception:
        return None

# AQI HELPERS

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
        "Hazardous": "Emergency conditions. Stay indoors and avoid all outdoor activity.",
    }
    return advice.get(category, "Check local guidelines.")


def get_alert_level(aqi: float) -> str:
    if aqi > 300:
        return "hazardous"
    if aqi > 200:
        return "very_unhealthy"
    if aqi > 150:
        return "unhealthy"
    return "normal"

#FETCHING DATA

@st.cache_data(ttl=300, show_spinner=False)
def fetch_current_data() -> dict:
    """
    Fetch current AQI and weather data directly.
    API keys are not exposed in UI error messages.
    """

    if not OPENWEATHER_KEY:
        raise ValueError("Missing OPENWEATHER_KEY in secrets.")

    try:
        if not AQICN_TOKEN:
            raise ValueError("Missing AQICN_TOKEN in secrets.")

        aqi_url = f"https://api.waqi.info/feed/{CITY}/?token={AQICN_TOKEN}"
        aqi_resp = requests.get(aqi_url, timeout=10)

        if aqi_resp.status_code != 200:
            raise ValueError("AQICN request failed.")

        aqi_json = aqi_resp.json()

        if aqi_json.get("status") != "ok":
            raise ValueError("AQICN returned invalid response.")

        iaqi = aqi_json["data"].get("iaqi", {})
        current_aqi = float(aqi_json["data"].get("aqi", 0))

        pm25 = float(iaqi.get("pm25", {}).get("v", 0))
        pm10 = float(iaqi.get("pm10", {}).get("v", 0))
        o3 = float(iaqi.get("o3", {}).get("v", 0))
        no2 = float(iaqi.get("no2", {}).get("v", 0))
        so2 = float(iaqi.get("so2", {}).get("v", 0))
        co = float(iaqi.get("co", {}).get("v", 0))

    except Exception:
        ap_url = (
            f"https://api.openweathermap.org/data/2.5/air_pollution"
            f"?lat={LAT}&lon={LON}&appid={OPENWEATHER_KEY}"
        )

        ap_resp = requests.get(ap_url, timeout=10)

        if ap_resp.status_code != 200:
            raise ValueError("OpenWeather Air Pollution API request failed. Check OPENWEATHER_KEY.")

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

    weather_url = (
        f"https://api.openweathermap.org/data/2.5/weather"
        f"?lat={LAT}&lon={LON}&appid={OPENWEATHER_KEY}&units=metric"
    )

    weather_resp = requests.get(weather_url, timeout=10)

    if weather_resp.status_code != 200:
        raise ValueError("OpenWeather Weather API request failed. Check OPENWEATHER_KEY.")

    weather_json = weather_resp.json()

    temp = float(weather_json["main"]["temp"])
    humidity = float(weather_json["main"]["humidity"])
    pressure = float(weather_json["main"]["pressure"])
    wind_speed = float(weather_json["wind"]["speed"])
    wind_deg = float(weather_json["wind"].get("deg", 0))

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

# PREDICTION

def build_feature_vector(data: dict, future_time: datetime) -> list:
    future_hour = future_time.hour
    future_day = future_time.weekday()
    future_month = future_time.month

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
        future_hour,
        future_day,
        future_month,
        int(future_day >= 5),
        float(np.sin(2 * np.pi * future_hour / 24)),
        float(np.cos(2 * np.pi * future_hour / 24)),
        float(np.sin(2 * np.pi * future_month / 12)),
        float(np.cos(2 * np.pi * future_month / 12)),
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
            try:
                pred_aqi = float(model.predict([features])[0])
            except Exception:
                noise = np.random.normal(0, 3)
                pred_aqi = current_data["current_aqi"] + noise
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
        (300, "Very Unhealthy", "#8f3f97"),
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
        yaxis=dict(gridcolor="rgba(255,255,255,0.05)", range=[0, 500]),
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


def make_shap_importance_chart(current_data: dict):
    model = load_model()

    if model is None:
        return None

    try:
        now = datetime.utcnow()

        rows = []
        for h in range(1, 73):
            future_time = now + timedelta(hours=h)
            rows.append(build_feature_vector(current_data, future_time))

        X = pd.DataFrame(rows, columns=FEATURE_COLS)

        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)
        except Exception:
            explainer = shap.Explainer(model.predict, X)
            shap_values = explainer(X).values

        if isinstance(shap_values, list):
            shap_values = shap_values[0]

        importance = np.abs(shap_values).mean(axis=0)

        imp_df = pd.DataFrame({
            "feature": FEATURE_COLS,
            "importance": importance,
        }).sort_values("importance", ascending=True).tail(12)

        fig = go.Figure(go.Bar(
            x=imp_df["importance"],
            y=imp_df["feature"],
            orientation="h",
            text=[f"{v:.3f}" for v in imp_df["importance"]],
            textposition="outside",
        ))

        fig.update_layout(
            title="SHAP Feature Importance",
            xaxis_title="Mean |SHAP value|",
            yaxis_title="Feature",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(13,20,40,0.8)",
            font=dict(color="white"),
            height=420,
            margin=dict(l=100, r=40, t=60, b=40),
            xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
            yaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
        )

        return fig

    except Exception:
        return None

def main():
    st.title("🌬️ AQI Predictor Dashboard")
    st.caption(
        f"Live air quality monitoring and 3-day forecast • "
        f"Last updated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}"
    )

    try:
        current = fetch_current_data()
    except Exception as e:
        st.error(f"Could not fetch current AQI/weather data: {e}")
        st.stop()

    predictions = get_forecast_cached(current)

    label, color = get_aqi_category(current["current_aqi"])
    advice = get_health_advice(label)

    model = load_model()
    if model is None:
        st.warning(
            "Model could not be loaded. Forecast is using fallback estimation. "
            "Check Hopsworks model registry or local model files."
        )

    # Current AQI alert system
    alert_level = get_alert_level(current["current_aqi"])

    if alert_level == "hazardous":
        st.markdown(
            f"""
            <div class="hazard-banner">
                ☠️ HAZARDOUS AQI ALERT — AQI {current["current_aqi"]:.0f}<br>
                Stay indoors, close windows, use masks or air purifiers, and avoid all outdoor activity.
            </div>
            """,
            unsafe_allow_html=True,
        )

    elif alert_level == "very_unhealthy":
        st.markdown(
            f"""
            <div class="very-unhealthy-banner">
                🚨 VERY UNHEALTHY AIR QUALITY — AQI {current["current_aqi"]:.0f}<br>
                Everyone should avoid prolonged outdoor exposure.
            </div>
            """,
            unsafe_allow_html=True,
        )

    elif alert_level == "unhealthy":
        st.markdown(
            f"""
            <div class="alert-banner">
                🚨 AIR QUALITY ALERT — {label.upper()} — Avoid outdoor activities!
            </div>
            """,
            unsafe_allow_html=True,
        )

    # AQI alert system based on forecast values
    forecast_max = max(p["aqi"] for p in predictions)
    forecast_peak = max(predictions, key=lambda p: p["aqi"])

    if forecast_max > 300:
        st.error(
            f"☠️ Forecast warning: Hazardous AQI is expected around "
            f"{forecast_peak['timestamp']} with AQI {forecast_max:.0f}. "
            "Stay indoors and avoid all outdoor activity."
        )
    elif forecast_max > 200:
        st.warning(
            f"🚨 Forecast warning: AQI may reach Very Unhealthy level around "
            f"{forecast_peak['timestamp']} with AQI {forecast_max:.0f}."
        )
    elif forecast_max > 150:
        st.warning(
            f"⚠️ AQI may reach {forecast_max:.0f} in the next 72 hours "
            f"around {forecast_peak['timestamp']} — Unhealthy level!"
        )
    elif forecast_max > 100:
        st.info(
            f"ℹ️ AQI is predicted to reach {forecast_max:.0f} in the next 72 hours "
            f"around {forecast_peak['timestamp']} — Sensitive groups alert."
        )
    else:
        st.success("✅ No major risk expected in the next 72 hours.")

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

    shap_tab, lime_tab = st.tabs(["SHAP Explanation", "LIME Explanation"])

    with shap_tab:
        st.markdown("### 🔍 SHAP Feature Importance")
        st.caption("Explains which features influence the 72-hour AQI predictions.")

        shap_fig = make_shap_importance_chart(current)

        if shap_fig is not None:
            st.plotly_chart(shap_fig, use_container_width=True)
        else:
            st.info("SHAP explanation unavailable because the trained model could not be loaded.")

    with lime_tab:
        st.markdown("### LIME Local Explanation")
        st.caption("Explains local feature importance for a sample AQI prediction.")

        lime_png_path = find_existing_file([
            "models/lime_explanation.png",
            "streamlit.app/models/lime_explanation.png",
            "../models/lime_explanation.png",
        ])

        lime_html_path = find_existing_file([
            "models/lime_explanation.html",
            "streamlit.app/models/lime_explanation.html",
            "../models/lime_explanation.html",
        ])

        if lime_png_path:
            st.image(
                lime_png_path,
                caption="LIME local explanation for a sample AQI prediction"
            )
        else:
            st.warning("LIME image not found. Please run training_pipeline.py first.")

        if lime_html_path:
            with st.expander("View interactive LIME explanation"):
                with open(lime_html_path, "r", encoding="utf-8") as f:
                    lime_html = f.read()
                components.html(lime_html, height=600, scrolling=True)

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
    st.caption("Powered by AQICN + OpenWeatherMap + Hopsworks + SHAP + LIME + Streamlit")


if __name__ == "__main__":
    main()