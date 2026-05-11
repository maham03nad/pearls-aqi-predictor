"""
STEP 5b: Streamlit Dashboard
Run: streamlit run app.py
"""

import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import os

# ─── CONFIG 
API_URL = os.getenv("API_URL", "http://localhost:8000")

# ─── PAGE SETUP 
st.set_page_config(
    page_title="AQI Predictor",
    page_icon="🌬️",
    layout="wide",
)

# ─── STYLING 
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;600;700&family=Share+Tech+Mono&display=swap');

html, body, [class*="css"] {
    font-family: 'Rajdhani', sans-serif;
    background-color: #0a0e1a;
    color: #e0e0e0;
}
h1, h2, h3 {
    font-family: 'Share Tech Mono', monospace;
    color: #00d4ff;
}
.alert-box {
    background: linear-gradient(90deg, #7b0000, #cc0000);
    border-left: 5px solid #ff0000;
    padding: 15px 20px;
    border-radius: 8px;
    color: white;
    font-size: 18px;
    font-weight: 700;
    margin-bottom: 20px;
    text-align: center;
    animation: pulse 1.5s infinite;
}
@keyframes pulse {
    0%   { opacity: 1.0; }
    50%  { opacity: 0.7; }
    100% { opacity: 1.0; }
}
.day-card {
    background: linear-gradient(135deg, #0f1e3d, #1a2f5a);
    border: 1px solid rgba(0, 212, 255, 0.2);
    border-radius: 14px;
    padding: 20px;
    text-align: center;
    margin: 5px;
}
.day-card .aqi-num {
    font-size: 42px;
    font-weight: 700;
    font-family: 'Share Tech Mono', monospace;
}
.day-card .aqi-label {
    font-size: 13px;
    margin-top: 4px;
}
.day-card .date-label {
    font-size: 12px;
    opacity: 0.6;
    margin-bottom: 8px;
}
.day-card .minmax {
    font-size: 12px;
    opacity: 0.5;
    margin-top: 8px;
}
.pollutant-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 10px;
    margin-top: 10px;
}
.pollutant-item {
    background: rgba(0, 212, 255, 0.05);
    border: 1px solid rgba(0, 212, 255, 0.15);
    border-radius: 10px;
    padding: 12px;
    text-align: center;
}
.pollutant-name {
    font-size: 11px;
    opacity: 0.6;
    text-transform: uppercase;
    letter-spacing: 1px;
}
.pollutant-value {
    font-size: 22px;
    font-weight: 700;
    color: #00d4ff;
    font-family: 'Share Tech Mono', monospace;
}
.weather-row {
    display: flex;
    justify-content: space-around;
    background: rgba(255,255,255,0.03);
    border-radius: 12px;
    padding: 15px;
    margin-top: 10px;
}
.weather-item {
    text-align: center;
}
.weather-value {
    font-size: 20px;
    font-weight: 600;
    color: #7eb8ff;
}
.weather-label {
    font-size: 11px;
    opacity: 0.5;
    text-transform: uppercase;
}
</style>
""", unsafe_allow_html=True)

# ─── DATA FETCHING
@st.cache_data(ttl=300)
def get_current():
    try:
        r = requests.get(f"{API_URL}/current", timeout=10)
        return r.json()
    except Exception as e:
        return {"error": str(e)}

@st.cache_data(ttl=3600)
def get_forecast():
    try:
        r = requests.get(f"{API_URL}/forecast", timeout=15)
        return r.json()
    except Exception as e:
        return {"error": str(e)}

# ─── CHART BUILDERS 
def build_gauge(aqi_value: float, color: str):
    fig = go.Figure(go.Indicator(
        mode  = "gauge+number",
        value = aqi_value,
        number = {
            "font": {"size": 52, "color": color,
                     "family": "Share Tech Mono"},
        },
        gauge = {
            "axis": {
                "range": [0, 500],
                "tickcolor": "rgba(255,255,255,0.3)",
                "tickfont":  {"color": "rgba(255,255,255,0.4)", "size": 10},
            },
            "bar": {"color": color, "thickness": 0.2},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
            "steps": [
                {"range": [0,   50],  "color": "rgba(0,100,0,0.4)"},
                {"range": [50,  100], "color": "rgba(100,100,0,0.4)"},
                {"range": [100, 150], "color": "rgba(120,60,0,0.4)"},
                {"range": [150, 200], "color": "rgba(120,0,0,0.4)"},
                {"range": [200, 300], "color": "rgba(80,0,120,0.4)"},
                {"range": [300, 500], "color": "rgba(60,0,20,0.4)"},
            ],
            "threshold": {
                "line":  {"color": color, "width": 3},
                "thickness": 0.8,
                "value": aqi_value,
            },
        },
    ))
    fig.update_layout(
        paper_bgcolor = "rgba(0,0,0,0)",
        plot_bgcolor  = "rgba(0,0,0,0)",
        height        = 260,
        margin        = dict(l=30, r=30, t=20, b=10),
        font          = {"color": "white"},
    )
    return fig

def build_forecast_chart(forecast_data: dict):
    rows = []
    for day in forecast_data.get("days", []):
        for h in day.get("hourly", []):
            rows.append({
                "timestamp": h["timestamp"],
                "aqi":       h["aqi"],
                "color":     h["color"],
                "category":  h["category"],
            })

    if not rows:
        return go.Figure()

    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    fig = go.Figure()

    # Shaded danger zone
    fig.add_hrect(y0=150, y1=500,
                  fillcolor="rgba(255,0,0,0.04)",
                  line_width=0,
                  annotation_text="Danger Zone",
                  annotation_position="top left",
                  annotation_font_color="rgba(255,100,100,0.5)")

    # Area fill
    fig.add_trace(go.Scatter(
        x=df["timestamp"],
        y=df["aqi"],
        mode="lines",
        line=dict(color="rgba(0,212,255,0.3)", width=0),
        fill="tozeroy",
        fillcolor="rgba(0,212,255,0.06)",
        showlegend=False,
        hoverinfo="skip",
    ))

    # Main line
    fig.add_trace(go.Scatter(
        x=df["timestamp"],
        y=df["aqi"],
        mode="lines+markers",
        line=dict(color="#00d4ff", width=2),
        marker=dict(
            size=5,
            color=df["color"].tolist(),
            line=dict(color="white", width=0.5),
        ),
        hovertemplate=(
            "<b>%{x|%b %d %H:%M}</b><br>"
            "AQI: <b>%{y}</b><br>"
            "<extra></extra>"
        ),
    ))

    # Threshold lines
    thresholds = [
        (50,  "Good",     "#00e400"),
        (100, "Moderate", "#ffff00"),
        (150, "Unhealthy","#ff7e00"),
    ]
    for val, label, c in thresholds:
        fig.add_hline(
            y=val, line_dash="dot",
            line_color=c, line_width=1,
            annotation_text=label,
            annotation_position="right",
            annotation_font_color=c,
            annotation_font_size=10,
        )

    fig.update_layout(
        title=dict(
            text="72-Hour AQI Forecast",
            font=dict(family="Share Tech Mono", color="#00d4ff", size=16),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor ="rgba(10,15,30,0.8)",
        height=360,
        margin=dict(l=10, r=80, t=50, b=10),
        xaxis=dict(
            gridcolor="rgba(255,255,255,0.04)",
            tickfont=dict(color="rgba(255,255,255,0.4)", size=10),
            showline=False,
        ),
        yaxis=dict(
            gridcolor="rgba(255,255,255,0.04)",
            tickfont=dict(color="rgba(255,255,255,0.4)", size=10),
            range=[0, max(df["aqi"].max() * 1.2, 200)],
            title=dict(text="AQI", font=dict(color="rgba(255,255,255,0.3)")),
        ),
        hovermode="x unified",
        showlegend=False,
    )
    return fig

# ─── MAIN APP 
def main():
    # Header
    st.markdown(
        "<h1 style='margin-bottom:0'>🌬️ AQI Predictor</h1>",
        unsafe_allow_html=True
    )
    st.caption(
        f"Live air quality monitoring and 3-day forecast • "
        f"Updated: {datetime.utcnow().strftime('%H:%M UTC')}"
    )

    # Fetch data
    current  = get_current()
    forecast = get_forecast()

    # Error check
    if "error" in current:
        st.error(
            f"Cannot connect to API at {API_URL}\n\n"
            f"Make sure api.py is running:\n"
            f"uvicorn api:app --reload --port 8000"
        )
        st.stop()

    aqi      = current["aqi"]
    category = current["category"]
    color    = current["color"]
    city     = current["city"]

    # Alert banner
    if current.get("alert"):
        st.markdown(
            f'<div class="alert-box">'
            f'🚨 AIR QUALITY ALERT — {category.upper()} in {city} — '
            f'Avoid all outdoor activities!'
            f'</div>',
            unsafe_allow_html=True
        )

    st.markdown("---")

    # ── Row 1: Gauge + Info 
    col_left, col_right = st.columns([1, 1.6])

    with col_left:
        st.markdown(f"### 📍 {city}")
        st.markdown(
            f'<div style="color:{color}; font-size:26px; '
            f'font-weight:700; font-family:Share Tech Mono">'
            f'{category}</div>',
            unsafe_allow_html=True
        )
        st.markdown(
            f'<div style="font-size:13px; opacity:0.6; margin-bottom:10px">'
            f'{current.get("advice", "")}</div>',
            unsafe_allow_html=True
        )
        st.plotly_chart(
            build_gauge(aqi, color),
            use_container_width=True
        )

    with col_right:
        # Weather
        w = current["weather"]
        st.markdown("### 🌤️ Weather")
        st.markdown(f"""
        <div class="weather-row">
            <div class="weather-item">
                <div class="weather-value">{w['temp']:.1f}°C</div>
                <div class="weather-label">Temperature</div>
            </div>
            <div class="weather-item">
                <div class="weather-value">{w['humidity']}%</div>
                <div class="weather-label">Humidity</div>
            </div>
            <div class="weather-item">
                <div class="weather-value">{w['pressure']}</div>
                <div class="weather-label">Pressure hPa</div>
            </div>
            <div class="weather-item">
                <div class="weather-value">{w['wind_speed']} m/s</div>
                <div class="weather-label">Wind</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Pollutants
        p = current["pollutants"]
        st.markdown("### 🧪 Pollutants (μg/m³)")
        st.markdown(f"""
        <div class="pollutant-grid">
            <div class="pollutant-item">
                <div class="pollutant-name">PM2.5</div>
                <div class="pollutant-value">{p['pm25']:.1f}</div>
            </div>
            <div class="pollutant-item">
                <div class="pollutant-name">PM10</div>
                <div class="pollutant-value">{p['pm10']:.1f}</div>
            </div>
            <div class="pollutant-item">
                <div class="pollutant-name">O3</div>
                <div class="pollutant-value">{p['o3']:.1f}</div>
            </div>
            <div class="pollutant-item">
                <div class="pollutant-name">NO2</div>
                <div class="pollutant-value">{p['no2']:.1f}</div>
            </div>
            <div class="pollutant-item">
                <div class="pollutant-name">SO2</div>
                <div class="pollutant-value">{p['so2']:.1f}</div>
            </div>
            <div class="pollutant-item">
                <div class="pollutant-name">CO</div>
                <div class="pollutant-value">{p['co']:.1f}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Row 2: Forecast Chart ─────────────────────────────────────
    if forecast and "days" in forecast:
        st.plotly_chart(
            build_forecast_chart(forecast),
            use_container_width=True
        )

        # Day cards
        st.markdown("### 📅 3-Day Summary")
        day_cols = st.columns(len(forecast["days"]))
        for i, day in enumerate(forecast["days"]):
            with day_cols[i]:
                # Format date nicely
                from datetime import datetime as dt
                try:
                    d = dt.strptime(day["date"], "%Y-%m-%d")
                    date_str = d.strftime("%A\n%b %d")
                except Exception:
                    date_str = day["date"]

                st.markdown(f"""
                <div class="day-card">
                    <div class="date-label">{date_str}</div>
                    <div class="aqi-num" style="color:{day['color']}">
                        {day['avg_aqi']}
                    </div>
                    <div class="aqi-label" style="color:{day['color']}">
                        {day['category']}
                    </div>
                    <div class="minmax">
                        ↑ {day['max_aqi']} &nbsp; ↓ {day['min_aqi']}
                    </div>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.warning("Forecast not available. Check API connection.")

    st.markdown("---")

    # ── AQI Reference Table 
    with st.expander("📖 AQI Health Scale Reference"):
        scale = {
            "Range":    ["0–50","51–100","101–150","151–200","201–300","301–500"],
            "Category": ["Good","Moderate","Unhealthy for Sensitive Groups",
                         "Unhealthy","Very Unhealthy","Hazardous"],
            "Who's at risk": [
                "No one",
                "Very sensitive individuals",
                "Children, elderly, lung/heart disease patients",
                "Everyone",
                "Everyone — serious effects",
                "Everyone — emergency conditions",
            ],
            "Action": [
                "Enjoy outdoors",
                "Sensitive people take it easy",
                "Sensitive groups stay indoors",
                "Everyone reduce outdoor time",
                "Avoid all outdoor activity",
                "Stay indoors, close windows",
            ],
        }
        st.dataframe(
            pd.DataFrame(scale),
            use_container_width=True,
            hide_index=True,
        )

if __name__ == "__main__":
    main()