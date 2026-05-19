# Project Report
# AQI Predictor

Air pollution is a one of the major environmental and public health concern in large cities such as Karachi. Air Quality Index (AQI) changes due to many facrors like pollutant concentration, weather conditions, time of day, traffic patterns, and other external factors.

The main purpose of this project is to build an end-to-end AQI prediction system that can collect real-time AQI and weather data, create ML features, train forecasting models, and predict future AQI values.

It is designed to support AQI forecasting for future time windows such as 3 hours, 24 hours, and 72 hours. It also includes an interactive dashboard, model explainability, feature store, and model registry.
---

## 2. Project Objectives

The main objectives of this project are:

- Collect AQI and weather data from external APIs.
- Clean and process raw environmental data.
- Engineer time-based, weather-based, pollutant-based, and rolling features.
- Store processed features in a feature store.
- Train and compare machine learning models.
- Select a production model using a suitable validation approach.
- Register the selected model in a model registry.
- Build an API for AQI prediction.
- Create an interactive dashboard for current and forecasted AQI.
- Add SHAP and LIME explainability.
- Provide forecast-based AQI alerts.
---

## 3. Data Sources

The project uses multiple data sources:

| Source | Purpose |
|---|---|
| AQICN API | Current AQI and pollutant values |
| OpenWeather API | Current weather data such as temperature, humidity, pressure, and wind |
| Open-Meteo Archive | Historical weather data for backfilling and model training |
| Hopsworks Feature Store | Stores processed model-ready features |

The feature pipeline fetches live AQI data from AQICN and live weather data from OpenWeather, then combines them into one structured feature row.

---

## 4. System Architecture

The project follows this end-to-end machine learning architecture:

```text
AQICN API + OpenWeather API + Historical Data
                    ↓
              Data Collection
                    ↓
              Data Cleaning
                    ↓
            Feature Engineering
                    ↓
        Hopsworks Feature Store
                    ↓
            Training Pipeline
                    ↓
            Model Comparison
                    ↓
              Model Registry
                    ↓
          FastAPI Prediction API
                    ↓
           Streamlit Dashboard
                    ↓
      SHAP/LIME Explainability
                    ↓
        Forecast-Based AQI Alerts
```
---

# 