#  Project Report---AQI Predictor

This project predicts Air Quality Index (AQI) for Karachi using pollutant, weather, time-based, and rolling AQI features.

## System Architecture

The project collects AQI and pollutant data from AQICN and weather data from OpenWeather. The data is processed through feature engineering and stored in Hopsworks Feature Store. ML models are trained through a training pipeline and the final model is registered in Hopsworks Model Registry. The predictions are displayed through a Streamlit dashboard.

```text
AQICN + OpenWeather
        ↓
Feature Pipeline
        ↓
Hopsworks Feature Store
        ↓
Feature View / Training Data
        ↓
Training Pipeline
        ↓
Hopsworks Model Registry
        ↓
Streamlit Dashboard

```

## Data Sources

- AQICN: AQI and pollutant data
- OpenWeather: weather data

## Feature Engineering

The project created pollutant features, weather features, time-based features, cyclic features, rolling AQI averages, and future AQI target columns.Feature Engineering

The project created contins 28 features including:

Pollutant features: PM2.5, PM10, O3, NO2, SO2, CO
Weather features: temperature, humidity, pressure, wind speed, wind direction
Time-based features: hour, day of week, month, weekend flag
Cyclical features: hour sine/cosine and month sine/cosine
Rolling features: AQI rolling average over 6 hours and 24 hours
Target columns: target_aqi_3h, target_aqi_24h, target_aqi_72h

Live feature rows store future target columns as NaN because future AQI is unknown at insertion time. Historical target values are created from past data using time-based shifting

## Hopsworks Feature Store

The engineered data was stored in Hopsworks Feature Store.

Feature Group: aqi_features
Feature View: aqi_feature_view
Training Data: version 1


## EDA

EDA was performed in `eda.ipynb`. It included missing value analysis, AQI statistics, AQI trends, pollutant relationships, correlation heatmap, and model comparison.

## Model Training

Multiple regression models were trained and compared:

- Linear Regression
- Ridge Regression
- Random Forest
- Gradient Boosting
- LSTM

Since AQI prediction is a regression taskn so the models were evaluated using MAE, RMSE, and R² score.
The final production model was selected based on evaluation metrics and registered in Hopsworks Model Registry.

## Results

The final registered model achieved:

- MAE: 11.76
- RMSE: 19.97
- R²: 0.738

R² = 0.738 means the model explains a good portion of AQI variation and performs reasonably well for AQI forecasting.

## Explainability

SHAP feature is used to explain global model behavior and LIME explanation is also added to explain an individual prediction.

## Deployment and Automation

The dashboard was deployed using Streamlit Cloud. GitHub Actions were used to automate the feature pipeline and training pipeline.

- Feature pipeline runs hourly.
- Training pipeline runs daily.
- Backfill pipeline can be triggered manually.

## Dashboard

The Streamlit dashboard displays:

Current AQI
AQI category
Weather conditions
Pollutant breakdown
72-hour AQI forecast
3-day summary
Forecast alerts
SHAP explanation
LIME explanation
AQI health reference

## Limitations

- AQI can be affected by external factors not included in the dataset.
- Live rows do not contain future target values because future AQI is unknown at insertion time.
- Historical future target values are generated from past data using time-based shifting.

## Conclusion

This project successfully implements an end-to-end AQI prediction system for Karachi with automated data pipelines, Hopsworks Feature Store, Hopsworks Model Registry, explainability using SHAP and LIME, and an interactive Streamlit dashboard for real-time and forecasted AQI monitoring.