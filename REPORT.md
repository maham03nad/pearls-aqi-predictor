#  Project Report---AQI Predictor

This project predicts Air Quality Index (AQI) for Karachi using pollutant, weather, time-based, and rolling AQI features.

## System Architecture

The project collects AQI and pollutant data from AQICN and weather data from OpenWeather. The data is processed through feature engineering and stored in Hopsworks Feature Store. ML models are trained through a training pipeline and the final model is registered in Hopsworks Model Registry. The predictions are displayed through a Streamlit dashboard.

## Data Sources

- AQICN: AQI and pollutant data
- OpenWeather: weather data

## Feature Engineering

The project created pollutant features, weather features, time-based features, cyclic features, rolling AQI averages, and future AQI target columns.

## EDA

EDA was performed in `eda.ipynb`. It included missing value analysis, AQI statistics, AQI trends, pollutant relationships, correlation heatmap, and model comparison.

## Model Training

Multiple regression models were trained and compared:

- Linear Regression
- Ridge Regression
- Random Forest
- Gradient Boosting
- LSTM

The final production model was selected based on evaluation metrics and registered in Hopsworks Model Registry.

## Results

The final registered model achieved:

- MAE: 11.76
- RMSE: 19.97
- R²: 0.738

R² = 0.738 means the model explains a good portion of AQI variation and performs reasonably well for AQI forecasting.

## Explainability

SHAP feature importance was used to explain global model behavior. LIME explanation was also added to explain an individual prediction.

## Deployment and Automation

The dashboard was deployed using Streamlit Cloud. GitHub Actions were used to automate the feature pipeline and training pipeline.

- Feature pipeline runs hourly.
- Training pipeline runs daily.
- Backfill pipeline can be triggered manually.

## Limitations

- AQI can be affected by external factors not included in the dataset.
- Live rows do not contain future target values because future AQI is unknown at insertion time.
- Historical future target values are generated from past data using time-based shifting.