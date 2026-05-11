"""
STEP 3: Training Pipeline
- Loads features + targets from Hopsworks Feature Store
- Trains Random Forest, Ridge Regression, XGBoost, LSTM
- Evaluates using RMSE, MAE, R²
- Saves the best model to Hopsworks Model Registry
- Uses SHAP for feature importance

Run daily via CI/CD:
    python training_pipeline.py
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import hopsworks
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble        import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model    import Ridge
from sklearn.preprocessing   import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics         import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline        import Pipeline

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# ─── CONFIG 
HOPSWORKS_KEY = os.getenv("HOPSWORKS_API_KEY", "YOUR_HW_KEY")
TARGET_COL    = "target_aqi_3h"      # predict AQI 3 hours ahead
FEATURE_COLS  = [
    "pm25", "pm10", "o3", "no2", "so2", "co",
    "temp", "humidity", "pressure", "wind_speed", "wind_deg",
    "hour", "day_of_week", "month", "is_weekend",
    "hour_sin", "hour_cos", "month_sin", "month_cos",
    "aqi_change_rate", "aqi_rolling_6h", "aqi_rolling_24h",
]
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

# ──  Load data from Feature Store
def load_features() -> pd.DataFrame:
    project = hopsworks.login(api_key_value=HOPSWORKS_KEY)
    fs = project.get_feature_store()
    fg = fs.get_feature_group("aqi_features", version=1)
    df = fg.read()
    print(f"[✓] Loaded {len(df)} rows from feature store")
    return df

# ── Prepare train/test split 
def prepare_data(df: pd.DataFrame):
    df = df.sort_values("timestamp").dropna(subset=FEATURE_COLS + [TARGET_COL])
    X = df[FEATURE_COLS].values
    y = df[TARGET_COL].values
    return train_test_split(X, y, test_size=0.2, shuffle=False)

# ──  Evaluate helper
def evaluate(name: str, y_true, y_pred) -> dict:
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    print(f"  [{name}]  RMSE={rmse:.2f}  MAE={mae:.2f}  R²={r2:.4f}")
    return {"model": name, "rmse": rmse, "mae": mae, "r2": r2}

# ── Train classic ML models 
def train_sklearn_models(X_train, X_test, y_train, y_test) -> list:
    results = []
    trained = {}

    models = {
        "RandomForest":   RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=42),
        "GradientBoost":  GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, random_state=42),
        "Ridge":          Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=10.0))]),
    }

    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        metrics = evaluate(name, y_test, preds)
        results.append(metrics)
        trained[name] = model

        # Save model
        path = os.path.join(MODELS_DIR, f"{name}.pkl")
        joblib.dump(model, path)

    return results, trained

# ── Train LSTM 
def train_lstm(X_train, X_test, y_train, y_test, seq_len=24) -> dict:
    """Reshape flat features into sequences for LSTM."""
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_train)
    X_te_s = scaler.transform(X_test)

    def make_sequences(X, y, seq):
        xs, ys = [], []
        for i in range(seq, len(X)):
            xs.append(X[i - seq:i])
            ys.append(y[i])
        return np.array(xs), np.array(ys)

    if len(X_tr_s) < seq_len + 10:
        print("  [LSTM] Not enough data, skipping.")
        return None, None

    X_tr_seq, y_tr_seq = make_sequences(X_tr_s, y_train, seq_len)
    X_te_seq, y_te_seq = make_sequences(X_te_s, y_test,  seq_len)

    model = Sequential([
        LSTM(64, input_shape=(seq_len, X_train.shape[1]), return_sequences=True),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(16, activation="relu"),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse")

    es = EarlyStopping(patience=5, restore_best_weights=True)
    print("Training LSTM...")
    model.fit(X_tr_seq, y_tr_seq,
              validation_split=0.1,
              epochs=50, batch_size=32,
              callbacks=[es], verbose=0)

    preds = model.predict(X_te_seq).flatten()
    metrics = evaluate("LSTM", y_te_seq, preds)

    model.save(os.path.join(MODELS_DIR, "lstm_model.h5"))
    joblib.dump(scaler, os.path.join(MODELS_DIR, "lstm_scaler.pkl"))

    return metrics, model

# ──  SHAP feature importance 
def explain_model(model, X_test, feature_names):
    print("Computing SHAP values...")
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test[:200])
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_test[:200],
                          feature_names=feature_names, show=False)
        path = os.path.join(MODELS_DIR, "shap_summary.png")
        plt.savefig(path, bbox_inches="tight")
        plt.close()
        print(f"  [✓] SHAP plot saved → {path}")
    except Exception as e:
        print(f"  [!] SHAP skipped: {e}")


# ── Push best model to Hopsworks Model Registry 
def register_best_model(results: list, sklearn_trained: dict):
    best = min(results, key=lambda x: x["rmse"])
    print(f"\n🏆 Best model: {best['model']}  (RMSE={best['rmse']:.2f})")

    project = hopsworks.login(api_key_value=HOPSWORKS_KEY)
    mr = project.get_model_registry()

    model_dir = MODELS_DIR
    hw_model = mr.python.create_model(
        name="aqi_predictor",
        metrics={"rmse": best["rmse"], "mae": best["mae"], "r2": best["r2"]},
        description=f"Best model: {best['model']}",
    )
    hw_model.save(model_dir)
    print(f"[✓] Model registered in Hopsworks Model Registry")
    return best

# ── MAIN ───
def run():
    print("=== Training Pipeline ===")

    df = load_features()
    X_train, X_test, y_train, y_test = prepare_data(df)
    print(f"Train: {X_train.shape}  Test: {X_test.shape}")

    all_results = []

    sk_results, sk_trained = train_sklearn_models(X_train, X_test, y_train, y_test)
    all_results.extend(sk_results)

    lstm_metrics, _ = train_lstm(X_train, X_test, y_train, y_test)
    if lstm_metrics:
        all_results.append(lstm_metrics)

    # SHAP on best sklearn model
    best_sk = min(sk_results, key=lambda x: x["rmse"])
    if best_sk["model"] in sk_trained:
        explain_model(sk_trained[best_sk["model"]], X_test, FEATURE_COLS)

    # Save metrics summary
    with open(os.path.join(MODELS_DIR, "metrics.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    register_best_model(all_results, sk_trained)
    print("=== Training complete ===")

if __name__ == "__main__":
    run()