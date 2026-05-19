import os
import hopsworks
from dotenv import load_dotenv

load_dotenv()

HOPSWORKS_KEY = os.getenv("HOPSWORKS_API_KEY")
HOPSWORKS_HOST = os.getenv("HOPSWORKS_HOST", "eu-west.cloud.hopsworks.ai")
HOPSWORKS_PROJECT = os.getenv("HOPSWORKS_PROJECT")
HOPSWORKS_PORT = int(os.getenv("HOPSWORKS_PORT", 443))

project = hopsworks.login(
    host=HOPSWORKS_HOST,
    port=HOPSWORKS_PORT,
    project=HOPSWORKS_PROJECT,
    api_key_value=HOPSWORKS_KEY,
)

fs = project.get_feature_store()

fg = fs.get_feature_group(
    name="aqi_features",
    version=1
)

query = fg.select_all()

feature_view = fs.create_feature_view(
    name="aqi_feature_view",
    version=1,
    query=query,
    labels=["target_aqi_72h"],
    description="Feature view for 72-hour AQI prediction using AQI, pollutant, weather, and time-based features."
)

print("Feature View created successfully:", feature_view.name)