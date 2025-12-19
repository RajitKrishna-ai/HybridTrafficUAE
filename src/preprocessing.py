from src.ingestion import minimal_cleaning
import pandas as pd
from src.features import add_proxy_incident_features, add_time_features, add_weather_features, add_traffic_dynamics_features
def create_features(df):    

    """
    Master feature engineering function.
    Used by training, inference, explainability.
    """

    # Ensure datetime
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"])

    # Apply feature blocks
    df = add_time_features(df)
    df = add_traffic_dynamics_features(df)
    df = add_weather_features(df)
    df = add_proxy_incident_features(df)

    return df
