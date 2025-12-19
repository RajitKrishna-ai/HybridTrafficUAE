import pandas as pd
import numpy as np
import os

def add_time_features(df):
    """
    Create time-based and calendar features from datetime including cyclical encoding.
    """

    # Hour / Day / Weekend
    df["hour"] = df["datetime"].dt.hour
    df["day_of_week"] = df["datetime"].dt.dayofweek  # Monday=0
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

    # Peak hour flags
    df["is_morning_peak"] = df["hour"].between(7, 10).astype(int)
    df["is_evening_peak"] = df["hour"].between(17, 20).astype(int)
    df["is_peak_hour"] = ((df["is_morning_peak"] == 1) | (df["is_evening_peak"] == 1)).astype(int)

    # Cyclical encoding for hour
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    return df

ROAD_SPEEDS = {
    "Highway": 100,
    "Arterial": 60,
    "Local": 40
}

def add_traffic_dynamics_features(df):
    """
    Engineer traffic dynamics features: moving averages, ratios, road-aware congestion.
    """

    df = df.sort_values(by=["road_id", "datetime"])

    # Moving averages of speed
    df["speed_ma_3"] = df.groupby("road_id")["avg_speed"].transform(lambda x: x.rolling(3, min_periods=1).mean())
    df["speed_ma_6"] = df.groupby("road_id")["avg_speed"].transform(lambda x: x.rolling(6, min_periods=1).mean())

    # Moving averages of vehiclecount
    df["vehicle_count_ma_3"] = df.groupby("road_id")["vehicle_count"].transform(lambda x: x.rolling(3, min_periods=1).mean())

    # Vehicle count ratio
    df["vehicle_count_ratio"] = df["vehicle_count"] / df["lane_count"]

    # Road-type speed ratio (for congestion calculation)
    df["road_type_speed_ratio"] = df.apply(
        lambda row: row["avg_speed"] / ROAD_SPEEDS.get(row["road_type"], 50), axis=1
    )
    return df


def add_weather_features(df):
    """
    Engineer weather severity features for traffic congestion modeling.
    """

    # Rain binary and severity
    df["is_raining"] = (df["rain_mm"] > 0).astype(int)
    df["rain_severity"] = pd.cut(
        df["rain_mm"],
        bins=[-0.01, 0, 2, 5, np.inf],
        labels=[0, 1, 2, 3]
    ).astype(int)

    # Wind severity
    df["wind_severity"] = pd.cut(
        df["wind_kph"],
        bins=[-0.01, 15, 25, np.inf],
        labels=[0, 1, 2]
    ).astype(int)

    # Temperature severity
    df["temperature_severity"] = ((df["temperature"] < 25) | (df["temperature"] > 40)).astype(int)

    # Combined weather flag
    df["weather_flag"] = ((df["is_raining"] == 1) | (df["wind_severity"] > 1) | (df["temperature_severity"] == 1)).astype(int)

    return df



def add_proxy_incident_features(df, seed=42):
    """
    Add synthetic incident features to traffic dataset.
    """
    np.random.seed(seed)

    df["incident_flag"] = 0
    df["incident_severity"] = 0

    for road in df["road_id"].unique():
        road_df = df[df["road_id"] == road]
        mean_vehicle_count = road_df["vehicle_count"].mean()

        # Assign random incidents based on peak and high traffic
        for idx, row in df[df["road_id"] == road].iterrows():
            prob = 0.01  # base probability
            if row["is_peak_hour"] == 1:
                prob += 0.02
            if row["vehicle_count"] > mean_vehicle_count:
                prob += 0.02
            if np.random.rand() < prob:
                df.at[idx, "incident_flag"] = 1
                # Severity proportional to vehicle_count drop
                df.at[idx, "incident_severity"] = np.random.choice([1,2,3], p=[0.6,0.3,0.1])

    # Holiday/Event impact (already binary)
    df["holiday_event_flag"] = df["is_holiday"]

    return df

def main():
    df = pd.read_csv("data/processed/traffic_with_weather_features.csv", parse_dates=["datetime"])

    df = add_proxy_incident_features(df)

    df.to_csv("data/processed/traffic_with_all_hybrid_features.csv", index=False)
    print("✅ Proxy incident features added and saved.")

if __name__ == "__main__":
    main()

