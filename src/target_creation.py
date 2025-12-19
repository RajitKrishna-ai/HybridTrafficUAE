import pandas as pd

ROAD_SPEEDS = {
    "Highway": 100,
    "Arterial": 60,
    "Local": 40
}

def create_congestion_level(df):
    # Calculate ratio
    df["congestion_ratio"] = df.apply(
        lambda row: row["avg_speed"] / ROAD_SPEEDS.get(row["road_type"], 50), axis=1
    )

    # Map to labels
    conditions = [
        (df["congestion_ratio"] >= 0.75),
        (df["congestion_ratio"] < 0.75) & (df["congestion_ratio"] >= 0.40),
        (df["congestion_ratio"] < 0.40)
    ]
    choices = ["Low", "Medium", "High"]
    df["congestion_level"] = pd.cut(
        df["congestion_ratio"],
        bins=[-1, 0.4, 0.75, 1.5],
        labels=["High", "Medium", "Low"]
    )

    return df

def create_congestion_cause(df):
    # Priority: Incident>Weather>Holiday/Event>Peak_Hour
    def assign_cause(row):
        if row["incident_flag"] == 1:
            return "Incident"
        elif row["weather_flag"] == 1:
            return "Weather"
        elif row["holiday_event_flag"] == 1:
            return "Holiday_Event"
        elif row["is_peak_hour"] == 1:
            return "Peak_Hour"
        else:
            return "None"

    df["congestion_cause"] = df.apply(assign_cause, axis=1)
    return df

def main():
    df = pd.read_csv("data/processed/traffic_with_all_hybrid_features.csv", parse_dates=["datetime"])

    df = create_congestion_level(df)
    df = create_congestion_cause(df)

    df.to_csv("data/processed/traffic_final_dataset.csv", index=False)
    print("✅ Target variables created. Dataset ready for modeling")

if __name__ == "__main__":
    main()
