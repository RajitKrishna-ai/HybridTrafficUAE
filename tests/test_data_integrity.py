import pandas as pd

def test_no_missing_critical_columns():
    df = pd.read_csv("data/processed/traffic_with_all_features.csv")

    critical_cols = ["datetime", "road_id", "avg_speed", "vehicle_count"]

    for col in critical_cols:
        assert col in df.columns, f"Missing column: {col}"
        assert df[col].isnull().sum() == 0, f"Null values in {col}"
