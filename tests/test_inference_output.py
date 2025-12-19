import pandas as pd

def test_final_output_schema():
    df = pd.read_csv("outputs/final_traffic_predictions.csv")

    expected_cols = [
        "datetime",
        "road_id",
        "congestion_level",
        "congestion_cause",
        "primary_driver",
        "confidence_proxy",
        "data_type"
    ]

    for col in expected_cols:
        assert col in df.columns, f"Missing output column: {col}"
