import pandas as pd
from src.features import add_time_features

def test_time_features_created():
    df = pd.DataFrame({
        "datetime": pd.to_datetime(["2025-01-01 08:00:00"])
    })

    df = add_time_features(df)

    assert "hour" in df.columns
    assert df.loc[0, "hour"] == 8
