import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix


# Configuration

DATA_PATH = "data/processed/traffic_final_dataset.csv"
RANDOM_STATE = 42

FEATURE_COLUMNS = [
    "hour",
    "day_of_week",
    "is_weekend",
    "is_peak_hour",
    "is_raining",
    "rain_severity",
    "wind_severity",
    "temperature_severity",
    "incident_flag",
    "incident_severity",
    "holiday_event_flag",
    "vehicle_count_ratio",
    "road_type_speed_ratio",
    "congestion_ratio"
]

TARGET_COLUMN = "congestion_cause"



def load_data():
    df = pd.read_csv(DATA_PATH)
    return df



def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=RANDOM_STATE,
        class_weight="balanced"
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("\n📊 Classification Report (Cause Model):\n")
    print(classification_report(y_test, y_pred))

    print("\n🧮 Confusion Matrix:\n")
    print(confusion_matrix(y_test, y_pred))

    return model


def main():
    df = load_data()

    # 🔥 CRITICAL FIX:
    # Cause only exists when congestion exists
    df = df[df[TARGET_COLUMN].notna()]

    # Safety checks
    assert df[TARGET_COLUMN].isna().sum() == 0, "Target still has NaN"
    assert len(df) > 0, "No data left after filtering"

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    model = train_model(X, y)



    # Ensure models folder exists
    os.makedirs("models", exist_ok=True)

    # Save the trained model
    joblib.dump(model, "models/cause_model.pkl")
    print("Cause model saved successfully!")

    
    cause_feature_cols = X.columns.tolist()
    joblib.dump(cause_feature_cols, "models/cause_feature_cols.pkl")

    print("\n✅ Congestion cause model training complete.")


if __name__ == "__main__":
    main()
