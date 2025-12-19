# src/4_model_congestion.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

def main():
    df = pd.read_csv("data/processed/traffic_final_dataset.csv", parse_dates=["datetime"])

    # Features
    feature_cols = [
        "avg_speed","vehicle_count","lane_count","road_type_speed_ratio",
        "hour","day_of_week","is_weekend","is_peak_hour","hour_sin","hour_cos",
        "is_raining","rain_severity","wind_severity","temperature_severity","weather_flag",
        "incident_flag","incident_severity","holiday_event_flag"
    ]
    X = df[feature_cols]
    y = df["congestion_level"]

    # Encode categorical if needed
    X = pd.get_dummies(X, columns=["lane_count"], drop_first=True)

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Train model
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=42,
        class_weight="balanced"
    )
    clf.fit(X_train, y_train)

    # Evaluate
    y_pred = clf.predict(X_test)
    print("✅ Classification Report:\n")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    # Save model
    joblib.dump(clf, "models/congestion_model.pkl")
    print("✅ Model saved: models/congestion_model.pkl")
    feature_cols = X.columns.tolist()
    joblib.dump(feature_cols, "models/congestion_feature_cols.pkl")

if __name__ == "__main__":
    main()
