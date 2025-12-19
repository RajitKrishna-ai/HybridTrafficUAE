import pandas as pd
import joblib
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"
DATA_PATH = BASE_DIR / "data" / "processed" / "traffic_final_dataset.csv"


def load_model():
    model_path = MODEL_DIR / "congestion_model.pkl"
    return joblib.load(model_path)


def load_data():
    df = pd.read_csv(DATA_PATH)

    # Keep only rows where cause is defined
    df = df[df["congestion_cause"].notna()]

    # Get feature columns dynamically based on model training
    # Assuming 'congestion_cause' is target and rest are features
    drop_cols = ["congestion_cause", "congestion_level", "congestion_ratio", "date", "datetime", "road_id"]
    feature_cols = [c for c in df.columns if c not in drop_cols]

    X = df[feature_cols]

    return X, feature_cols


def compute_feature_importance(model, X, feature_cols=None):
    """
    Compute feature importance and return a sorted DataFrame.
    """

    # Ensure model has feature_importances_
    if not hasattr(model, "feature_importances_"):
        raise ValueError("Model does not havefeature_importances_ attribute")

    importance = model.feature_importances_

    # If feature_cols is None, take from X
    if feature_cols is None:
        feature_cols = X.columns.tolist()

    # Safety check: trim to min length
    min_len = min(len(feature_cols), len(importance))
    feature_cols = feature_cols[:min_len]
    importance = importance[:min_len]

    fi_df = pd.DataFrame({
        "feature": feature_cols,
        "importance": importance
    }).sort_values(by="importance", ascending=False)

    return fi_df



def main():
    print("Loading model...")
    model = load_model()

    print("Loading data...")
    X, feature_cols = load_data()

    print("Computing feature importance...")
    fi_df = compute_feature_importance(model, X, feature_cols)

    output_path = BASE_DIR / "reports" / "feature_importance_cause.csv"
    fi_df.to_csv(output_path, index=False)

    print("\nTop Feature Importance (Congestion Cause):")
    print(fi_df.head(10))


if __name__ == "__main__":
    main()
