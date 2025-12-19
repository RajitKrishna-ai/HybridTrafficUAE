# inference_pipeline.py
import sys
import os
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_DIR)
import numpy as np
import pandas as pd
import joblib 
import shap
import matplotlib.pyplot as plt
from src.ingestion import minimal_cleaning
from src.preprocessing import create_features

# Import feature engineering from preprocessing / ingestion
# -------------------------------
from src.ingestion import minimal_cleaning
from src.features import (
    add_time_features,
    add_traffic_dynamics_features,
    add_weather_features,
    add_proxy_incident_features
)

# Paths
# -------------------------------
raw_data_path = "data/processed/traffic_with_all_features.csv"
congestion_model_path = "models/congestion_model.pkl"
congestion_features_path = "models/congestion_feature_cols.pkl"
cause_model_path = "models/cause_model.pkl"
cause_features_path = "models/cause_feature_cols.pkl"

output_csv_path = "data/processed/traffic_predictions.csv"

# Load raw data
# -------------------------------
print("Loading raw traffic data...")
raw_df = pd.read_csv(raw_data_path, parse_dates=["datetime"])

# Minimal cleaning
# -------------------------------
print("Applying minimal cleaning...")
cleaned_df = minimal_cleaning(raw_df)

# Feature engineering
# -------------------------------
print("Creating features...")
df = add_time_features(cleaned_df)
df = add_traffic_dynamics_features(df)
df = add_weather_features(df)
df = add_proxy_incident_features(df)

# Load models and feature columns
# -------------------------------
print("Loading models and feature columns...")
congestion_model = joblib.load(congestion_model_path)
congestion_features = joblib.load(congestion_features_path)

cause_model = joblib.load(cause_model_path)
cause_features = joblib.load(cause_features_path)

# Align features for model input
# -------------------------------
print("Aligning features for model input...")

# Function to align dataframe columns to model training columns
def align_features(df, feature_cols):
    df_copy = df.copy()
    for col in feature_cols:
        if col not in df_copy.columns:
            df_copy[col] = 0  # Add missing columns as 0
    df_copy = df_copy[feature_cols]
    return df_copy

X_congestion = align_features(df, congestion_features)
X_cause = align_features(df, cause_features)

# Predict congestion and cause
# -------------------------------
print("Predicting congestion level...")
df["congestion_level"] = congestion_model.predict(X_congestion)

print("Predicting congestion cause...")
df["congestion_cause"] = cause_model.predict(X_cause)


# SHAP EXPLAINABILITY


print("Calculating SHAP values for congestion model...")

# SHAP for Congestion Model (MULTI-CLASS SAFE)
# -------------------------------
explainer_cong = shap.Explainer(congestion_model, X_congestion)
shap_values_cong = explainer_cong(X_congestion)

# Get class labels in correct order
congestion_classes = list(congestion_model.classes_)

# Map predicted labels to class indices
predicted_class_indices = [
    congestion_classes.index(label)
    for label in df["congestion_level"]
]

# Extract SHAP values corresponding to predicted class
shap_values_cong_selected = np.array([
    shap_values_cong.values[i, :, class_idx]
    for i, class_idx in enumerate(predicted_class_indices)
])

# Convert toDataFrame
shap_df_cong = pd.DataFrame(
    shap_values_cong_selected,
    columns=X_congestion.columns
)

# Top contributing feature per prediction
df["top_feature_congestion"] = shap_df_cong.abs().idxmax(axis=1)

print("✅ SHAP congestion explanations calculated.")

# SHAP for Cause Model (multi-class)
# -------------------------------
print("Calculating SHAP values for cause model...")

explainer_cause = shap.Explainer(cause_model, X_cause)
shap_values_cause = explainer_cause(X_cause)

cause_classes = list(cause_model.classes_)

predicted_cause_indices = [
    cause_classes.index(label)
    for label in df["congestion_cause"]
]

shap_values_cause_selected = np.array([
    shap_values_cause.values[i, :, class_idx]
    for i, class_idx in enumerate(predicted_cause_indices)
])

shap_df_cause = pd.DataFrame(
    shap_values_cause_selected,
    columns=X_cause.columns
)

df["top_feature_cause"] = shap_df_cause.abs().idxmax(axis=1)

print("✅ SHAP cause explanations calculated.")

# SHAP SUMMARY PLOTS (global)
# -------------------------------
print("Generating SHAP summary plots.")

os.makedirs("outputs", exist_ok=True)

# Congestion summary plot
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_df_cong.values, X_congestion, plot_type="bar", show=False)
plt.tight_layout()
plt.savefig("outputs/shap_summary_congestion.png")
plt.close()

# Cause summary plot
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_df_cause.values, X_cause, plot_type="bar", show=False)
plt.tight_layout()
plt.savefig("outputs/shap_summary_cause.png")
plt.close()

print("✅ SHAP summary plots saved in outputs/")


# # Save structured predictions
# # -------------------------------
# output_df = pd.DataFrame({
#     "datetime": df["datetime"],
#     "road_id": df["road_id"],
#     "predicted_congestion": df["congestion_level"],
#     "predicted_cause": df["congestion_cause"]
# })

# #include numeric features for reference / explainability
# numeric_features = [col for col in df.columns if col not in ["datetime", "road_id","congestion_level", "congestion_cause"]]
# output_df = pd.concat([output_df, df[numeric_features]], axis=1)

# output_csv_path = "data/processed/traffic_predictions.csv"
# output_df.to_csv(output_csv_path, index=False)

# print(f"Structured predictions saved to {output_csv_path}!")
# print("Sample predictions:")
# print(output_df.head())



# LOCAL SHAP EXPLANATION

print("Generating local SHAP explanation...")

# a sample index (for demo purpose) we can change this to any index 
sample_index = 100  

# Congestion Local Explanation
# -------------------------------
shap_value_single_cong = shap.Explanation(
    values=shap_df_cong.iloc[sample_index].values,
    base_values=shap_values_cong.base_values[sample_index][
        congestion_classes.index(df.loc[sample_index, "congestion_level"])
    ],
    data=X_congestion.iloc[sample_index].values,
    feature_names=X_congestion.columns
)

plt.figure(figsize=(10, 6))
shap.waterfall_plot(shap_value_single_cong, show=False)
plt.tight_layout()
plt.savefig("outputs/shap_local_congestion.png")
plt.close()

print("Local SHAP congestion explanation saved.")


# Cause Local Explanation
# -------------------------------
shap_value_single_cause = shap.Explanation(
    values=shap_df_cause.iloc[sample_index].values,
    base_values=shap_values_cause.base_values[sample_index][
        cause_classes.index(df.loc[sample_index, "congestion_cause"])
    ],
    data=X_cause.iloc[sample_index].values,
    feature_names=X_cause.columns
)

plt.figure(figsize=(10, 6))
shap.waterfall_plot(shap_value_single_cause, show=False)
plt.tight_layout()
plt.savefig("outputs/shap_local_cause.png")
plt.close()

print("Local SHAP cause explanation saved.")



# Final Prediction Output CSV


print("Building final production output..")

final_output = pd.DataFrame({
    "datetime": df["datetime"],
    "road_id": df["road_id"],
    "congestion_level": df["congestion_level"],
    "congestion_cause": df["congestion_cause"],
    "primary_driver": df["top_feature_congestion"],
})

# Confidence proxy
if hasattr(congestion_model, "predict_proba"):
    final_output["confidence_proxy"] = np.max(
        congestion_model.predict_proba(X_congestion), axis=1
    )
else:
    final_output["confidence_proxy"] = np.nan

# Data origin classification
def classify_data_origin(row):
    if row.get("incident_flag", 0) == 1:
        return "proxy_incident"
    elif row.get("holiday_event_flag", 0) == 1:
        return "real_holiday"
    else:
        return "real_traffic_weather"

final_output["data_type"] = df.apply(classify_data_origin, axis=1)

# Save final output
final_output_path = "outputs/final_traffic_predictions.csv"
final_output.to_csv(final_output_path, index=False)

print(f"✅ Final production output saved to {final_output_path}")
print(final_output.head())



