import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import shap

pred_path = "../outputs/final_traffic_predictions.csv"
pred_df = pd.read_csv(pred_path, parse_dates=["datetime"])
pred_df.head()

pred_df["congestion_level"].value_counts(normalize=True)

pred_df["hour"] = pred_df["datetime"].dt.hour

plt.figure(figsize=(8,4))
sns.countplot(
    data=pred_df,
    x="hour",
    hue="congestion_level"
)
plt.title("Congestion Level by Hour")
plt.show()

pred_df["congestion_cause"].value_counts()

top_drivers = pred_df["primary_driver"].value_counts().head(10)

plt.figure(figsize=(8,4))
top_drivers.plot(kind="bar")
plt.title("Top SHAP Drivers Across Predictions")
plt.ylabel("Frequency")
plt.show()

pred_df["data_type"].value_counts(normalize=True)
