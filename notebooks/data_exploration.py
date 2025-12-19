import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use("default")
pd.set_option("display.max_columns", 50)

data_path = "../data/processed/traffic_with_all_features.csv"
df = pd.read_csv(data_path, parse_dates=["datetime"])

print("Dataset shape:", df.shape)
df.head()

print("Start date:", df["datetime"].min())
print("End date:", df["datetime"].max())

missing = df.isnull().sum()
missing[missing > 0]

df["hour"] = df["datetime"].dt.hour

hourly_speed = df.groupby("hour")["avg_speed"].mean()

plt.figure(figsize=(8,4))
hourly_speed.plot()
plt.xlabel("Hour of Day")
plt.ylabel("Average Speed")
plt.title("Average Speed by Hour")
plt.show()

plt.figure(figsize=(6,4))
sns.scatterplot(
    data=df.sample(2000, random_state=42),
    x="vehicle_count",
    y="avg_speed",
    alpha=0.5
)
plt.title("Vehicle Count vs Speed")
plt.show()

plt.figure(figsize=(6,4))
sns.boxplot(data=df, x="rain_mm", y="avg_speed")
plt.title("Speed Distribution vs Rain")
plt.show()
df["incident_flag"].value_counts(normalize=True)
