import pandas as pd
import numpy as np
import os

# Create folder if it doesn't exist
os.makedirs("data/raw", exist_ok=True)

# Generate datetime index (hourly for 1 month)
dates = pd.date_range(start="2025-11-01", end="2025-11-30 23:00:00", freq="H")

# Example road segments
roads = [
    {"road_id": "R1", "lane_count": 4, "road_type": "Highway"},
    {"road_id": "R2", "lane_count": 3, "road_type": "Arterial"},
    {"road_id": "R3", "lane_count": 2, "road_type": "Local"},
    {"road_id": "R4", "lane_count": 3, "road_type": "Arterial"},
    {"road_id": "R5", "lane_count": 2, "road_type": "Local"}
]

# Generate synthetic traffic data
rows = []
for r in roads:
    for dt in dates:
        hour = dt.hour
        # Base speed per road type
        base_speed = {"Highway": 100, "Arterial": 60, "Local": 40}[r["road_type"]]
        # Peak hours: 7-10 AM & 5-8 PM slower traffic
        peak_factor = 0.6 if (7 <= hour <= 10 or 17 <= hour <= 20) else 1.0
        # Add some random daily traffic variation
        daily_variation = np.random.normal(0, 5)
        avg_speed = base_speed * peak_factor + daily_variation
        vehicle_count = int(100 * peak_factor + np.random.randint(0, 50))
        rows.append({
            "datetime": dt,
            "road_id": r["road_id"],
            "avg_speed": max(avg_speed, 5),  # minimum speed 5 km/h
            "vehicle_count": vehicle_count,
            "lane_count": r["lane_count"],
            "road_type": r["road_type"],
            "synthetic_flag": True
        })

# Convert to DataFrame
synthetic_traffic = pd.DataFrame(rows)

# Save as CSV
synthetic_traffic.to_csv("data/raw/synthetic_traffic.csv", index=False)

print("Synthetic traffic CSV for full month created successfully at data/raw/synthetic_traffic.csv")
print(synthetic_traffic.head())
