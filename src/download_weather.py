
import os
import pandas as pd
from datetime import datetime
from meteostat import Point, Daily

# -----------------------------
# Settings
# -----------------------------
# Dubai coordinates
lat, lon = 25.276987, 55.296249
dubai = Point(lat, lon)

# Time range
start = datetime(2025, 1, 1)
end = datetime(2025, 12, 31)

# Output file path
output_path = os.path.join("data", "raw", "weather.csv")

# -----------------------------
# Fetch weather data
# -----------------------------
data = Daily(dubai, start, end).fetch()

# Keep only useful columns
data = data[['tavg','tmin','tmax','prcp','wspd','pres','tsun']]
data.reset_index(inplace=True)

# -----------------------------
# Save to CSV
# -----------------------------
# Ensure folder exists (safe)
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# Save CSV
data.to_csv(output_path, index=False)
print(f"✅ Weather data saved to: {output_path}")
