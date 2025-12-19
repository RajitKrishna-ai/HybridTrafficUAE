import pandas as pd
import os

# Define holidays and minor events
holidays = [
    # 2025 Official Public Holidays
    {"date": "2025-01-01", "event_name": "New Year", "event_type": "Public Holiday", "minor_event": False},
    {"date": "2025-03-30", "event_name": "Eid Al Fitr 1", "event_type": "Public Holiday", "minor_event": False},
    {"date": "2025-03-31", "event_name": "Eid Al Fitr 2", "event_type": "Public Holiday", "minor_event": False},
    {"date": "2025-06-28", "event_name": "Eid Al-Adha", "event_type": "Public Holiday", "minor_event": False},
    {"date": "2025-06-29", "event_name": "Eid Al-Adha Holiday", "event_type": "Public Holiday", "minor_event": False},
    {"date": "2025-12-02", "event_name": "UAE National Day", "event_type": "Public Holiday", "minor_event": False},
    {"date": "2025-12-03", "event_name": "UAE National Day Holiday", "event_type": "Public Holiday", "minor_event": False},
    
    # 2025 Minor/Other Events (optional)
    {"date": "2025-02-14", "event_name": "Valentine's Day", "event_type": "Minor Event", "minor_event": True},
    {"date": "2025-03-21", "event_name": "Mother's Day (Eastern)", "event_type": "Minor Event", "minor_event": True},

    # 2026 Official Public Holidays
    {"date": "2026-01-01", "event_name": "New Year", "event_type": "Public Holiday", "minor_event": False},
    {"date": "2026-02-17", "event_name": "Eid Al Fitr 1", "event_type": "Public Holiday", "minor_event": False},
    {"date": "2026-02-18", "event_name": "Eid Al Fitr 2", "event_type": "Public Holiday", "minor_event": False},
    {"date": "2026-07-17", "event_name": "Eid Al-Adha", "event_type": "Public Holiday", "minor_event": False},
    {"date": "2026-07-18", "event_name": "Eid Al-Adha Holiday", "event_type": "Public Holiday", "minor_event": False},
    {"date": "2026-12-02", "event_name": "UAE National Day", "event_type": "Public Holiday", "minor_event": False},
    {"date": "2026-12-03", "event_name": "UAE National Day Holiday", "event_type": "Public Holiday", "minor_event": False},
]

# Convert to DataFrame
df = pd.DataFrame(holidays)

# Convert date to ISO format (YYYY-MM-DD)
df['date'] = pd.to_datetime(df['date']).dt.date

# Ensure folder exists
os.makedirs(os.path.join("data", "raw"), exist_ok=True)

# Save CSV
output_file = os.path.join("data", "raw", "uae_holidays.csv")
df.to_csv(output_file, index=False)

print(f"✅ UAE holidays CSV saved at {output_file}")
print(df)
