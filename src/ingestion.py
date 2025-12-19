import pandas as pd
import os

def load_data():
    '''
    loading all the data
    '''
    traffic = pd.read_csv('data/raw/synthetic_traffic.csv',parse_dates= ['datetime'])
    weather= pd.read_csv('data/raw/weather.csv')
    holidays= pd.read_csv('data/raw/holidays.csv')
    return traffic, weather,holidays

def merge_datasets(traffic,weather,holidays):
    weather.rename(columns={
        'time':'date',
        'tavg':'temperature',
        'prcp':'rain_mm',
        'wspd':'wind_kph'
    },inplace= True)

    weather["date"] = pd.to_datetime(weather["date"], format="%d-%m-%Y").dt.date
    traffic['date'] = traffic['datetime'].dt.date
    merged = pd.merge(traffic,weather, on= 'date', how ='left')

    holidays["date"] = pd.to_datetime(holidays["date"]).dt.date
    holidays['is_holiday'] =1

    merged = pd.merge(
        merged,
        holidays[['date','is_holiday']],
        on= 'date',
        how= 'left'
    )
    
    merged['is_holiday'] = merged['is_holiday'].fillna(0).astype(int)
    return merged

def minimal_cleaning(df):
    # Weather proxies
    df['temperature'] = df['temperature'].fillna(df['temperature'].mean())
    df['rain_mm']= df['rain_mm'].fillna(0)
    df['wind_kph']= df['wind_kph'].fillna(15)
    if "humidity" not in df.columns:
        df["humidity"] = 59
    else:
        df["humidity"] = df["humidity"].fillna(59)

    df['synthetic_flag']= df['synthetic_flag']. astype(bool)
    df['road_type']= df['road_type'].astype(str)

    df.drop_duplicates(subset =['datetime', 'road_id'],inplace = True)
    return df

def main():
    os.makedirs('data/processes',exist_ok= True)

    traffic,weather,holidays = load_data()
    merged= merge_datasets(traffic, weather, holidays)
    cleaned = minimal_cleaning(merged)
    
    cleaned.to_csv('data/processed/traffic_merged',index = False)
    print("✅ Processed dataset saved to data/processed/traffic_merged.csv")


if __name__=='__main__':
    main()