import numpy as np
import pandas as pd
import random
from datetime import datetime, timedelta

# Function to generate random weather conditions
def generate_weather(season):
    if season == 'Winter':
        weather_conditions = ['Snowy', 'Cloudy', 'Sunny']
    elif season == 'Summer':
        weather_conditions = ['Sunny', 'Cloudy', 'Rainy']
    elif season == 'Spring':
        weather_conditions = ['Rainy', 'Cloudy', 'Sunny']
    else:  # Autumn
        weather_conditions = ['Cloudy', 'Rainy', 'Sunny']
    return random.choice(weather_conditions)

# Function to determine the season based on the month
def get_season(month):
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    else:
        return 'Autumn'

# Function to simulate appliance usage data
def generate_data(num_days=30):
    data = []
    start_date = datetime.now()

    appliances = ['HVAC', 'Washing Machine', 'Refrigerator', 'Lights', 'Oven']

    for day in range(num_days):
        date = start_date + timedelta(days=day)
        season = get_season(date.month)
        is_weekend = date.weekday() >= 5

        for hour in range(24):
            time_of_day = hour
            weather = generate_weather(season)
            num_occupants = random.randint(1, 5)

            for appliance in appliances:
                # Base usage per appliance
                base_usage = {
                    'HVAC': random.uniform(1.0, 3.0) if season in ['Winter', 'Summer'] else random.uniform(0.5, 1.5),
                    'Washing Machine': random.uniform(0.5, 1.0) if hour in [7, 19] else 0,
                    'Refrigerator': random.uniform(0.8, 1.2),
                    'Lights': random.uniform(0.2, 0.8) if 18 <= hour <= 23 else random.uniform(0.1, 0.3),
                    'Oven': random.uniform(0.5, 1.5) if hour in [12, 18] else 0
                }

                # Add randomness and anomalies
                anomaly = random.uniform(0, 2) if random.random() < 0.01 else 0
                occupant_factor = num_occupants * random.uniform(0.05, 0.2)
                weather_factor = 0.5 if weather in ['Rainy', 'Snowy'] else 0.2

                power_usage = base_usage[appliance] + occupant_factor + weather_factor + anomaly

                data.append({
                    'Date': date.strftime('%Y-%m-%d'),
                    'Hour': hour,
                    'Appliance': appliance,
                    'Weather': weather,
                    'Season': season,
                    'Is_Weekend': is_weekend,
                    'Num_Occupants': num_occupants,
                    'Power_Usage_kWh': round(power_usage, 2)
                })

    return pd.DataFrame(data)

# Generate data and save to CSV
data = generate_data(num_days=365)
data.to_csv('enhanced_household_power_usage.csv', index=False)
print("Enhanced data generation complete. File saved as 'enhanced_household_power_usage.csv'.")