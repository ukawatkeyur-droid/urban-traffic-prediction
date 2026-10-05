"""
generate_data.py
----------------
Creates the sample dataset  data/traffic_data.csv  used by the dashboard.

The data is SYNTHETIC but built from realistic rules instead of pure randomness:
  * two rush-hour peaks on weekdays (morning and evening), quiet nights
  * a flatter, lower pattern on weekends (different for each location)
  * every location has its own capacity, free-flow speed and peak times
  * rain reduces traffic volume and vehicle speed, and raises accident risk
  * speed falls as traffic approaches the road capacity
  * a few missing values, duplicate rows and invalid values are added on purpose
    so that the cleaning step of the project has something real to do

Run it (from the project folder) with:
    python generate_data.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
START_DATE = "2025-09-01"  # a Monday
NUMBER_OF_DAYS = 21        # three full weeks
OUTPUT_PATH = Path(__file__).parent / "data" / "traffic_data.csv"

# Characteristics of each location
#   base      : approximate maximum vehicles per hour on the road
#   speed     : free-flow speed (km/h) when the road is empty
#   rush      : how strongly the location reacts to weekday rush hours (0-1)
#   weekend   : weekend traffic as a fraction of weekday traffic
#   morning / evening : hour of the morning / evening peak
LOCATIONS = {
    "Central Business District": dict(base=2400, speed=45, rush=1.00, weekend=0.60, morning=9.0, evening=18.0),
    "Railway Station Road":      dict(base=2000, speed=40, rush=0.95, weekend=0.80, morning=8.0, evening=18.5),
    "University Avenue":         dict(base=1500, speed=50, rush=0.85, weekend=0.45, morning=9.0, evening=16.5),
    "Highway Junction":          dict(base=3000, speed=70, rush=0.80, weekend=0.90, morning=8.0, evening=17.5),
    "Residential Ring Road":     dict(base=1100, speed=50, rush=0.70, weekend=0.85, morning=7.5, evening=19.0),
}

VEHICLE_TYPES = ["Two-Wheeler", "Car", "Bus", "Truck", "Auto Rickshaw"]


def bump(hour, centre, width):
    """A smooth bell-shaped curve centred on `centre` (used to build rush-hour peaks)."""
    return np.exp(-0.5 * ((hour - centre) / width) ** 2)


def hourly_profile(hour, is_weekend, loc):
    """Relative traffic level (about 0 to 1) for one hour at one location."""
    if is_weekend:
        profile = 0.08 + 0.50 * bump(hour, 13.0, 3.5) + 0.40 * bump(hour, 19.0, 2.5)
        return profile * loc["weekend"] / 0.8
    morning_peak = 0.95 * bump(hour, loc["morning"], 1.3)
    evening_peak = 0.90 * bump(hour, loc["evening"], 1.7)
    midday = 0.40 * bump(hour, 13.0, 2.8)
    return 0.07 + loc["rush"] * (morning_peak + evening_peak) + midday


def build_weather(rng, timestamps):
    """City-wide hourly weather: rain falls in 'episodes' on rainy days."""
    states = rng.choice(
        ["Clear", "Cloudy", "Drizzle", "Rain", "Storm"],
        size=NUMBER_OF_DAYS,
        p=[0.40, 0.27, 0.13, 0.15, 0.05],
    )
    rows = []
    for day_index in range(NUMBER_OF_DAYS):
        state = states[day_index]
        episode_start = rng.integers(5, 17)
        episode_length = rng.integers(4, 11)
        for hour in range(24):
            raining = state in ("Drizzle", "Rain", "Storm") and episode_start <= hour < episode_start + episode_length
            if state == "Drizzle" and raining:
                rain_mm, label = rng.uniform(0.2, 2.0), "Drizzle"
            elif state == "Rain" and raining:
                rain_mm, label = rng.uniform(2.0, 9.0), "Rain"
            elif state == "Storm" and raining:
                rain_mm, label = rng.uniform(9.0, 25.0), "Storm"
            elif state in ("Clear", "Cloudy"):
                rain_mm, label = 0.0, state
            else:
                rain_mm, label = 0.0, "Cloudy"  # dry hours of a rainy day
            daily_temp = 27 + 4.5 * np.sin((hour - 9) / 24 * 2 * np.pi)
            temperature = daily_temp - (2.5 if rain_mm > 0 else 0) + rng.normal(0, 0.8)
            rows.append((label, round(float(temperature), 1), round(float(rain_mm), 1)))
    weather = pd.DataFrame(rows, columns=["Weather", "Temperature", "Rainfall"])
    weather.index = timestamps
    return weather


def generate():
    rng = np.random.default_rng(SEED)
    timestamps = pd.date_range(START_DATE, periods=NUMBER_OF_DAYS * 24, freq="h")
    weather = build_weather(rng, timestamps)

    records = []
    for timestamp in timestamps:
        hour = timestamp.hour
        is_weekend = timestamp.dayofweek >= 5
        rain = weather.loc[timestamp, "Rainfall"]

        for location, loc in LOCATIONS.items():
            profile = hourly_profile(hour, is_weekend, loc)
            rain_effect = 1 - 0.012 * min(rain, 12)           # heavier rain -> fewer vehicles
            noise = rng.lognormal(mean=0, sigma=0.07)          # natural variation
            volume = loc["base"] * 0.85 * profile * rain_effect * noise
            volume = int(max(volume, 15))

            utilisation = min(volume / (loc["base"] * 1.1), 1.2)  # how full the road is
            speed = loc["speed"] * (1 - 0.60 * utilisation ** 1.6) * (1 - 0.012 * min(rain, 12))
            speed = float(np.clip(speed + rng.normal(0, 2.0), 8, loc["speed"] + 5))

            accident_rate = 0.01 + 0.10 * utilisation ** 2 + 0.015 * min(rain, 10) * (utilisation + 0.2)
            accidents = int(rng.poisson(accident_rate))

            night = hour >= 22 or hour <= 5
            probabilities = [0.20, 0.25, 0.10, 0.40, 0.05] if night else [0.45, 0.33, 0.09, 0.07, 0.06]
            vehicle_type = rng.choice(VEHICLE_TYPES, p=probabilities)

            records.append({
                "Date": timestamp.strftime("%Y-%m-%d"),
                "Time": timestamp.strftime("%H:%M"),
                "Location": location,
                "Day": timestamp.day_name(),
                "Traffic_Volume": volume,
                "Average_Speed": round(speed, 1),
                "Weather": weather.loc[timestamp, "Weather"],
                "Temperature": weather.loc[timestamp, "Temperature"],
                "Rainfall": rain,
                "Vehicle_Type": vehicle_type,
                "Accidents": accidents,
            })

    data = pd.DataFrame(records)

    # ---- Add realistic "mess" so the cleaning step is meaningful ----
    missing_speed = rng.choice(data.index, size=20, replace=False)
    missing_temperature = rng.choice(data.index, size=15, replace=False)
    data.loc[missing_speed, "Average_Speed"] = np.nan
    data.loc[missing_temperature, "Temperature"] = np.nan

    invalid_rows = rng.choice(data.index, size=3, replace=False)
    data.loc[invalid_rows, "Traffic_Volume"] = -1                      # sensor error

    duplicates = data.sample(8, random_state=SEED)
    data = pd.concat([data, duplicates]).sort_values(
        ["Date", "Time", "Location"], kind="stable"
    ).reset_index(drop=True)
    return data


if __name__ == "__main__":
    dataset = generate()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {len(dataset)} rows to {OUTPUT_PATH}")
