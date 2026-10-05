"""
congestion.py
-------------
Simple congestion classification: Low / Moderate / High.

Idea
    1. Build a Congestion_Score between 0 and 1.
         - high traffic volume  -> higher score
         - low average speed    -> higher score   (only if Average_Speed exists)
       When both columns exist, the score is the average of the two parts.
    2. Compute the thresholds FROM THE DATA using the mean and standard deviation:
         Low       : score <= mean - 0.5 * std
         High      : score >= mean + 0.5 * std
         Moderate  : everything in between
    Because the thresholds come from the dataset, the labels are RELATIVE
    ("high compared with the rest of this dataset"), not absolute traffic-engineering limits.
"""

import numpy as np
import pandas as pd

LEVELS = ["Low", "Moderate", "High"]
LEVEL_LABELS = {"Low": "🟢 Low", "Moderate": "🟡 Moderate", "High": "🔴 High"}
LEVEL_COLORS = {"Low": "#2ECC71", "Moderate": "#F1C40F", "High": "#E74C3C"}

STD_MULTIPLIER = 0.5  # how many standard deviations away from the mean define Low / High


def _scale_0_to_1(series):
    """Min-max scaling. The 1st and 99th percentiles are used so extreme outliers do not squash the scale."""
    low, high = series.quantile([0.01, 0.99])
    if high == low:
        return pd.Series(0.5, index=series.index)
    return (series.clip(low, high) - low) / (high - low)


def compute_congestion_score(df):
    """Return a Series (0 = free flowing, 1 = very congested)."""
    parts = []
    if "Traffic_Volume" in df.columns:
        parts.append(_scale_0_to_1(df["Traffic_Volume"]))
    if "Average_Speed" in df.columns:
        parts.append(1 - _scale_0_to_1(df["Average_Speed"]))
    if not parts:
        raise ValueError("Congestion analysis needs Traffic_Volume and/or Average_Speed.")
    return sum(parts) / len(parts)


def mean_std_thresholds(values):
    """Low / High cut-offs calculated from the mean and standard deviation."""
    mean, std = values.mean(), values.std()
    return mean - STD_MULTIPLIER * std, mean + STD_MULTIPLIER * std


def classify_values(values, low_cutoff, high_cutoff):
    """Turn numbers into the labels Low / Moderate / High."""
    labels = np.select([values <= low_cutoff, values >= high_cutoff], ["Low", "High"], default="Moderate")
    return pd.Categorical(labels, categories=LEVELS, ordered=True)


def classify_single_value(value, low_cutoff, high_cutoff):
    """Label for one number (used by the prediction page)."""
    if value <= low_cutoff:
        return "Low"
    if value >= high_cutoff:
        return "High"
    return "Moderate"


def add_congestion_columns(df):
    """Add Congestion_Score and Congestion_Level. Returns (new_df, info_dict)."""
    df = df.copy()
    df["Congestion_Score"] = compute_congestion_score(df)
    low_cutoff, high_cutoff = mean_std_thresholds(df["Congestion_Score"])
    df["Congestion_Level"] = classify_values(df["Congestion_Score"], low_cutoff, high_cutoff)

    # Thresholds based only on traffic volume (used to label model predictions)
    volume_low, volume_high = mean_std_thresholds(df["Traffic_Volume"])

    uses_speed = "Average_Speed" in df.columns
    info = {
        "low_cutoff": float(low_cutoff),
        "high_cutoff": float(high_cutoff),
        "volume_low_cutoff": float(volume_low),
        "volume_high_cutoff": float(volume_high),
        "uses_speed": uses_speed,
        "method": ("traffic volume and average speed" if uses_speed else "traffic volume only"),
    }
    return df, info


def congestion_distribution(df):
    """Count and percentage of records in each congestion level."""
    counts = df["Congestion_Level"].value_counts().reindex(LEVELS, fill_value=0)
    return pd.DataFrame({
        "Level": LEVELS,
        "Records": counts.values,
        "Percentage": (counts.values / max(len(df), 1) * 100).round(1),
    })


def location_congestion_share(df):
    """Percentage of Low / Moderate / High records for each location (sorted by High %)."""
    if "Location" not in df.columns:
        return pd.DataFrame()
    shares = pd.crosstab(df["Location"], df["Congestion_Level"], normalize="index") * 100
    shares = shares.reindex(columns=LEVELS, fill_value=0)
    return shares.sort_values("High", ascending=False)


def location_congestion_table(df):
    """Summary table per location, most congested first."""
    if "Location" not in df.columns:
        return pd.DataFrame()
    shares = location_congestion_share(df)
    grouped = df.groupby("Location")
    table = pd.DataFrame({"Average Traffic Volume": grouped["Traffic_Volume"].mean().round(0)})
    if "Average_Speed" in df.columns:
        table["Average Speed"] = grouped["Average_Speed"].mean().round(1)
    table["Average Congestion Score"] = grouped["Congestion_Score"].mean().round(3)
    table["High Congestion (%)"] = shares["High"].reindex(table.index).round(1)
    return table.sort_values("High Congestion (%)", ascending=False).reset_index()


def hourly_high_congestion_share(df):
    """Percentage of records with High congestion for every hour of the day."""
    if "Hour" not in df.columns:
        return pd.Series(dtype=float)
    return df.groupby("Hour")["Congestion_Level"].apply(lambda levels: (levels == "High").mean() * 100)
