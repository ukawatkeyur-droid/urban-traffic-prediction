"""
analysis.py
-----------
Statistics, peak analysis, correlation analysis and automatic insights.

Every sentence produced here is built from numbers calculated on the
dataset - nothing is written by hand in advance.
"""

import numpy as np
import pandas as pd

from modules.data_processing import DAY_ORDER, NUMERIC_COLUMNS

RAIN_BINS = [-np.inf, 0, 2.5, 7.6, np.inf]
RAIN_LABELS = ["No rain", "Light rain", "Moderate rain", "Heavy rain"]


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------
def format_hour(hour):
    """8 -> '8 AM', 0 -> '12 AM', 17 -> '5 PM'."""
    hour = int(hour)
    suffix = "AM" if hour < 12 else "PM"
    display = hour % 12 or 12
    return f"{display} {suffix}"


def format_hour_range(start, end):
    """Hours 8 to 9 (inclusive) -> '08:00-10:00'."""
    return f"{start:02d}:00-{(end + 1) % 24:02d}:00"


def percent_change(new_value, old_value):
    """Percentage difference of new_value compared with old_value."""
    if old_value == 0 or pd.isna(old_value):
        return np.nan
    return (new_value - old_value) / old_value * 100


def find_busy_windows(hourly_average, share=0.9):
    """Groups of consecutive hours whose average is at least `share` x the peak average."""
    busy_hours = sorted(hourly_average.index[hourly_average >= share * hourly_average.max()])
    windows = []
    for hour in busy_hours:
        if windows and hour == windows[-1][1] + 1:
            windows[-1][1] = hour
        else:
            windows.append([hour, hour])
    return [(start, end) for start, end in windows]


def numeric_columns_available(df):
    """Numeric columns from the standard list that exist in the dataset."""
    return [column for column in NUMERIC_COLUMNS if column in df.columns]


# --------------------------------------------------------------------------
# Descriptive statistics
# --------------------------------------------------------------------------
def descriptive_statistics(series):
    """Mean, median, mode, min, max, range, std, variance and quartiles of one column."""
    values = series.dropna()
    modes = values.mode()
    q1, q3 = values.quantile(0.25), values.quantile(0.75)
    return {
        "Mean": values.mean(),
        "Median": values.median(),
        "Mode": modes.iloc[0] if len(modes) else np.nan,
        "Minimum": values.min(),
        "Maximum": values.max(),
        "Range": values.max() - values.min(),
        "Std Deviation": values.std(),
        "Variance": values.var(),
        "Q1 (25%)": q1,
        "Q3 (75%)": q3,
        "IQR": q3 - q1,
    }


def statistics_table(df, columns):
    """One row per column, one column per statistic."""
    table = pd.DataFrame({column: descriptive_statistics(df[column]) for column in columns}).T
    return table.round(2)


def explain_statistics(df):
    """Plain-language sentences about Traffic_Volume, built from the calculated statistics."""
    stats = descriptive_statistics(df["Traffic_Volume"])
    mean, median, std = stats["Mean"], stats["Median"], stats["Std Deviation"]
    sentences = []

    if mean > median * 1.05:
        shape = (f"The mean ({mean:,.0f}) is higher than the median ({median:,.0f}), so the data is "
                 "right-skewed: a few very busy periods pull the average up.")
    elif mean < median * 0.95:
        shape = (f"The mean ({mean:,.0f}) is lower than the median ({median:,.0f}), so the data is "
                 "left-skewed: a few very quiet periods pull the average down.")
    else:
        shape = (f"The mean ({mean:,.0f}) and median ({median:,.0f}) are close, so the distribution "
                 "is fairly symmetric.")
    sentences.append(shape)

    variation = std / mean * 100 if mean else np.nan
    level = "low" if variation < 25 else "moderate" if variation < 50 else "high"
    sentences.append(
        f"The standard deviation is {std:,.0f} ({variation:.0f}% of the mean), which shows {level} variability "
        "in traffic volume between records."
    )
    sentences.append(
        f"The middle 50% of records lie between {stats['Q1 (25%)']:,.0f} and {stats['Q3 (75%)']:,.0f} vehicles "
        f"(IQR = {stats['IQR']:,.0f})."
    )

    if "Is_Weekend" in df.columns and df["Is_Weekend"].nunique() == 2:
        weekday = df.loc[df["Is_Weekend"] == 0, "Traffic_Volume"].mean()
        weekend = df.loc[df["Is_Weekend"] == 1, "Traffic_Volume"].mean()
        change = percent_change(weekend, weekday)
        direction = "lower" if change < 0 else "higher"
        sentences.append(
            f"Average weekday traffic is {weekday:,.0f} versus {weekend:,.0f} at weekends "
            f"({abs(change):.0f}% {direction} at weekends)."
        )
    return sentences


# --------------------------------------------------------------------------
# Group averages used by the charts and the peak analysis
# --------------------------------------------------------------------------
def hourly_average(df):
    return df.groupby("Hour")["Traffic_Volume"].mean()


def daily_average(df):
    return df.groupby("Day")["Traffic_Volume"].mean().reindex(DAY_ORDER).dropna()


def location_average(df):
    return df.groupby("Location")["Traffic_Volume"].mean().sort_values(ascending=False)


def add_rain_category(df):
    """Add a Rain_Category column (No / Light / Moderate / Heavy) from the Rainfall column."""
    df = df.copy()
    df["Rain_Category"] = pd.cut(df["Rainfall"], bins=RAIN_BINS, labels=RAIN_LABELS).astype(str)
    return df


def weather_impact_table(df):
    """Average traffic per weather condition (or per rain category if Weather is missing).

    Returns (table, grouping_name) or (None, None) when no weather information exists.
    """
    if "Weather" in df.columns:
        group_column, source = "Weather", df
    elif "Rainfall" in df.columns:
        group_column, source = "Rain_Category", add_rain_category(df)
    else:
        return None, None
    grouped = source.groupby(group_column)["Traffic_Volume"]
    table = pd.DataFrame({"Average Traffic Volume": grouped.mean().round(1), "Records": grouped.size()})
    if group_column == "Rain_Category":
        table = table.reindex([label for label in RAIN_LABELS if label in table.index])
    else:
        table = table.sort_values("Average Traffic Volume", ascending=False)
    return table, group_column.replace("_", " ")


def get_peak_summary(df):
    """Peak / lowest hour and day, most / least congested location. Missing parts are left out."""
    summary = {}
    if "Hour" in df.columns:
        hourly = hourly_average(df)
        summary["peak_hour"] = int(hourly.idxmax())
        summary["peak_hour_volume"] = float(hourly.max())
        summary["lowest_hour"] = int(hourly.idxmin())
        summary["lowest_hour_volume"] = float(hourly.min())
    if "Day" in df.columns:
        daily = daily_average(df)
        if len(daily):
            summary["peak_day"] = daily.idxmax()
            summary["peak_day_volume"] = float(daily.max())
            summary["lowest_day"] = daily.idxmin()
            summary["lowest_day_volume"] = float(daily.min())
    if "Location" in df.columns:
        metric = "Congestion_Score" if "Congestion_Score" in df.columns else "Traffic_Volume"
        by_location = df.groupby("Location")[metric].mean()
        summary["most_congested_location"] = by_location.idxmax()
        summary["least_congested_location"] = by_location.idxmin()
        summary["location_metric"] = "average congestion score" if metric == "Congestion_Score" else "average traffic volume"
    return summary


# --------------------------------------------------------------------------
# Correlation analysis
# --------------------------------------------------------------------------
def get_correlation_matrix(df):
    """Correlation matrix of the numeric columns that exist and actually vary."""
    candidates = [c for c in ["Traffic_Volume", "Average_Speed", "Temperature", "Rainfall", "Accidents", "Is_Weekend"]
                  if c in df.columns]
    usable = [c for c in candidates if df[c].nunique() > 1]
    if len(usable) < 2 or "Traffic_Volume" not in usable:
        return pd.DataFrame()
    return df[usable].corr()


def describe_strength(r):
    size = abs(r)
    if size >= 0.7:
        return "strong"
    if size >= 0.4:
        return "moderate"
    if size >= 0.2:
        return "weak"
    return "very weak"


def explain_correlations(corr):
    """Automatic explanation of the strongest correlations."""
    if corr.empty:
        return ["Not enough numeric columns with variation to calculate correlations."]
    sentences = []
    with_volume = corr["Traffic_Volume"].drop("Traffic_Volume").sort_values(key=abs, ascending=False)
    for column, r in with_volume.items():
        direction = "positive" if r > 0 else "negative"
        meaning = "increase together" if r > 0 else "move in opposite directions"
        sentences.append(
            f"Traffic Volume and {column.replace('_', ' ')} show a {describe_strength(r)} {direction} "
            f"correlation (r = {r:.2f}): they tend to {meaning}."
        )

    # Strongest pair in the whole matrix (upper triangle only, so each pair appears once)
    pairs = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool)).stack()
    if len(pairs):
        strongest_pair = pairs.abs().idxmax()      # e.g. ("Traffic_Volume", "Average_Speed")
        first, second = strongest_pair
        r = pairs.loc[strongest_pair]
        sentences.append(
            f"The strongest relationship in the dataset is between {first.replace('_', ' ')} and "
            f"{second.replace('_', ' ')} (r = {r:.2f})."
        )
    if "Temperature" in corr.columns:
        sentences.append("Temperature follows the daily cycle (cooler at night, warmer by day), so its correlation "
                         "with traffic may simply reflect the time of day rather than a direct effect.")
    sentences.append("Correlation measures how two variables move together; it does not prove that one causes the other.")
    return sentences


# --------------------------------------------------------------------------
# Automatic insights
# --------------------------------------------------------------------------
def relative_traffic(df):
    """Traffic divided by what is normal for the same location, hour and day type.

    A value of 1.0 means 'normal'. This removes the effect of rush hours and location
    so we can compare, for example, rainy and dry periods more fairly.
    """
    group_columns = [c for c in ["Location", "Hour", "Is_Weekend"] if c in df.columns]
    if group_columns:
        expected = df.groupby(group_columns)["Traffic_Volume"].transform("mean")
    else:
        expected = pd.Series(df["Traffic_Volume"].mean(), index=df.index)
    return df["Traffic_Volume"] / expected.replace(0, np.nan)


def generate_insights(df):
    """Return a list of {'category': ..., 'text': ...} dictionaries built from the data."""
    insights = []

    def add(category, text):
        insights.append({"category": category, "text": text})

    volume = "Traffic_Volume"

    # ---- Time of day ----
    if "Hour" in df.columns and df["Hour"].nunique() > 1:
        hourly = hourly_average(df)
        windows = find_busy_windows(hourly)
        window_text = " and ".join(format_hour_range(start, end) for start, end in windows)
        add("Time of day",
            f"Traffic is highest during {window_text} (hours within 10% of the peak). The single busiest hour is "
            f"{format_hour(hourly.idxmax())} with an average of {hourly.max():,.0f} vehicles.")
        add("Time of day",
            f"Traffic is lowest around {format_hour(hourly.idxmin())}, averaging {hourly.min():,.0f} vehicles "
            f"({hourly.max() / max(hourly.min(), 1):.1f} times lower than the peak hour).")

    # ---- Days ----
    if "Day" in df.columns:
        daily = daily_average(df)
        if len(daily) > 1:
            add("Days", f"{daily.idxmax()} is the busiest day ({daily.max():,.0f} vehicles on average) and "
                        f"{daily.idxmin()} is the quietest ({daily.min():,.0f}).")
    if "Is_Weekend" in df.columns and df["Is_Weekend"].nunique() == 2:
        weekday = df.loc[df["Is_Weekend"] == 0, volume].mean()
        weekend = df.loc[df["Is_Weekend"] == 1, volume].mean()
        change = percent_change(weekend, weekday)
        word = "lower" if change < 0 else "higher"
        add("Days", f"Weekend traffic is {abs(change):.1f}% {word} than weekday traffic "
                    f"({weekend:,.0f} vs {weekday:,.0f} vehicles on average).")

    # ---- Locations ----
    if "Location" in df.columns and df["Location"].nunique() > 1:
        by_location = location_average(df)
        add("Locations", f"{by_location.index[0]} has the highest average traffic ({by_location.iloc[0]:,.0f} vehicles), "
                         f"while {by_location.index[-1]} has the lowest ({by_location.iloc[-1]:,.0f}).")
        if "Congestion_Level" in df.columns:
            high_share = df.groupby("Location")["Congestion_Level"].apply(lambda s: (s == "High").mean() * 100)
            add("Locations", f"{high_share.idxmax()} spends the largest share of time in High congestion "
                             f"({high_share.max():.1f}% of its records).")

    # ---- Weather ----
    traffic_index = relative_traffic(df)
    if "Rainfall" in df.columns:
        rainy = df["Rainfall"] > 0
        if rainy.sum() >= 10 and (~rainy).sum() >= 10:
            change = (traffic_index[rainy].mean() - traffic_index[~rainy].mean()) * 100
            if abs(change) < 2:
                add("Weather", "After adjusting for location and time of day, traffic during rain is about the same "
                               f"as in dry periods (difference {change:+.1f}%).")
            else:
                word = "lower" if change < 0 else "higher"
                add("Weather", f"After adjusting for location and time of day, traffic in periods with rainfall is "
                               f"about {abs(change):.1f}% {word} than in dry periods (association, not proof of cause).")
            heavy_limit = df.loc[rainy, "Rainfall"].quantile(0.75)
            heavy = df["Rainfall"] >= heavy_limit
            if heavy.sum() >= 10:
                heavy_change = (traffic_index[heavy].mean() - traffic_index[~rainy].mean()) * 100
                word = "lower" if heavy_change < 0 else "higher"
                add("Weather", f"Heavy rainfall (top 25% of rainy records, at least {heavy_limit:.1f}) is associated with "
                               f"{abs(heavy_change):.1f}% {word} traffic than dry periods.")
    if "Weather" in df.columns and df["Weather"].nunique() > 1:
        by_weather = df.groupby("Weather")[volume].mean()
        add("Weather", f"Without any adjustment, the weather condition with the highest average traffic is "
                       f"{by_weather.idxmax()} ({by_weather.max():,.0f}) and the lowest is {by_weather.idxmin()} "
                       f"({by_weather.min():,.0f}). This raw comparison is also affected by WHEN each weather "
                       "type occurs (for example rain during rush hour).")

    # ---- Speed ----
    if "Average_Speed" in df.columns:
        r = df[volume].corr(df["Average_Speed"])
        if not pd.isna(r):
            word = "falls" if r < 0 else "rises"
            add("Speed", f"Average speed generally {word} as traffic volume increases (correlation r = {r:.2f}).")
        if "Hour" in df.columns and df["Hour"].nunique() > 1:
            hourly = hourly_average(df)
            speed_by_hour = df.groupby("Hour")["Average_Speed"].mean()
            add("Speed", f"Average speed is {speed_by_hour[hourly.idxmax()]:.1f} at the busiest hour "
                         f"({format_hour(hourly.idxmax())}) compared with {speed_by_hour[hourly.idxmin()]:.1f} at the "
                         f"quietest hour ({format_hour(hourly.idxmin())}).")

    # ---- Accidents ----
    if "Accidents" in df.columns and df["Accidents"].sum() > 0:
        total = int(df["Accidents"].sum())
        text = f"The dataset records {total:,} accidents in total."
        if "Hour" in df.columns:
            by_hour = df.groupby("Hour")["Accidents"].sum()
            text += f" Most occur around {format_hour(by_hour.idxmax())} ({int(by_hour.max())} accidents)."
        if "Location" in df.columns:
            by_location = df.groupby("Location")["Accidents"].sum()
            text += f" {by_location.idxmax()} has the most ({int(by_location.max())})."
        add("Accidents", text)
        r = df[volume].corr(df["Accidents"])
        if not pd.isna(r):
            add("Accidents", f"Accidents have a {describe_strength(r)} {'positive' if r > 0 else 'negative'} "
                             f"correlation with traffic volume (r = {r:.2f}).")

    # ---- Congestion ----
    if "Congestion_Level" in df.columns:
        shares = df["Congestion_Level"].value_counts(normalize=True) * 100
        add("Congestion", f"{shares.get('High', 0):.1f}% of records are High congestion, "
                          f"{shares.get('Moderate', 0):.1f}% Moderate and {shares.get('Low', 0):.1f}% Low "
                          "(relative to this dataset).")
        if "Hour" in df.columns:
            high_by_hour = df.groupby("Hour")["Congestion_Level"].apply(lambda s: (s == "High").mean() * 100)
            add("Congestion", f"High congestion is most common at {format_hour(high_by_hour.idxmax())} "
                              f"({high_by_hour.max():.0f}% of records at that hour).")

    # ---- Trend over time ----
    if "Date" in df.columns and df["Date"].nunique() >= 7:
        daily_series = df.groupby("Date")[volume].mean()
        slope = np.polyfit(np.arange(len(daily_series)), daily_series.values, 1)[0]
        total_change = slope * (len(daily_series) - 1) / daily_series.mean() * 100
        if abs(total_change) < 3:
            add("Trend", f"Daily average traffic is fairly stable over the {len(daily_series)} days "
                         f"(fitted change {total_change:+.1f}%).")
        else:
            word = "an upward" if total_change > 0 else "a downward"
            add("Trend", f"Daily average traffic shows {word} trend over the {len(daily_series)} days "
                         f"(fitted change {total_change:+.1f}%).")

    if not insights:
        add("General", "Not enough columns were found to generate detailed insights.")
    return insights
