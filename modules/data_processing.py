"""
data_processing.py
------------------
Everything related to LOADING and CLEANING the traffic dataset.

Main functions
    load_csv()             -> read a CSV file (path or uploaded file)
    standardize_columns()  -> rename columns such as "traffic volume" to "Traffic_Volume"
    get_data_quality_report() / missing_values_table() -> describe problems in raw data
    clean_data()           -> full cleaning pipeline, returns (clean_df, log)

Only ONE column is mandatory: Traffic_Volume.
All other columns are optional; the rest of the project checks what is available.
"""

import numpy as np
import pandas as pd

DAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
WEEKEND_DAYS = ["Saturday", "Sunday"]

NUMERIC_COLUMNS = ["Traffic_Volume", "Average_Speed", "Temperature", "Rainfall", "Accidents"]
CATEGORICAL_COLUMNS = ["Location", "Weather", "Vehicle_Type"]

# Different spellings that people commonly use -> the standard name used in this project
COLUMN_ALIASES = {
    "date": "Date",
    "time": "Time",
    "hour": "Hour",
    "location": "Location",
    "road": "Location",
    "junction": "Location",
    "traffic_volume": "Traffic_Volume",
    "trafficvolume": "Traffic_Volume",
    "volume": "Traffic_Volume",
    "vehicle_count": "Traffic_Volume",
    "average_speed": "Average_Speed",
    "avg_speed": "Average_Speed",
    "speed": "Average_Speed",
    "weather": "Weather",
    "weather_condition": "Weather",
    "temperature": "Temperature",
    "temp": "Temperature",
    "rainfall": "Rainfall",
    "rain": "Rainfall",
    "day": "Day",
    "day_of_week": "Day",
    "vehicle_type": "Vehicle_Type",
    "accidents": "Accidents",
    "accident": "Accidents",
    "accident_count": "Accidents",
}

# Values that should be treated as "missing" when reading a CSV
MISSING_VALUE_MARKERS = ["", " ", "NA", "N/A", "n/a", "NaN", "nan", "null", "NULL", "None", "-", "?"]

# Obvious invalid values: anything outside (min, max) is removed
VALID_RANGES = {
    "Traffic_Volume": (0, 1_000_000),
    "Average_Speed": (0, 200),
    "Temperature": (-50, 60),
    "Rainfall": (0, 500),
    "Accidents": (0, 1000),
}

_DAY_LOOKUP = {day[:3].lower(): day for day in DAY_ORDER}


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def load_csv(source):
    """Read a CSV from a file path or an uploaded file object.

    Raises ValueError with a friendly message if the file cannot be used.
    """
    try:
        data = pd.read_csv(source, na_values=MISSING_VALUE_MARKERS, skipinitialspace=True)
    except pd.errors.EmptyDataError:
        raise ValueError("The file is empty. Please upload a CSV that contains data.")
    except FileNotFoundError:
        raise ValueError(f"File not found: {source}")
    except Exception as error:  # unreadable / wrong format
        raise ValueError(f"Could not read the file as a CSV ({error}).")

    if data.empty:
        raise ValueError("The CSV has column names but no rows.")
    return data


def standardize_columns(df):
    """Rename columns to the standard names and remove duplicated column names."""
    renamed = {}
    for column in df.columns:
        key = str(column).strip().lower().replace("-", "_").replace(" ", "_")
        renamed[column] = COLUMN_ALIASES.get(key, str(column).strip())
    df = df.rename(columns=renamed)
    return df.loc[:, ~df.columns.duplicated()].copy()


# --------------------------------------------------------------------------
# Data quality report (used on the "Data" page for the RAW data)
# --------------------------------------------------------------------------
def missing_values_table(df):
    """Missing value count and percentage for every column."""
    missing = df.isna().sum()
    table = pd.DataFrame({
        "Column": missing.index,
        "Missing Values": missing.values,
        "Missing (%)": (missing.values / max(len(df), 1) * 100).round(2),
    })
    return table


def get_data_quality_report(df):
    """Simple summary dictionary describing the raw dataset."""
    return {
        "rows": len(df),
        "columns": df.shape[1],
        "missing_total": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
    }


def data_types_table(df):
    """Column names with their data types (as text, so it displays safely)."""
    return pd.DataFrame({"Column": df.columns.astype(str), "Data Type": df.dtypes.astype(str).values})


# --------------------------------------------------------------------------
# Helpers for dates, times and day names
# --------------------------------------------------------------------------
def parse_dates(series):
    """Convert text to dates. Unreadable values become NaT (missing)."""
    parsed = pd.to_datetime(series, errors="coerce")
    if parsed.isna().mean() > 0.5:  # first attempt failed for most rows -> try other formats
        parsed = pd.to_datetime(series, errors="coerce", format="mixed", dayfirst=True)
    return parsed


def parse_hours(series):
    """Extract the hour (0-23) from values like '08:30', '8:30 AM' or plain numbers like 8."""
    text = series.astype("string").str.strip()
    hours = pd.Series(np.nan, index=series.index, dtype="float64")
    for time_format in ("%H:%M", "%H:%M:%S", "%I:%M %p", "%I:%M:%S %p", "%I %p"):
        parsed = pd.to_datetime(text, format=time_format, errors="coerce")
        hours = hours.fillna(parsed.dt.hour.astype("float64"))
    numbers = pd.to_numeric(text, errors="coerce")
    numbers = numbers.where((numbers >= 0) & (numbers <= 23))
    return hours.fillna(numbers.astype("float64"))


def normalise_day_names(series):
    """Convert 'mon', 'MONDAY', 'Mon ' ... into 'Monday'."""
    short = series.astype("string").str.strip().str.lower().str[:3]
    return short.map(_DAY_LOOKUP)


def has_columns(df, *columns):
    """True when every given column exists in the DataFrame."""
    return all(column in df.columns for column in columns)


# --------------------------------------------------------------------------
# Cleaning pipeline
# --------------------------------------------------------------------------
def clean_data(raw_df):
    """Clean the raw dataset and return (clean_df, log).

    Steps
        1. standardise column names
        2. remove duplicate rows
        3. convert numeric columns (bad text becomes missing)
        4. remove rows without Traffic_Volume and rows with obviously invalid values
        5. convert Date / Time columns, derive Hour, Day and Is_Weekend
        6. fill remaining missing values (median for numbers, most frequent value for text)
    `log` is a dictionary with before/after statistics shown on the Data page.
    """
    if raw_df is None or raw_df.empty:
        raise ValueError("The dataset is empty. Please upload a CSV file that contains data.")

    df = standardize_columns(raw_df)
    if "Traffic_Volume" not in df.columns:
        raise ValueError(
            "The dataset must contain a traffic volume column (named Traffic_Volume, Volume or "
            f"Vehicle_Count). Columns found: {', '.join(map(str, df.columns))}"
        )

    log = {
        "rows_before": len(df),
        "columns_before": df.shape[1],
        "missing_before": int(df.isna().sum().sum()),
        "duplicates_before": int(df.duplicated().sum()),
    }

    # 1. Duplicates
    df = df.drop_duplicates().reset_index(drop=True)
    log["duplicates_removed"] = log["rows_before"] - len(df)

    # 2. Numeric conversion and tidy text columns
    for column in NUMERIC_COLUMNS + ["Hour"]:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    for column in CATEGORICAL_COLUMNS:
        if column in df.columns:
            df[column] = df[column].astype("string").str.strip()

    # 3. Rows without the main measurement cannot be used
    missing_target = df["Traffic_Volume"].isna()
    log["missing_target_removed"] = int(missing_target.sum())
    df = df.loc[~missing_target].reset_index(drop=True)

    # 4. Obviously invalid values (negative volume, speed above 200, ...)
    invalid = pd.Series(False, index=df.index)
    for column, (lowest, highest) in VALID_RANGES.items():
        if column in df.columns:
            invalid |= (df[column] < lowest) | (df[column] > highest)
    log["invalid_removed"] = int(invalid.sum())
    df = df.loc[~invalid].reset_index(drop=True)

    # 5. Date and time
    log["invalid_dates_removed"] = 0
    log["invalid_times_removed"] = 0
    if "Date" in df.columns:
        df["Date"] = parse_dates(df["Date"])
        bad_dates = df["Date"].isna()
        log["invalid_dates_removed"] = int(bad_dates.sum())
        df = df.loc[~bad_dates].reset_index(drop=True)

    if "Time" in df.columns:
        df["Hour"] = parse_hours(df["Time"])
    elif "Hour" not in df.columns and "Date" in df.columns and (df["Date"].dt.hour != 0).any():
        df["Hour"] = df["Date"].dt.hour.astype("float64")  # Date column also contains the time
    if "Date" in df.columns:
        df["Date"] = df["Date"].dt.normalize()

    if "Hour" in df.columns:
        bad_hours = df["Hour"].isna() | (df["Hour"] < 0) | (df["Hour"] > 23)
        log["invalid_times_removed"] = int(bad_hours.sum())
        df = df.loc[~bad_hours].reset_index(drop=True)
        df["Hour"] = df["Hour"].astype(int)

    if df.empty:
        raise ValueError("No valid rows remain after cleaning. Please check the dataset values.")

    # 6. Day names and weekend flag
    if "Date" in df.columns:
        df["Day"] = df["Date"].dt.day_name()
    elif "Day" in df.columns:
        df["Day"] = normalise_day_names(df["Day"])
        if df["Day"].isna().all():
            df = df.drop(columns="Day")
    if "Day" in df.columns:
        df["Day"] = df["Day"].fillna(df["Day"].mode().iloc[0]).astype(str)
        df["Is_Weekend"] = df["Day"].isin(WEEKEND_DAYS).astype(int)

    # 7. Fill remaining missing values
    filled = {}
    for column in NUMERIC_COLUMNS:
        if column in df.columns and df[column].isna().any():
            filled[column] = int(df[column].isna().sum())
            df[column] = df[column].fillna(df[column].median())
    for column in CATEGORICAL_COLUMNS:
        if column in df.columns:
            if df[column].isna().any():
                filled[column] = int(df[column].isna().sum())
                mode = df[column].mode()
                df[column] = df[column].fillna(mode.iloc[0] if len(mode) else "Unknown")
            df[column] = df[column].astype(str)
    if "Accidents" in df.columns:
        df["Accidents"] = df["Accidents"].round().astype(int)
    log["filled_values"] = filled

    # 8. Tidy ordering
    sort_columns = [c for c in ["Date", "Hour"] if c in df.columns]
    if sort_columns:
        df = df.sort_values(sort_columns, kind="stable")
    df = df.reset_index(drop=True)

    log["rows_after"] = len(df)
    log["columns_after"] = df.shape[1]
    log["missing_after"] = int(df.isna().sum().sum())
    log["duplicates_after"] = int(df.duplicated().sum())
    log["columns_added"] = [c for c in ["Hour", "Is_Weekend"] if c in df.columns and c not in raw_df.columns]
    return df, log
