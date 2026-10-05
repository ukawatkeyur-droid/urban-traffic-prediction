"""
visualization.py
----------------
All charts of the project (Matplotlib + Seaborn) in one dark theme.

Every function takes a DataFrame (or table) and RETURNS a Matplotlib figure.
The Streamlit app is responsible for displaying it.
"""

import matplotlib

matplotlib.use("Agg")  # no pop-up windows; Streamlit displays the figure

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from modules.congestion import LEVELS, LEVEL_COLORS
from modules.data_processing import DAY_ORDER, WEEKEND_DAYS

# Colour palette (matches the dark dashboard)
BACKGROUND = "#0E1117"
PANEL = "#161B22"
TEXT = "#E6EDF3"
MUTED = "#9DA7B3"
GRID = "#30363D"
ACCENT = "#4DA3FF"
ACCENT_2 = "#FF7B72"


def _new_figure(width=8, height=4):
    figure, axes = plt.subplots(figsize=(width, height))
    figure.patch.set_facecolor(BACKGROUND)
    return figure, axes


def _style_axes(axes, title, x_label="", y_label="", grid=True):
    axes.set_facecolor(PANEL)
    axes.set_title(title, color=TEXT, fontsize=13, fontweight="bold", pad=12)
    axes.set_xlabel(x_label, color=MUTED)
    axes.set_ylabel(y_label, color=MUTED)
    axes.tick_params(colors=MUTED)
    for spine in axes.spines.values():
        spine.set_color(GRID)
    if grid:
        axes.grid(True, color=GRID, alpha=0.5, linewidth=0.6)
    else:
        axes.grid(False)
    axes.set_axisbelow(True)


def _style_legend(axes):
    legend = axes.legend(facecolor=PANEL, edgecolor=GRID)
    for text in legend.get_texts():
        text.set_color(TEXT)


def _style_colorbar(axes, label):
    colorbar = axes.collections[0].colorbar
    colorbar.ax.tick_params(colors=MUTED)
    colorbar.set_label(label, color=MUTED)
    colorbar.outline.set_edgecolor(GRID)


# --------------------------------------------------------------------------
# Exploratory data analysis charts
# --------------------------------------------------------------------------
def plot_traffic_by_hour(df):
    hourly = df.groupby("Hour")["Traffic_Volume"].mean()
    figure, axes = _new_figure()
    axes.plot(hourly.index, hourly.values, color=ACCENT, marker="o", linewidth=2)
    axes.fill_between(hourly.index, hourly.values, color=ACCENT, alpha=0.15)
    peak_hour = hourly.idxmax()
    axes.annotate(f"Peak: {peak_hour}:00", (peak_hour, hourly.max()), textcoords="offset points",
                  xytext=(0, 10), ha="center", color=TEXT, fontsize=9)
    axes.set_xticks(range(0, 24, 2))
    _style_axes(axes, "Average Traffic Volume by Hour of Day", "Hour of day", "Average vehicles")
    figure.tight_layout()
    return figure


def plot_traffic_by_day(df):
    from modules.analysis import daily_average
    daily = daily_average(df)
    colors = [ACCENT_2 if day in WEEKEND_DAYS else ACCENT for day in daily.index]
    figure, axes = _new_figure()
    bars = axes.bar(daily.index, daily.values, color=colors)
    axes.bar_label(bars, fmt="%.0f", color=TEXT, fontsize=8, padding=2)
    axes.tick_params(axis="x", rotation=30)
    _style_axes(axes, "Average Traffic Volume by Day (weekend in red)", "Day", "Average vehicles")
    figure.tight_layout()
    return figure


def plot_traffic_by_location(df):
    by_location = df.groupby("Location")["Traffic_Volume"].mean().sort_values()
    figure, axes = _new_figure(height=max(3.5, 0.6 * len(by_location) + 1.5))
    bars = axes.barh(by_location.index, by_location.values, color=ACCENT)
    axes.bar_label(bars, fmt="%.0f", color=TEXT, fontsize=8, padding=3)
    _style_axes(axes, "Average Traffic Volume by Location", "Average vehicles", "")
    figure.tight_layout()
    return figure


def plot_traffic_trend(df):
    daily = df.groupby("Date")["Traffic_Volume"].mean()
    figure, axes = _new_figure(width=10)
    axes.plot(daily.index, daily.values, color=ACCENT, marker="o", markersize=3, linewidth=1.5, label="Daily average")
    if len(daily) >= 14:
        axes.plot(daily.index, daily.rolling(7).mean(), color=ACCENT_2, linewidth=2.2, linestyle="--",
                  label="7-day moving average")
    _style_axes(axes, "Traffic Trend Over Time", "Date", "Average vehicles")
    _style_legend(axes)
    figure.autofmt_xdate()
    figure.tight_layout()
    return figure


def plot_speed_vs_traffic(df):
    sample = df.sample(min(len(df), 2000), random_state=42)
    figure, axes = _new_figure()
    axes.scatter(sample["Traffic_Volume"], sample["Average_Speed"], s=14, alpha=0.4, color=ACCENT)
    if sample["Traffic_Volume"].nunique() > 1:
        slope, intercept = np.polyfit(sample["Traffic_Volume"], sample["Average_Speed"], 1)
        x_values = np.linspace(sample["Traffic_Volume"].min(), sample["Traffic_Volume"].max(), 50)
        axes.plot(x_values, slope * x_values + intercept, color=ACCENT_2, linewidth=2, label="Trend line")
        _style_legend(axes)
    _style_axes(axes, "Average Speed vs Traffic Volume", "Traffic volume (vehicles)", "Average speed")
    figure.tight_layout()
    return figure


def plot_weather_impact(table, group_name):
    figure, axes = _new_figure()
    bars = axes.bar(table.index.astype(str), table["Average Traffic Volume"], color=ACCENT)
    axes.bar_label(bars, fmt="%.0f", color=TEXT, fontsize=8, padding=2)
    axes.tick_params(axis="x", rotation=20)
    _style_axes(axes, f"Average Traffic Volume by {group_name}", group_name, "Average vehicles")
    figure.tight_layout()
    return figure


def plot_day_hour_heatmap(df):
    pivot = df.pivot_table(index="Day", columns="Hour", values="Traffic_Volume", aggfunc="mean")
    pivot = pivot.reindex([day for day in DAY_ORDER if day in pivot.index])
    figure, axes = _new_figure(width=12, height=4.5)
    sns.heatmap(pivot, cmap="magma", ax=axes, linewidths=0.3, linecolor=BACKGROUND,
                cbar_kws={"label": "Average vehicles"})
    _style_axes(axes, "Traffic Volume Heatmap: Day x Hour", "Hour of day", "", grid=False)
    axes.tick_params(axis="y", rotation=0)
    _style_colorbar(axes, "Average vehicles")
    figure.tight_layout()
    return figure


# --------------------------------------------------------------------------
# Correlation
# --------------------------------------------------------------------------
def plot_correlation_heatmap(corr):
    labels = [name.replace("_", " ") for name in corr.columns]
    corr = corr.copy()
    corr.index, corr.columns = labels, labels
    size = max(5, 1.0 * len(labels))
    figure, axes = _new_figure(width=size + 1, height=size)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, center=0, ax=axes,
                linewidths=0.5, linecolor=BACKGROUND, cbar_kws={"label": "Correlation (r)"})
    _style_axes(axes, "Correlation Matrix", "", "", grid=False)
    axes.tick_params(axis="x", rotation=30)
    _style_colorbar(axes, "Correlation (r)")
    figure.tight_layout()
    return figure


# --------------------------------------------------------------------------
# Congestion
# --------------------------------------------------------------------------
def plot_congestion_distribution(distribution):
    figure, axes = _new_figure(height=3.6)
    colors = [LEVEL_COLORS[level] for level in distribution["Level"]]
    bars = axes.bar(distribution["Level"], distribution["Percentage"], color=colors)
    axes.bar_label(bars, fmt="%.1f%%", color=TEXT, padding=2)
    _style_axes(axes, "Share of Records by Congestion Level", "", "Percentage of records")
    axes.set_ylim(0, max(distribution["Percentage"].max() * 1.2, 10))
    figure.tight_layout()
    return figure


def plot_location_congestion(shares):
    ordered = shares.iloc[::-1]  # highest High-% ends up at the top
    figure, axes = _new_figure(height=max(3.5, 0.6 * len(ordered) + 1.5))
    left = np.zeros(len(ordered))
    for level in LEVELS:
        axes.barh(ordered.index, ordered[level], left=left, color=LEVEL_COLORS[level], label=level)
        left += ordered[level].values
    _style_axes(axes, "Congestion Mix by Location", "Percentage of records", "")
    axes.set_xlim(0, 100)
    _style_legend(axes)
    figure.tight_layout()
    return figure


def plot_congestion_by_hour(high_share):
    figure, axes = _new_figure()
    axes.plot(high_share.index, high_share.values, color=LEVEL_COLORS["High"], marker="o", linewidth=2)
    axes.fill_between(high_share.index, high_share.values, color=LEVEL_COLORS["High"], alpha=0.15)
    axes.set_xticks(range(0, 24, 2))
    _style_axes(axes, "How Often Congestion Is High, by Hour", "Hour of day", "% of records with High congestion")
    figure.tight_layout()
    return figure


# --------------------------------------------------------------------------
# Machine learning
# --------------------------------------------------------------------------
def plot_actual_vs_predicted(y_actual, y_predicted, r2):
    figure, axes = _new_figure(height=4.5)
    axes.scatter(y_actual, y_predicted, s=14, alpha=0.45, color=ACCENT)
    limit_low = min(y_actual.min(), y_predicted.min())
    limit_high = max(y_actual.max(), y_predicted.max())
    axes.plot([limit_low, limit_high], [limit_low, limit_high], color=ACCENT_2, linestyle="--",
              linewidth=2, label="Perfect prediction")
    axes.text(0.04, 0.92, f"R² = {r2:.3f}", transform=axes.transAxes, color=TEXT, fontsize=11,
              bbox=dict(facecolor=BACKGROUND, edgecolor=GRID, boxstyle="round"))
    _style_axes(axes, "Actual vs Predicted Traffic Volume (test data)", "Actual vehicles", "Predicted vehicles")
    _style_legend(axes)
    figure.tight_layout()
    return figure


def plot_feature_importance(importance_table, title):
    ordered = importance_table.iloc[::-1]
    figure, axes = _new_figure(height=max(3.5, 0.45 * len(ordered) + 1.5))
    axes.barh(ordered["Feature"], ordered["Importance"], color=ACCENT)
    _style_axes(axes, title, "Importance", "")
    figure.tight_layout()
    return figure
