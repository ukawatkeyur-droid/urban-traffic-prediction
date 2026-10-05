"""
app.py  -  Urban Traffic Intelligence System
--------------------------------------------
Main Streamlit file. It only handles the USER INTERFACE (pages, widgets, layout).
All calculations live in the  modules/  folder:

    modules/data_processing.py  - loading and cleaning
    modules/analysis.py         - statistics, correlation, peaks, insights
    modules/congestion.py       - Low / Moderate / High classification
    modules/ml_model.py         - machine learning
    modules/visualization.py    - all charts

Run with:   streamlit run app.py
"""

import html
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from modules import analysis, congestion, data_processing, ml_model, visualization

# set_page_config must be the first Streamlit command
st.set_page_config(page_title="Urban Traffic Intelligence", page_icon="🚦", layout="wide")

BASE_DIR = Path(__file__).parent
SAMPLE_DATA_PATH = BASE_DIR / "data" / "traffic_data.csv"
CSS_PATH = BASE_DIR / "assets" / "styles.css"

PAGES = ["🏠 Home", "📂 Data", "📊 EDA", "📈 Statistics", "🚦 Congestion", "🤖 Prediction", "💡 Insights"]
DISCLAIMER = ("Predictions and analytical insights are based on historical data and should not be "
              "treated as guaranteed future outcomes.")


# ==========================================================================
# Small helper functions
# ==========================================================================
def load_css():
    """Apply the custom dashboard styling from assets/styles.css."""
    if CSS_PATH.exists():
        st.markdown(f"<style>{CSS_PATH.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def kpi_card(label, value, note="", style=""):
    """Return the HTML of one KPI card. `style` can be kpi-low, kpi-mid or kpi-high."""
    note_html = f'<div class="kpi-note">{html.escape(str(note))}</div>' if note else ""
    return (f'<div class="kpi-card {style}"><div class="kpi-label">{html.escape(str(label))}</div>'
            f'<div class="kpi-value">{html.escape(str(value))}</div>{note_html}</div>')


def show_kpi_row(cards):
    """Show several KPI cards side by side. `cards` is a list of (label, value, note, style) tuples."""
    columns = st.columns(len(cards))
    for column, card in zip(columns, cards):
        with column:
            st.markdown(kpi_card(*card), unsafe_allow_html=True)


def show_figure(figure):
    """Display a Matplotlib figure and free its memory."""
    st.pyplot(figure)
    plt.close(figure)


def show_unavailable(chart_name, needed_columns):
    st.info(f"**{chart_name}** needs: {needed_columns}. This column is not available in the current dataset.")


def page_title(title, subtitle=""):
    st.title(title)
    if subtitle:
        st.markdown(f'<div class="page-subtitle">{html.escape(subtitle)}</div>', unsafe_allow_html=True)


def display_frame(df):
    """Copy of a DataFrame that looks nice in a table (dates without 00:00:00)."""
    shown = df.copy()
    if "Date" in shown.columns:
        shown["Date"] = shown["Date"].dt.strftime("%Y-%m-%d")
    return shown


# ==========================================================================
# Loading data into the session
# ==========================================================================
def init_state():
    st.session_state.setdefault("uploader_key", 0)
    st.session_state.setdefault("data_version", 0)


def load_dataset(source, source_name):
    """Read + clean + classify congestion. Session data is replaced ONLY if every step succeeds."""
    raw = data_processing.load_csv(source)
    clean, log = data_processing.clean_data(raw)
    clean, congestion_info = congestion.add_congestion_columns(clean)
    st.session_state["raw_df"] = raw
    st.session_state["clean_df"] = clean
    st.session_state["cleaning_log"] = log
    st.session_state["congestion_info"] = congestion_info
    st.session_state["source_name"] = source_name
    st.session_state["data_version"] += 1


def ensure_data_loaded():
    """On first start, load the built-in sample dataset."""
    if "clean_df" in st.session_state:
        return
    if not SAMPLE_DATA_PATH.exists():
        st.error("Sample dataset not found at `data/traffic_data.csv`. Run `python generate_data.py` "
                 "once to create it, then refresh this page.")
        st.stop()
    try:
        load_dataset(SAMPLE_DATA_PATH, "Sample dataset")
    except ValueError as error:
        st.error(f"The sample dataset could not be loaded: {error}")
        st.stop()


def get_data():
    """Return the cleaned dataset (stops the page with a message if it is empty)."""
    df = st.session_state["clean_df"]
    if df is None or df.empty:
        st.error("The dataset is empty. Please upload a valid CSV on the Data page.")
        st.stop()
    return df


# ==========================================================================
# Page: Home
# ==========================================================================
def render_home():
    df = get_data()
    peaks = analysis.get_peak_summary(df)

    page_title("Urban Traffic Intelligence System",
               "Data-driven analysis of urban traffic patterns and congestion.")
    st.write(
        "This project analyses historical urban traffic records to discover **when** and **where** traffic is "
        "heaviest, how weather and speed relate to traffic, and how congestion can be classified and estimated. "
        "It follows the standard data science workflow: **Data → Cleaning → EDA → Statistics → Visualization → "
        "Machine Learning → Insights**."
    )

    average_speed = f"{df['Average_Speed'].mean():.1f}" if "Average_Speed" in df.columns else "N/A"
    peak_hour = analysis.format_hour(peaks["peak_hour"]) if "peak_hour" in peaks else "N/A"
    show_kpi_row([
        ("Total Records", f"{len(df):,}", f"Source: {st.session_state['source_name']}", ""),
        ("Average Traffic Volume", f"{df['Traffic_Volume'].mean():,.0f}", "vehicles per record", ""),
        ("Average Speed", average_speed, "units as in the dataset" if average_speed != "N/A" else "column not available", ""),
        ("Peak Traffic Hour", peak_hour, f"avg {peaks['peak_hour_volume']:,.0f} vehicles" if "peak_hour" in peaks else "time column not available", ""),
        ("Most Congested Location", peaks.get("most_congested_location", "N/A"),
         "highest average congestion score" if "most_congested_location" in peaks else "location column not available", ""),
    ])

    st.warning(DISCLAIMER)

    st.subheader("How to use this dashboard")
    st.markdown(
        "- **Data** – upload your own CSV or keep the sample dataset; see the cleaning steps.\n"
        "- **EDA** – charts for hours, days, locations, weather and a Day × Hour heatmap.\n"
        "- **Statistics** – mean, median, mode, variance, quartiles and correlations.\n"
        "- **Congestion** – peak analysis and Low / Moderate / High classification.\n"
        "- **Prediction** – a simple machine learning model that estimates traffic volume.\n"
        "- **Insights** – findings written automatically from the calculated numbers."
    )


# ==========================================================================
# Page: Data (upload + cleaning)
# ==========================================================================
def render_data():
    page_title("📂 Data Upload & Cleaning", "Load a dataset, inspect its problems and see how it is cleaned.")

    with st.expander("Expected CSV format"):
        st.markdown(
            "Only **Traffic_Volume** is required. Other columns are optional and unlock more analysis:\n\n"
            "`Date, Time, Location, Traffic_Volume, Average_Speed, Weather, Temperature, Rainfall, Day, "
            "Vehicle_Type, Accidents`\n\n"
            "Column names such as *traffic volume*, *avg_speed* or *temp* are recognised automatically."
        )

    uploaded = st.file_uploader("Upload a traffic CSV file", type=["csv"],
                                key=f"uploader_{st.session_state['uploader_key']}")
    if uploaded is not None:
        signature = f"{uploaded.name}-{uploaded.size}"
        if st.session_state.get("upload_signature") != signature:  # only process a new file once
            st.session_state["upload_signature"] = signature
            try:
                load_dataset(uploaded, uploaded.name)
                st.session_state["upload_status"] = ("success", f"Loaded and cleaned **{uploaded.name}**.")
            except ValueError as error:
                st.session_state["upload_status"] = ("error", str(error))
        status = st.session_state.get("upload_status")
        if status:
            (st.success if status[0] == "success" else st.error)(status[1])

    if st.button("↩ Use the sample dataset"):
        try:
            load_dataset(SAMPLE_DATA_PATH, "Sample dataset")
        except ValueError as error:
            st.error(str(error))
        else:
            st.session_state["uploader_key"] += 1  # clears the upload box
            st.session_state["upload_signature"] = None
            st.session_state["upload_status"] = None
            st.rerun()

    raw = st.session_state["raw_df"]
    clean = st.session_state["clean_df"]
    log = st.session_state["cleaning_log"]
    quality = data_processing.get_data_quality_report(raw)

    # ---- 1. Raw data ----
    st.subheader("1. Raw dataset")
    st.caption(f"Current dataset: {st.session_state['source_name']}")
    show_kpi_row([
        ("Rows", f"{quality['rows']:,}", "", ""),
        ("Columns", quality["columns"], "", ""),
        ("Missing values", f"{quality['missing_total']:,}", "", ""),
        ("Duplicate rows", f"{quality['duplicate_rows']:,}", "", ""),
    ])
    st.dataframe(raw.head(10))
    left, right = st.columns(2)
    with left:
        st.markdown("**Missing values per column**")
        st.dataframe(data_processing.missing_values_table(raw), hide_index=True)
    with right:
        st.markdown("**Data types**")
        st.dataframe(data_processing.data_types_table(raw), hide_index=True)

    # ---- 2. Cleaning ----
    st.subheader("2. Cleaning: before vs after")
    summary = pd.DataFrame({
        "Metric": ["Rows", "Columns", "Missing values", "Duplicate rows"],
        "Before cleaning": [log["rows_before"], log["columns_before"], log["missing_before"], log["duplicates_before"]],
        "After cleaning": [log["rows_after"], log["columns_after"], log["missing_after"], log["duplicates_after"]],
    })
    st.dataframe(summary, hide_index=True)

    st.markdown("**What the cleaning step did**")
    actions = [
        f"Removed **{log['duplicates_removed']}** duplicate rows.",
        f"Removed **{log['missing_target_removed']}** rows with no Traffic_Volume value.",
        f"Removed **{log['invalid_removed']}** rows with obviously invalid values (for example negative traffic volume).",
    ]
    if log["invalid_dates_removed"] or log["invalid_times_removed"]:
        actions.append(f"Removed **{log['invalid_dates_removed']}** rows with unreadable dates and "
                       f"**{log['invalid_times_removed']}** rows with unreadable times.")
    for column, count in log["filled_values"].items():
        actions.append(f"Filled **{count}** missing values in `{column}` (median for numbers, most frequent value for text).")
    if log["columns_added"]:
        actions.append("Converted Date/Time and created new columns: " + ", ".join(f"`{c}`" for c in log["columns_added"]) + ".")
    for action in actions:
        st.markdown(f"- {action}")

    # ---- 3. Cleaned data ----
    st.subheader("3. Cleaned dataset")
    cleaned_view = display_frame(clean.drop(columns=["Congestion_Score", "Congestion_Level"], errors="ignore"))
    st.dataframe(cleaned_view.head(10))
    st.download_button("⬇ Download cleaned CSV", data=cleaned_view.to_csv(index=False).encode("utf-8"),
                       file_name="cleaned_traffic_data.csv", mime="text/csv")


# ==========================================================================
# Page: EDA
# ==========================================================================
def render_eda():
    df = get_data()
    page_title("📊 Exploratory Data Analysis", "Visual exploration of traffic patterns.")

    left, right = st.columns(2)
    with left:
        st.subheader("Traffic by Hour")
        if "Hour" in df.columns:
            show_figure(visualization.plot_traffic_by_hour(df))
        else:
            show_unavailable("Traffic by Hour", "a Time (or Hour) column")
    with right:
        st.subheader("Traffic by Day")
        if "Day" in df.columns:
            show_figure(visualization.plot_traffic_by_day(df))
        else:
            show_unavailable("Traffic by Day", "a Date (or Day) column")

    left, right = st.columns(2)
    with left:
        st.subheader("Traffic by Location")
        if "Location" in df.columns:
            show_figure(visualization.plot_traffic_by_location(df))
        else:
            show_unavailable("Traffic by Location", "a Location column")
    with right:
        st.subheader("Speed vs Traffic")
        if "Average_Speed" in df.columns:
            show_figure(visualization.plot_speed_vs_traffic(df))
        else:
            show_unavailable("Speed vs Traffic", "an Average_Speed column")

    st.subheader("Traffic Trend")
    if "Date" in df.columns and df["Date"].nunique() > 1:
        show_figure(visualization.plot_traffic_trend(df))
    else:
        show_unavailable("Traffic Trend", "a Date column with at least two different dates")

    st.subheader("Weather Impact")
    weather_table, group_name = analysis.weather_impact_table(df)
    if weather_table is not None:
        left, right = st.columns([3, 2])
        with left:
            show_figure(visualization.plot_weather_impact(weather_table, group_name))
        with right:
            st.dataframe(weather_table)
            st.caption("Groups with few records give less reliable averages.")
    else:
        show_unavailable("Weather Impact", "a Weather or Rainfall column")

    st.subheader("Heatmap: Day × Hour")
    if "Day" in df.columns and "Hour" in df.columns:
        show_figure(visualization.plot_day_hour_heatmap(df))
        st.caption("Brighter cells mean higher average traffic volume.")
    else:
        show_unavailable("Day × Hour heatmap", "both Day (or Date) and Time (or Hour) columns")


# ==========================================================================
# Page: Statistics
# ==========================================================================
def render_statistics():
    df = get_data()
    page_title("📈 Statistical Analysis", "Descriptive statistics and correlations calculated from the dataset.")
    tab_descriptive, tab_correlation = st.tabs(["Descriptive Statistics", "Correlation Analysis"])

    with tab_descriptive:
        columns = analysis.numeric_columns_available(df)
        st.subheader("Summary statistics")
        st.dataframe(analysis.statistics_table(df, columns))
        st.caption("Standard deviation and variance are sample statistics (divide by n − 1). Q1 and Q3 are the 25th and 75th percentiles.")

        st.subheader("What the numbers say about traffic volume")
        for sentence in analysis.explain_statistics(df):
            st.markdown(f"- {sentence}")

        group_options = [c for c in ["Location", "Day", "Weather", "Hour", "Vehicle_Type"] if c in df.columns]
        if group_options:
            st.subheader("Traffic volume statistics by group")
            group = st.selectbox("Group by", options=group_options)
            grouped = df.groupby(group)["Traffic_Volume"].agg(["count", "mean", "median", "std", "min", "max"]).round(1)
            if group == "Day":
                grouped = grouped.reindex(data_processing.DAY_ORDER).dropna(how="all")
            st.dataframe(grouped)

    with tab_correlation:
        corr = analysis.get_correlation_matrix(df)
        if corr.empty:
            st.info("Correlation analysis needs Traffic_Volume plus at least one other numeric column with variation.")
        else:
            left, right = st.columns([3, 2])
            with left:
                show_figure(visualization.plot_correlation_heatmap(corr))
            with right:
                st.markdown("**Automatic explanation**")
                for sentence in analysis.explain_correlations(corr):
                    st.markdown(f"- {sentence}")
            st.markdown("**Correlation table**")
            st.dataframe(corr.round(3))


# ==========================================================================
# Page: Congestion (peak analysis + classification)
# ==========================================================================
def render_congestion():
    df = get_data()
    info = st.session_state["congestion_info"]
    peaks = analysis.get_peak_summary(df)
    page_title("🚦 Congestion & Peak Traffic Analysis", "When and where traffic is heaviest.")

    # ---- Peak analysis ----
    st.subheader("Peak traffic analysis")

    def hour_card(label, key, volume_key):
        if key in peaks:
            return (label, analysis.format_hour(peaks[key]), f"avg {peaks[volume_key]:,.0f} vehicles", "")
        return (label, "N/A", "time column not available", "")

    def day_card(label, key, volume_key):
        if key in peaks:
            return (label, peaks[key], f"avg {peaks[volume_key]:,.0f} vehicles", "")
        return (label, "N/A", "date/day column not available", "")

    def location_card(label, key):
        if key in peaks:
            return (label, peaks[key], peaks["location_metric"], "")
        return (label, "N/A", "location column not available", "")

    show_kpi_row([
        hour_card("Peak Hour", "peak_hour", "peak_hour_volume"),
        hour_card("Lowest Traffic Hour", "lowest_hour", "lowest_hour_volume"),
        day_card("Peak Day", "peak_day", "peak_day_volume"),
    ])
    show_kpi_row([
        day_card("Lowest Traffic Day", "lowest_day", "lowest_day_volume"),
        location_card("Most Congested Location", "most_congested_location"),
        location_card("Least Congested Location", "least_congested_location"),
    ])

    # ---- Classification ----
    st.subheader("Congestion classification")
    st.markdown(
        f"Each record gets a **congestion score** from 0 (free flowing) to 1 (very congested), based on "
        f"**{info['method']}**. The thresholds are calculated from the data using the mean ± 0.5 × standard deviation "
        f"of the score:"
    )
    st.markdown(
        f"- 🟢 **Low**: score ≤ {info['low_cutoff']:.3f}\n"
        f"- 🟡 **Moderate**: between {info['low_cutoff']:.3f} and {info['high_cutoff']:.3f}\n"
        f"- 🔴 **High**: score ≥ {info['high_cutoff']:.3f}"
    )
    st.caption("These labels are relative to this dataset: 'High' means high compared with the other records here.")

    distribution = congestion.congestion_distribution(df)
    styles = {"Low": "kpi-low", "Moderate": "kpi-mid", "High": "kpi-high"}
    show_kpi_row([
        (f"{congestion.LEVEL_LABELS[row.Level]} congestion", f"{row.Percentage:.1f}%", f"{row.Records:,} records", styles[row.Level])
        for row in distribution.itertuples()
    ])

    left, right = st.columns(2)
    with left:
        show_figure(visualization.plot_congestion_distribution(distribution))
    with right:
        if "Hour" in df.columns:
            show_figure(visualization.plot_congestion_by_hour(congestion.hourly_high_congestion_share(df)))
        else:
            show_unavailable("Congestion by hour", "a Time (or Hour) column")

    st.subheader("Locations with the highest congestion")
    if "Location" in df.columns:
        left, right = st.columns([2, 3])
        with left:
            st.dataframe(congestion.location_congestion_table(df), hide_index=True)
        with right:
            show_figure(visualization.plot_location_congestion(congestion.location_congestion_share(df)))
    else:
        show_unavailable("Location congestion", "a Location column")


# ==========================================================================
# Page: Prediction (machine learning)
# ==========================================================================
def get_trained_model(df, model_name, test_size):
    """Train a model once per dataset/settings and reuse it (stored in the session)."""
    version = st.session_state["data_version"]
    cache = st.session_state.setdefault("model_cache", {})
    if st.session_state.get("model_cache_version") != version:  # new dataset -> forget old models
        cache.clear()
        st.session_state["model_cache_version"] = version
    key = (model_name, test_size)
    if key not in cache:
        cache[key] = ml_model.train_model(df, model_name, test_size)
    return cache[key]


def prediction_input(df, feature):
    """Create the right input widget for one model feature and return its value."""
    key = f"predict_{feature}"
    if feature == "Hour":
        default_hour = int(analysis.hourly_average(df).idxmax())
        return st.slider("Hour of day (0-23)", min_value=0, max_value=23, value=default_hour, key=key)
    if feature == "Day":
        days = [day for day in data_processing.DAY_ORDER if day in set(df["Day"])]
        return st.selectbox("Day of week", options=days, key=key)
    if feature in ("Location", "Weather", "Vehicle_Type"):
        options = sorted(df[feature].astype(str).unique())
        return st.selectbox(feature.replace("_", " "), options=options, key=key)
    if feature == "Temperature":
        return st.number_input("Temperature", value=float(round(df["Temperature"].mean(), 1)),
                               min_value=float(df["Temperature"].min() - 10), max_value=float(df["Temperature"].max() + 10),
                               step=0.5, key=key)
    if feature == "Rainfall":
        return st.number_input("Rainfall", value=0.0, min_value=0.0,
                               max_value=float(max(df["Rainfall"].max() * 2, 1.0)), step=0.5, key=key)
    raise ValueError(f"No input widget defined for feature: {feature}")


def render_prediction():
    df = get_data()
    info = st.session_state["congestion_info"]
    page_title("🤖 Traffic Prediction", "A historical-pattern model that estimates traffic volume from similar past situations.")
    st.warning(DISCLAIMER)

    with st.expander("How does the model work? (simple explanation)"):
        st.markdown(
            "**Goal:** estimate *Traffic_Volume* (vehicles) from information known in advance: "
            "hour, day, location, weather, temperature and rainfall.\n\n"
            "**Train/test split:** the records are divided into a training set and a test set. "
            "The test records are kept separate while the historical-pattern method is evaluated.\n\n"
            "**Historical Pattern:** the system compares the requested situation with historical records. "
            "Records that are more similar receive greater weight, and the estimated traffic volume is calculated "
            "from the strongest historical matches. This makes the prediction easy to explain.\n\n"
            "**Not used as inputs:** average speed and accidents, because they are only known *after* the traffic "
            "has happened. Using them as prediction inputs would cause data leakage." 
        )

    left, right = st.columns(2)
    with left:
        model_name = st.selectbox("Prediction Method", options=ml_model.MODEL_NAMES, index=0)
    with right:
        test_percent = st.slider("Test set size (%)", min_value=10, max_value=40, value=20, step=5)
    test_size = test_percent / 100

    try:
        with st.spinner("Training models..."):
            results = {name: get_trained_model(df, name, test_size) for name in ml_model.MODEL_NAMES}
    except ValueError as error:
        st.error(f"The model could not be trained: {error}")
        return
    selected = results[model_name]
    test_metrics = selected["test_metrics"]

    # ---- Training summary ----
    st.subheader("Model performance")
    features = selected["numeric_features"] + selected["categorical_features"]
    st.markdown(f"**Training rows:** {selected['train_rows']:,} &nbsp;|&nbsp; **Test rows:** {selected['test_rows']:,} "
                f"&nbsp;|&nbsp; **Input features:** {', '.join(features)}")
    show_kpi_row([
        ("R² score (test)", f"{test_metrics['R2']:.3f}", "1.0 = perfect, 0 = no better than the average", ""),
        ("MAE (test)", f"{test_metrics['MAE']:,.0f}", "average error in vehicles", ""),
        ("RMSE (test)", f"{test_metrics['RMSE']:,.0f}", "error, punishes big mistakes more", ""),
    ])
    st.markdown(
        f"On data the model has never seen, **{model_name}** explains about **{max(test_metrics['R2'], 0) * 100:.0f}%** "
        f"of the variation in traffic volume, and its predictions are off by about **{test_metrics['MAE']:,.0f} vehicles** "
        "on average."
    )
    if selected["train_metrics"]["R2"] - test_metrics["R2"] > 0.1:
        st.warning("The training R² is much higher than the test R². The model may be over-fitting (memorising the "
                   "training data).")

    st.markdown("**Prediction method comparison (same train/test split)**")
    st.dataframe(ml_model.metrics_table(results), hide_index=True)

    left, right = st.columns(2)
    with left:
        show_figure(visualization.plot_actual_vs_predicted(selected["y_test"], selected["y_predicted"], test_metrics["R2"]))
    with right:
        title = "Feature relationship importance"
        show_figure(visualization.plot_feature_importance(ml_model.get_feature_importance(selected), title))

    # ---- Prediction form ----
    st.subheader("Try a prediction")
    with st.form("prediction_form"):
        input_columns = st.columns(2)
        inputs = {}
        for position, feature in enumerate(features):
            with input_columns[position % 2]:
                inputs[feature] = prediction_input(df, feature)
        submitted = st.form_submit_button("Predict traffic")

    if submitted:
        volume = ml_model.predict_traffic(selected, inputs)
        level = congestion.classify_single_value(volume, info["volume_low_cutoff"], info["volume_high_cutoff"])
        level_style = {"Low": "kpi-low", "Moderate": "kpi-mid", "High": "kpi-high"}[level]
        show_kpi_row([
            ("Estimated Traffic Volume", f"{volume:,.0f} vehicles", f"historical-pattern estimate ({model_name})", ""),
            ("Predicted Congestion", f"{congestion.LEVEL_LABELS[level]} Congestion",
             "volume compared with the historical distribution", level_style),
        ])

        # Compare with what actually happened in similar situations
        similar = df
        if "Location" in inputs:
            similar = similar[similar["Location"] == inputs["Location"]]
        if "Hour" in inputs:
            similar = similar[similar["Hour"] == inputs["Hour"]]
        if "Day" in inputs and "Is_Weekend" in similar.columns:
            similar = similar[similar["Is_Weekend"] == int(inputs["Day"] in data_processing.WEEKEND_DAYS)]
        if len(similar) >= 3:
            st.caption(f"For comparison, {len(similar)} similar historical records (same location, hour and day type) "
                       f"averaged {similar['Traffic_Volume'].mean():,.0f} vehicles. Typical model error (RMSE): "
                       f"±{test_metrics['RMSE']:,.0f} vehicles.")
        st.caption("This is a model-based estimate from historical data, not a guaranteed outcome.")


# ==========================================================================
# Page: Insights
# ==========================================================================
def render_insights():
    df = get_data()
    page_title("💡 Traffic Insights", "Findings generated automatically from the calculated values.")
    insights = analysis.generate_insights(df)
    for item in insights:
        st.markdown(
            f'<div class="insight-card"><span class="insight-tag">{html.escape(item["category"].upper())}</span>'
            f'<div>{html.escape(item["text"])}</div></div>',
            unsafe_allow_html=True,
        )
    st.caption("Every sentence above is built from numbers calculated on the current dataset, so the insights change "
               "when you upload different data. They describe association in historical data, not proven causes.")
    st.warning(DISCLAIMER)


# ==========================================================================
# Navigation
# ==========================================================================
PAGE_FUNCTIONS = {
    "🏠 Home": render_home,
    "📂 Data": render_data,
    "📊 EDA": render_eda,
    "📈 Statistics": render_statistics,
    "🚦 Congestion": render_congestion,
    "🤖 Prediction": render_prediction,
    "💡 Insights": render_insights,
}


def main():
    init_state()
    load_css()
    ensure_data_loaded()

    st.sidebar.title("🚦 Urban Traffic Intelligence")
    st.sidebar.caption("Foundation of Data Science project")
    page = st.sidebar.radio("Navigation", PAGES, label_visibility="collapsed")
    st.sidebar.divider()
    st.sidebar.markdown(f"**Dataset:** {st.session_state['source_name']}")
    st.sidebar.markdown(f"**Records:** {len(st.session_state['clean_df']):,}")
    st.sidebar.caption("For educational and analytical purposes only.")

    PAGE_FUNCTIONS[page]()


if __name__ == "__main__":
    main()
