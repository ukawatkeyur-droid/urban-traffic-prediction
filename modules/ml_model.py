"""
ml_model.py
-----------
Sklearn-free traffic volume prediction.

This module uses historical pattern matching instead of scikit-learn.
It predicts traffic volume from:
    Hour, Day, Location, Weather, Temperature, Rainfall, Vehicle_Type

No Average_Speed or Accidents are used because they are outcomes of traffic,
not information known in advance.
"""

import numpy as np
import pandas as pd


LINEAR_REGRESSION = "Historical Pattern"
RANDOM_FOREST = "Historical Pattern"
MODEL_NAMES = [LINEAR_REGRESSION]

MIN_ROWS_FOR_TRAINING = 50


def get_feature_lists(df, model_name=None):
    """Return available numeric and categorical prediction features."""

    numeric = [
        c for c in ["Hour", "Temperature", "Rainfall"]
        if c in df.columns
    ]

    categorical = [
        c for c in ["Day", "Location", "Weather", "Vehicle_Type"]
        if c in df.columns
    ]

    if not numeric and not categorical:
        raise ValueError(
            "No input columns are available for prediction. "
            "The dataset needs at least one of: "
            "Time/Hour, Day/Date, Location, Weather, Temperature, Rainfall."
        )

    return numeric, categorical


def make_input_frame(df, numeric, categorical):
    """Select and prepare feature columns."""

    columns = numeric + categorical

    frame = df[columns].copy()

    for column in categorical:
        frame[column] = frame[column].astype(str)

    return frame


def _calculate_similarity(training_df, input_row):
    """
    Calculate similarity between a new situation and historical records.

    Categorical matches receive strong weight.
    Numeric differences receive normalized weights.
    """

    scores = pd.Series(0.0, index=training_df.index)

    # Hour is highly important for traffic.
    if "Hour" in training_df.columns and "Hour" in input_row:
        hour_difference = np.abs(
            training_df["Hour"].astype(float)
            - float(input_row["Hour"])
        )

        # Handle circular time: 23:00 is close to 00:00.
        hour_difference = np.minimum(
            hour_difference,
            24 - hour_difference
        )

        scores += np.exp(-hour_difference / 3.0) * 4.0

    # Categorical features.
    categorical_columns = [
        "Day",
        "Location",
        "Weather",
        "Vehicle_Type"
    ]

    for column in categorical_columns:

        if column in training_df.columns and column in input_row:

            matches = (
                training_df[column].astype(str)
                == str(input_row[column])
            )

            scores += matches.astype(float) * 3.0

    # Temperature similarity.
    if (
        "Temperature" in training_df.columns
        and "Temperature" in input_row
    ):
        values = training_df["Temperature"].astype(float)

        scale = max(values.std(), 1.0)

        difference = np.abs(
            values - float(input_row["Temperature"])
        ) / scale

        scores += np.exp(-difference) * 1.0

    # Rainfall similarity.
    if (
        "Rainfall" in training_df.columns
        and "Rainfall" in input_row
    ):
        values = training_df["Rainfall"].astype(float)

        scale = max(values.std(), 1.0)

        difference = np.abs(
            values - float(input_row["Rainfall"])
        ) / scale

        scores += np.exp(-difference) * 1.0

    return scores


def train_model(df, model_name=RANDOM_FOREST, test_size=0.2):
    """
    Prepare the historical dataset for prediction.

    No sklearn model is trained.
    Historical records are stored and later used for similarity-based prediction.
    """

    if df is None or len(df) < MIN_ROWS_FOR_TRAINING:
        raise ValueError(
            f"At least {MIN_ROWS_FOR_TRAINING} rows are needed "
            "to create a prediction model."
        )

    if "Traffic_Volume" not in df.columns:
        raise ValueError(
            "Traffic_Volume column is required."
        )

    if df["Traffic_Volume"].nunique() < 2:
        raise ValueError(
            "Traffic_Volume has no variation, "
            "so prediction cannot be performed."
        )

    data = df.copy()

    numeric, categorical = get_feature_lists(data, model_name)

    features = make_input_frame(
        data,
        numeric,
        categorical
    )

    target = pd.to_numeric(
        data["Traffic_Volume"],
        errors="coerce"
    )

    valid = target.notna()

    features = features.loc[valid].reset_index(drop=True)
    target = target.loc[valid].reset_index(drop=True)

    # Deterministic train/test split.
    split_point = int(len(features) * (1 - test_size))

    split_point = max(
        1,
        min(split_point, len(features) - 1)
    )

    x_train = features.iloc[:split_point].copy()
    x_test = features.iloc[split_point:].copy()

    y_train = target.iloc[:split_point].copy()
    y_test = target.iloc[split_point:].copy()

    # Evaluate the historical-pattern predictor on test data.
    predictions = []

    training_data = x_train.copy()
    training_data["Traffic_Volume"] = y_train.values

    for _, row in x_test.iterrows():

        prediction = _predict_from_history(
            training_data,
            row
        )

        predictions.append(prediction)

    predictions = np.asarray(predictions)

    train_predictions = []

    for _, row in x_train.iterrows():

        prediction = _predict_from_history(
            training_data,
            row,
            exclude_matching_row=False
        )

        train_predictions.append(prediction)

    train_predictions = np.asarray(train_predictions)

    return {
        "model_name": "Historical Pattern",
        "pipeline": training_data,
        "numeric_features": numeric,
        "categorical_features": categorical,
        "train_rows": len(x_train),
        "test_rows": len(x_test),
        "y_test": y_test.reset_index(drop=True),
        "y_predicted": pd.Series(predictions),
        "train_metrics": evaluate(
            y_train,
            train_predictions
        ),
        "test_metrics": evaluate(
            y_test,
            predictions
        ),
    }


def _predict_from_history(
    training_data,
    input_row,
    exclude_matching_row=False
):
    """Predict using the most similar historical records."""

    if training_data.empty:
        return 0.0

    scores = _calculate_similarity(
        training_data,
        input_row
    )

    # Take the strongest historical matches.
    top_count = min(
        15,
        len(training_data)
    )

    top_indices = scores.nlargest(top_count).index

    top_scores = scores.loc[top_indices]

    traffic_values = training_data.loc[
        top_indices,
        "Traffic_Volume"
    ].astype(float)

    # Prevent zero total weight.
    if top_scores.sum() <= 0:
        prediction = traffic_values.mean()

    else:
        prediction = np.average(
            traffic_values,
            weights=top_scores + 0.001
        )

    return float(max(prediction, 0))


def evaluate(y_true, y_predicted):
    """Calculate R², MAE and RMSE without sklearn."""

    y_true = np.asarray(y_true, dtype=float)
    y_predicted = np.asarray(y_predicted, dtype=float)

    if len(y_true) == 0:
        return {
            "R2": 0.0,
            "MAE": 0.0,
            "RMSE": 0.0
        }

    errors = y_true - y_predicted

    mae = np.mean(np.abs(errors))

    rmse = np.sqrt(
        np.mean(errors ** 2)
    )

    ss_res = np.sum(errors ** 2)

    ss_tot = np.sum(
        (y_true - np.mean(y_true)) ** 2
    )

    if ss_tot == 0:
        r2 = 0.0

    else:
        r2 = 1 - (ss_res / ss_tot)

    return {
        "R2": float(r2),
        "MAE": float(mae),
        "RMSE": float(rmse)
    }


def metrics_table(results):
    """Create a model evaluation table."""

    rows = []

    for name, result in results.items():

        rows.append({
            "Model": result.get(
                "model_name",
                name
            ),

            "R² (train)": round(
                result["train_metrics"]["R2"],
                3
            ),

            "R² (test)": round(
                result["test_metrics"]["R2"],
                3
            ),

            "MAE (test)": round(
                result["test_metrics"]["MAE"],
                1
            ),

            "RMSE (test)": round(
                result["test_metrics"]["RMSE"],
                1
            )
        })

    return pd.DataFrame(rows)


def predict_traffic(result, inputs):
    """
    Predict traffic volume for one situation.

    inputs:
        Dictionary containing feature values.
    """

    numeric = result["numeric_features"]
    categorical = result["categorical_features"]

    required = numeric + categorical

    missing = [
        column
        for column in required
        if column not in inputs
    ]

    if missing:
        raise ValueError(
            "Missing input values for: "
            + ", ".join(missing)
        )

    row = pd.Series(inputs)

    prediction = _predict_from_history(
        result["pipeline"],
        row
    )

    return float(max(prediction, 0))


def get_feature_importance(result, top_n=10):
    """
    Estimate feature importance using correlation with
    historical traffic volume.

    This is not sklearn feature importance.
    """

    data = result["pipeline"].copy()

    if data.empty:
        return pd.DataFrame(
            columns=["Feature", "Importance"]
        )

    target = data["Traffic_Volume"].astype(float)

    scores = []

    for feature in (
        result["numeric_features"]
        + result["categorical_features"]
    ):

        if feature not in data.columns:
            continue

        if pd.api.types.is_numeric_dtype(
            data[feature]
        ):

            correlation = data[
                [feature]
            ].assign(
                Traffic_Volume=target
            ).corr().iloc[0, 1]

        else:

            grouped = data.groupby(
                feature
            )["Traffic_Volume"].mean()

            encoded = data[
                feature
            ].map(grouped)

            correlation = encoded.corr(target)

        if pd.isna(correlation):
            importance = 0.0

        else:
            importance = abs(float(correlation))

        scores.append({
            "Feature": feature,
            "Importance": importance
        })

    table = pd.DataFrame(scores)

    if table.empty:
        return table

    return (
        table
        .sort_values(
            "Importance",
            ascending=False
        )
        .head(top_n)
        .reset_index(drop=True)
    )