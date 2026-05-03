import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.monitoring.monitoring_config import (
    BATCH_SCORED_CLAIMS_PATH,
    CATEGORICAL_FEATURES_TO_MONITOR,
    DRIFT_THRESHOLDS,
    FEATURE_DRIFT_REPORT_PATH,
    MONITORING_REPORT_DIR,
    MONITORING_SUMMARY_PATH,
    NUMERIC_FEATURES_TO_MONITOR,
    PREDICTION_COLUMNS_TO_MONITOR,
    PREDICTION_DRIFT_REPORT_PATH,
    PREDICTION_LOG_PATH,
    TRAINING_BASELINE_PATH,
)


def utc_now_string() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        print(f"File not found: {path}")
        return pd.DataFrame()

    return pd.read_csv(path)


def load_prediction_logs() -> pd.DataFrame:
    if not PREDICTION_LOG_PATH.exists():
        print(f"Prediction log not found: {PREDICTION_LOG_PATH}")
        return pd.DataFrame()

    rows = []

    with PREDICTION_LOG_PATH.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue

            event = json.loads(line)
            request = event.get("request", {})
            response = event.get("response", {})

            row = {
                "prediction_timestamp_utc": event.get("prediction_timestamp_utc"),
                "claim_id": event.get("claim_id"),
                **request,
                **response,
            }

            rows.append(row)

    return pd.DataFrame(rows)


def safe_pct_change(current_value: float, baseline_value: float) -> float:
    if baseline_value == 0 or pd.isna(baseline_value):
        return 0.0

    return (current_value - baseline_value) / abs(baseline_value)


def calculate_numeric_feature_drift(
    baseline_df: pd.DataFrame,
    current_df: pd.DataFrame,
    column: str,
) -> dict[str, Any]:
    baseline_series = pd.to_numeric(baseline_df[column], errors="coerce")
    current_series = pd.to_numeric(current_df[column], errors="coerce")

    baseline_mean = baseline_series.mean()
    current_mean = current_series.mean()

    baseline_std = baseline_series.std()
    current_std = current_series.std()

    baseline_missing_rate = baseline_series.isna().mean()
    current_missing_rate = current_series.isna().mean()

    mean_change_pct = safe_pct_change(current_mean, baseline_mean)
    std_change_pct = safe_pct_change(current_std, baseline_std)
    missing_rate_change = current_missing_rate - baseline_missing_rate

    drift_detected = (
        abs(mean_change_pct) >= DRIFT_THRESHOLDS["mean_change_pct"]
        or abs(std_change_pct) >= DRIFT_THRESHOLDS["std_change_pct"]
        or abs(missing_rate_change) >= DRIFT_THRESHOLDS["missing_rate_change"]
    )

    return {
        "column": column,
        "feature_type": "numeric",
        "baseline_mean": baseline_mean,
        "current_mean": current_mean,
        "mean_change_pct": mean_change_pct,
        "baseline_std": baseline_std,
        "current_std": current_std,
        "std_change_pct": std_change_pct,
        "baseline_missing_rate": baseline_missing_rate,
        "current_missing_rate": current_missing_rate,
        "missing_rate_change": missing_rate_change,
        "drift_detected": drift_detected,
    }


def get_top_category_share(df: pd.DataFrame, column: str) -> tuple[str | None, float]:
    if column not in df.columns or df.empty:
        return None, 0.0

    value_counts = df[column].astype("string").value_counts(normalize=True, dropna=False)

    if value_counts.empty:
        return None, 0.0

    top_category = str(value_counts.index[0])
    top_share = float(value_counts.iloc[0])

    return top_category, top_share


def calculate_categorical_feature_drift(
    baseline_df: pd.DataFrame,
    current_df: pd.DataFrame,
    column: str,
) -> dict[str, Any]:
    baseline_top_category, baseline_top_share = get_top_category_share(
        baseline_df,
        column,
    )
    current_top_category, current_top_share = get_top_category_share(
        current_df,
        column,
    )

    share_change = current_top_share - baseline_top_share

    baseline_unique_count = (
        baseline_df[column].nunique(dropna=False) if column in baseline_df.columns else 0
    )
    current_unique_count = (
        current_df[column].nunique(dropna=False) if column in current_df.columns else 0
    )

    drift_detected = (
        baseline_top_category != current_top_category
        or abs(share_change) >= DRIFT_THRESHOLDS["category_share_change"]
        or current_unique_count > baseline_unique_count
    )

    return {
        "column": column,
        "feature_type": "categorical",
        "baseline_top_category": baseline_top_category,
        "current_top_category": current_top_category,
        "baseline_top_share": baseline_top_share,
        "current_top_share": current_top_share,
        "top_share_change": share_change,
        "baseline_unique_count": baseline_unique_count,
        "current_unique_count": current_unique_count,
        "drift_detected": drift_detected,
    }


def build_feature_drift_report(
    baseline_df: pd.DataFrame,
    current_df: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for column in NUMERIC_FEATURES_TO_MONITOR:
        if column in baseline_df.columns and column in current_df.columns:
            rows.append(
                calculate_numeric_feature_drift(
                    baseline_df=baseline_df,
                    current_df=current_df,
                    column=column,
                )
            )

    for column in CATEGORICAL_FEATURES_TO_MONITOR:
        if column in baseline_df.columns and column in current_df.columns:
            rows.append(
                calculate_categorical_feature_drift(
                    baseline_df=baseline_df,
                    current_df=current_df,
                    column=column,
                )
            )

    return pd.DataFrame(rows)


def calculate_prediction_distribution_drift(
    baseline_df: pd.DataFrame,
    current_df: pd.DataFrame,
    column: str,
) -> dict[str, Any]:
    baseline_top_category, baseline_top_share = get_top_category_share(
        baseline_df,
        column,
    )
    current_top_category, current_top_share = get_top_category_share(
        current_df,
        column,
    )

    share_change = current_top_share - baseline_top_share

    drift_detected = (
        baseline_top_category != current_top_category
        or abs(share_change) >= DRIFT_THRESHOLDS["prediction_share_change"]
    )

    return {
        "column": column,
        "metric_type": "prediction_distribution",
        "baseline_top_category": baseline_top_category,
        "current_top_category": current_top_category,
        "baseline_top_share": baseline_top_share,
        "current_top_share": current_top_share,
        "top_share_change": share_change,
        "drift_detected": drift_detected,
    }


def calculate_numeric_prediction_drift(
    baseline_df: pd.DataFrame,
    current_df: pd.DataFrame,
    column: str,
) -> dict[str, Any]:
    baseline_series = pd.to_numeric(baseline_df[column], errors="coerce")
    current_series = pd.to_numeric(current_df[column], errors="coerce")

    baseline_mean = baseline_series.mean()
    current_mean = current_series.mean()

    mean_change_pct = safe_pct_change(current_mean, baseline_mean)

    drift_detected = abs(mean_change_pct) >= DRIFT_THRESHOLDS["mean_change_pct"]

    return {
        "column": column,
        "metric_type": "numeric_prediction",
        "baseline_mean": baseline_mean,
        "current_mean": current_mean,
        "mean_change_pct": mean_change_pct,
        "drift_detected": drift_detected,
    }


def build_prediction_drift_report(
    baseline_scored_df: pd.DataFrame,
    current_prediction_df: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for column in PREDICTION_COLUMNS_TO_MONITOR:
        if column not in baseline_scored_df.columns or column not in current_prediction_df.columns:
            continue

        if column in ["fraud_risk_probability", "recommended_reserve"]:
            rows.append(
                calculate_numeric_prediction_drift(
                    baseline_df=baseline_scored_df,
                    current_df=current_prediction_df,
                    column=column,
                )
            )
        else:
            rows.append(
                calculate_prediction_distribution_drift(
                    baseline_df=baseline_scored_df,
                    current_df=current_prediction_df,
                    column=column,
                )
            )

    return pd.DataFrame(rows)


def build_monitoring_summary(
    feature_drift_df: pd.DataFrame,
    prediction_drift_df: pd.DataFrame,
    baseline_count: int,
    current_count: int,
) -> dict[str, Any]:
    feature_drift_count = (
        int(feature_drift_df["drift_detected"].sum())
        if not feature_drift_df.empty and "drift_detected" in feature_drift_df.columns
        else 0
    )

    prediction_drift_count = (
        int(prediction_drift_df["drift_detected"].sum())
        if not prediction_drift_df.empty and "drift_detected" in prediction_drift_df.columns
        else 0
    )

    status = "healthy"

    if feature_drift_count > 0 or prediction_drift_count > 0:
        status = "review_required"

    return {
        "monitoring_timestamp_utc": utc_now_string(),
        "status": status,
        "baseline_record_count": baseline_count,
        "current_prediction_record_count": current_count,
        "feature_drift_count": feature_drift_count,
        "prediction_drift_count": prediction_drift_count,
        "feature_drift_report_path": str(FEATURE_DRIFT_REPORT_PATH),
        "prediction_drift_report_path": str(PREDICTION_DRIFT_REPORT_PATH),
    }


def run_monitoring() -> None:
    print("Starting monitoring and drift detection...")

    MONITORING_REPORT_DIR.mkdir(parents=True, exist_ok=True)

    baseline_features_df = read_csv_if_exists(TRAINING_BASELINE_PATH)
    baseline_scored_df = read_csv_if_exists(BATCH_SCORED_CLAIMS_PATH)
    current_prediction_df = load_prediction_logs()

    if baseline_features_df.empty:
        print("No baseline feature data found. Run Gold feature engineering first.")
        return

    if baseline_scored_df.empty:
        print("No batch scored claims found. Run batch scoring first.")
        return

    if current_prediction_df.empty:
        print("No prediction logs found. Run real-time predictions first.")
        return

    feature_drift_df = build_feature_drift_report(
        baseline_df=baseline_features_df,
        current_df=current_prediction_df,
    )

    prediction_drift_df = build_prediction_drift_report(
        baseline_scored_df=baseline_scored_df,
        current_prediction_df=current_prediction_df,
    )

    feature_drift_df.to_csv(FEATURE_DRIFT_REPORT_PATH, index=False)
    prediction_drift_df.to_csv(PREDICTION_DRIFT_REPORT_PATH, index=False)

    summary = build_monitoring_summary(
        feature_drift_df=feature_drift_df,
        prediction_drift_df=prediction_drift_df,
        baseline_count=len(baseline_features_df),
        current_count=len(current_prediction_df),
    )

    with MONITORING_SUMMARY_PATH.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print("Monitoring completed.")
    print(f"Feature drift report: {FEATURE_DRIFT_REPORT_PATH}")
    print(f"Prediction drift report: {PREDICTION_DRIFT_REPORT_PATH}")
    print(f"Monitoring summary: {MONITORING_SUMMARY_PATH}")
    print(summary)


if __name__ == "__main__":
    run_monitoring()