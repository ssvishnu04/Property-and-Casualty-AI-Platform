import json
from pathlib import Path

import pandas as pd


PORTFOLIO_SUMMARY_PATH = Path("data/gold/portfolio_summary/portfolio_summary.csv")
REINSURANCE_EXPOSURE_PATH = Path(
    "data/gold/reinsurance_exposure/reinsurance_exposure.csv"
)
PREDICTION_LOG_PATH = Path("data/prediction_logs/prediction_log.jsonl")
MODEL_COMPARISON_PATH = Path("reports/model_metrics/model_comparison_summary.csv")
RAGAS_RESULTS_PATH = Path("reports/ragas/ragas_results.csv")
CHAMPION_METADATA_PATH = Path("data/model_artifacts/champion_metadata.json")
MONITORING_SUMMARY_PATH = Path("reports/monitoring/monitoring_summary.json")
FEATURE_DRIFT_REPORT_PATH = Path("reports/monitoring/feature_drift_report.csv")
PREDICTION_DRIFT_REPORT_PATH = Path("reports/monitoring/prediction_drift_report.csv")
SCORED_CLAIMS_PATH = Path("data/gold/scored_claims/scored_claims.csv")

def load_scored_claims() -> pd.DataFrame:
    return read_csv_if_exists(SCORED_CLAIMS_PATH)

def load_monitoring_summary() -> dict:
    if not MONITORING_SUMMARY_PATH.exists():
        return {}

    with MONITORING_SUMMARY_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_feature_drift_report() -> pd.DataFrame:
    return read_csv_if_exists(FEATURE_DRIFT_REPORT_PATH)


def load_prediction_drift_report() -> pd.DataFrame:
    return read_csv_if_exists(PREDICTION_DRIFT_REPORT_PATH)
def read_csv_if_exists(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def load_portfolio_summary() -> pd.DataFrame:
    return read_csv_if_exists(PORTFOLIO_SUMMARY_PATH)


def load_reinsurance_exposure() -> pd.DataFrame:
    return read_csv_if_exists(REINSURANCE_EXPOSURE_PATH)


def load_model_comparison() -> pd.DataFrame:
    return read_csv_if_exists(MODEL_COMPARISON_PATH)


def load_ragas_results() -> pd.DataFrame:
    return read_csv_if_exists(RAGAS_RESULTS_PATH)


def load_champion_metadata() -> dict:
    if not CHAMPION_METADATA_PATH.exists():
        return {}

    with CHAMPION_METADATA_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_prediction_logs() -> pd.DataFrame:
    if not PREDICTION_LOG_PATH.exists():
        return pd.DataFrame()

    rows = []

    with PREDICTION_LOG_PATH.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue

            event = json.loads(line)
            response = event.get("response", {})

            rows.append(
                {
                    "prediction_timestamp_utc": event.get(
                        "prediction_timestamp_utc"
                    ),
                    "claim_id": event.get("claim_id"),
                    "severity_prediction": response.get("severity_prediction"),
                    "fraud_risk_prediction": response.get("fraud_risk_prediction"),
                    "fraud_risk_probability": response.get(
                        "fraud_risk_probability"
                    ),
                    "recommended_reserve": response.get("recommended_reserve"),
                    "triage_priority": response.get("triage_priority"),
                    "retention_breach_flag": response.get("retention_breach_flag"),
                    "calculated_ceded_loss": response.get("calculated_ceded_loss"),
                    "high_priority_claim_flag": response.get(
                        "high_priority_claim_flag"
                    ),
                }
            )

    return pd.DataFrame(rows)


def format_currency(value) -> str:
    try:
        return f"${float(value):,.0f}"
    except Exception:
        return "$0"


def format_percent(value) -> str:
    try:
        return f"{float(value) * 100:.1f}%"
    except Exception:
        return "0.0%"