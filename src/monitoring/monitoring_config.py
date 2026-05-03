from pathlib import Path


TRAINING_BASELINE_PATH = Path("data/gold/claim_features/claim_features.csv")
BATCH_SCORED_CLAIMS_PATH = Path("data/gold/scored_claims/scored_claims.csv")
PREDICTION_LOG_PATH = Path("data/prediction_logs/prediction_log.jsonl")

MONITORING_REPORT_DIR = Path("reports/monitoring")

FEATURE_DRIFT_REPORT_PATH = MONITORING_REPORT_DIR / "feature_drift_report.csv"
PREDICTION_DRIFT_REPORT_PATH = MONITORING_REPORT_DIR / "prediction_drift_report.csv"
MONITORING_SUMMARY_PATH = MONITORING_REPORT_DIR / "monitoring_summary.json"


NUMERIC_FEATURES_TO_MONITOR = [
    "claim_amount",
    "reserve_amount",
    "prior_claim_count",
    "retention",
    "treaty_limit",
    "calculated_ceded_loss",
    "reinsurance_recovery_ratio",
    "combined_risk_score",
    "loss_year",
    "loss_month",
    "loss_quarter",
]


CATEGORICAL_FEATURES_TO_MONITOR = [
    "line_of_business",
    "loss_type",
    "state",
    "treaty_type",
]


PREDICTION_COLUMNS_TO_MONITOR = [
    "severity_prediction",
    "fraud_risk_prediction",
    "fraud_risk_probability",
    "recommended_reserve",
    "triage_priority",
]


DRIFT_THRESHOLDS = {
    "mean_change_pct": 0.25,
    "std_change_pct": 0.50,
    "missing_rate_change": 0.10,
    "category_share_change": 0.20,
    "prediction_share_change": 0.20,
}