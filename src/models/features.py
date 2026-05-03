from pathlib import Path


GOLD_FEATURES_PATH = Path("data/gold/claim_features/claim_features.csv")

MODEL_ARTIFACT_DIR = Path("data/model_artifacts")
METRICS_REPORT_DIR = Path("reports/model_metrics")

MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT_NAME = "SpecialtyRe_AI_Model_Experimentation"


CATEGORICAL_FEATURES = [
    "line_of_business",
    "loss_type",
    "state",
    "treaty_type",
]


NUMERIC_FEATURES = [
    "claim_amount",
    "prior_claim_count",
    "litigation_flag",
    "cat_exposure",
    "retention",
    "treaty_limit",
    "retention_breach_flag",
    "calculated_ceded_loss",
    "reinsurance_recovery_ratio",
    "treaty_exhaustion_ratio",
    "prior_claim_frequency_risk",
    "litigation_risk_flag",
    "catastrophe_risk_flag",
    "suspicious_claim_flag",
    "combined_risk_score",
    "high_priority_claim_flag",
    "text_fraud_signal",
    "text_litigation_signal",
    "text_cat_signal",
    "text_injury_signal",
    "source_document_count",
    "loss_year",
    "loss_month",
    "loss_quarter",
]


SEVERITY_TARGET = "severity_label"
FRAUD_TARGET = "suspicious_flag"
RESERVE_TARGET = "reserve_amount"