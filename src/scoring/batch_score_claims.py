import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import pandas as pd


GOLD_FEATURES_PATH = Path("data/gold/claim_features/claim_features.csv")
MODEL_DIR = Path("data/model_artifacts")
CHAMPION_METADATA_PATH = MODEL_DIR / "champion_metadata.json"

SEVERITY_MODEL_PATH = MODEL_DIR / "severity_champion.pkl"
FRAUD_MODEL_PATH = MODEL_DIR / "fraud_champion.pkl"
RESERVE_MODEL_PATH = MODEL_DIR / "reserve_champion.pkl"

SCORED_OUTPUT_DIR = Path("data/gold/scored_claims")
SCORED_CLAIMS_PATH = SCORED_OUTPUT_DIR / "scored_claims.csv"
BATCH_AUDIT_PATH = SCORED_OUTPUT_DIR / "batch_scoring_audit.jsonl"


FEATURE_COLUMNS = [
    "line_of_business",
    "loss_type",
    "state",
    "treaty_type",
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


SEVERITY_LABEL_MAP = {
    0: "High",
    1: "Low",
    2: "Medium",
}


def utc_now_string() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_model(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Model artifact not found: {path}")

    artifact = joblib.load(path)

    if isinstance(artifact, dict) and "model" in artifact:
        return artifact["model"]

    return artifact


def load_champion_metadata() -> dict[str, Any]:
    if CHAMPION_METADATA_PATH.exists():
        with CHAMPION_METADATA_PATH.open("r", encoding="utf-8") as file:
            return json.load(file)

    return {}


def read_gold_features() -> pd.DataFrame:
    if not GOLD_FEATURES_PATH.exists():
        raise FileNotFoundError(
            f"Gold features not found: {GOLD_FEATURES_PATH}. "
            "Run Step 5 before batch scoring."
        )

    return pd.read_csv(GOLD_FEATURES_PATH)


def ensure_feature_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for column in FEATURE_COLUMNS:
        if column not in df.columns:
            df[column] = None

    return df


def normalize_severity(value: Any) -> str:
    try:
        return SEVERITY_LABEL_MAP.get(int(value), str(value))
    except Exception:
        return str(value)


def assign_batch_triage_priority(row: pd.Series) -> str:
    severity = row.get("severity_prediction")
    fraud_probability = float(row.get("fraud_risk_probability", 0))
    high_priority_claim_flag = int(row.get("high_priority_claim_flag", 0))

    if severity == "High" or fraud_probability >= 0.70 or high_priority_claim_flag == 1:
        return "Urgent"

    if severity == "Medium" or fraud_probability >= 0.40:
        return "Review"

    return "Standard"


def score_claims_batch(df: pd.DataFrame) -> pd.DataFrame:
    df = ensure_feature_columns(df)

    severity_model = load_model(SEVERITY_MODEL_PATH)
    fraud_model = load_model(FRAUD_MODEL_PATH)
    reserve_model = load_model(RESERVE_MODEL_PATH)

    X = df[FEATURE_COLUMNS]

    scored_df = df.copy()

    severity_raw = severity_model.predict(X)
    scored_df["severity_prediction"] = [
        normalize_severity(value) for value in severity_raw
    ]

    fraud_predictions = fraud_model.predict(X)
    scored_df["fraud_risk_prediction"] = fraud_predictions.astype(int)

    if hasattr(fraud_model, "predict_proba"):
        fraud_probabilities = fraud_model.predict_proba(X)

        if fraud_probabilities.shape[1] == 2:
            scored_df["fraud_risk_probability"] = fraud_probabilities[:, 1]
        else:
            scored_df["fraud_risk_probability"] = 0.0
    else:
        scored_df["fraud_risk_probability"] = 0.0

    scored_df["recommended_reserve"] = reserve_model.predict(X)

    scored_df["triage_priority"] = scored_df.apply(
        assign_batch_triage_priority,
        axis=1,
    )

    scored_df["batch_scoring_timestamp_utc"] = utc_now_string()

    return scored_df


def write_batch_audit(
    input_count: int,
    output_count: int,
    output_path: Path,
    metadata: dict[str, Any],
) -> None:
    SCORED_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    audit_event = {
        "batch_scoring_timestamp_utc": utc_now_string(),
        "input_record_count": input_count,
        "output_record_count": output_count,
        "output_path": str(output_path),
        "champion_metadata": metadata,
    }

    with BATCH_AUDIT_PATH.open("a", encoding="utf-8") as file:
        file.write(json.dumps(audit_event) + "\n")


def run_batch_scoring() -> None:
    print("Starting batch claim scoring...")

    SCORED_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    gold_df = read_gold_features()

    if gold_df.empty:
        print("No gold claim features found.")
        return

    metadata = load_champion_metadata()

    scored_df = score_claims_batch(gold_df)

    scored_df.to_csv(SCORED_CLAIMS_PATH, index=False)

    write_batch_audit(
        input_count=len(gold_df),
        output_count=len(scored_df),
        output_path=SCORED_CLAIMS_PATH,
        metadata=metadata,
    )

    print(f"Batch scoring completed. Scored {len(scored_df)} claims.")
    print(f"Output written to: {SCORED_CLAIMS_PATH}")
    print(f"Audit written to: {BATCH_AUDIT_PATH}")


if __name__ == "__main__":
    run_batch_scoring()