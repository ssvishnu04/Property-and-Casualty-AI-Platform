import pandas as pd

from src.scoring.batch_score_claims import (
    assign_batch_triage_priority,
    ensure_feature_columns,
)


def test_ensure_feature_columns_adds_missing_columns():
    df = pd.DataFrame(
        {
            "claim_id": ["CLM-1"],
            "claim_amount": [100000],
        }
    )

    result = ensure_feature_columns(df)

    assert "loss_year" in result.columns
    assert "loss_month" in result.columns
    assert "loss_quarter" in result.columns
    assert "retention_breach_flag" in result.columns


def test_assign_batch_triage_priority_urgent_for_high_severity():
    row = pd.Series(
        {
            "severity_prediction": "High",
            "fraud_risk_probability": 0.1,
            "high_priority_claim_flag": 0,
        }
    )

    assert assign_batch_triage_priority(row) == "Urgent"


def test_assign_batch_triage_priority_review_for_medium_severity():
    row = pd.Series(
        {
            "severity_prediction": "Medium",
            "fraud_risk_probability": 0.1,
            "high_priority_claim_flag": 0,
        }
    )

    assert assign_batch_triage_priority(row) == "Review"


def test_assign_batch_triage_priority_standard_for_low_risk():
    row = pd.Series(
        {
            "severity_prediction": "Low",
            "fraud_risk_probability": 0.1,
            "high_priority_claim_flag": 0,
        }
    )

    assert assign_batch_triage_priority(row) == "Standard"