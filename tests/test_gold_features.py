import pandas as pd

from src.feature_engineering.gold_features import (
    add_reinsurance_features,
    add_risk_indicator_features,
    create_portfolio_summary,
)


def test_add_reinsurance_features_calculates_ceded_loss():
    df = pd.DataFrame(
        {
            "claim_id": ["CLM-1"],
            "claim_amount": [425000],
            "retention": [250000],
            "treaty_limit": [2000000],
            "ceded_loss": [0],
        }
    )

    result = add_reinsurance_features(df)

    assert result["retention_breach_flag"].iloc[0] == 1
    assert result["calculated_ceded_loss"].iloc[0] == 175000


def test_add_risk_indicator_features_creates_high_priority_flag():
    df = pd.DataFrame(
        {
            "claim_id": ["CLM-1"],
            "claim_amount": [425000],
            "prior_claim_count": [3],
            "litigation_flag": [1],
            "cat_exposure": [1],
            "suspicious_flag": [1],
            "retention_breach_flag": [1],
        }
    )

    result = add_risk_indicator_features(df)

    assert result["combined_risk_score"].iloc[0] >= 0.5
    assert result["high_priority_claim_flag"].iloc[0] == 1


def test_create_portfolio_summary_groups_by_lob():
    df = pd.DataFrame(
        {
            "claim_id": ["CLM-1", "CLM-2"],
            "line_of_business": ["Commercial Property", "Commercial Property"],
            "claim_amount": [100000, 200000],
            "reserve_amount": [120000, 220000],
            "calculated_ceded_loss": [0, 50000],
            "large_loss_flag": [0, 0],
            "high_priority_claim_flag": [1, 0],
            "combined_risk_score": [0.6, 0.2],
        }
    )

    result = create_portfolio_summary(df)

    assert len(result) == 1
    assert result["claim_count"].iloc[0] == 2
    assert result["total_claim_amount"].iloc[0] == 300000