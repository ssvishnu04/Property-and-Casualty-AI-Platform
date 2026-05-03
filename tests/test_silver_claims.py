import pandas as pd

from src.data_processing.silver_claims import (
    add_quality_flags,
    deduplicate_claims,
    standardize_column_names,
)


def test_standardize_column_names():
    df = pd.DataFrame(
        {
            "Claim ID": ["CLM-1"],
            "Policy-ID": ["POL-1"],
        }
    )

    result = standardize_column_names(df)

    assert "claim_id" in result.columns
    assert "policy_id" in result.columns


def test_add_quality_flags_valid_record():
    df = pd.DataFrame(
        {
            "claim_id": ["CLM-1"],
            "policy_id": ["POL-1"],
            "line_of_business": ["Commercial Property"],
            "loss_type": ["Hail"],
            "claim_amount": [100000],
            "reserve_amount": [120000],
        }
    )

    result = add_quality_flags(df)

    assert result["dq_has_error"].iloc[0] == False


def test_add_quality_flags_invalid_claim_amount():
    df = pd.DataFrame(
        {
            "claim_id": ["CLM-1"],
            "policy_id": ["POL-1"],
            "line_of_business": ["Commercial Property"],
            "loss_type": ["Hail"],
            "claim_amount": [-100],
            "reserve_amount": [120000],
        }
    )

    result = add_quality_flags(df)

    assert result["dq_has_error"].iloc[0] == True


def test_deduplicate_claims_keeps_latest():
    df = pd.DataFrame(
        {
            "claim_id": ["CLM-1", "CLM-1"],
            "claim_amount": [100000, 200000],
            "updated_timestamp_utc": pd.to_datetime(
                ["2026-04-28T10:00:00Z", "2026-04-28T11:00:00Z"]
            ),
        }
    )

    result = deduplicate_claims(df)

    assert len(result) == 1
    assert result["claim_amount"].iloc[0] == 200000