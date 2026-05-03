import pandas as pd

from src.monitoring.drift_monitor import (
    calculate_numeric_feature_drift,
    calculate_categorical_feature_drift,
    safe_pct_change,
)


def test_safe_pct_change():
    result = safe_pct_change(125, 100)
    assert result == 0.25


def test_numeric_feature_drift_detects_mean_change():
    baseline_df = pd.DataFrame({"claim_amount": [100, 100, 100]})
    current_df = pd.DataFrame({"claim_amount": [200, 200, 200]})

    result = calculate_numeric_feature_drift(
        baseline_df=baseline_df,
        current_df=current_df,
        column="claim_amount",
    )

    assert bool(result["drift_detected"]) is True


def test_categorical_feature_drift_detects_top_category_change():
    baseline_df = pd.DataFrame({"state": ["TX", "TX", "FL"]})
    current_df = pd.DataFrame({"state": ["CA", "CA", "TX"]})

    result = calculate_categorical_feature_drift(
        baseline_df=baseline_df,
        current_df=current_df,
        column="state",
    )

    assert bool(result["drift_detected"]) is True