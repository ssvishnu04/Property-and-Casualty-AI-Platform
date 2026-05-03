from pathlib import Path

import numpy as np
import pandas as pd


SILVER_CLAIMS_CURRENT_PATH = Path("data/silver/claims_current/claims_current.csv")
SILVER_CAT_EVENTS_CLEAN_PATH = Path("data/silver/cat_events_clean/cat_events_clean.csv")
SILVER_POLICIES_CLEAN_PATH = Path("data/silver/policies_clean/policies_clean.csv")
SILVER_TREATIES_CLEAN_PATH = Path("data/silver/treaties_clean/treaties_clean.csv")
SILVER_EXPOSURES_CLEAN_PATH = Path("data/silver/exposures_clean/exposures_clean.csv")
SILVER_ACORD_CLAIMS_PATH = Path("data/silver/acord_claims_clean/acord_claims_clean.csv")
SILVER_TEXT_FEATURES_PATH = Path("data/silver/text_features_clean/text_features_clean.csv")

GOLD_CLAIM_FEATURES_PATH = Path("data/gold/claim_features/claim_features.csv")
GOLD_REINSURANCE_EXPOSURE_PATH = Path("data/gold/reinsurance_exposure/reinsurance_exposure.csv")
GOLD_PORTFOLIO_SUMMARY_PATH = Path("data/gold/portfolio_summary/portfolio_summary.csv")



def read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        print(f"File not found: {path}")
        return pd.DataFrame()
    return pd.read_csv(path)


def safe_divide(numerator, denominator):
    denominator = np.where(denominator == 0, np.nan, denominator)
    return numerator / denominator


def enrich_with_policy_data(claims_df: pd.DataFrame, policies_df: pd.DataFrame) -> pd.DataFrame:
    if policies_df.empty or "policy_id" not in policies_df.columns:
        return claims_df

    cols = [c for c in ["policy_id", "policy_limit", "deductible"] if c in policies_df.columns]
    if len(cols) <= 1:
        return claims_df

    return claims_df.merge(
        policies_df[cols].drop_duplicates("policy_id"),
        on="policy_id",
        how="left",
    )


def enrich_with_treaty_data(claims_df: pd.DataFrame, treaties_df: pd.DataFrame) -> pd.DataFrame:
    if treaties_df.empty or "treaty_id" not in claims_df.columns or "treaty_id" not in treaties_df.columns:
        return claims_df

    cols = [c for c in ["treaty_id", "treaty_type", "retention", "treaty_limit"] if c in treaties_df.columns]
    if len(cols) <= 1:
        return claims_df

    df = claims_df.merge(
        treaties_df[cols].drop_duplicates("treaty_id"),
        on="treaty_id",
        how="left",
        suffixes=("", "_treaty"),
    )

    for col in ["treaty_type", "retention", "treaty_limit"]:
        treaty_col = f"{col}_treaty"
        if treaty_col in df.columns:
            if col in df.columns:
                df[col] = df[col].fillna(df[treaty_col])
            else:
                df[col] = df[treaty_col]
            df = df.drop(columns=[treaty_col])

    return df


def enrich_with_exposure_data(claims_df: pd.DataFrame, exposures_df: pd.DataFrame) -> pd.DataFrame:
    if exposures_df.empty:
        return claims_df

    join_key = None
    if "policy_id" in claims_df.columns and "policy_id" in exposures_df.columns:
        join_key = "policy_id"
    elif "claim_id" in claims_df.columns and "claim_id" in exposures_df.columns:
        join_key = "claim_id"

    if not join_key:
        return claims_df

    cols = [
        c for c in [
            join_key,
            "cat_exposure",
            "exposure_amount",
            "location_risk_score",
        ]
        if c in exposures_df.columns
    ]

    if len(cols) <= 1:
        return claims_df

    df = claims_df.merge(
        exposures_df[cols].drop_duplicates(join_key),
        on=join_key,
        how="left",
        suffixes=("", "_exposure"),
    )

    if "cat_exposure_exposure" in df.columns:
        if "cat_exposure" in df.columns:
            df["cat_exposure"] = df["cat_exposure"].fillna(df["cat_exposure_exposure"])
        else:
            df["cat_exposure"] = df["cat_exposure_exposure"]
        df = df.drop(columns=["cat_exposure_exposure"])

    return df


def enrich_with_cat_event_data(claims_df: pd.DataFrame, cat_events_df: pd.DataFrame) -> pd.DataFrame:
    df = claims_df.copy()

    if "active_cat_event_flag" not in df.columns:
        df["active_cat_event_flag"] = 0

    if cat_events_df.empty or "state" not in df.columns or "state" not in cat_events_df.columns:
        return df

    cat_events_df = cat_events_df.copy()
    cat_events_df["state"] = cat_events_df["state"].astype("string").str.upper()

    active_states = set(cat_events_df["state"].dropna().unique())

    df["active_cat_event_flag"] = (
        df["state"].astype("string").str.upper().isin(active_states).astype(int)
    )

    if "cat_exposure" in df.columns:
        df["cat_exposure"] = np.where(df["active_cat_event_flag"] == 1, 1, df["cat_exposure"])
    else:
        df["cat_exposure"] = df["active_cat_event_flag"]

    return df


def add_claim_financial_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in ["policy_limit", "deductible"]:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["claim_amount"] = pd.to_numeric(df["claim_amount"], errors="coerce").fillna(0)
    df["reserve_amount"] = pd.to_numeric(df["reserve_amount"], errors="coerce").fillna(0)

    df["claim_to_policy_limit_ratio"] = safe_divide(df["claim_amount"], df["policy_limit"])
    df["reserve_to_claim_ratio"] = safe_divide(df["reserve_amount"], df["claim_amount"]).fillna(0)
    df["deductible_to_claim_ratio"] = safe_divide(df["deductible"], df["claim_amount"]).fillna(0)

    df["large_loss_flag"] = (df["claim_amount"] >= 250000).astype(int)
    df["very_large_loss_flag"] = (df["claim_amount"] >= 1000000).astype(int)

    return df


def add_reinsurance_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for column in ["retention", "treaty_limit", "claim_amount", "ceded_loss"]:
        if column not in df.columns:
            df[column] = 0
        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0)

    df["retention_breach_flag"] = (df["claim_amount"] > df["retention"]).astype(int)

    df["net_retained_loss"] = np.where(
        df["retention_breach_flag"] == 1,
        df["retention"],
        df["claim_amount"],
    )

    df["calculated_ceded_loss"] = np.where(
        df["retention_breach_flag"] == 1,
        np.minimum(df["claim_amount"] - df["retention"], df["treaty_limit"]),
        0,
    )

    df["reinsurance_recovery_ratio"] = safe_divide(
        df["calculated_ceded_loss"],
        df["claim_amount"],
    ).fillna(0)

    df["treaty_exhaustion_ratio"] = safe_divide(
        df["calculated_ceded_loss"],
        df["treaty_limit"],
    ).fillna(0)

    df["high_reinsurance_impact_flag"] = (
        df["reinsurance_recovery_ratio"] >= 0.5
    ).astype(int)

    return df

def combine_api_and_acord_claims(
    api_claims_df: pd.DataFrame,
    acord_claims_df: pd.DataFrame,
) -> pd.DataFrame:
    if acord_claims_df.empty:
        return api_claims_df

    common_columns = list(set(api_claims_df.columns).union(set(acord_claims_df.columns)))

    api_aligned = api_claims_df.reindex(columns=common_columns)
    acord_aligned = acord_claims_df.reindex(columns=common_columns)

    combined_df = pd.concat([api_aligned, acord_aligned], ignore_index=True)

    if "claim_id" in combined_df.columns:
        combined_df = combined_df.drop_duplicates(subset=["claim_id"], keep="last")

    return combined_df

def enrich_with_text_features(
    claims_df: pd.DataFrame,
    text_features_df: pd.DataFrame,
) -> pd.DataFrame:
    df = claims_df.copy()

    text_feature_columns = [
        "claim_id",
        "text_fraud_signal",
        "text_litigation_signal",
        "text_cat_signal",
        "text_injury_signal",
        "source_document_count",
    ]

    if text_features_df.empty or "claim_id" not in text_features_df.columns:
        for column in text_feature_columns:
            if column != "claim_id" and column not in df.columns:
                df[column] = 0
        return df

    available_columns = [
        column for column in text_feature_columns
        if column in text_features_df.columns
    ]

    df = df.merge(
        text_features_df[available_columns],
        on="claim_id",
        how="left",
    )

    for column in text_feature_columns:
        if column != "claim_id" and column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0)

    return df

def add_risk_indicator_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for column in [
        "prior_claim_count",
        "litigation_flag",
        "cat_exposure",
        "suspicious_flag",
        "active_cat_event_flag",
        "text_fraud_signal",
        "text_litigation_signal",
        "text_cat_signal",
        "text_injury_signal",
        "source_document_count",
    ]:
        if column not in df.columns:
            df[column] = 0
        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0)

    df["prior_claim_frequency_risk"] = np.where(df["prior_claim_count"] >= 3, 1, 0)
    df["litigation_risk_flag"] = (
        (df["litigation_flag"] == 1) | (df["text_litigation_signal"] == 1)
    ).astype(int)

    df["catastrophe_risk_flag"] = (
        (df["cat_exposure"] == 1) | (df["active_cat_event_flag"] == 1) | (df["text_cat_signal"] == 1)
    ).astype(int)

    df["suspicious_claim_flag"] = ((df["suspicious_flag"] == 1) | (df["text_fraud_signal"] == 1)
    ).astype(int)

    df["combined_risk_score"] = (
        df["prior_claim_frequency_risk"] * 0.25
        + df["litigation_risk_flag"] * 0.25
        + df["catastrophe_risk_flag"] * 0.20
        + df["suspicious_claim_flag"] * 0.30
    )

    df["high_priority_claim_flag"] = (
        (df["combined_risk_score"] >= 0.50)
        | (df["claim_amount"] >= 300000)
        | (df["retention_breach_flag"] == 1)
    ).astype(int)

    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "loss_date" in df.columns:
        df["loss_date"] = pd.to_datetime(df["loss_date"], errors="coerce", utc=True)
        df["loss_year"] = df["loss_date"].dt.year
        df["loss_month"] = df["loss_date"].dt.month
        df["loss_quarter"] = df["loss_date"].dt.quarter
    else:
        df["loss_year"] = 2026
        df["loss_month"] = 1
        df["loss_quarter"] = 1

    df["loss_year"] = df["loss_year"].fillna(2026).astype(int)
    df["loss_month"] = df["loss_month"].fillna(1).astype(int)
    df["loss_quarter"] = df["loss_quarter"].fillna(1).astype(int)

    return df


def create_claim_features(
    claims_df: pd.DataFrame,
    policies_df: pd.DataFrame,
    treaties_df: pd.DataFrame,
    exposures_df: pd.DataFrame,
    cat_events_df: pd.DataFrame,
    text_features_df: pd.DataFrame,
) -> pd.DataFrame:
    df = claims_df.copy()

    df = enrich_with_policy_data(df, policies_df)
    df = enrich_with_treaty_data(df, treaties_df)
    df = enrich_with_exposure_data(df, exposures_df)
    df = enrich_with_cat_event_data(df, cat_events_df)
    df = enrich_with_text_features(df, text_features_df)

    df = add_claim_financial_features(df)
    df = add_reinsurance_features(df)
    df = add_risk_indicator_features(df)
    df = add_time_features(df)

    return df


def create_reinsurance_exposure(df: pd.DataFrame) -> pd.DataFrame:
    grouping_columns = ["line_of_business", "treaty_type"]

    summary = (
        df.groupby(grouping_columns, dropna=False)
        .agg(
            claim_count=("claim_id", "count"),
            total_claim_amount=("claim_amount", "sum"),
            total_reserve_amount=("reserve_amount", "sum"),
            total_ceded_loss=("calculated_ceded_loss", "sum"),
            avg_reinsurance_recovery_ratio=("reinsurance_recovery_ratio", "mean"),
            retention_breach_count=("retention_breach_flag", "sum"),
            high_reinsurance_impact_count=("high_reinsurance_impact_flag", "sum"),
        )
        .reset_index()
    )

    summary["ceded_loss_ratio"] = safe_divide(
        summary["total_ceded_loss"],
        summary["total_claim_amount"],
    ).fillna(0)

    return summary


def create_portfolio_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby("line_of_business", dropna=False)
        .agg(
            claim_count=("claim_id", "count"),
            total_claim_amount=("claim_amount", "sum"),
            average_claim_amount=("claim_amount", "mean"),
            total_reserve_amount=("reserve_amount", "sum"),
            total_ceded_loss=("calculated_ceded_loss", "sum"),
            large_loss_count=("large_loss_flag", "sum"),
            high_priority_claim_count=("high_priority_claim_flag", "sum"),
            average_combined_risk_score=("combined_risk_score", "mean"),
        )
        .reset_index()
    )

    summary["high_priority_rate"] = safe_divide(
        summary["high_priority_claim_count"],
        summary["claim_count"],
    ).fillna(0)

    return summary


def write_dataframe(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Wrote {len(df)} records to {path}")


def run_gold_feature_engineering() -> None:
    print("Starting Gold feature engineering...")

    silver_claims_df = read_csv_if_exists(SILVER_CLAIMS_CURRENT_PATH)

    if silver_claims_df.empty:
        print("No Silver current claims data found.")
        return

    policies_df = read_csv_if_exists(SILVER_POLICIES_CLEAN_PATH)
    treaties_df = read_csv_if_exists(SILVER_TREATIES_CLEAN_PATH)
    exposures_df = read_csv_if_exists(SILVER_EXPOSURES_CLEAN_PATH)
    cat_events_df = read_csv_if_exists(SILVER_CAT_EVENTS_CLEAN_PATH)
    acord_claims_df = read_csv_if_exists(SILVER_ACORD_CLAIMS_PATH)
    text_features_df = read_csv_if_exists(SILVER_TEXT_FEATURES_PATH)

    combined_claims_df = combine_api_and_acord_claims(
       api_claims_df=silver_claims_df,
       acord_claims_df=acord_claims_df,
    )

    claim_features_df = create_claim_features(
         claims_df=combined_claims_df,
         policies_df=policies_df,
         treaties_df=treaties_df,
         exposures_df=exposures_df,
         cat_events_df=cat_events_df,
         text_features_df=text_features_df,
   )

    reinsurance_exposure_df = create_reinsurance_exposure(claim_features_df)
    portfolio_summary_df = create_portfolio_summary(claim_features_df)

    write_dataframe(claim_features_df, GOLD_CLAIM_FEATURES_PATH)
    write_dataframe(reinsurance_exposure_df, GOLD_REINSURANCE_EXPOSURE_PATH)
    write_dataframe(portfolio_summary_df, GOLD_PORTFOLIO_SUMMARY_PATH)

    print("Gold feature engineering completed successfully.")


if __name__ == "__main__":
    run_gold_feature_engineering()