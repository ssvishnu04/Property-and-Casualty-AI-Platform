from pathlib import Path

import pandas as pd


BRONZE_CLAIMS_PATH = Path("data/bronze/claims_raw/claims_bronze.csv")
BRONZE_FNOL_PATH = Path("data/bronze/fnol_raw/fnol_bronze.csv")
BRONZE_CAT_EVENTS_PATH = Path("data/bronze/cat_events_raw/cat_events_bronze.csv")
BRONZE_POLICIES_PATH = Path("data/bronze/policies_raw/policies_bronze.csv")
BRONZE_TREATIES_PATH = Path("data/bronze/treaties_raw/treaties_bronze.csv")
BRONZE_EXPOSURES_PATH = Path("data/bronze/exposures_raw/exposures_bronze.csv")

SILVER_CLAIMS_CLEAN_PATH = Path("data/silver/claims_clean/claims_clean.csv")
SILVER_CLAIMS_CURRENT_PATH = Path("data/silver/claims_current/claims_current.csv")
SILVER_FNOL_CLEAN_PATH = Path("data/silver/fnol_clean/fnol_clean.csv")
SILVER_CAT_EVENTS_CLEAN_PATH = Path("data/silver/cat_events_clean/cat_events_clean.csv")
SILVER_POLICIES_CLEAN_PATH = Path("data/silver/policies_clean/policies_clean.csv")
SILVER_TREATIES_CLEAN_PATH = Path("data/silver/treaties_clean/treaties_clean.csv")
SILVER_EXPOSURES_CLEAN_PATH = Path("data/silver/exposures_clean/exposures_clean.csv")

QUARANTINE_CLAIMS_PATH = Path("data/silver/quarantine/claims_quarantine.csv")


REQUIRED_CLAIM_COLUMNS = [
    "claim_id",
    "policy_id",
    "line_of_business",
    "loss_type",
    "state",
    "claim_amount",
    "reserve_amount",
    "severity_label",
    "suspicious_flag",
]


NUMERIC_COLUMNS = [
    "claim_amount",
    "reserve_amount",
    "prior_claim_count",
    "litigation_flag",
    "cat_exposure",
    "suspicious_flag",
    "retention",
    "treaty_limit",
    "reinsurance_triggered",
    "ceded_loss",
    "policy_limit",
    "deductible",
    "exposure_amount",
    "location_risk_score",
]


TIMESTAMP_COLUMNS = [
    "loss_date",
    "created_timestamp_utc",
    "updated_timestamp_utc",
    "_ingestion_timestamp_utc",
    "event_timestamp_utc",
    "effective_date",
    "expiration_date",
]


STRING_COLUMNS = [
    "claim_id",
    "policy_id",
    "treaty_id",
    "event_id",
    "exposure_id",
    "line_of_business",
    "loss_type",
    "state",
    "severity_label",
    "claim_status",
    "treaty_type",
    "cat_event_type",
    "event_name",
]


def read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        print(f"File not found: {path}")
        return pd.DataFrame()

    return pd.read_csv(path)


def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("-", "_")
    )

    return df


def ensure_required_columns(df: pd.DataFrame, required_columns: list[str]) -> pd.DataFrame:
    df = df.copy()

    for column in required_columns:
        if column not in df.columns:
            df[column] = None

    return df


def cast_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for column in NUMERIC_COLUMNS:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    return df


def cast_timestamp_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for column in TIMESTAMP_COLUMNS:
        if column in df.columns:
            df[column] = pd.to_datetime(df[column], errors="coerce", utc=True)

    return df


def normalize_string_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for column in STRING_COLUMNS:
        if column in df.columns:
            df[column] = df[column].astype("string").str.strip()

    if "state" in df.columns:
        df["state"] = df["state"].str.upper()

    if "severity_label" in df.columns:
        df["severity_label"] = df["severity_label"].str.title()

    if "claim_status" in df.columns:
        df["claim_status"] = df["claim_status"].fillna("Unknown").str.title()

    return df


def add_quality_flags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["dq_missing_claim_id"] = df["claim_id"].isna() | (
        df["claim_id"].astype(str).str.len() == 0
    )
    df["dq_missing_policy_id"] = df["policy_id"].isna() | (
        df["policy_id"].astype(str).str.len() == 0
    )
    df["dq_invalid_claim_amount"] = df["claim_amount"].isna() | (
        df["claim_amount"] <= 0
    )
    df["dq_invalid_reserve_amount"] = df["reserve_amount"].isna() | (
        df["reserve_amount"] < 0
    )
    df["dq_missing_lob"] = df["line_of_business"].isna() | (
        df["line_of_business"].astype(str).str.len() == 0
    )
    df["dq_missing_loss_type"] = df["loss_type"].isna() | (
        df["loss_type"].astype(str).str.len() == 0
    )

    dq_columns = [
        "dq_missing_claim_id",
        "dq_missing_policy_id",
        "dq_invalid_claim_amount",
        "dq_invalid_reserve_amount",
        "dq_missing_lob",
        "dq_missing_loss_type",
    ]

    df["dq_has_error"] = df[dq_columns].any(axis=1)

    return df


def split_valid_and_quarantine(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    valid_df = df[df["dq_has_error"] == False].copy()
    quarantine_df = df[df["dq_has_error"] == True].copy()

    return valid_df, quarantine_df


def deduplicate_by_key(df: pd.DataFrame, key: str) -> pd.DataFrame:
    df = df.copy()

    if key not in df.columns:
        return df

    sort_columns = []

    if "updated_timestamp_utc" in df.columns:
        sort_columns.append("updated_timestamp_utc")

    if "_ingestion_timestamp_utc" in df.columns:
        sort_columns.append("_ingestion_timestamp_utc")

    if sort_columns:
        df = df.sort_values(sort_columns)

    return df.drop_duplicates(subset=[key], keep="last")


def create_current_claims_view(df: pd.DataFrame) -> pd.DataFrame:
    current_df = deduplicate_by_key(df, "claim_id")

    if "claim_status" in current_df.columns:
        current_df = current_df[current_df["claim_status"].isin(["Open", "Unknown"])]

    return current_df


def drop_sensitive_raw_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    sensitive_columns = [
        "claimant_name",
        "claimant_email",
        "claimant_phone",
        "claimant_address",
    ]

    columns_to_drop = [column for column in sensitive_columns if column in df.columns]

    if columns_to_drop:
        df = df.drop(columns=columns_to_drop)

    return df


def clean_generic_data(df: pd.DataFrame, dedupe_key: str | None = None) -> pd.DataFrame:
    df = standardize_column_names(df)
    df = cast_numeric_columns(df)
    df = cast_timestamp_columns(df)
    df = normalize_string_columns(df)
    df = drop_sensitive_raw_columns(df)

    if dedupe_key and dedupe_key in df.columns:
        df = deduplicate_by_key(df, dedupe_key)

    return df


def write_dataframe(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Wrote {len(df)} records to {path}")


def process_claims_to_silver() -> None:
    print("Starting Silver claims processing...")

    bronze_claims_df = read_csv_if_exists(BRONZE_CLAIMS_PATH)

    if bronze_claims_df.empty:
        print("No Bronze claims data found.")
        return

    df = standardize_column_names(bronze_claims_df)
    df = ensure_required_columns(df, REQUIRED_CLAIM_COLUMNS)
    df = cast_numeric_columns(df)
    df = cast_timestamp_columns(df)
    df = normalize_string_columns(df)
    df = add_quality_flags(df)

    valid_df, quarantine_df = split_valid_and_quarantine(df)

    valid_df = drop_sensitive_raw_columns(valid_df)
    quarantine_df = drop_sensitive_raw_columns(quarantine_df)

    current_df = create_current_claims_view(valid_df)

    write_dataframe(valid_df, SILVER_CLAIMS_CLEAN_PATH)
    write_dataframe(current_df, SILVER_CLAIMS_CURRENT_PATH)

    if not quarantine_df.empty:
        write_dataframe(quarantine_df, QUARANTINE_CLAIMS_PATH)
    else:
        print("No quarantined claim records.")


def process_fnol_to_silver() -> None:
    print("Starting Silver FNOL processing...")

    df = read_csv_if_exists(BRONZE_FNOL_PATH)

    if df.empty:
        print("No Bronze FNOL data found.")
        return

    df = clean_generic_data(df, dedupe_key="event_id")
    write_dataframe(df, SILVER_FNOL_CLEAN_PATH)


def process_cat_events_to_silver() -> None:
    print("Starting Silver CAT events processing...")

    df = read_csv_if_exists(BRONZE_CAT_EVENTS_PATH)

    if df.empty:
        print("No Bronze CAT events data found.")
        return

    dedupe_key = "event_id" if "event_id" in df.columns else None
    df = clean_generic_data(df, dedupe_key=dedupe_key)

    write_dataframe(df, SILVER_CAT_EVENTS_CLEAN_PATH)


def process_policies_to_silver() -> None:
    print("Starting Silver policies processing...")

    df = read_csv_if_exists(BRONZE_POLICIES_PATH)

    if df.empty:
        print("No Bronze policies data found.")
        return

    df = clean_generic_data(df, dedupe_key="policy_id")
    write_dataframe(df, SILVER_POLICIES_CLEAN_PATH)


def process_treaties_to_silver() -> None:
    print("Starting Silver treaties processing...")

    df = read_csv_if_exists(BRONZE_TREATIES_PATH)

    if df.empty:
        print("No Bronze treaties data found.")
        return

    df = clean_generic_data(df, dedupe_key="treaty_id")
    write_dataframe(df, SILVER_TREATIES_CLEAN_PATH)


def process_exposures_to_silver() -> None:
    print("Starting Silver exposures processing...")

    df = read_csv_if_exists(BRONZE_EXPOSURES_PATH)

    if df.empty:
        print("No Bronze exposures data found.")
        return

    dedupe_key = "exposure_id" if "exposure_id" in df.columns else None
    df = clean_generic_data(df, dedupe_key=dedupe_key)

    write_dataframe(df, SILVER_EXPOSURES_CLEAN_PATH)


def run_silver_processing() -> None:
    print("Starting Silver layer processing...")

    process_claims_to_silver()
    process_fnol_to_silver()
    process_cat_events_to_silver()
    process_policies_to_silver()
    process_treaties_to_silver()
    process_exposures_to_silver()

    print("Silver layer processing completed successfully.")


if __name__ == "__main__":
    run_silver_processing()