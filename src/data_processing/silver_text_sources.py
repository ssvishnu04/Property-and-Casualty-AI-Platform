from pathlib import Path

import pandas as pd


BRONZE_FILE_SOURCES_PATH = Path("data/bronze/file_sources_raw/file_sources_bronze.csv")

SILVER_ACORD_CLAIMS_PATH = Path("data/silver/acord_claims_clean/acord_claims_clean.csv")
SILVER_TEXT_FEATURES_PATH = Path("data/silver/text_features_clean/text_features_clean.csv")


FRAUD_KEYWORDS = [
    "suspicious",
    "inconsistent",
    "changed the loss description",
    "late reporting",
    "siu",
    "fraud",
]

LITIGATION_KEYWORDS = [
    "attorney",
    "litigation",
    "demand letter",
    "legal",
    "lawsuit",
    "coverage dispute",
]

CAT_KEYWORDS = [
    "storm",
    "hail",
    "hurricane",
    "wildfire",
    "flood",
    "catastrophe",
    "wind",
]

INJURY_KEYWORDS = [
    "injury",
    "bodily injury",
    "medical",
    "hospital",
]


def read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        print(f"File not found: {path}")
        return pd.DataFrame()

    return pd.read_csv(path)


def keyword_flag(text: str, keywords: list[str]) -> int:
    if not isinstance(text, str):
        return 0

    text_lower = text.lower()

    return int(any(keyword in text_lower for keyword in keywords))


def clean_acord_xml_records(df: pd.DataFrame) -> pd.DataFrame:
    acord_df = df[df["file_type"] == "xml"].copy()

    if acord_df.empty:
        return acord_df

    numeric_columns = [
        "claim_amount",
        "reserve_amount",
        "prior_claim_count",
        "litigation_flag",
        "cat_exposure",
        "suspicious_flag",
        "retention",
        "treaty_limit",
    ]

    for column in numeric_columns:
        if column in acord_df.columns:
            acord_df[column] = pd.to_numeric(acord_df[column], errors="coerce")

    if "loss_date" in acord_df.columns:
        acord_df["loss_date"] = pd.to_datetime(
            acord_df["loss_date"],
            errors="coerce",
            utc=True,
        )

    string_columns = [
        "claim_id",
        "policy_id",
        "treaty_id",
        "line_of_business",
        "loss_type",
        "state",
        "treaty_type",
    ]

    for column in string_columns:
        if column in acord_df.columns:
            acord_df[column] = acord_df[column].astype("string").str.strip()

    if "state" in acord_df.columns:
        acord_df["state"] = acord_df["state"].str.upper()

    return acord_df


def build_text_features(df: pd.DataFrame) -> pd.DataFrame:
    text_df = df[df["file_type"] == "txt"].copy()

    if text_df.empty:
        return pd.DataFrame()

    text_df["raw_text"] = text_df["raw_text"].fillna("")

    text_df["text_fraud_signal"] = text_df["raw_text"].apply(
        lambda text: keyword_flag(text, FRAUD_KEYWORDS)
    )
    text_df["text_litigation_signal"] = text_df["raw_text"].apply(
        lambda text: keyword_flag(text, LITIGATION_KEYWORDS)
    )
    text_df["text_cat_signal"] = text_df["raw_text"].apply(
        lambda text: keyword_flag(text, CAT_KEYWORDS)
    )
    text_df["text_injury_signal"] = text_df["raw_text"].apply(
        lambda text: keyword_flag(text, INJURY_KEYWORDS)
    )

    grouped = (
        text_df.groupby("claim_id", dropna=False)
        .agg(
            text_fraud_signal=("text_fraud_signal", "max"),
            text_litigation_signal=("text_litigation_signal", "max"),
            text_cat_signal=("text_cat_signal", "max"),
            text_injury_signal=("text_injury_signal", "max"),
            source_document_count=("file_name", "count"),
        )
        .reset_index()
    )

    grouped = grouped[grouped["claim_id"].notna()]

    return grouped


def write_dataframe(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if df.empty:
        print(f"No records to write for {path}")
        return

    df.to_csv(path, index=False)
    print(f"Wrote {len(df)} records to {path}")


def run_silver_text_processing() -> None:
    print("Starting Silver text/XML source processing...")

    bronze_df = read_csv_if_exists(BRONZE_FILE_SOURCES_PATH)

    if bronze_df.empty:
        print("No Bronze file source data found.")
        return

    acord_df = clean_acord_xml_records(bronze_df)
    text_features_df = build_text_features(bronze_df)

    write_dataframe(acord_df, SILVER_ACORD_CLAIMS_PATH)
    write_dataframe(text_features_df, SILVER_TEXT_FEATURES_PATH)

    print("Silver text/XML processing completed successfully.")


if __name__ == "__main__":
    run_silver_text_processing()