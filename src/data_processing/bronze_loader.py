import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.config.settings import settings


LANDING_ROOT = Path(settings.DATA_PATH)
BRONZE_ROOT = Path("data/bronze")


BRONZE_OUTPUTS = {
    "claims": BRONZE_ROOT / "claims_raw" / "claims_bronze.csv",
    "fnol": BRONZE_ROOT / "fnol_raw" / "fnol_bronze.csv",
    "cat_events": BRONZE_ROOT / "cat_events_raw" / "cat_events_bronze.csv",
    "policies": BRONZE_ROOT / "policies_raw" / "policies_bronze.csv",
    "treaties": BRONZE_ROOT / "treaties_raw" / "treaties_bronze.csv",
    "exposures": BRONZE_ROOT / "exposures_raw" / "exposures_bronze.csv",
    "ingestion_audit": BRONZE_ROOT / "ingestion_audit" / "ingestion_audit_bronze.csv",
}


def read_json_file(file_path: Path) -> dict[str, Any]:
    with file_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def flatten_landing_file(file_path: Path) -> list[dict[str, Any]]:
    payload = read_json_file(file_path)

    metadata = payload.get("ingestion_metadata", {})
    records = payload.get("data", [])

    flattened_records = []

    for record in records:
        row = record.copy()

        row["_source_file"] = str(file_path)
        row["_source_name"] = metadata.get("source_name")
        row["_source_endpoint"] = metadata.get("source_endpoint")
        row["_ingestion_timestamp_utc"] = metadata.get("ingestion_timestamp_utc")
        row["_ingestion_mode"] = metadata.get("ingestion_mode")
        row["_record_count_in_file"] = metadata.get("record_count")
        row["_raw_format"] = metadata.get("raw_format")
        row["_watermark_used"] = metadata.get("watermark_used")

        flattened_records.append(row)

    return flattened_records


def collect_landing_records(source_name: str) -> list[dict[str, Any]]:
    source_path = LANDING_ROOT / source_name

    if not source_path.exists():
        print(f"No landing folder found for source: {source_name}")
        return []

    json_files = list(source_path.rglob("*.json"))

    if not json_files:
        print(f"No JSON files found for source: {source_name}")
        return []

    all_records = []

    for json_file in json_files:
        try:
            records = flatten_landing_file(json_file)
            all_records.extend(records)
        except Exception as exc:
            print(f"Failed to process file {json_file}: {exc}")

    return all_records


def write_bronze_csv(records: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not records:
        print(f"No records to write for {output_path}")
        return

    df = pd.DataFrame(records)
    df.to_csv(output_path, index=False)

    print(f"Wrote {len(df)} records to {output_path}")


def load_source_to_bronze(source_name: str) -> None:
    records = collect_landing_records(source_name)
    output_path = BRONZE_OUTPUTS[source_name]
    write_bronze_csv(records, output_path)


def load_audit_log_to_bronze() -> None:
    audit_file = LANDING_ROOT / "_audit" / "ingestion_audit_log.jsonl"

    if not audit_file.exists():
        print("No audit log found.")
        return

    records = []

    with audit_file.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                records.append(json.loads(line))

    write_bronze_csv(records, BRONZE_OUTPUTS["ingestion_audit"])


def run_bronze_load() -> None:
    print("Starting Bronze load...")

    for source_name in [
        "claims",
        "fnol",
        "cat_events",
        "policies",
        "treaties",
        "exposures",
    ]:
        load_source_to_bronze(source_name)

    load_audit_log_to_bronze()

    print("Bronze load completed successfully.")


if __name__ == "__main__":
    run_bronze_load()