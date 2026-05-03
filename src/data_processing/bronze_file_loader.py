import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import pandas as pd


FILE_LANDING_ROOT = Path("data/local_landing/file_sources")
BRONZE_FILE_OUTPUT_PATH = Path("data/bronze/file_sources_raw/file_sources_bronze.csv")


def read_text_file(path: Path) -> str:
    with path.open("r", encoding="utf-8") as file:
        return file.read()


def parse_xml_file(path: Path) -> dict[str, Any]:
    tree = ET.parse(path)
    root = tree.getroot()

    parsed = {}

    for child in root:
        key = child.tag.strip()
        value = child.text.strip() if child.text else None
        parsed[key] = value

    return parsed


def extract_claim_id_from_text(text: str, fallback_name: str) -> str | None:
    match = re.search(r"(CLM-[A-Z0-9-]+)", text, re.IGNORECASE)

    if match:
        return match.group(1).upper()

    file_match = re.search(r"(CLM-[A-Z0-9-]+)", fallback_name, re.IGNORECASE)

    if file_match:
        return file_match.group(1).upper()

    return None


def normalize_xml_keys(parsed_xml: dict[str, Any]) -> dict[str, Any]:
    key_map = {
        "ClaimId": "claim_id",
        "PolicyId": "policy_id",
        "TreatyId": "treaty_id",
        "LineOfBusiness": "line_of_business",
        "LossType": "loss_type",
        "State": "state",
        "LossDate": "loss_date",
        "ClaimAmount": "claim_amount",
        "ReserveAmount": "reserve_amount",
        "PriorClaimCount": "prior_claim_count",
        "LitigationFlag": "litigation_flag",
        "CatExposure": "cat_exposure",
        "SuspiciousFlag": "suspicious_flag",
        "TreatyType": "treaty_type",
        "Retention": "retention",
        "TreatyLimit": "treaty_limit",
    }

    normalized = {}

    for original_key, value in parsed_xml.items():
        normalized_key = key_map.get(original_key, original_key)
        normalized[normalized_key] = value

    return normalized


def load_file_records() -> list[dict[str, Any]]:
    records = []

    for source_folder in FILE_LANDING_ROOT.glob("*"):
        if not source_folder.is_dir():
            continue

        source_name = source_folder.name

        for file_path in source_folder.rglob("*"):
            if file_path.suffix.lower() not in [".txt", ".xml"]:
                continue

            if file_path.suffix.lower() == ".xml":
                parsed_xml = normalize_xml_keys(parse_xml_file(file_path))
                record = {
                    "source_name": source_name,
                    "file_name": file_path.name,
                    "file_path": str(file_path),
                    "file_type": "xml",
                    "claim_id": parsed_xml.get("claim_id"),
                    "raw_text": None,
                    **parsed_xml,
                }
            else:
                raw_text = read_text_file(file_path)
                record = {
                    "source_name": source_name,
                    "file_name": file_path.name,
                    "file_path": str(file_path),
                    "file_type": "txt",
                    "claim_id": extract_claim_id_from_text(raw_text, file_path.name),
                    "raw_text": raw_text,
                }

            records.append(record)

    return records


def run_bronze_file_load() -> None:
    print("Starting Bronze file source load...")

    records = load_file_records()

    if not records:
        print("No file source records found.")
        return

    BRONZE_FILE_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(records)
    df.to_csv(BRONZE_FILE_OUTPUT_PATH, index=False)

    print(f"Wrote {len(df)} file source records to {BRONZE_FILE_OUTPUT_PATH}")


if __name__ == "__main__":
    run_bronze_file_load()