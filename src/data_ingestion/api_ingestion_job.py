import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

from src.config.settings import settings
from src.data_ingestion.audit_logger import write_audit_log
from src.data_ingestion.watermark_manager import get_watermark, update_watermark


BASE_URL = f"http://{settings.API_HOST}:{settings.API_PORT}"


def utc_now_string() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def file_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def call_api(endpoint: str) -> dict[str, Any]:
    url = f"{BASE_URL}{endpoint}"
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return response.json()


def extract_records(response: dict[str, Any], possible_keys: list[str]) -> list[dict[str, Any]]:
    for key in possible_keys:
        value = response.get(key)
        if isinstance(value, list):
            return value

    if isinstance(response, list):
        return response

    return []


def write_json_to_landing(
    source_name: str,
    ingestion_mode: str,
    payload: dict[str, Any],
) -> Path:
    landing_root = Path(settings.DATA_PATH)

    source_folder = landing_root / source_name / ingestion_mode
    source_folder.mkdir(parents=True, exist_ok=True)

    output_file = source_folder / f"{source_name}_{ingestion_mode}_{file_timestamp()}.json"

    with output_file.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return output_file


def build_landing_payload(
    source_name: str,
    endpoint: str,
    ingestion_mode: str,
    records: list[dict[str, Any]],
    watermark_used: str | None = None,
) -> dict[str, Any]:
    return {
        "ingestion_metadata": {
            "source_name": source_name,
            "source_endpoint": endpoint,
            "ingestion_timestamp_utc": utc_now_string(),
            "record_count": len(records),
            "ingestion_mode": ingestion_mode,
            "raw_format": "json",
            "watermark_used": watermark_used,
        },
        "data": records,
    }


def ingest_generic_source(
    source_name: str,
    endpoint: str,
    ingestion_mode: str,
    response_keys: list[str],
    watermark_key: str | None = None,
    update_watermark_after_success: bool = False,
) -> None:
    watermark_used = get_watermark(watermark_key) if watermark_key else None

    try:
        response = call_api(endpoint)
        records = extract_records(response, response_keys)

        payload = build_landing_payload(
            source_name=source_name,
            endpoint=endpoint,
            ingestion_mode=ingestion_mode,
            records=records,
            watermark_used=watermark_used,
        )

        output_file = write_json_to_landing(source_name, ingestion_mode, payload)

        if watermark_key and update_watermark_after_success:
            update_watermark(watermark_key)

        write_audit_log(
            {
                "source_name": source_name,
                "ingestion_mode": ingestion_mode,
                "status": "success",
                "record_count": len(records),
                "watermark_used": watermark_used,
                "output_file": str(output_file),
            }
        )

        print(f"{source_name} {ingestion_mode} ingested: {len(records)} → {output_file}")

    except Exception as exc:
        write_audit_log(
            {
                "source_name": source_name,
                "ingestion_mode": ingestion_mode,
                "status": "failed",
                "watermark_used": watermark_used,
                "error_message": str(exc),
            }
        )
        raise


def ingest_claims_full_batch() -> None:
    ingest_generic_source(
        source_name="claims",
        ingestion_mode="full_batch",
        endpoint="/claims/new?limit=25",
        response_keys=["claims"],
    )


def ingest_claims_delta() -> None:
    watermark_key = "claims_delta"
    since_timestamp = get_watermark(watermark_key)

    ingest_generic_source(
        source_name="claims",
        ingestion_mode="delta",
        endpoint=f"/claims/updated?since_timestamp={since_timestamp}&limit=25",
        response_keys=["claims"],
        watermark_key=watermark_key,
        update_watermark_after_success=True,
    )


def ingest_open_claims_updated_today() -> None:
    ingest_generic_source(
        source_name="claims",
        ingestion_mode="open_claims",
        endpoint="/claims/open?limit=25",
        response_keys=["claims"],
    )


def ingest_fnol_micro_batch() -> None:
    watermark_key = "fnol_micro_batch"

    ingest_generic_source(
        source_name="fnol",
        ingestion_mode="micro_batch",
        endpoint="/fnol/events?minutes=15",
        response_keys=["fnol_events", "events"],
        watermark_key=watermark_key,
        update_watermark_after_success=True,
    )


def ingest_cat_events_full_batch() -> None:
    ingest_generic_source(
        source_name="cat_events",
        ingestion_mode="full_batch",
        endpoint="/cat-events/",
        response_keys=["cat_events", "events"],
    )


def ingest_policies_full_batch() -> None:
    ingest_generic_source(
        source_name="policies",
        ingestion_mode="full_batch",
        endpoint="/policies/",
        response_keys=["policies"],
    )


def ingest_treaties_full_batch() -> None:
    ingest_generic_source(
        source_name="treaties",
        ingestion_mode="full_batch",
        endpoint="/treaties/",
        response_keys=["treaties"],
    )


def ingest_exposures_full_batch() -> None:
    ingest_generic_source(
        source_name="exposures",
        ingestion_mode="full_batch",
        endpoint="/exposures/",
        response_keys=["exposures"],
    )


def ingest_reference_data_full_batch() -> None:
    ingest_policies_full_batch()
    ingest_treaties_full_batch()
    ingest_exposures_full_batch()
    ingest_cat_events_full_batch()


def run_mode(mode: str) -> None:
    if mode == "full_batch":
        ingest_claims_full_batch()
    elif mode == "delta":
        ingest_claims_delta()
    elif mode == "fnol":
        ingest_fnol_micro_batch()
    elif mode == "open_claims":
        ingest_open_claims_updated_today()
    elif mode == "cat_events":
        ingest_cat_events_full_batch()
    elif mode == "policies":
        ingest_policies_full_batch()
    elif mode == "treaties":
        ingest_treaties_full_batch()
    elif mode == "exposures":
        ingest_exposures_full_batch()
    elif mode == "reference":
        ingest_reference_data_full_batch()
    elif mode == "all":
        ingest_claims_full_batch()
        ingest_claims_delta()
        ingest_open_claims_updated_today()
        ingest_fnol_micro_batch()
        ingest_reference_data_full_batch()
    else:
        raise ValueError(
            "Invalid mode. Use one of: full_batch, delta, fnol, open_claims, "
            "cat_events, policies, treaties, exposures, reference, all"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SpecialtyRe AI API ingestion job")

    parser.add_argument(
        "--mode",
        required=True,
        choices=[
            "full_batch",
            "delta",
            "fnol",
            "open_claims",
            "cat_events",
            "policies",
            "treaties",
            "exposures",
            "reference",
            "all",
        ],
        help="Ingestion mode to run",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_mode(args.mode)