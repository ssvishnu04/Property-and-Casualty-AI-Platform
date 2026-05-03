import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.data_ingestion.audit_logger import write_audit_log


EXTERNAL_ROOT = Path("data/external")
LANDING_ROOT = Path("data/local_landing/file_sources")


SOURCE_CONFIG = {
    "acord_xml": {
        "source_path": EXTERNAL_ROOT / "acord_xml",
        "file_pattern": "*.xml",
    },
    "adjuster_notes": {
        "source_path": EXTERNAL_ROOT / "adjuster_notes",
        "file_pattern": "*.txt",
    },
    "claim_emails": {
        "source_path": EXTERNAL_ROOT / "claim_emails",
        "file_pattern": "*.txt",
    },
}


def utc_now_string() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def file_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def build_file_metadata(
    source_name: str,
    source_file: Path,
    landing_file: Path,
) -> dict[str, Any]:
    return {
        "source_name": source_name,
        "source_file_name": source_file.name,
        "source_file_path": str(source_file),
        "landing_file_path": str(landing_file),
        "file_extension": source_file.suffix.lower(),
        "file_size_bytes": source_file.stat().st_size,
        "ingestion_timestamp_utc": utc_now_string(),
        "raw_format": source_file.suffix.lower().replace(".", ""),
    }


def ingest_source_files(source_name: str) -> None:
    config = SOURCE_CONFIG[source_name]
    source_path = config["source_path"]
    file_pattern = config["file_pattern"]

    if not source_path.exists():
        print(f"Source folder not found: {source_path}")
        return

    files = list(source_path.glob(file_pattern))

    if not files:
        print(f"No files found for {source_name}")
        return

    landing_source_folder = LANDING_ROOT / source_name / "full_batch"
    landing_source_folder.mkdir(parents=True, exist_ok=True)

    metadata_records = []

    for source_file in files:
        landing_file = landing_source_folder / (
            f"{source_file.stem}_{file_timestamp()}{source_file.suffix}"
        )

        shutil.copy2(source_file, landing_file)

        metadata = build_file_metadata(
            source_name=source_name,
            source_file=source_file,
            landing_file=landing_file,
        )

        metadata_records.append(metadata)

    metadata_file = landing_source_folder / f"{source_name}_metadata_{file_timestamp()}.json"

    with metadata_file.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "ingestion_metadata": {
                    "source_name": source_name,
                    "ingestion_timestamp_utc": utc_now_string(),
                    "record_count": len(metadata_records),
                    "ingestion_mode": "full_batch",
                    "raw_format": "file",
                },
                "data": metadata_records,
            },
            file,
            indent=2,
        )

    write_audit_log(
        {
            "source_name": source_name,
            "ingestion_mode": "file_full_batch",
            "status": "success",
            "record_count": len(metadata_records),
            "output_file": str(metadata_file),
        }
    )

    print(f"Ingested {len(metadata_records)} {source_name} files → {landing_source_folder}")


def run_file_ingestion() -> None:
    print("Starting file source ingestion...")

    for source_name in SOURCE_CONFIG:
        ingest_source_files(source_name)

    print("File source ingestion completed successfully.")


if __name__ == "__main__":
    run_file_ingestion()