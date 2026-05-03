import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.config.settings import settings


AUDIT_DIR = Path(settings.DATA_PATH) / "_audit"
AUDIT_FILE = AUDIT_DIR / "ingestion_audit_log.jsonl"


def utc_now_string() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def write_audit_log(event: dict[str, Any]) -> None:
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)

    event["audit_timestamp_utc"] = utc_now_string()

    with AUDIT_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event) + "\n")