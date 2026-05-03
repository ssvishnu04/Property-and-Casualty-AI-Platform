import json
from datetime import datetime, timezone
from pathlib import Path

from src.config.settings import settings


WATERMARK_DIR = Path(settings.DATA_PATH) / "_watermarks"
WATERMARK_FILE = WATERMARK_DIR / "watermarks.json"


DEFAULT_WATERMARK = "1970-01-01T00:00:00Z"


def utc_now_string() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def ensure_watermark_file_exists() -> None:
    WATERMARK_DIR.mkdir(parents=True, exist_ok=True)

    if not WATERMARK_FILE.exists():
        WATERMARK_FILE.write_text("{}", encoding="utf-8")


def load_watermarks() -> dict:
    ensure_watermark_file_exists()

    with WATERMARK_FILE.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_watermark(source_name: str) -> str:
    watermarks = load_watermarks()

    return watermarks.get(source_name, DEFAULT_WATERMARK)


def update_watermark(source_name: str, watermark_value: str | None = None) -> None:
    watermarks = load_watermarks()

    watermarks[source_name] = watermark_value or utc_now_string()

    with WATERMARK_FILE.open("w", encoding="utf-8") as f:
        json.dump(watermarks, f, indent=2)