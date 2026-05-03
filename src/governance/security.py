import hashlib
from copy import deepcopy


PII_FIELDS = [
    "claimant_name",
    "claimant_email",
    "claimant_phone",
    "claimant_address",
]


def mask_string(value: str, visible_chars: int = 2) -> str:
    if value is None:
        return value

    value = str(value)

    if len(value) <= visible_chars:
        return "*" * len(value)

    return value[:visible_chars] + "*" * (len(value) - visible_chars)


def hash_value(value: str) -> str:
    if value is None:
        return value

    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def mask_pii_record(record: dict) -> dict:
    masked = deepcopy(record)

    for field in PII_FIELDS:
        if field in masked:
            masked[field] = mask_string(masked[field])

    if "claimant_id" in masked:
        masked["claimant_id_hash"] = hash_value(masked["claimant_id"])
        del masked["claimant_id"]

    return masked