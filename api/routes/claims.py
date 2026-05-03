from datetime import datetime, timezone, timedelta

from fastapi import APIRouter, Query

from src.config.settings import settings
from src.data_ingestion.data_generator import generate_claim, generate_claims
from src.governance.security import mask_pii_record


router = APIRouter()


def enrich_claim_runtime_fields(claim: dict, minutes_ago: int = 0) -> dict:
    now = datetime.now(timezone.utc)

    claim["created_timestamp_utc"] = (
        now - timedelta(minutes=minutes_ago)
    ).strftime("%Y-%m-%dT%H:%M:%SZ")

    claim["updated_timestamp_utc"] = now.strftime("%Y-%m-%dT%H:%M:%SZ")

    claim["claim_status"] = "Open"

    return claim


@router.get("/new")
def get_new_claims(limit: int = Query(default=5, ge=1, le=100)):
    claims = generate_claims(limit)

    enriched_claims = [
        enrich_claim_runtime_fields(claim, minutes_ago=i * 10)
        for i, claim in enumerate(claims)
    ]

    if settings.MASK_PII:
        enriched_claims = [mask_pii_record(claim) for claim in enriched_claims]

    return {
        "source_system": "claims-admin-system",
        "record_count": len(enriched_claims),
        "claims": enriched_claims,
    }


@router.get("/updated")
def get_updated_claims(
    since_timestamp: str,
    limit: int = Query(default=10, ge=1, le=100),
):
    claims = generate_claims(limit)

    enriched_claims = [
        enrich_claim_runtime_fields(claim, minutes_ago=i * 5)
        for i, claim in enumerate(claims)
    ]

    if settings.MASK_PII:
        enriched_claims = [mask_pii_record(claim) for claim in enriched_claims]

    return {
        "source_system": "claims-admin-system",
        "since_timestamp": since_timestamp,
        "record_count": len(enriched_claims),
        "claims": enriched_claims,
    }


@router.get("/open")
def get_open_claims_updated_today(limit: int = Query(default=10, ge=1, le=100)):
    claims = generate_claims(limit)

    enriched_claims = [
        enrich_claim_runtime_fields(claim, minutes_ago=i * 20)
        for i, claim in enumerate(claims)
    ]

    for claim in enriched_claims:
        claim["claim_status"] = "Open"

    if settings.MASK_PII:
        enriched_claims = [mask_pii_record(claim) for claim in enriched_claims]

    return {
        "source_system": "claims-admin-system",
        "filter": "open_claims_updated_today",
        "record_count": len(enriched_claims),
        "claims": enriched_claims,
    }


@router.get("/{claim_id}")
def get_claim_by_id(claim_id: str):
    claim = generate_claim()
    claim["claim_id"] = claim_id
    claim = enrich_claim_runtime_fields(claim)

    if settings.MASK_PII:
        claim = mask_pii_record(claim)

    return {
        "source_system": "claims-admin-system",
        "claim": claim,
    }