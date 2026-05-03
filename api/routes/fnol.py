from datetime import datetime, timezone, timedelta
import random

from fastapi import APIRouter, Query

from src.data_ingestion.data_generator import generate_claim


router = APIRouter()


@router.get("/events")
def get_fnol_events(minutes: int = Query(default=15, ge=1, le=120)):
    event_count = random.randint(3, 8)
    now = datetime.now(timezone.utc)

    events = []

    for i in range(event_count):
        claim = generate_claim()

        event_timestamp = now - timedelta(minutes=random.randint(0, minutes))

        events.append(
            {
                "event_id": f"FNOL-{random.randint(100000, 999999)}",
                "event_type": "FIRST_NOTICE_OF_LOSS",
                "event_timestamp_utc": event_timestamp.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "claim_id": claim["claim_id"],
                "policy_id": claim["policy_id"],
                "treaty_id": claim["treaty_id"],
                "line_of_business": claim["line_of_business"],
                "loss_type": claim["loss_type"],
                "state": claim["state"],
                "claim_amount": claim["claim_amount"],
                "claim_note": claim["claim_note"],
                "source_system": "fnol-event-service",
            }
        )

    return {
        "source_system": "fnol-event-service",
        "window_minutes": minutes,
        "record_count": len(events),
        "fnol_events": events,
    }