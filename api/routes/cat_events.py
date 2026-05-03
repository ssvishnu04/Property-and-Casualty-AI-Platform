from fastapi import APIRouter, Query

from src.data_ingestion.data_generator import generate_cat_event


router = APIRouter()


@router.get("/")
def get_cat_events(limit: int = Query(default=5, ge=1, le=50)):
    events = [generate_cat_event() for _ in range(limit)]

    return {
        "source_system": "cat-event-feed",
        "record_count": len(events),
        "cat_events": events,
    }