from fastapi import APIRouter

from src.data_ingestion.data_generator import generate_exposure


router = APIRouter()


@router.get("/")
def get_all_exposures():
    exposures = []

    for i in range(1, 101):
        exposures.append(
            {
                "exposure_id": f"EXP-{1000 + i}",
                "policy_id": f"POL-{1000 + i}",
                "state": "TX",
                "cat_exposure": 1 if i % 3 == 0 else 0,
                "exposure_amount": 500000 + (i * 25000),
                "location_risk_score": round(0.25 + (i % 10) * 0.05, 2),
            }
        )

    return {"exposures": exposures}