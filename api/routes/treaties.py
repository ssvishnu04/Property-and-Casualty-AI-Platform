from fastapi import APIRouter

from src.data_ingestion.data_generator import generate_treaty


router = APIRouter()


@router.get("/")
def get_all_treaties():
    treaties = []

    for i in range(1, 101):
        treaties.append(
            {
                "treaty_id": f"TRT-{1000 + i}",
                "treaty_type": "Excess of Loss",
                "retention": 250000,
                "treaty_limit": 2000000,
            }
        )

    return {"treaties": treaties}