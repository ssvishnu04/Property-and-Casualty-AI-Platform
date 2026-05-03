from src.data_ingestion.data_generator import generate_policy
from fastapi import APIRouter

router = APIRouter()

@router.get("/")
def get_all_policies():
    policies = []

    for i in range(1, 101):
        policies.append(
            {
                "policy_id": f"POL-{1000 + i}",
                "line_of_business": "Commercial Property",
                "policy_limit": 1000000 + (i * 10000),
                "deductible": 5000,
                "state": "TX",
            }
        )

    return {"policies": policies}