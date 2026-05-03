from fastapi import APIRouter

from api.schemas import ClaimRiskRequest, ClaimRiskResponse
from src.models.inference import inference_service
from pathlib import Path
import json
from fastapi import Query


router = APIRouter()


@router.get("/health")
def prediction_health():
    return {
        "status": "healthy",
        "service": "claim-risk-inference",
    }


@router.get("/model-info")
def model_info():
    return inference_service.model_info()


@router.post("/claim-risk", response_model=ClaimRiskResponse)
def predict_claim_risk(request: ClaimRiskRequest):
    claim = request.model_dump()
    prediction = inference_service.predict(claim)

    return prediction

PREDICTION_LOG_PATH = Path("data/prediction_logs/prediction_log.jsonl")


@router.get("/audit-logs")
def get_prediction_audit_logs(limit: int = Query(default=50, ge=1, le=500)):
    if not PREDICTION_LOG_PATH.exists():
        return {"records": []}

    records = []

    with PREDICTION_LOG_PATH.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                records.append(json.loads(line))

    records = records[-limit:]
    records.reverse()

    return {"records": records}