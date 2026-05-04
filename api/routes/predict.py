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


def flatten_prediction_log(record: dict) -> dict:
    claim = record.get("claim", {})
    response = record.get("response", {})
    prediction = record.get("prediction", {})

    # Some versions may store outputs under response, some under prediction
    output = response if response else prediction

    return {
        "prediction_timestamp_utc": record.get("prediction_timestamp_utc")
        or record.get("timestamp_utc")
        or record.get("timestamp"),
        "claim_id": record.get("claim_id") or claim.get("claim_id"),
        "policy_id": claim.get("policy_id"),
        "treaty_id": claim.get("treaty_id"),
        "line_of_business": claim.get("line_of_business"),
        "loss_type": claim.get("loss_type"),
        "state": claim.get("state"),
        "claim_amount": claim.get("claim_amount"),
        "reserve_amount": claim.get("reserve_amount"),
        "severity_prediction": output.get("severity_prediction"),
        "fraud_risk_prediction": output.get("fraud_risk_prediction"),
        "fraud_risk_probability": output.get("fraud_risk_probability"),
        "recommended_reserve": output.get("recommended_reserve"),
        "triage_priority": output.get("triage_priority"),
        "retention_breach_flag": output.get("retention_breach_flag"),
        "calculated_ceded_loss": output.get("calculated_ceded_loss"),
        "reinsurance_recovery_ratio": output.get("reinsurance_recovery_ratio"),
        "high_priority_claim_flag": output.get("high_priority_claim_flag"),
    }


@router.get("/audit-logs")
def get_prediction_audit_logs(limit: int = Query(default=50, ge=1, le=500)):
    if not PREDICTION_LOG_PATH.exists():
        return {"records": []}

    records = []

    with PREDICTION_LOG_PATH.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                raw_record = json.loads(line)
                records.append(flatten_prediction_log(raw_record))

    records = records[-limit:]
    records.reverse()

    return {"records": records}