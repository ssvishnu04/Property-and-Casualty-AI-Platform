from fastapi import APIRouter

from api.schemas import ClaimRiskRequest, ClaimRiskResponse
from src.models.inference import inference_service


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