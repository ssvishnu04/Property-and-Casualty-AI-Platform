from fastapi import APIRouter, HTTPException

from src.genai.rag_pipeline import rag_service
from src.models.inference import inference_service


router = APIRouter()


@router.post("/explain-claim")
def explain_claim(request: dict):
    claim = request.get("claim")
    question = request.get("question", "Explain this claim risk")
    prediction = request.get("prediction")

    if not claim:
        raise HTTPException(
            status_code=400,
            detail="Missing 'claim' in request payload",
        )

    if prediction is None:
        prediction = inference_service.predict(claim, log_prediction=False)

    rag_response = rag_service.answer(
        question=question,
        context_data=prediction,
    )

    return {
        "prediction": prediction,
        "explanation": rag_response["answer"],
        "retrieved_sources": rag_response["retrieved_sources"],
        "retrieved_contexts": rag_response["retrieved_contexts"],
    }