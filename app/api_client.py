import requests
import os


API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")


def predict_claim_risk(payload: dict) -> dict:
    response = requests.post(
        f"{API_BASE_URL}/predict/claim-risk",
        json=payload,
        timeout=60,
    )
    response.raise_for_status()
    return response.json()


def explain_claim(payload: dict) -> dict:
    response = requests.post(
        f"{API_BASE_URL}/genai/explain-claim",
        json=payload,
        timeout=120,
    )
    response.raise_for_status()
    return response.json()


def get_model_info() -> dict:
    response = requests.get(
        f"{API_BASE_URL}/predict/model-info",
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def check_api_health() -> bool:
    try:
        response = requests.get(f"{API_BASE_URL}/predict/health", timeout=10)
        return response.status_code == 200
    except Exception:
        return False