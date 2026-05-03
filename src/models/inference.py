import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import pandas as pd


MODEL_DIR = Path("data/model_artifacts")
PREDICTION_LOG_DIR = Path("data/prediction_logs")
PREDICTION_LOG_PATH = PREDICTION_LOG_DIR / "prediction_log.jsonl"
METADATA_PATH = MODEL_DIR / "champion_metadata.json"

SEVERITY_MODEL_PATH = MODEL_DIR / "severity_champion.pkl"
FRAUD_MODEL_PATH = MODEL_DIR / "fraud_champion.pkl"
RESERVE_MODEL_PATH = MODEL_DIR / "reserve_champion.pkl"


FLAG_INTERPRETATIONS = {
    "cat_exposure": {
        "0": "No catastrophe exposure identified",
        "1": "Claim is catastrophe-exposed, such as hail, wind, flood, wildfire, or hurricane exposure",
    },
    "litigation_flag": {
        "0": "No litigation or attorney involvement reported",
        "1": "Litigation or attorney involvement reported",
    },
    "suspicious_flag": {
        "0": "No suspicious claim indicators identified",
        "1": "Suspicious claim indicators identified",
    },
    "retention_breach_flag": {
        "0": "Claim does not exceed reinsurance retention",
        "1": "Claim exceeds reinsurance retention and may trigger reinsurance recovery",
    },
    "high_priority_claim_flag": {
        "0": "Claim does not meet high-priority triage rules",
        "1": "Claim meets high-priority triage rules based on risk, severity, or reinsurance impact",
    },
    "prior_claim_frequency_risk": {
        "0": "Prior claim count is below high-frequency threshold",
        "1": "Prior claim count is high and may indicate increased risk",
    },
    "litigation_risk_flag": {
        "0": "No litigation risk signal",
        "1": "Litigation risk signal present",
    },
    "catastrophe_risk_flag": {
        "0": "No catastrophe risk signal",
        "1": "Catastrophe risk signal present",
    },
    "suspicious_claim_flag": {
        "0": "No suspicious claim risk signal",
        "1": "Suspicious claim risk signal present",
    },
}


def utc_now_string() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_model_artifact(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Model artifact not found: {path}")

    artifact = joblib.load(path)

    if isinstance(artifact, dict) and "model" in artifact:
        return artifact["model"]

    return artifact


class ClaimRiskInferenceService:
    def __init__(self):
        self.severity_model = load_model_artifact(SEVERITY_MODEL_PATH)
        self.fraud_model = load_model_artifact(FRAUD_MODEL_PATH)
        self.reserve_model = load_model_artifact(RESERVE_MODEL_PATH)
        self.metadata = self.load_metadata()

    def load_metadata(self) -> dict[str, Any]:
        if METADATA_PATH.exists():
            with METADATA_PATH.open("r", encoding="utf-8") as file:
                return json.load(file)

        return {}

    def prepare_features(self, claim: dict[str, Any]) -> pd.DataFrame:
        claim_amount = float(claim.get("claim_amount", 0))
        retention = float(claim.get("retention", 0))
        treaty_limit = float(claim.get("treaty_limit", 0))

        prior_claim_count = int(claim.get("prior_claim_count", 0))
        litigation_flag = int(claim.get("litigation_flag", 0))
        cat_exposure = int(claim.get("cat_exposure", 0))
        suspicious_flag = int(claim.get("suspicious_flag", 0))
        text_fraud_signal = int(claim.get("text_fraud_signal", 0))
        text_litigation_signal = int(claim.get("text_litigation_signal", 0))
        text_cat_signal = int(claim.get("text_cat_signal", 0))
        text_injury_signal = int(claim.get("text_injury_signal", 0))
        source_document_count = int(claim.get("source_document_count", 0))

        retention_breach_flag = int(claim_amount > retention)

        calculated_ceded_loss = (
            max(0, min(claim_amount - retention, treaty_limit))
            if retention_breach_flag == 1
            else 0
        )

        reinsurance_recovery_ratio = (
            calculated_ceded_loss / claim_amount if claim_amount > 0 else 0
        )

        treaty_exhaustion_ratio = (
            calculated_ceded_loss / treaty_limit if treaty_limit > 0 else 0
        )

        prior_claim_frequency_risk = int(prior_claim_count >= 3)
        litigation_risk_flag = int(litigation_flag == 1 or text_litigation_signal == 1)
        catastrophe_risk_flag = int(cat_exposure == 1 or text_cat_signal == 1)
        suspicious_claim_flag = int(suspicious_flag == 1 or text_fraud_signal == 1)

        combined_risk_score = (
            prior_claim_frequency_risk * 0.25
            + litigation_risk_flag * 0.25
            + catastrophe_risk_flag * 0.20
            + suspicious_claim_flag * 0.30
        )

        high_priority_claim_flag = int(
            combined_risk_score >= 0.50
            or claim_amount >= 300000
            or retention_breach_flag == 1
        )

        features = {
            "line_of_business": claim.get("line_of_business"),
            "loss_type": claim.get("loss_type"),
            "state": claim.get("state"),
            "treaty_type": claim.get("treaty_type"),
            "claim_amount": claim_amount,
            "prior_claim_count": prior_claim_count,
            "litigation_flag": litigation_flag,
            "cat_exposure": cat_exposure,
            "retention": retention,
            "treaty_limit": treaty_limit,
            "retention_breach_flag": retention_breach_flag,
            "calculated_ceded_loss": calculated_ceded_loss,
            "reinsurance_recovery_ratio": reinsurance_recovery_ratio,
            "treaty_exhaustion_ratio": treaty_exhaustion_ratio,
            "prior_claim_frequency_risk": prior_claim_frequency_risk,
            "litigation_risk_flag": litigation_risk_flag,
            "catastrophe_risk_flag": catastrophe_risk_flag,
            "suspicious_claim_flag": suspicious_claim_flag,
            "combined_risk_score": combined_risk_score,
            "high_priority_claim_flag": high_priority_claim_flag,
            "text_fraud_signal": text_fraud_signal,
            "text_litigation_signal": text_litigation_signal,
            "text_cat_signal": text_cat_signal,
            "text_injury_signal": text_injury_signal,
            "source_document_count": source_document_count,
            "loss_month": int(claim.get("loss_month", 1)),
            "loss_quarter": int(claim.get("loss_quarter", 1)),
            "loss_year": int(claim.get("loss_year", 2026)),
        }

        return pd.DataFrame([features])

    def get_fraud_probability(self, features_df: pd.DataFrame) -> float:
        if hasattr(self.fraud_model, "predict_proba"):
            probabilities = self.fraud_model.predict_proba(features_df)

            if probabilities.shape[1] == 2:
                return float(probabilities[0][1])

        return 0.0

    def normalize_severity_prediction(self, value: Any) -> str:
        label_map = {
            0: "High",
            1: "Low",
            2: "Medium",
        }

        try:
            return label_map.get(int(value), str(value))
        except Exception:
            return str(value)

    def assign_triage_priority(
        self,
        severity_prediction: str,
        fraud_probability: float,
        high_priority_claim_flag: int,
    ) -> str:
        if (
            severity_prediction == "High"
            or fraud_probability >= 0.70
            or high_priority_claim_flag == 1
        ):
            return "Urgent"

        if severity_prediction == "Medium" or fraud_probability >= 0.40:
            return "Review"

        return "Standard"

    def build_risk_explanations(
        self,
        claim: dict[str, Any],
        feature_row: dict[str, Any],
        severity_prediction: str,
        fraud_probability: float,
        reserve_prediction: float,
    ) -> list[str]:
        explanations = []

        if int(claim.get("cat_exposure", 0)) == 1:
            explanations.append(
                "Catastrophe exposure is present, which may increase claim severity and portfolio risk."
            )

        if int(claim.get("litigation_flag", 0)) == 1:
            explanations.append(
                "Litigation or attorney involvement is reported, which may increase claim complexity and reserve needs."
            )

        if int(claim.get("suspicious_flag", 0)) == 1:
            explanations.append(
                "Suspicious claim indicators are present, so the claim may require SIU or enhanced review."
            )

        if int(feature_row["prior_claim_frequency_risk"]) == 1:
            explanations.append(
                "Prior claim frequency is high, which may indicate elevated customer or policy risk."
            )

        if int(feature_row["retention_breach_flag"]) == 1:
            explanations.append(
                "Claim amount exceeds reinsurance retention, so reinsurance recovery may be triggered."
            )

        if float(feature_row["calculated_ceded_loss"]) > 0:
            explanations.append(
                f"Estimated ceded loss is ${float(feature_row['calculated_ceded_loss']):,.2f}, subject to treaty limit and coverage review."
            )

        if int(feature_row["high_priority_claim_flag"]) == 1:
            explanations.append(
                "Claim is flagged as high priority based on risk score, claim size, or reinsurance impact."
            )

        if severity_prediction == "High":
            explanations.append(
                "Severity model predicts High severity, so senior adjuster review is recommended."
            )

        if fraud_probability >= 0.70:
            explanations.append(
                "Fraud risk probability is high, so investigation review is recommended."
            )

        if reserve_prediction > float(claim.get("claim_amount", 0)):
            explanations.append(
                "Recommended reserve is higher than the claim amount, indicating possible loss development or uncertainty."
            )

        if not explanations:
            explanations.append(
                "No major risk indicators were triggered based on the submitted claim attributes."
            )

        return explanations

    def predict(self, claim: dict[str, Any], log_prediction: bool = True) -> dict[str, Any]:
        features_df = self.prepare_features(claim)
        feature_row = features_df.iloc[0].to_dict()

        severity_raw = self.severity_model.predict(features_df)[0]
        severity_prediction = self.normalize_severity_prediction(severity_raw)

        fraud_prediction = int(self.fraud_model.predict(features_df)[0])
        fraud_probability = self.get_fraud_probability(features_df)

        reserve_prediction = float(self.reserve_model.predict(features_df)[0])

        triage_priority = self.assign_triage_priority(
            severity_prediction=severity_prediction,
            fraud_probability=fraud_probability,
            high_priority_claim_flag=int(feature_row["high_priority_claim_flag"]),
        )

        risk_explanations = self.build_risk_explanations(
            claim=claim,
            feature_row=feature_row,
            severity_prediction=severity_prediction,
            fraud_probability=fraud_probability,
            reserve_prediction=reserve_prediction,
        )

        response = {
            "claim_id": claim.get("claim_id"),
            "severity_prediction": severity_prediction,
            "fraud_risk_prediction": fraud_prediction,
            "fraud_risk_probability": round(fraud_probability, 4),
            "recommended_reserve": round(reserve_prediction, 2),
            "retention_breach_flag": int(feature_row["retention_breach_flag"]),
            "calculated_ceded_loss": round(
                float(feature_row["calculated_ceded_loss"]), 2
            ),
            "reinsurance_recovery_ratio": round(
                float(feature_row["reinsurance_recovery_ratio"]), 4
            ),
            "high_priority_claim_flag": int(feature_row["high_priority_claim_flag"]),
            "triage_priority": triage_priority,
            "flag_interpretations": FLAG_INTERPRETATIONS,
            "risk_explanations": risk_explanations,
        }

        if log_prediction:
           self.log_prediction(claim, response)

        return response

    def log_prediction(self, claim: dict[str, Any], response: dict[str, Any]) -> None:
        PREDICTION_LOG_DIR.mkdir(parents=True, exist_ok=True)

        event = {
            "prediction_timestamp_utc": utc_now_string(),
            "claim_id": claim.get("claim_id"),
            "request": claim,
            "response": response,
        }

        with PREDICTION_LOG_PATH.open("a", encoding="utf-8") as file:
            file.write(json.dumps(event) + "\n")

    def model_info(self) -> dict[str, Any]:
        return {
            "severity_model": self.metadata.get("severity", {}),
            "fraud_model": self.metadata.get("fraud", {}),
            "reserve_model": self.metadata.get("reserve", {}),
            "flag_interpretations": FLAG_INTERPRETATIONS,
            "artifact_paths": {
                "severity": str(SEVERITY_MODEL_PATH),
                "fraud": str(FRAUD_MODEL_PATH),
                "reserve": str(RESERVE_MODEL_PATH),
            },
        }


inference_service = ClaimRiskInferenceService()