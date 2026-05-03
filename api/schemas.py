from typing import Optional
from pydantic import BaseModel, Field


class ClaimRiskRequest(BaseModel):
    claim_id: str = Field(..., example="CLM-TEST-001")
    policy_id: str = Field(..., example="POL-TEST-001")
    treaty_id: Optional[str] = Field(default="TRT-TEST-001")

    line_of_business: str = Field(..., example="Commercial Property")
    loss_type: str = Field(..., example="Hail")
    state: str = Field(..., example="TX")
    treaty_type: str = Field(default="Excess of Loss")

    claim_amount: float = Field(..., example=425000)
    reserve_amount: float = Field(default=500000)

    prior_claim_count: int = Field(default=3)
    litigation_flag: int = Field(default=1)
    cat_exposure: int = Field(default=1)
    suspicious_flag: int = Field(default=1)

    retention: float = Field(default=250000)
    treaty_limit: float = Field(default=2000000)

    loss_month: int = Field(default=4)
    loss_quarter: int = Field(default=2)
    loss_year: int = Field(default=2026)


class ClaimRiskResponse(BaseModel):
    claim_id: str

    severity_prediction: str
    fraud_risk_prediction: int
    fraud_risk_probability: float
    recommended_reserve: float

    retention_breach_flag: int
    calculated_ceded_loss: float
    reinsurance_recovery_ratio: float
    high_priority_claim_flag: int
    triage_priority: str

    flag_interpretations: dict
    risk_explanations: list[str]