"""Pydantic request/response schemas for the credit scoring API."""

from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


class ReasonCode(BaseModel):
    code: str
    feature: str
    direction: str   # "risk_increase" | "risk_decrease"
    text: str
    shap_value: float


class CreditScoreResponse(BaseModel):
    wallet: str
    timestamp: str
    score: int = Field(..., ge=0, le=1000, description="Credit score 0–1000 (higher = better)")
    pd_30d: Optional[float] = Field(None, description="Probability of default in next 30 days")
    pd_60d: Optional[float] = Field(None, description="Probability of default in next 60 days")
    pd_90d: float = Field(..., description="Probability of default in next 90 days")
    risk_grade: str = Field(..., description="A / B / C / D / E")
    top_reason_codes: list[ReasonCode] = []
    feature_summary: dict = {}
    model_version: str
    calibration: str


class BatchScoreRequest(BaseModel):
    wallets: list[str] = Field(..., max_length=100)
    horizon_days: int = Field(default=90, ge=30, le=90)
    explain: bool = Field(default=True)


class HealthResponse(BaseModel):
    status: str
    models_loaded: list[str]
    version: str
