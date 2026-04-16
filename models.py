from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, Field


class GeoSignal(BaseModel):
    score: float = Field(ge=-2, le=2)
    tier: int = Field(ge=1, le=3)
    status: Literal["pending", "confirmed", "verified"]
    reason: str
    translated_title: Optional[str] = None
    source: str
    timestamp: datetime


class MacroFactors(BaseModel):
    oil_price: float
    dxy: float
    etf_flow: float


class FinalDecision(BaseModel):
    regime: Literal["RISK_ON", "RISK_OFF", "NEUTRAL"]
    macro_score: float
    signal: Literal["LONG", "SHORT", "NO_TRADE"]
    factor_details: Dict[str, Any] = Field(default_factory=dict)
