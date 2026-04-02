"""
Observation schema for AITEA.

This is the structured state the agent sees at each step.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from aitea.schemas.common import (
    MarketSnapshot,
    PortfolioSnapshot,
    TreasurySnapshot,
    RiskSnapshot,
    NewsEvent,
    PendingOrder,
)


class Observation(BaseModel):
    timestamp: int

    market: MarketSnapshot
    portfolio: PortfolioSnapshot
    treasury: TreasurySnapshot
    risk: RiskSnapshot

    news: List[NewsEvent] = Field(default_factory=list)
    pending_orders: List[PendingOrder] = Field(default_factory=list)

    recent_history_summary: Dict[str, float] = Field(default_factory=dict)
    task_context: Optional[Dict[str, float]] = None

    class Config:
        extra = "forbid"
