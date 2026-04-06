"""Observation schema for AITEA."""

from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import Field

from .common import (
    NewsSignal,
    PendingOrder,
    PortfolioSummary,
    PricePoint,
    RegimeSignal,
    RiskSummary,
    _BaseSchema,
)


class Observation(_BaseSchema):
    step: int = Field(..., ge=0)
    timestamp: str = Field(..., description="ISO-8601 timestamp string.")
    task_name: str = Field(..., description="Current task identifier.")
    episode_id: str = Field(default="")
    market: Dict[str, PricePoint] = Field(default_factory=dict)
    portfolio: PortfolioSummary = Field(default_factory=PortfolioSummary)
    risk: RiskSummary = Field(default_factory=RiskSummary)
    news: List[NewsSignal] = Field(default_factory=list)
    regime: RegimeSignal = Field(default_factory=RegimeSignal)
    pending_orders: List[PendingOrder] = Field(default_factory=list)
    recent_actions: List[str] = Field(default_factory=list)
    recent_rewards: List[float] = Field(default_factory=list)
    history_summary: str = Field(default="")
    market_status: str = Field(default="open")
    benchmark_return: float = Field(default=0.0)
    metadata: Dict[str, str] = Field(default_factory=dict)
