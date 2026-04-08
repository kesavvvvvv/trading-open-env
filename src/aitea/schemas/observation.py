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
    # ===== Core Metadata =====
    step: int = Field(..., ge=0)
    timestamp: str = Field(..., description="ISO-8601 timestamp string.")
    task_name: str = Field(..., description="Current task identifier.")
    episode_id: str = Field(default="")

    # ===== Market State =====
    market: Dict[str, PricePoint] = Field(default_factory=dict)

    # ===== Portfolio State =====
    portfolio: PortfolioSummary = Field(default_factory=PortfolioSummary)

    # ===== Risk State =====
    risk: RiskSummary = Field(default_factory=RiskSummary)

    # ===== Signals =====
    news: List[NewsSignal] = Field(default_factory=list)
    regime: RegimeSignal = Field(default_factory=RegimeSignal)

    # ===== Execution State =====
    pending_orders: List[PendingOrder] = Field(default_factory=list)

    # ===== Agent Memory =====
    recent_actions: List[str] = Field(default_factory=list)
    recent_rewards: List[float] = Field(default_factory=list)
    history_summary: str = Field(default="")

    # ===== Market Meta =====
    market_status: str = Field(default="open")
    benchmark_return: float = Field(default=0.0)

    # ===== 🔥 CRITICAL FIX (FOR BASELINE + GRADERS) =====
    task_metrics: Dict[str, float] = Field(
        default_factory=dict,
        description="Task-specific metrics (target_remaining, fx_exposure, tracking_error, etc.)"
    )

    # ===== 🔥 BONUS (DEBUG + JUDGE VISIBILITY) =====
    info_metrics: Dict[str, float] = Field(
        default_factory=dict,
        description="Execution diagnostics (slippage, turnover, pnl_delta, etc.)"
    )

    # ===== Misc =====
    metadata: Dict[str, str] = Field(default_factory=dict)
