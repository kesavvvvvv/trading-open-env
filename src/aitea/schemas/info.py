"""Info schema for AITEA."""

from __future__ import annotations

from typing import Dict, List

from pydantic import Field

from .common import _BaseSchema


class Info(_BaseSchema):
    step: int = Field(default=0, ge=0)
    episode_done: bool = Field(default=False)
    execution_cost: float = Field(default=0.0)
    slippage: float = Field(default=0.0)
    fill_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    turnover: float = Field(default=0.0, ge=0.0)
    pnl_delta: float = Field(default=0.0)
    drawdown: float = Field(default=0.0, ge=0.0)
    gross_exposure: float = Field(default=0.0, ge=0.0)
    net_exposure: float = Field(default=0.0)
    constraint_violations: List[str] = Field(default_factory=list)
    task_metrics: Dict[str, float] = Field(default_factory=dict)
    metadata: Dict[str, str] = Field(default_factory=dict)
