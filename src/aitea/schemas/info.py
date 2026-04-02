"""
Info schema for AITEA.

This contains rich diagnostics returned by env.step().
"""

from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from aitea.schemas.common import ComplianceStatus


class Info(BaseModel):
    pnl_delta: float = 0.0
    total_pnl: float = 0.0

    execution_cost: float = 0.0
    slippage: float = 0.0

    compliance_status: ComplianceStatus = ComplianceStatus.OK
    violations: List[str] = Field(default_factory=list)

    drawdown: float = 0.0
    exposure: float = 0.0

    step_index: int = 0
    episode_reward: float = 0.0

    task_metrics: Dict[str, float] = Field(default_factory=dict)

    class Config:
        extra = "forbid"
