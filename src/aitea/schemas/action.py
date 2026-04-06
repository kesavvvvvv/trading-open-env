"""Action schema for AITEA."""

from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import Field, confloat

from .common import OrderInstruction, _BaseSchema


class Action(_BaseSchema):
    orders: List[OrderInstruction] = Field(default_factory=list)
    cancel_order_ids: List[str] = Field(default_factory=list)
    rebalance_targets: Dict[str, confloat(ge=0.0, le=1.0)] = Field(default_factory=dict)
    hedge_targets: Dict[str, float] = Field(default_factory=dict)
    risk_reduction: float = Field(default=0.0, ge=0.0, le=1.0)
    flatten_all: bool = Field(default=False)
    hold_position: bool = Field(default=False)
    comment: Optional[str] = Field(default=None, max_length=500)
    strategy_tag: Optional[str] = Field(default=None, max_length=100)
