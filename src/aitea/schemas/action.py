"""
Action schema for AITEA.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from aitea.schemas.common import (
    OrderType,
    OrderSide,
    OrderUrgency,
    TimeInForce,
    HedgeInstruction,
    AllocationAdjustment,
)


class Order(BaseModel):
    instrument: str
    side: OrderSide
    order_type: OrderType
    quantity: float

    price: Optional[float] = None
    urgency: OrderUrgency = OrderUrgency.NORMAL
    time_in_force: TimeInForce = TimeInForce.DAY
    allow_partial_fill: bool = True

    class Config:
        extra = "forbid"


class Action(BaseModel):
    orders: List[Order] = Field(default_factory=list)

    hedge_instructions: List[HedgeInstruction] = Field(default_factory=list)
    allocation_adjustment: Optional[AllocationAdjustment] = None

    # Optional high-level controls
    pause_trading: bool = False
    risk_off: bool = False
    metadata: Dict[str, str] = Field(default_factory=dict)

    class Config:
        extra = "forbid"
