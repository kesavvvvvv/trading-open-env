"""
Common schema types and reusable models for AITEA.
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


# -----------------------------
# Typed aliases
# -----------------------------

Price = float
Quantity = float
Timestamp = int
Score = float


# -----------------------------
# Enums
# -----------------------------

class InstrumentId(str, Enum):
    AAPL = "AAPL"
    MSFT = "MSFT"
    GOOG = "GOOG"
    EURUSD = "EURUSD"


class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"


class MarketRegime(str, Enum):
    BULL = "bull"
    BEAR = "bear"
    CHAOTIC = "chaotic"


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"


class OrderUrgency(str, Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"


class TimeInForce(str, Enum):
    DAY = "day"
    IOC = "ioc"   # immediate or cancel
    GTC = "gtc"   # good till cancelled


class ComplianceStatus(str, Enum):
    OK = "ok"
    WARNING = "warning"
    VIOLATION = "violation"


# -----------------------------
# Reusable nested models
# -----------------------------

class MarketSnapshot(BaseModel):
    prices: Dict[str, Price] = Field(default_factory=dict)
    volatility: float = 0.0
    spread: float = 0.0
    regime_hint: Optional[MarketRegime] = None
    liquidity: Dict[str, float] = Field(default_factory=dict)

    class Config:
        extra = "forbid"


class PortfolioSnapshot(BaseModel):
    cash: float = 0.0
    positions: Dict[str, Quantity] = Field(default_factory=dict)
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    total_value: float = 0.0

    class Config:
        extra = "forbid"


class TreasurySnapshot(BaseModel):
    upcoming_outflows: Dict[str, float] = Field(default_factory=dict)
    upcoming_inflows: Dict[str, float] = Field(default_factory=dict)
    minimum_cash_required: float = 0.0
    current_liquidity_buffer: float = 0.0

    class Config:
        extra = "forbid"


class RiskSnapshot(BaseModel):
    drawdown: float = 0.0
    exposure: float = 0.0
    risk_level: RiskLevel = RiskLevel.LOW
    compliance_status: ComplianceStatus = ComplianceStatus.OK

    class Config:
        extra = "forbid"


class NewsEvent(BaseModel):
    headline: str
    sentiment: float = 0.0  # -1.0 to +1.0
    severity: float = 0.0   # 0.0 to 1.0
    affected_assets: List[str] = Field(default_factory=list)

    class Config:
        extra = "forbid"


class PendingOrder(BaseModel):
    order_id: str
    instrument: str
    side: OrderSide
    order_type: OrderType
    quantity: Quantity
    price: Optional[Price] = None
    filled_quantity: Quantity = 0.0
    status: str = "open"

    class Config:
        extra = "forbid"


class HedgeInstruction(BaseModel):
    instrument: str
    target_hedge_ratio: float = 0.0
    urgency: OrderUrgency = OrderUrgency.NORMAL

    class Config:
        extra = "forbid"


class AllocationAdjustment(BaseModel):
    target_weights: Dict[str, float] = Field(default_factory=dict)
    urgency: OrderUrgency = OrderUrgency.NORMAL

    class Config:
        extra = "forbid"
