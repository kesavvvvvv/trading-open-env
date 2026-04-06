"""Shared schema components for AITEA."""

from __future__ import annotations

from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field, PositiveInt, confloat


class _BaseSchema(BaseModel):
    class Config:
        extra = "forbid"
        validate_assignment = True
        use_enum_values = True


class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    CANCEL = "cancel"
    MODIFY = "modify"


class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"
    TWAP = "twap"
    VWAP = "vwap"
    PASSIVE = "passive"


class MarketRegime(str, Enum):
    CALM = "calm"
    NORMAL = "normal"
    VOLATILE = "volatile"
    CRISIS = "crisis"
    RECOVERY = "recovery"


class RiskLevel(str, Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


class InstrumentSpec(_BaseSchema):
    symbol: str = Field(..., description="Instrument ticker or identifier.")
    asset_class: str = Field(default="equity", description="Asset class label.")
    currency: str = Field(default="USD", description="Pricing currency.")


class PricePoint(_BaseSchema):
    bid: float = Field(..., ge=0.0)
    ask: float = Field(..., ge=0.0)
    last: float = Field(..., ge=0.0)
    mid: float = Field(..., ge=0.0)
    spread: float = Field(..., ge=0.0)
    volume: float = Field(default=0.0, ge=0.0)


class Position(_BaseSchema):
    symbol: str
    quantity: float = Field(default=0.0)
    average_price: float = Field(default=0.0, ge=0.0)
    market_price: float = Field(default=0.0, ge=0.0)
    market_value: float = Field(default=0.0)
    unrealized_pnl: float = Field(default=0.0)
    realized_pnl: float = Field(default=0.0)


class PortfolioSummary(_BaseSchema):
    cash: float = Field(default=0.0)
    equity: float = Field(default=0.0)
    gross_exposure: float = Field(default=0.0, ge=0.0)
    net_exposure: float = Field(default=0.0)
    leverage: float = Field(default=1.0, ge=0.0)
    positions: List[Position] = Field(default_factory=list)


class RiskSummary(_BaseSchema):
    risk_level: RiskLevel = RiskLevel.LOW
    drawdown_pct: float = Field(default=0.0, ge=0.0)
    gross_exposure_pct: float = Field(default=0.0, ge=0.0)
    net_exposure_pct: float = Field(default=0.0, ge=0.0)
    max_position_pct: float = Field(default=0.0, ge=0.0)
    violation_count: int = Field(default=0, ge=0)


class NewsSignal(_BaseSchema):
    headline: str = Field(default="")
    severity: float = Field(default=0.0, ge=0.0, le=1.0)
    sentiment: float = Field(default=0.0, ge=-1.0, le=1.0)
    affected_symbols: List[str] = Field(default_factory=list)


class RegimeSignal(_BaseSchema):
    regime: MarketRegime = MarketRegime.NORMAL
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    transition_probability: float = Field(default=0.0, ge=0.0, le=1.0)


class OrderInstruction(_BaseSchema):
    symbol: str = Field(..., description="Instrument ticker.")
    side: OrderSide = Field(..., description="Order direction.")
    quantity: PositiveInt = Field(..., description="Order quantity in shares or units.")
    order_type: OrderType = Field(default=OrderType.MARKET)
    limit_price: Optional[float] = Field(default=None, ge=0.0)
    target_weight: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    urgency: float = Field(default=0.5, ge=0.0, le=1.0)
    time_in_force: str = Field(default="day")
    tag: Optional[str] = None


class PendingOrder(_BaseSchema):
    order_id: str
    symbol: str
    side: OrderSide
    quantity: PositiveInt
    filled_quantity: int = Field(default=0, ge=0)
    remaining_quantity: int = Field(default=0, ge=0)
    order_type: OrderType = Field(default=OrderType.MARKET)
    status: str = Field(default="open")
