"""State management and serialization for AITEA."""

from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from ..config import AITEAConfig, get_config
from ..schemas import (
    Info,
    MarketRegime,
    NewsSignal,
    Observation,
    PendingOrder,
    PortfolioSummary,
    Position,
    PricePoint,
    RegimeSignal,
    Reward,
    RiskLevel,
    RiskSummary,
)


class Transition(BaseModel):
    """Standard step/reset response object."""

    class Config:
        extra = "forbid"
        validate_assignment = True

    observation: Observation
    reward: float = Field(default=0.0)
    done: bool = Field(default=False)
    info: Info = Field(default_factory=Info)
    error: Optional[str] = Field(default=None)
    reward_detail: Reward = Field(default_factory=Reward)


@dataclass
class AITEAState:
    """Authoritative internal environment state."""

    episode_id: str
    task_name: str
    task_profile: Dict[str, Any] = field(default_factory=dict)
    step: int = 0

    cash: float = 0.0
    starting_cash: float = 0.0
    equity: float = 0.0
    peak_equity: float = 0.0

    prices: Dict[str, float] = field(default_factory=dict)
    previous_prices: Dict[str, float] = field(default_factory=dict)

    positions: Dict[str, float] = field(default_factory=dict)
    avg_cost: Dict[str, float] = field(default_factory=dict)
    realized_pnl_by_symbol: Dict[str, float] = field(default_factory=dict)

    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    gross_exposure: float = 0.0
    net_exposure: float = 0.0
    drawdown_pct: float = 0.0

    regime: MarketRegime = MarketRegime.NORMAL
    regime_confidence: float = 0.5
    hidden_volatility: float = 0.01

    news_queue: List[NewsSignal] = field(default_factory=list)
    pending_orders: List[PendingOrder] = field(default_factory=list)

    recent_actions: List[str] = field(default_factory=list)
    recent_rewards: List[float] = field(default_factory=list)

    task_metrics: Dict[str, Any] = field(default_factory=dict)
    done: bool = False
    last_error: Optional[str] = None

    rng: random.Random = field(default_factory=random.Random, repr=False)


def _timestamp_for_step(step: int) -> str:
    base = datetime(2025, 1, 1, 9, 30, 0)
    return (base + timedelta(minutes=step)).isoformat()


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _safe_numeric_metrics(metrics: Dict[str, Any]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for key, value in metrics.items():
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            out[key] = float(value)
    return out


def update_derived_state(state: AITEAState) -> None:
    """Recompute portfolio metrics in-place."""
    gross_exposure = 0.0
    net_exposure = 0.0
    market_value = 0.0
    unrealized = 0.0

    for symbol, qty in state.positions.items():
        price = state.prices.get(symbol, 0.0)
        avg_cost = state.avg_cost.get(symbol, price)
        position_value = qty * price
        gross_exposure += abs(position_value)
        net_exposure += position_value
        market_value += position_value
        unrealized += (price - avg_cost) * qty

    state.unrealized_pnl = unrealized
    state.realized_pnl = float(sum(state.realized_pnl_by_symbol.values()))
    state.equity = state.cash + market_value
    state.gross_exposure = gross_exposure
    state.net_exposure = net_exposure

    if state.equity > state.peak_equity:
        state.peak_equity = state.equity

    if state.peak_equity > 0:
        state.drawdown_pct = max(0.0, (state.peak_equity - state.equity) / state.peak_equity)
    else:
        state.drawdown_pct = 0.0


def _risk_level_from_drawdown(drawdown: float) -> RiskLevel:
    if drawdown < 0.05:
        return RiskLevel.LOW
    if drawdown < 0.10:
        return RiskLevel.MODERATE
    if drawdown < 0.15:
        return RiskLevel.HIGH
    return RiskLevel.CRITICAL


def build_observation(state: AITEAState, config: AITEAConfig | None = None) -> Observation:
    """Convert internal state into an agent-facing Observation."""
    cfg = config or get_config()
    update_derived_state(state)

    market: Dict[str, PricePoint] = {}
    for i, (symbol, price) in enumerate(state.prices.items()):
        spread_pct = max(0.0008, 0.0015 + 0.6 * state.hidden_volatility)
        spread = max(0.01, price * spread_pct)
        bid = max(0.01, price - spread / 2.0)
        ask = price + spread / 2.0
        volume_base = 10_000.0 * float(state.task_profile.get("liquidity_scale", 1.0))
        volume = max(100.0, volume_base * (1.0 + 0.05 * math.sin(state.step + i)))
        market[symbol] = PricePoint(
            bid=bid,
            ask=ask,
            last=price,
            mid=price,
            spread=spread,
            volume=volume,
        )

    positions: List[Position] = []
    for symbol in cfg.assets:
        qty = float(state.positions.get(symbol, 0.0))
        price = float(state.prices.get(symbol, 0.0))
        avg_cost = float(state.avg_cost.get(symbol, price))
        market_value = qty * price
        unrealized = (price - avg_cost) * qty
        positions.append(
            Position(
                symbol=symbol,
                quantity=qty,
                average_price=avg_cost,
                market_price=price,
                market_value=market_value,
                unrealized_pnl=unrealized,
                realized_pnl=float(state.realized_pnl_by_symbol.get(symbol, 0.0)),
            )
        )

    leverage = state.gross_exposure / state.equity if state.equity > 0 else 0.0
    portfolio = PortfolioSummary(
        cash=state.cash,
        equity=state.equity,
        gross_exposure=state.gross_exposure,
        net_exposure=state.net_exposure,
        leverage=leverage,
        positions=positions,
    )

    risk = RiskSummary(
        risk_level=_risk_level_from_drawdown(state.drawdown_pct),
        drawdown_pct=state.drawdown_pct,
        gross_exposure_pct=state.gross_exposure / state.equity if state.equity > 0 else 0.0,
        net_exposure_pct=abs(state.net_exposure) / state.equity if state.equity > 0 else 0.0,
        max_position_pct=float(state.task_profile.get("max_position_pct", 0.0)),
        violation_count=int(state.task_metrics.get("violation_count", 0)),
    )

    news_items = state.news_queue[-3:]
    regime_confidence = max(0.0, min(1.0, float(state.regime_confidence)))
    regime = RegimeSignal(
        regime=state.regime,
        confidence=regime_confidence,
        transition_probability=float(state.task_profile.get("regime_flip_probability", 0.0)),
    )

    rewards = list(state.recent_rewards[-5:])
    recent_actions = list(state.recent_actions[-5:])
    history_summary = " | ".join(recent_actions[-3:]) if recent_actions else "No recent actions"

    benchmark_return = 0.0
    if state.previous_prices:
        returns: List[float] = []
        for symbol, prev_price in state.previous_prices.items():
            new_price = state.prices.get(symbol, prev_price)
            if prev_price > 0:
                returns.append((new_price - prev_price) / prev_price)
        if returns:
            benchmark_return = sum(returns) / len(returns)

    return Observation(
        step=state.step,
        timestamp=_timestamp_for_step(state.step),
        task_name=state.task_name,
        episode_id=state.episode_id,
        market=market,
        portfolio=portfolio,
        risk=risk,
        news=news_items,
        regime=regime,
        pending_orders=copy.deepcopy(state.pending_orders),
        recent_actions=recent_actions,
        recent_rewards=rewards,
        history_summary=history_summary,
        market_status="done" if state.done else "open",
        benchmark_return=benchmark_return,
        metadata={
            "task_kind": str(state.task_profile.get("kind", "generic")),
            "episode_id": state.episode_id,
            "task_name": state.task_name,
        },
    )


def build_info(
    state: AITEAState,
    *,
    execution_cost: float = 0.0,
    slippage: float = 0.0,
    fill_ratio: float = 1.0,
    turnover: float = 0.0,
    pnl_delta: float = 0.0,
    violations: Optional[List[str]] = None,
) -> Info:
    """Create a diagnostic info payload."""
    numeric_metrics = _safe_numeric_metrics(state.task_metrics)
    numeric_metrics.setdefault("equity", state.equity)
    numeric_metrics.setdefault("drawdown_pct", state.drawdown_pct)
    numeric_metrics.setdefault("gross_exposure", state.gross_exposure)
    numeric_metrics.setdefault("net_exposure", state.net_exposure)

    return Info(
        step=state.step,
        episode_done=state.done,
        execution_cost=execution_cost,
        slippage=slippage,
        fill_ratio=max(0.0, min(1.0, fill_ratio)),
        turnover=turnover,
        pnl_delta=pnl_delta,
        drawdown=state.drawdown_pct,
        gross_exposure=state.gross_exposure,
        net_exposure=state.net_exposure,
        constraint_violations=violations or [],
        task_metrics=numeric_metrics,
        metadata={
            "episode_id": state.episode_id,
            "task_name": state.task_name,
        },
    )


class StateManager:
    """Utility wrapper around state serialization and snapshots."""

    def __init__(self, config: AITEAConfig | None = None) -> None:
        self.config = config or get_config()

    def snapshot(self, state: AITEAState) -> AITEAState:
        return copy.deepcopy(state)

    def observation(self, state: AITEAState) -> Observation:
        return build_observation(state, self.config)

    def info(
        self,
        state: AITEAState,
        *,
        execution_cost: float = 0.0,
        slippage: float = 0.0,
        fill_ratio: float = 1.0,
        turnover: float = 0.0,
        pnl_delta: float = 0.0,
        violations: Optional[List[str]] = None,
    ) -> Info:
        return build_info(
            state,
            execution_cost=execution_cost,
            slippage=slippage,
            fill_ratio=fill_ratio,
            turnover=turnover,
            pnl_delta=pnl_delta,
            violations=violations,
        )
