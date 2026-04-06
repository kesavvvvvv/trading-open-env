"""Deterministic baseline policy for AITEA."""

from __future__ import annotations

from typing import Dict, List

from ..schemas import Action, OrderInstruction, OrderSide, OrderType, Observation


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _top_symbol(observation: Observation) -> str:
    if observation.market:
        return next(iter(observation.market.keys()))
    return "AAPL"


def _portfolio_weight(observation: Observation, symbol: str) -> float:
    equity = max(1.0, _safe_float(observation.portfolio.equity, 0.0))
    pos = next((p for p in observation.portfolio.positions if p.symbol == symbol), None)
    if pos is None:
        return 0.0
    return (float(pos.quantity) * float(pos.market_price)) / equity


def baseline_action(observation: Observation) -> Action:
    """
    Simple, reproducible baseline.

    Behavior:
    - For execution tasks: move gradually toward the target symbol.
    - For liquidity tasks: use small market orders to avoid impact.
    - For hedge tasks: reduce the dominant exposure.
    - For rebalance tasks: nudge toward target weights.
    - For news/regime tasks: reduce risk and avoid aggressive trading.
    """
    kind = str(observation.metadata.get("task_kind", "generic"))
    orders: List[OrderInstruction] = []
    hold_position = False
    flatten_all = False
    rebalance_targets: Dict[str, float] = {}
    hedge_targets: Dict[str, float] = {}

    symbols = list(observation.market.keys()) or ["AAPL", "MSFT", "GOOG", "TSLA", "AMZN"]
    target_symbol = symbols[0]

    # Conservative defaults
    urgency = 0.35
    quantity = 20

    if kind == "execution":
        target_symbol = str(observation.metadata.get("target_symbol", target_symbol))
        remaining = 0.0
        if observation.recent_actions:
            remaining = max(0.0, _safe_float(observation.task_metrics.get("target_remaining", 0.0), 0.0))
        if remaining <= 1.0:
            hold_position = True
        else:
            quantity = max(1, min(100, int(max(10.0, remaining * 0.02))))
            orders.append(
                OrderInstruction(
                    symbol=target_symbol,
                    side=OrderSide.BUY,
                    quantity=quantity,
                    order_type=OrderType.MARKET,
                    urgency=urgency,
                    tag="baseline_execution",
                )
            )

    elif kind == "liquidity":
        target_symbol = str(observation.metadata.get("target_symbol", target_symbol))
        remaining = max(0.0, _safe_float(observation.task_metrics.get("target_remaining", 0.0), 0.0))
        if remaining <= 1.0:
            hold_position = True
        else:
            quantity = max(1, min(50, int(max(5.0, remaining * 0.01))))
            orders.append(
                OrderInstruction(
                    symbol=target_symbol,
                    side=OrderSide.BUY,
                    quantity=quantity,
                    order_type=OrderType.MARKET,
                    urgency=0.25,
                    tag="baseline_liquidity",
                )
            )

    elif kind == "hedge":
        hedge_symbol = str(observation.metadata.get("hedge_symbol", target_symbol))
        fx_exposure = max(0.0, _safe_float(observation.task_metrics.get("fx_exposure", 0.0), 0.0))
        if fx_exposure <= 1000.0:
            hold_position = True
        else:
            hedge_targets[hedge_symbol] = -min(fx_exposure * 0.001, 100.0)

    elif kind == "rebalance":
        target_weights = observation.metadata.get("target_weights")
        if isinstance(target_weights, dict) and target_weights:
            for symbol, w in target_weights.items():
                rebalance_targets[str(symbol)] = max(0.0, min(1.0, float(w)))
        else:
            # Safe fallback: keep existing positions, reduce noise.
            hold_position = True

    elif kind in {"news", "regime"}:
        # Risk-off response: reduce risk and avoid aggressive trading.
        flatten_all = True
        hold_position = False

    else:
        hold_position = True

    return Action(
        orders=orders,
        rebalance_targets=rebalance_targets,
        hedge_targets=hedge_targets,
        flatten_all=flatten_all,
        hold_position=hold_position if not orders and not rebalance_targets and not hedge_targets and not flatten_all else False,
        strategy_tag="baseline_rules",
        comment=f"kind={kind}",
    )
