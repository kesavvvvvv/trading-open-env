"""Order execution engine for AITEA."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Tuple

from ..config import AITEAConfig, get_config
from ..schemas import OrderInstruction, OrderSide, OrderType, PendingOrder
from ..env.state_manager import AITEAState


@dataclass
class ExecutionResult:
    symbol: str
    side: str
    requested_quantity: int
    filled_quantity: int
    remaining_quantity: int
    average_fill_price: float
    execution_cost: float
    slippage_cost: float
    notional: float
    status: str
    violation: str | None = None


def _clamp_min(value: float, minimum: float = 0.01) -> float:
    return value if value >= minimum else minimum


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


class ExecutionEngine:
    """Deterministic market execution with slippage, costs, and partial fills."""

    def __init__(self, config: AITEAConfig | None = None) -> None:
        self.config = config or get_config()

    def _liquidity_capacity(self, state: AITEAState, symbol: str, order: OrderInstruction) -> int:
        base = int(max(1, round(1200.0 * float(state.task_profile.get("liquidity_scale", 0.75)))))
        if order.order_type == OrderType.MARKET:
            return base
        return max(1, int(base * 0.5))

    def _fill_price(self, price: float, side: OrderSide, filled_qty: int, capacity: int, hidden_volatility: float) -> float:
        side_mult = 1.0 if side == OrderSide.BUY else -1.0
        slippage_pct = self.config.slippage_coefficient * (filled_qty / max(1, capacity)) * (1.0 + 0.5 * hidden_volatility)
        return _clamp_min(price * (1.0 + side_mult * slippage_pct))

    def execute_order(
        self,
        state: AITEAState,
        order: OrderInstruction,
        liquidity_budget: Dict[str, int] | None = None,
    ) -> ExecutionResult:
        liquidity_budget = dict(liquidity_budget or {})
        symbol = order.symbol

        if symbol not in state.prices:
            return ExecutionResult(
                symbol=symbol,
                side=str(order.side),
                requested_quantity=int(order.quantity),
                filled_quantity=0,
                remaining_quantity=int(order.quantity),
                average_fill_price=0.0,
                execution_cost=0.0,
                slippage_cost=0.0,
                notional=0.0,
                status="rejected",
                violation="unknown_symbol",
            )

        requested = max(0, int(order.quantity))
        if requested <= 0:
            return ExecutionResult(
                symbol=symbol,
                side=str(order.side),
                requested_quantity=0,
                filled_quantity=0,
                remaining_quantity=0,
                average_fill_price=0.0,
                execution_cost=0.0,
                slippage_cost=0.0,
                notional=0.0,
                status="rejected",
                violation="non_positive_quantity",
            )

        capacity = int(liquidity_budget.get(symbol, self._liquidity_capacity(state, symbol, order)))
        capacity = max(0, capacity)
        current_position = int(round(_safe_float(state.positions.get(symbol, 0.0), 0.0)))
        price = _safe_float(state.prices[symbol], 0.0)

        if order.side == OrderSide.SELL and current_position <= 0:
            return ExecutionResult(
                symbol=symbol,
                side=str(order.side),
                requested_quantity=requested,
                filled_quantity=0,
                remaining_quantity=requested,
                average_fill_price=0.0,
                execution_cost=0.0,
                slippage_cost=0.0,
                notional=0.0,
                status="rejected",
                violation="no_position_to_sell",
            )

        filled_qty = min(requested, capacity)
        if order.side == OrderSide.SELL:
            filled_qty = min(filled_qty, current_position)

        if filled_qty <= 0:
            pending = PendingOrder(
                order_id=f"{getattr(state, 'episode_id', 'episode')}-{getattr(state, 'step', 0)}-{symbol}",
                symbol=symbol,
                side=order.side,
                quantity=requested,
                filled_quantity=0,
                remaining_quantity=requested,
                order_type=order.order_type,
                status="pending",
            )
            state.pending_orders.append(pending)
            return ExecutionResult(
                symbol=symbol,
                side=str(order.side),
                requested_quantity=requested,
                filled_quantity=0,
                remaining_quantity=requested,
                average_fill_price=0.0,
                execution_cost=0.0,
                slippage_cost=0.0,
                notional=0.0,
                status="pending",
                violation="liquidity_constraint",
            )

        fill_price = self._fill_price(price, order.side, filled_qty, max(1, capacity), _safe_float(state.hidden_volatility, 0.0))
        notional = fill_price * filled_qty
        execution_cost = notional * self.config.transaction_cost_pct
        slippage_cost = abs(fill_price - price) * filled_qty

        if order.side == OrderSide.BUY:
            total_cost = notional + execution_cost
            state.cash -= total_cost
            old_qty = _safe_float(state.positions.get(symbol, 0.0), 0.0)
            new_qty = old_qty + filled_qty
            prev_avg = _safe_float(state.avg_cost.get(symbol, price), price)
            if new_qty > 0:
                state.avg_cost[symbol] = (prev_avg * old_qty + fill_price * filled_qty) / new_qty
            state.positions[symbol] = new_qty
        else:
            old_qty = _safe_float(state.positions.get(symbol, 0.0), 0.0)
            sold_qty = min(old_qty, float(filled_qty))
            proceeds = fill_price * sold_qty
            state.cash += max(0.0, proceeds - execution_cost)
            avg_cost = _safe_float(state.avg_cost.get(symbol, price), price)
            realized = (fill_price - avg_cost) * sold_qty
            state.realized_pnl_by_symbol[symbol] = _safe_float(state.realized_pnl_by_symbol.get(symbol, 0.0), 0.0) + realized
            state.positions[symbol] = max(0.0, old_qty - sold_qty)
            filled_qty = int(sold_qty)

        remaining = max(0, requested - filled_qty)
        status = "filled" if remaining == 0 else "partial"

        if remaining > 0:
            state.pending_orders.append(
                PendingOrder(
                    order_id=f"{getattr(state, 'episode_id', 'episode')}-{getattr(state, 'step', 0)}-{symbol}",
                    symbol=symbol,
                    side=order.side,
                    quantity=requested,
                    filled_quantity=filled_qty,
                    remaining_quantity=remaining,
                    order_type=order.order_type,
                    status="partial",
                )
            )

        return ExecutionResult(
            symbol=symbol,
            side=str(order.side),
            requested_quantity=requested,
            filled_quantity=filled_qty,
            remaining_quantity=remaining,
            average_fill_price=fill_price,
            execution_cost=execution_cost,
            slippage_cost=slippage_cost,
            notional=notional,
            status=status,
            violation=None,
        )

    def execute_orders(
        self,
        state: AITEAState,
        orders: Iterable[OrderInstruction],
        liquidity_budget: Dict[str, int] | None = None,
    ) -> List[ExecutionResult]:
        results: List[ExecutionResult] = []
        budget = dict(liquidity_budget or {})
        for order in orders:
            result = self.execute_order(state, order, budget)
            results.append(result)
            if result.remaining_quantity > 0:
                budget[result.symbol] = max(0, budget.get(result.symbol, 0) - result.filled_quantity)
        return results
