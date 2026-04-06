"""Trade execution engine for AITEA."""

from __future__ import annotations

from typing import Dict, List, Tuple

from ..config import AITEAConfig, get_config
from ..schemas import Action, OrderInstruction, OrderSide, OrderType, PendingOrder
from ..env.state_manager import AITEAState


class ExecutionEngine:
    """Simulate fills, partial fills, slippage, and transaction costs."""

    def __init__(self, config: AITEAConfig | None = None) -> None:
        self.config = config or get_config()

    def _order_list(self, action: Action) -> List[OrderInstruction]:
        orders = list(action.orders)
        if action.flatten_all:
            for symbol, qty in action.rebalance_targets.items():
                if qty > 0:
                    side = OrderSide.SELL if qty > 0 else OrderSide.BUY
                    orders.append(
                        OrderInstruction(
                            symbol=symbol,
                            side=side,
                            quantity=max(1, int(round(abs(qty)))),
                            order_type=OrderType.MARKET,
                            urgency=1.0,
                            tag="flatten",
                        )
                    )
        return orders

    def execute(
        self,
        state: AITEAState,
        action: Action,
        liquidity_budget: Dict[str, int] | None = None,
    ) -> Dict[str, float]:
        """
        Execute the provided action against the current state.

        Mutates state in place and returns diagnostics.
        """
        budget = dict(liquidity_budget or {})
        orders = self._order_list(action)

        execution_cost = 0.0
        slippage_cost = 0.0
        filled_total = 0
        requested_total = 0
        violations: List[str] = []

        for order in orders:
            symbol = order.symbol
            if symbol not in state.prices:
                violations.append(f"unknown_symbol:{symbol}")
                continue

            price = float(state.prices[symbol])
            requested = int(order.quantity)
            requested_total += requested
            if requested <= 0:
                violations.append("non_positive_quantity")
                continue

            capacity = int(budget.get(symbol, requested))
            filled = min(requested, max(0, capacity))
            if filled <= 0:
                state.pending_orders.append(
                    PendingOrder(
                        order_id=f"{state.episode_id}-{state.step}-{symbol}",
                        symbol=symbol,
                        side=order.side,
                        quantity=requested,
                        filled_quantity=0,
                        remaining_quantity=requested,
                        order_type=order.order_type,
                        status="pending",
                    )
                )
                continue

            budget[symbol] = max(0, capacity - filled)
            filled_total += filled

            side_mult = 1.0 if order.side == OrderSide.BUY else -1.0
            slippage_pct = self.config.slippage_coefficient * (filled / max(1, capacity)) * (1.0 + state.hidden_volatility)
            fill_price = max(0.01, price * (1.0 + side_mult * slippage_pct))
            notional = fill_price * filled
            fee = notional * self.config.transaction_cost_pct
            execution_cost += fee
            slippage_cost += abs(fill_price - price) * filled

            if order.side == OrderSide.BUY:
                state.cash -= (notional + fee)
                old_qty = float(state.positions.get(symbol, 0.0))
                new_qty = old_qty + filled
                prev_avg = float(state.avg_cost.get(symbol, price))
                if new_qty > 0:
                    state.avg_cost[symbol] = (prev_avg * old_qty + fill_price * filled) / new_qty
                state.positions[symbol] = new_qty
            else:
                held = float(state.positions.get(symbol, 0.0))
                sold = min(held, float(filled))
                proceeds = fill_price * sold
                state.cash += (proceeds - fee)
                avg_cost = float(state.avg_cost.get(symbol, price))
                realized = (fill_price - avg_cost) * sold
                state.realized_pnl_by_symbol[symbol] = state.realized_pnl_by_symbol.get(symbol, 0.0) + realized
                state.positions[symbol] = max(0.0, held - sold)
                if sold < filled:
                    violations.append(f"sell_clipped_to_position:{symbol}")

            if filled < requested:
                state.pending_orders.append(
                    PendingOrder(
                        order_id=f"{state.episode_id}-{state.step}-{symbol}",
                        symbol=symbol,
                        side=order.side,
                        quantity=requested,
                        filled_quantity=filled,
                        remaining_quantity=requested - filled,
                        order_type=order.order_type,
                        status="partial",
                    )
                )

        state.pending_orders = state.pending_orders[-10:]

        return {
            "execution_cost": execution_cost,
            "slippage_cost": slippage_cost,
            "fill_ratio": (filled_total / requested_total) if requested_total > 0 else 1.0,
            "filled_total": float(filled_total),
            "requested_total": float(requested_total),
            "violation_count": float(len(violations)),
        }
