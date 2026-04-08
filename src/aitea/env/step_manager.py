"""One-step simulation update for AITEA."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple

from ..config import AITEAConfig, get_config
from ..reward import RewardModel
from ..schemas import (
    Action,
    MarketRegime,
    NewsSignal,
    OrderInstruction,
    OrderSide,
    OrderType,
    PendingOrder,
    Reward,
)
from .state_manager import (
    AITEAState,
    StateManager,
    Transition,
    build_observation,
    update_derived_state,
)


def _validate_action(action: Any) -> Action:
    if isinstance(action, Action):
        return action
    validator = getattr(Action, "model_validate", None)
    if callable(validator):
        return validator(action)
    return Action.parse_obj(action)


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(v)))


def _dedupe_preserve_order(values: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out


class StepManager:
    """Execute one simulation step."""

    def __init__(self, config: AITEAConfig | None = None) -> None:
        self.config = config or get_config()
        self.state_manager = StateManager(self.config)
        self.reward_model = RewardModel(self.config)

    def _append_recent(self, state: AITEAState, action_text: str, reward_value: float) -> None:
        state.recent_actions.append(action_text)
        state.recent_rewards.append(float(reward_value))
        state.recent_actions = state.recent_actions[-10:]
        state.recent_rewards = state.recent_rewards[-10:]

    def _ensure_state(self, state: AITEAState) -> None:
        if not hasattr(state, "task_metrics") or state.task_metrics is None:
            state.task_metrics = {}
        if not hasattr(state, "realized_pnl_by_symbol") or state.realized_pnl_by_symbol is None:
            state.realized_pnl_by_symbol = {}
        if not hasattr(state, "avg_cost") or state.avg_cost is None:
            state.avg_cost = {}
        if not hasattr(state, "pending_orders") or state.pending_orders is None:
            state.pending_orders = []
        if not hasattr(state, "recent_actions") or state.recent_actions is None:
            state.recent_actions = []
        if not hasattr(state, "recent_rewards") or state.recent_rewards is None:
            state.recent_rewards = []
        if not hasattr(state, "news_queue") or state.news_queue is None:
            state.news_queue = []
        if not hasattr(state, "prices") or state.prices is None:
            state.prices = {}
        if not hasattr(state, "previous_prices") or state.previous_prices is None:
            state.previous_prices = {}
        if not hasattr(state, "positions") or state.positions is None:
            state.positions = {}
        if not hasattr(state, "cash"):
            state.cash = 0.0
        if not hasattr(state, "equity"):
            state.equity = 0.0
        if not hasattr(state, "starting_cash"):
            state.starting_cash = max(1.0, float(state.equity or 1.0))
        if not hasattr(state, "done"):
            state.done = False
        if not hasattr(state, "last_error"):
            state.last_error = None

    def _generate_news(self, state: AITEAState) -> None:
        profile = state.task_profile
        p = float(profile.get("news_probability", 0.03))
        if state.rng.random() < p:
            severity = _clamp(abs(state.rng.gauss(0.4, 0.25)), 0.05, 1.0)
            sentiment = _clamp(state.rng.uniform(-1.0, 1.0), -1.0, 1.0)
            headline = f"{state.task_name} market event at step {state.step}"
            state.news_queue.append(
                NewsSignal(
                    headline=headline,
                    severity=severity,
                    sentiment=sentiment,
                    affected_symbols=list(state.prices.keys())[:2],
                )
            )
            state.news_queue = state.news_queue[-5:]

    def _maybe_switch_regime(self, state: AITEAState) -> None:
        profile = state.task_profile
        p = float(profile.get("regime_flip_probability", 0.02))
        if state.rng.random() >= p:
            return

        choices = [
            MarketRegime.CALM,
            MarketRegime.NORMAL,
            MarketRegime.VOLATILE,
            MarketRegime.CRISIS,
            MarketRegime.RECOVERY,
        ]
        new_regime = choices[state.rng.randrange(len(choices))]
        state.regime = new_regime
        if new_regime == MarketRegime.CALM:
            state.hidden_volatility = max(0.004, state.hidden_volatility * 0.85)
            state.regime_confidence = 0.78
        elif new_regime == MarketRegime.NORMAL:
            state.hidden_volatility = max(0.006, state.hidden_volatility * 1.0)
            state.regime_confidence = 0.72
        elif new_regime == MarketRegime.VOLATILE:
            state.hidden_volatility = max(0.010, state.hidden_volatility * 1.5)
            state.regime_confidence = 0.55
        elif new_regime == MarketRegime.CRISIS:
            state.hidden_volatility = max(0.015, state.hidden_volatility * 2.0)
            state.regime_confidence = 0.42
        elif new_regime == MarketRegime.RECOVERY:
            state.hidden_volatility = max(0.008, state.hidden_volatility * 1.15)
            state.regime_confidence = 0.60

    def _advance_prices(self, state: AITEAState) -> None:
        state.previous_prices = dict(state.prices)
        latest_news = state.news_queue[-1] if state.news_queue else None

        if state.regime == MarketRegime.CALM:
            regime_mult = 0.65
            drift_base = 0.00010
        elif state.regime == MarketRegime.NORMAL:
            regime_mult = 1.00
            drift_base = 0.00015
        elif state.regime == MarketRegime.VOLATILE:
            regime_mult = 1.50
            drift_base = 0.00005
        elif state.regime == MarketRegime.CRISIS:
            regime_mult = 2.10
            drift_base = -0.00010
        elif state.regime == MarketRegime.RECOVERY:
            regime_mult = 0.90
            drift_base = 0.00020
        else:
            regime_mult = 1.00
            drift_base = 0.00010

        for idx, symbol in enumerate(list(state.prices.keys())):
            prev_price = float(state.prices[symbol])
            seasonal = 0.00005 * math.sin((state.step + 1) / 4.0 + idx)

            news_drift = 0.0
            news_vol_mult = 1.0
            if latest_news is not None and symbol in latest_news.affected_symbols:
                news_drift = latest_news.sentiment * latest_news.severity * 0.010
                news_vol_mult = 1.0 + latest_news.severity * 1.5

            sigma = max(0.0005, state.hidden_volatility * regime_mult * news_vol_mult)
            shock = state.rng.gauss(drift_base + seasonal + news_drift, sigma)
            state.prices[symbol] = max(0.01, prev_price * (1.0 + shock))

    def _target_error(self, state: AITEAState) -> float:
        kind = state.task_profile.get("kind", "generic")
        equity = max(1.0, state.equity)

        if kind in {"execution", "liquidity"}:
            symbols = list(state.prices.keys())
            default_symbol = symbols[0] if symbols else "AAPL"
            target_symbol = str(state.task_profile.get("target_symbol", default_symbol))
            target_qty = float(state.task_profile.get("target_quantity", 0.0))
            held = float(state.positions.get(target_symbol, 0.0))
            return abs(target_qty - held) / max(1.0, target_qty)

        if kind == "hedge":
            fx_exposure = abs(float(state.task_metrics.get("fx_exposure", 0.0)))
            return fx_exposure / max(1.0, float(state.task_profile.get("fx_exposure", 1.0)))

        if kind == "rebalance":
            target_weights = dict(state.task_profile.get("target_weights", {}))
            if not target_weights:
                return 0.0
            error = 0.0
            for symbol, target_w in target_weights.items():
                current_w = 0.0
                if equity > 0:
                    current_w = (state.positions.get(symbol, 0.0) * state.prices.get(symbol, 0.0)) / equity
                error += abs(current_w - float(target_w))
            return error / max(1, len(target_weights))

        if kind in {"news", "regime"}:
            return state.drawdown_pct

        return state.drawdown_pct

    def _orders_from_action(self, state: AITEAState, action: Action) -> List[OrderInstruction]:
        orders: List[OrderInstruction] = list(action.orders)

        if action.flatten_all:
            for symbol, qty in state.positions.items():
                if qty > 0:
                    orders.append(
                        OrderInstruction(
                            symbol=symbol,
                            side=OrderSide.SELL,
                            quantity=max(1, int(round(qty))),
                            order_type=OrderType.MARKET,
                            urgency=1.0,
                            tag="flatten_all",
                        )
                    )

        if action.rebalance_targets:
            equity = max(1.0, state.equity)
            for symbol, target_w in action.rebalance_targets.items():
                price = max(1.0, float(state.prices.get(symbol, 1.0)))
                current_value = float(state.positions.get(symbol, 0.0)) * price
                desired_value = float(target_w) * equity
                delta_value = desired_value - current_value
                if abs(delta_value) < price:
                    continue
                side = OrderSide.BUY if delta_value > 0 else OrderSide.SELL
                qty = max(1, int(round(abs(delta_value) / price)))
                orders.append(
                    OrderInstruction(
                        symbol=symbol,
                        side=side,
                        quantity=qty,
                        order_type=OrderType.MARKET,
                        urgency=0.8,
                        tag="rebalance",
                    )
                )

        if action.hedge_targets:
            for symbol, delta in action.hedge_targets.items():
                price = max(1.0, float(state.prices.get(symbol, 1.0)))
                side = OrderSide.BUY if delta > 0 else OrderSide.SELL
                qty = max(1, int(round(abs(float(delta)) / price)))
                orders.append(
                    OrderInstruction(
                        symbol=symbol,
                        side=side,
                        quantity=qty,
                        order_type=OrderType.MARKET,
                        urgency=0.7,
                        tag="hedge",
                    )
                )

        return orders

    def _apply_order(
        self,
        state: AITEAState,
        order: OrderInstruction,
        liquidity_budget: Dict[str, int],
        violations: List[str],
    ) -> Tuple[float, float, int, int, float]:
        symbol = order.symbol
        if symbol not in state.prices:
            violations.append("invalid_action")
            violations.append(f"unknown_symbol:{symbol}")
            return 0.0, 0.0, 0, 0, 0.0

        requested = int(order.quantity)
        if requested <= 0:
            violations.append("invalid_action")
            violations.append("non_positive_quantity")
            return 0.0, 0.0, 0, 0, 0.0

        capacity = int(liquidity_budget.get(symbol, requested))
        filled = min(requested, max(0, capacity))
        if filled <= 0:
            pending = PendingOrder(
                order_id=f"{state.episode_id}-{state.step}-{symbol}",
                symbol=symbol,
                side=order.side,
                quantity=requested,
                filled_quantity=0,
                remaining_quantity=requested,
                order_type=order.order_type,
                status="pending",
            )
            state.pending_orders.append(pending)
            violations.append("liquidity_constraint")
            return 0.0, 0.0, 0, requested, 0.0

        liquidity_budget[symbol] = max(0, capacity - filled)
        price = float(state.prices[symbol])
        side_mult = 1.0 if order.side == OrderSide.BUY else -1.0

        if order.side == OrderSide.SELL and state.positions.get(symbol, 0.0) <= 0:
            violations.append("invalid_action")
            violations.append(f"no_position_to_sell:{symbol}")
            return 0.0, 0.0, 0, requested, 0.0

        available = int(round(state.positions.get(symbol, 0.0))) if order.side == OrderSide.SELL else filled
        if order.side == OrderSide.SELL:
            filled = min(filled, max(0, available))
            if filled <= 0:
                violations.append("invalid_action")
                violations.append(f"insufficient_position:{symbol}")
                return 0.0, 0.0, 0, requested, 0.0
            if filled < requested:
                violations.append(f"partial_fill_position_cap:{symbol}")

        slippage_pct = self.config.slippage_coefficient * (filled / max(1, capacity)) * (1.0 + 0.5 * state.hidden_volatility)
        fill_price = max(0.01, price * (1.0 + side_mult * slippage_pct))
        notional = fill_price * filled
        execution_cost = notional * self.config.transaction_cost_pct
        slippage_cost = abs(fill_price - price) * filled
        turnover_notional = notional

        if order.side == OrderSide.BUY:
            total_cost = notional + execution_cost
            state.cash -= total_cost
            old_qty = float(state.positions.get(symbol, 0.0))
            new_qty = old_qty + filled
            prev_avg = float(state.avg_cost.get(symbol, price))
            if new_qty > 0:
                state.avg_cost[symbol] = (prev_avg * old_qty + fill_price * filled) / new_qty
            state.positions[symbol] = new_qty
        else:
            old_qty = float(state.positions.get(symbol, 0.0))
            sold_qty = min(old_qty, float(filled))
            proceeds = fill_price * sold_qty
            total_cost = proceeds - execution_cost
            state.cash += total_cost
            avg_cost = float(state.avg_cost.get(symbol, price))
            realized = (fill_price - avg_cost) * sold_qty
            state.realized_pnl_by_symbol[symbol] = state.realized_pnl_by_symbol.get(symbol, 0.0) + realized
            state.positions[symbol] = max(0.0, old_qty - sold_qty)
            if sold_qty < filled:
                violations.append(f"sell_clipped_to_position:{symbol}")
            filled = int(sold_qty)

        return execution_cost, slippage_cost, filled, requested, turnover_notional

    def _build_reward(
        self,
        state: AITEAState,
        prev_equity: float,
        pnl_delta: float,
        execution_cost_total: float,
        slippage_cost_total: float,
        filled_total: int,
        requested_total: int,
        turnover_notional: float,
        target_error_before: float,
        target_error_after: float,
        violations: List[str],
        action_obj: Action,
    ) -> Reward:
        violations = _dedupe_preserve_order(list(violations))

        fill_ratio = (filled_total / requested_total) if requested_total > 0 else 1.0
        completion_progress = max(0.0, target_error_before - target_error_after)
        turnover = turnover_notional / max(1.0, prev_equity)

        if action_obj.hold_position and not action_obj.orders and not action_obj.rebalance_targets and not action_obj.hedge_targets and not action_obj.flatten_all:
            active_task = state.task_profile.get("kind", "generic")
            if active_task in {"execution", "liquidity", "hedge", "rebalance"} and target_error_after > 0.01:
                violations.append("no_op_abuse")

        if state.gross_exposure > self.config.max_gross_exposure_pct * max(1.0, state.starting_cash):
            violations.append("gross_exposure_breach")
        if state.drawdown_pct > self.config.max_drawdown_pct:
            violations.append("drawdown_breach")
        if any(
            abs(v) > self.config.max_position_pct * max(1.0, state.starting_cash)
            for v in (state.positions.get(sym, 0.0) * state.prices.get(sym, 0.0) for sym in state.positions)
        ):
            violations.append("position_limit_breach")

        violations = _dedupe_preserve_order(list(violations))

        state.task_metrics["violation_count"] = float(len(violations))
        state.task_metrics["execution_cost"] = float(execution_cost_total)
        state.task_metrics["slippage_cost"] = float(slippage_cost_total)
        state.task_metrics["fill_ratio"] = float(fill_ratio)
        state.task_metrics["turnover"] = float(turnover)
        state.task_metrics["pnl_delta"] = float(pnl_delta)
        state.task_metrics["target_error"] = float(target_error_after)
        state.task_metrics["progress"] = float(completion_progress)
        state.task_metrics["market_drag"] = float(slippage_cost_total)
        state.task_metrics["cash"] = float(state.cash)
        state.task_metrics["equity"] = float(state.equity)
        state.task_metrics["gross_exposure"] = float(state.gross_exposure)
        state.task_metrics["net_exposure"] = float(state.net_exposure)
        state.task_metrics["drawdown_pct"] = float(state.drawdown_pct)
        state.task_metrics["equity_peak"] = float(state.peak_equity)

        if state.task_profile.get("kind") in {"execution", "liquidity"}:
            state.task_metrics["target_remaining"] = max(
                0.0,
                float(state.task_profile.get("target_quantity", 0.0)) * target_error_after,
            )
        elif state.task_profile.get("kind") == "hedge":
            hedge_symbol = str(state.task_profile.get("hedge_symbol", "MSFT"))
            state.task_metrics["hedge_error"] = max(0.0, float(state.task_metrics.get("fx_exposure", 0.0)))
            state.task_metrics["hedge_symbol"] = hedge_symbol
        elif state.task_profile.get("kind") == "rebalance":
            state.task_metrics["tracking_error"] = float(target_error_after)

        try:
            reward = self.reward_model.compute(
                state,
                pnl_delta=pnl_delta,
                execution_cost=execution_cost_total,
                slippage=slippage_cost_total,
                fill_ratio=fill_ratio,
                turnover=turnover,
                market_drag=slippage_cost_total,
                target_error=target_error_after,
                completion_progress=completion_progress,
                violations=violations,
            )
        except Exception as exc:
            fallback_total = (
                (pnl_delta / max(1.0, prev_equity))
                - (execution_cost_total / max(1.0, prev_equity))
                - (slippage_cost_total / max(1.0, prev_equity))
                - 0.25 * len(violations)
            )
            fallback_total = _clamp(fallback_total, self.config.reward_clip_min, self.config.reward_clip_max)
            reward = Reward(
                total=fallback_total,
                normalized_score=_clamp((math.tanh(fallback_total) + 1.0) / 2.0, 0.0, 1.0),
                components={
                    "pnl": pnl_delta / max(1.0, prev_equity),
                    "fill_ratio": fill_ratio,
                    "completion_progress": completion_progress,
                },
                penalties={
                    "fallback_error": 1.0,
                    "violation_count": float(len(violations)),
                },
                raw_total=fallback_total,
                clipped_total=fallback_total,
            )
            state.last_error = f"reward_fallback:{type(exc).__name__}"

        reward.total = _clamp(reward.total, self.config.reward_clip_min, self.config.reward_clip_max)
        reward.clipped_total = reward.total
        reward.normalized_score = _clamp(reward.normalized_score, 0.0, 1.0)
        return reward

    def step(self, state: AITEAState, action: Any) -> Transition:
        try:
            self._ensure_state(state)
            state.last_error = None

            if state.done:
                obs = build_observation(state, self.config)
                info = self.state_manager.info(state, violations=["episode_already_done"])
                reward_model = Reward(
                    total=0.0,
                    normalized_score=0.0,
                    components={},
                    penalties={"episode_already_done": 1.0},
                    raw_total=0.0,
                    clipped_total=0.0,
                )
                return Transition(
                    observation=obs,
                    reward=0.0,
                    done=True,
                    info=info,
                    error="episode_already_done",
                    reward_detail=reward_model,
                )

            try:
                action_obj = _validate_action(action)
            except Exception as exc:
                obs = build_observation(state, self.config)
                info = self.state_manager.info(state, violations=["invalid_action"])
                reward_detail = Reward(
                    total=0.0,
                    normalized_score=0.0,
                    components={},
                    penalties={"invalid_action": 1.0},
                    raw_total=0.0,
                    clipped_total=0.0,
                )
                return Transition(
                    observation=obs,
                    reward=0.0,
                    done=False,
                    info=info,
                    error=f"invalid_action:{type(exc).__name__}",
                    reward_detail=reward_detail,
                )

            prev_equity = float(state.equity)
            target_error_before = self._target_error(state)

            execution_cost_total = 0.0
            slippage_cost_total = 0.0
            filled_total = 0
            requested_total = 0
            turnover_notional = 0.0
            violations: List[str] = []

            orders = self._orders_from_action(state, action_obj)

            if (
                (not action_obj.orders or len(action_obj.orders) == 0)
                and not action_obj.flatten_all
                and not action_obj.rebalance_targets
                and not action_obj.hedge_targets
            ):
                violations.append("no_op_abuse")
            # HARD PENALTY FOR NO-OP
            if "no_op_abuse" in violations:
                state.task_metrics["force_negative_reward"] = 1.0

            valid_symbols = set(state.prices.keys())
            filtered_orders: List[OrderInstruction] = []
            for order in orders:
                if order.symbol in valid_symbols:
                    filtered_orders.append(order)
                else:
                    violations.append("invalid_symbol")
                    violations.append(f"unknown_symbol:{order.symbol}")
            orders = filtered_orders

            if action_obj.cancel_order_ids:
                remaining: List[PendingOrder] = []
                cancelled = set(action_obj.cancel_order_ids)
                for po in state.pending_orders:
                    if po.order_id in cancelled:
                        continue
                    remaining.append(po)
                if len(remaining) == len(state.pending_orders):
                    violations.append("cancel_id_not_found")
                state.pending_orders = remaining

            liquidity_scale = float(state.task_profile.get("liquidity_scale", 0.75))
            liquidity_budget: Dict[str, int] = {
                symbol: int(max(1, round(1200.0 * liquidity_scale)))
                for symbol in state.prices
            }

            carry_over: List[PendingOrder] = []
            for pending in state.pending_orders:
                order = OrderInstruction(
                    symbol=pending.symbol,
                    side=pending.side,
                    quantity=int(pending.remaining_quantity),
                    order_type=pending.order_type,
                    urgency=0.5,
                    tag="pending",
                )
                ec, sc, filled, requested, notional = self._apply_order(state, order, liquidity_budget, violations)
                execution_cost_total += ec
                slippage_cost_total += sc
                filled_total += filled
                requested_total += requested
                turnover_notional += notional
                remaining_qty = max(0, requested - filled)
                if remaining_qty > 0:
                    carry_over.append(
                        PendingOrder(
                            order_id=pending.order_id,
                            symbol=pending.symbol,
                            side=pending.side,
                            quantity=pending.quantity,
                            filled_quantity=pending.filled_quantity + filled,
                            remaining_quantity=remaining_qty,
                            order_type=pending.order_type,
                            status="pending",
                        )
                    )
            state.pending_orders = carry_over

            for order in orders:
                ec, sc, filled, requested, notional = self._apply_order(state, order, liquidity_budget, violations)
                execution_cost_total += ec
                slippage_cost_total += sc
                filled_total += filled
                requested_total += requested
                turnover_notional += notional
                if filled < requested:
                    remaining_qty = requested - filled
                    state.pending_orders.append(
                        PendingOrder(
                            order_id=f"{state.episode_id}-{state.step}-{order.symbol}",
                            symbol=order.symbol,
                            side=order.side,
                            quantity=requested,
                            filled_quantity=filled,
                            remaining_quantity=remaining_qty,
                            order_type=order.order_type,
                            status="partial",
                        )
                    )

            state.pending_orders = state.pending_orders[-10:]
            state.recent_actions = state.recent_actions[-10:]
            state.recent_rewards = state.recent_rewards[-10:]
            state.news_queue = state.news_queue[-5:]

            kind = state.task_profile.get("kind", "generic")
            if kind == "hedge":
                hedge_symbol = str(state.task_profile.get("hedge_symbol", next(iter(state.prices.keys()), "MSFT")))
                trade_notional = 0.0
                for order in orders:
                    if order.symbol == hedge_symbol:
                        trade_notional += float(order.quantity) * float(state.prices.get(hedge_symbol, 1.0))
                fx_exposure = float(state.task_metrics.get("fx_exposure", 0.0))
                fx_exposure = max(0.0, fx_exposure - 0.35 * trade_notional)
                state.task_metrics["fx_exposure"] = fx_exposure
                state.task_metrics["hedge_error"] = fx_exposure

            self._generate_news(state)
            self._maybe_switch_regime(state)
            self._advance_prices(state)
            update_derived_state(state)

            pnl_delta = state.equity - prev_equity
            target_error_after = self._target_error(state)

            reward_detail = self._build_reward(
                state,
                prev_equity=prev_equity,
                pnl_delta=pnl_delta,
                execution_cost_total=execution_cost_total,
                slippage_cost_total=slippage_cost_total,
                filled_total=filled_total,
                requested_total=requested_total,
                turnover_notional=turnover_notional,
                target_error_before=target_error_before,
                target_error_after=target_error_after,
                violations=violations,
                action_obj=action_obj,
            )

            state.step += 1
            state.task_metrics["step"] = float(state.step)

            horizon = int(state.task_profile.get("horizon", self.config.episode_length))
            success = False
            if kind in {"execution", "liquidity"}:
                success = float(state.task_metrics.get("target_remaining", 1.0)) <= 1.0
            elif kind == "hedge":
                success = float(state.task_metrics.get("fx_exposure", 1.0)) <= 1_000.0
            elif kind == "rebalance":
                success = float(state.task_metrics.get("tracking_error", 1.0)) <= 0.08
            elif kind in {"news", "regime"}:
                success = state.drawdown_pct <= self.config.max_drawdown_pct * 0.75

            state.done = bool(state.step >= horizon or success or "drawdown_breach" in violations)

            violations = _dedupe_preserve_order(violations)
            state.last_error = ";".join(violations) if violations else None

            action_text = f"orders={len(orders)} hold={action_obj.hold_position} flatten={action_obj.flatten_all}"
            self._append_recent(state, action_text, float(reward_detail.total))

            info = self.state_manager.info(
                state,
                execution_cost=execution_cost_total,
                slippage=slippage_cost_total,
                fill_ratio=(filled_total / requested_total) if requested_total > 0 else 1.0,
                turnover=(turnover_notional / max(1.0, prev_equity)),
                pnl_delta=pnl_delta,
                violations=violations,
            )

            obs = build_observation(state, self.config)
            return Transition(
                observation=obs,
                reward=float(reward_detail.total),
                done=state.done,
                info=info,
                error=state.last_error,
                reward_detail=reward_detail,
            )

        except Exception as exc:
            self._ensure_state(state)
            state.last_error = f"step_fallback:{type(exc).__name__}"
            obs = build_observation(state, self.config)
            info = self.state_manager.info(state, violations=[state.last_error])
            reward_detail = Reward(
                total=0.0,
                normalized_score=0.0,
                components={},
                penalties={"step_error": 1.0},
                raw_total=0.0,
                clipped_total=0.0,
            )
            return Transition(
                observation=obs,
                reward=0.0,
                done=False,
                info=info,
                error=state.last_error,
                reward_detail=reward_detail,
            )
