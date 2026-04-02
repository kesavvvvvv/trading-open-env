"""
Step manager for AITEA.

This orchestrates one full environment transition.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Iterable, Tuple

from aitea.schemas.action import Action, Order
from aitea.schemas.common import ComplianceStatus, OrderSide, OrderType, RiskLevel
from aitea.schemas.info import Info
from aitea.schemas.observation import Observation
from aitea.schemas.reward import Reward


class StepManager:
    def step(self, env: Any, raw_action: Any) -> tuple[Observation, Reward, bool, Info]:
        env.ensure_open()
        env.ensure_not_done()

        action = env.validate_action(raw_action)
        state = env.state_data
        config = env.config
        rng = env.rng

        execution_metrics = self._apply_execution(env, action)
        self._apply_market_update(env, action)
        treasury_metrics = self._apply_treasury_update(env)
        self._apply_news_update(env)
        self._apply_regime_update(env)
        self._apply_multi_agent_update(env)
        risk_metrics = self._apply_risk_update(env)

        self._revalue_portfolio(env)

        reward = self._compute_reward(env, execution_metrics, treasury_metrics, risk_metrics)
        env.episode_reward += reward.value

        env.step_count += 1
        state["timestamp"] = env.step_count
        state["step_index"] = env.step_count
        state["episode_reward"] = env.episode_reward

        done = self._check_done(env, action, risk_metrics)
        env.done = done
        state["done"] = done

        observation = env.state_manager.build_observation(env)

        info = Info(
            pnl_delta=float(execution_metrics.get("pnl_delta", 0.0)),
            total_pnl=float(state["portfolio"].get("realized_pnl", 0.0) + state["portfolio"].get("unrealized_pnl", 0.0)),
            execution_cost=float(execution_metrics.get("execution_cost", 0.0)),
            slippage=float(execution_metrics.get("slippage", 0.0)),
            compliance_status=risk_metrics.get("compliance_status", ComplianceStatus.OK),
            violations=list(risk_metrics.get("violations", [])),
            drawdown=float(risk_metrics.get("drawdown", 0.0)),
            exposure=float(risk_metrics.get("exposure", 0.0)),
            step_index=env.step_count,
            episode_reward=env.episode_reward,
            task_metrics=self._build_task_metrics(env, execution_metrics, treasury_metrics, risk_metrics),
        )

        env.last_info = info
        env.current_observation = observation

        env.state_manager.update_recent_history(
            env,
            {
                "step": env.step_count,
                "reward": reward.value,
                "slippage": info.slippage,
                "pnl_delta": info.pnl_delta,
                "execution_cost": info.execution_cost,
            },
        )
        env.state_manager.rebuild_history_summary(env)

        env.last_action_error = None

        task = env.task
        if task is not None:
            on_step = getattr(task, "on_step", None)
            if callable(on_step):
                try:
                    on_step(env.state_data, action, info)
                except TypeError:
                    try:
                        on_step(env.state_data)
                    except TypeError:
                        on_step()

        return observation, reward, done, info

    def _apply_execution(self, env: Any, action: Action) -> Dict[str, float]:
        if action.pause_trading:
            return {"pnl_delta": 0.0, "execution_cost": 0.0, "slippage": 0.0}

        engine = getattr(env, "execution_engine", None)
        if engine is not None:
            for method_name in ("execute", "step", "apply"):
                method = getattr(engine, method_name, None)
                if callable(method):
                    result = self._call_engine_method(method, env, action)
                    if isinstance(result, dict):
                        self._merge_state_from_dict(env, result)
                        return {
                            "pnl_delta": float(result.get("pnl_delta", 0.0)),
                            "execution_cost": float(result.get("execution_cost", 0.0)),
                            "slippage": float(result.get("slippage", 0.0)),
                        }
                    break

        return self._fallback_execution(env, action)

    def _fallback_execution(self, env: Any, action: Action) -> Dict[str, float]:
        state = env.state_data
        market_prices = state["market"]["prices"]
        portfolio = state["portfolio"]

        pnl_delta = 0.0
        execution_cost = 0.0
        slippage = 0.0
        pending_orders = state.setdefault("pending_orders", [])

        for order in action.orders:
            if order.quantity <= 0:
                state["last_action_error"] = "order_quantity_must_be_positive"
                state.setdefault("violations", []).append("invalid_quantity")
                continue

            if order.instrument not in market_prices:
                state["last_action_error"] = f"unknown_instrument:{order.instrument}"
                state.setdefault("violations", []).append("unknown_instrument")
                continue

            mid_price = float(market_prices[order.instrument])
            spread = float(state["market"].get("spread", 0.0))
            qty = float(order.quantity)

            fill_qty = self._estimate_fill_quantity(env, order, mid_price)
            if fill_qty <= 0:
                pending_orders.append(
                    {
                        "order_id": f"po_{env.step_count}_{len(pending_orders)}",
                        "instrument": order.instrument,
                        "side": order.side,
                        "order_type": order.order_type,
                        "quantity": qty,
                        "price": order.price,
                        "filled_quantity": 0.0,
                        "status": "queued",
                    }
                )
                continue

            trade_price = self._compute_fill_price(mid_price, spread, order.side, order.order_type, order.price)
            notional = fill_qty * trade_price
            cost = abs(notional) * env.config.transaction_cost_pct
            impact = abs(trade_price - mid_price) * fill_qty

            if order.side == OrderSide.BUY:
                portfolio["cash"] -= (notional + cost)
                portfolio["positions"][order.instrument] = float(portfolio["positions"].get(order.instrument, 0.0)) + fill_qty
                pnl_delta -= (cost + impact)
            else:
                portfolio["cash"] += (notional - cost)
                portfolio["positions"][order.instrument] = float(portfolio["positions"].get(order.instrument, 0.0)) - fill_qty
                pnl_delta -= (cost + impact)

            execution_cost += cost
            slippage += impact

            state.setdefault("metrics", {})["last_trade_price"] = trade_price
            state["metrics"]["last_trade_qty"] = fill_qty

        state["portfolio"]["realized_pnl"] = float(state["portfolio"].get("realized_pnl", 0.0) + pnl_delta)
        return {"pnl_delta": pnl_delta, "execution_cost": execution_cost, "slippage": slippage}

    def _apply_market_update(self, env: Any, action: Action) -> None:
        engine = getattr(env, "market_engine", None)
        if engine is not None:
            for method_name in ("update", "step", "advance"):
                method = getattr(engine, method_name, None)
                if callable(method):
                    result = self._call_engine_method(method, env, action)
                    if isinstance(result, dict):
                        self._merge_state_from_dict(env, result)
                    return

        self._fallback_market_update(env)

    def _fallback_market_update(self, env: Any) -> None:
        state = env.state_data
        rng = env.rng
        regime = state.get("hidden_regime", "bull")
        news_signal = self._aggregate_news_signal(state.get("news", []))

        for asset, price in list(state["market"]["prices"].items()):
            price = float(price)
            regime_drift = {
                "bull": 0.0008,
                "bear": -0.0008,
                "chaotic": 0.0,
            }.get(regime, 0.0)

            shock = news_signal * 0.002
            noise = rng.gauss(0.0, state["market"].get("volatility", 0.02) * 0.5)
            new_price = max(0.01, price * (1.0 + regime_drift + shock + noise))
            state["market"]["prices"][asset] = new_price

            liquidity = float(state["market"]["liquidity"].get(asset, 1.0))
            state["market"]["liquidity"][asset] = max(0.1, min(2.0, liquidity + rng.gauss(0.0, 0.03)))

        vol = float(state["market"].get("volatility", 0.02))
        spread = float(state["market"].get("spread", 0.001))
        state["market"]["volatility"] = max(0.001, min(0.20, vol + rng.gauss(0.0, 0.002) + abs(news_signal) * 0.01))
        state["market"]["spread"] = max(0.0001, min(0.05, spread + rng.gauss(0.0, 0.0002) + abs(news_signal) * 0.0005))

    def _apply_treasury_update(self, env: Any) -> Dict[str, float]:
        engine = getattr(env, "treasury_engine", None)
        if engine is not None:
            for method_name in ("update", "step", "advance"):
                method = getattr(engine, method_name, None)
                if callable(method):
                    result = self._call_engine_method(method, env)
                    if isinstance(result, dict):
                        self._merge_state_from_dict(env, result)
                        return {
                            "treasury_penalty": float(result.get("treasury_penalty", 0.0)),
                            "cash_delta": float(result.get("cash_delta", 0.0)),
                        }
                    return {"treasury_penalty": 0.0, "cash_delta": 0.0}

        return self._fallback_treasury_update(env)

    def _fallback_treasury_update(self, env: Any) -> Dict[str, float]:
        state = env.state_data
        treasury = state["treasury"]
        portfolio = state["portfolio"]

        cash_delta = 0.0
        penalty = 0.0

        outflows = treasury.get("upcoming_outflows", {})
        inflows = treasury.get("upcoming_inflows", {})

        step_key = str(env.step_count)
        if step_key in outflows:
            amt = float(outflows.pop(step_key))
            portfolio["cash"] -= amt
            cash_delta -= amt
            if portfolio["cash"] < treasury.get("minimum_cash_required", 0.0):
                penalty += 1.0
                state.setdefault("violations", []).append("cash_below_minimum")

        if step_key in inflows:
            amt = float(inflows.pop(step_key))
            portfolio["cash"] += amt
            cash_delta += amt

        treasury["current_liquidity_buffer"] = max(0.0, float(portfolio["cash"]) * 0.15)

        return {"treasury_penalty": penalty, "cash_delta": cash_delta}

    def _apply_risk_update(self, env: Any) -> Dict[str, Any]:
        engine = getattr(env, "risk_engine", None)
        if engine is not None:
            for method_name in ("update", "step", "assess"):
                method = getattr(engine, method_name, None)
                if callable(method):
                    result = self._call_engine_method(method, env)
                    if isinstance(result, dict):
                        self._merge_state_from_dict(env, result)
                        return self._normalize_risk_result(result)
                    break

        return self._fallback_risk_update(env)

    def _fallback_risk_update(self, env: Any) -> Dict[str, Any]:
        state = env.state_data
        portfolio = state["portfolio"]
        market_prices = state["market"]["prices"]

        total_position_value = 0.0
        gross_exposure = 0.0

        for instrument, qty in portfolio["positions"].items():
            price = float(market_prices.get(instrument, 0.0))
            value = float(qty) * price
            total_position_value += value
            gross_exposure += abs(value)

        total_value = max(1e-9, float(portfolio["cash"]) + total_position_value)
        peak = max(float(state.get("peak_portfolio_value", total_value)), total_value)
        drawdown = max(0.0, (peak - total_value) / peak)

        state["peak_portfolio_value"] = peak
        state["risk"]["drawdown"] = drawdown
        state["risk"]["exposure"] = gross_exposure / total_value

        violations = state.setdefault("violations", [])
        compliance_status = ComplianceStatus.OK
        risk_level = RiskLevel.LOW

        if drawdown > env.config.max_drawdown:
            compliance_status = ComplianceStatus.VIOLATION
            risk_level = RiskLevel.HIGH
            violations.append("drawdown_limit_breached")
        elif gross_exposure / total_value > 0.60:
            compliance_status = ComplianceStatus.WARNING
            risk_level = RiskLevel.MEDIUM

        if float(portfolio["cash"]) < state["treasury"].get("minimum_cash_required", 0.0):
            compliance_status = ComplianceStatus.VIOLATION
            violations.append("liquidity_shortfall")

        state["risk"]["compliance_status"] = compliance_status
        state["risk"]["risk_level"] = risk_level

        return {
            "drawdown": drawdown,
            "exposure": gross_exposure / total_value,
            "compliance_status": compliance_status,
            "risk_level": risk_level,
            "violations": violations,
        }

    def _apply_news_update(self, env: Any) -> None:
        engine = getattr(env, "news_engine", None)
        if engine is not None:
            for method_name in ("update", "step", "advance"):
                method = getattr(engine, method_name, None)
                if callable(method):
                    result = self._call_engine_method(method, env)
                    if isinstance(result, dict):
                        self._merge_state_from_dict(env, result)
                    return

        # Lightweight fallback news generation
        if env.config.enable_news and env.step_count % 7 == 0:
            regime = env.state_data.get("hidden_regime", "bull")
            headline = {
                "bull": "Positive macro update improves risk sentiment",
                "bear": "Risk-off headlines pressure markets",
                "chaotic": "Uncertain market conditions persist",
            }.get(regime, "Neutral market conditions")
            env.state_data.setdefault("news", []).append(
                {
                    "headline": headline,
                    "sentiment": 0.35 if regime == "bull" else (-0.35 if regime == "bear" else 0.0),
                    "severity": 0.4,
                    "affected_assets": list(env.config.assets[:2]),
                }
            )
            env.state_data["news"] = env.state_data["news"][-5:]

    def _apply_regime_update(self, env: Any) -> None:
        engine = getattr(env, "regime_engine", None)
        if engine is not None:
            for method_name in ("update", "step", "advance"):
                method = getattr(engine, method_name, None)
                if callable(method):
                    result = self._call_engine_method(method, env)
                    if isinstance(result, dict):
                        self._merge_state_from_dict(env, result)
                    return

        if env.config.enable_regimes:
            if env.step_count % 11 == 0 and env.step_count > 0:
                current = env.state_data.get("hidden_regime", "bull")
                next_regime = {
                    "bull": "bear",
                    "bear": "chaotic",
                    "chaotic": "bull",
                }.get(current, "bull")
                env.state_data["hidden_regime"] = next_regime
                env.state_data["market"]["regime_hint"] = next_regime

    def _apply_multi_agent_update(self, env: Any) -> None:
        engine = getattr(env, "multi_agent_engine", None)
        if engine is not None:
            for method_name in ("update", "step", "advance"):
                method = getattr(engine, method_name, None)
                if callable(method):
                    result = self._call_engine_method(method, env)
                    if isinstance(result, dict):
                        self._merge_state_from_dict(env, result)
                    return

        if env.config.enable_multi_agent:
            noise = env.rng.gauss(0.0, env.config.noise_trader_strength)
            for asset, price in list(env.state_data["market"]["prices"].items()):
                env.state_data["market"]["prices"][asset] = max(0.01, float(price) * (1.0 + noise * 0.001))

    def _compute_reward(self, env: Any, execution_metrics: Dict[str, float], treasury_metrics: Dict[str, float], risk_metrics: Dict[str, Any]) -> Reward:
        reward_model = getattr(env, "reward_model", None)
        if reward_model is not None:
            for method_name in ("compute", "step_reward", "reward", "calculate"):
                method = getattr(reward_model, method_name, None)
                if callable(method):
                    result = self._call_engine_method(method, env, execution_metrics, treasury_metrics, risk_metrics)
                    if isinstance(result, Reward):
                        return result
                    if isinstance(result, dict):
                        if hasattr(Reward, "model_validate"):
                            return Reward.model_validate(result)
                        return Reward.parse_obj(result)
                    if isinstance(result, (int, float)):
                        return Reward(
                            value=float(result),
                            components={"raw": float(result)},
                            penalties={},
                            normalized_score=self._normalize_reward(float(result)),
                        )

        value = (
            float(execution_metrics.get("pnl_delta", 0.0))
            - float(execution_metrics.get("execution_cost", 0.0))
            - float(execution_metrics.get("slippage", 0.0))
            - float(treasury_metrics.get("treasury_penalty", 0.0))
            - self._risk_penalty(risk_metrics)
        )

        components = {
            "pnl_delta": float(execution_metrics.get("pnl_delta", 0.0)),
            "execution_cost": -float(execution_metrics.get("execution_cost", 0.0)),
            "slippage": -float(execution_metrics.get("slippage", 0.0)),
            "treasury_penalty": -float(treasury_metrics.get("treasury_penalty", 0.0)),
            "risk_penalty": -self._risk_penalty(risk_metrics),
        }

        penalties = {
            "compliance": float(1.0 if risk_metrics.get("compliance_status") == ComplianceStatus.VIOLATION else 0.0),
            "drawdown": float(risk_metrics.get("drawdown", 0.0)),
        }

        return Reward(
            value=value,
            components=components,
            penalties=penalties,
            normalized_score=self._normalize_reward(value),
        )

    def _check_done(self, env: Any, action: Action, risk_metrics: Dict[str, Any]) -> bool:
        if env.step_count >= env.config.episode_horizon:
            return True

        if risk_metrics.get("compliance_status") == ComplianceStatus.VIOLATION:
            return True

        task = env.task
        if task is not None:
            for hook_name in ("should_terminate", "is_done", "terminal_condition", "done"):
                hook = getattr(task, hook_name, None)
                if callable(hook):
                    try:
                        result = hook(env.state_data, action, risk_metrics)
                    except TypeError:
                        try:
                            result = hook(env.state_data)
                        except TypeError:
                            result = hook()
                    if isinstance(result, bool):
                        return result
                    break

        return False

    def _build_task_metrics(self, env: Any, execution_metrics: Dict[str, float], treasury_metrics: Dict[str, float], risk_metrics: Dict[str, Any]) -> Dict[str, float]:
        return {
            "pnl_delta": float(execution_metrics.get("pnl_delta", 0.0)),
            "execution_cost": float(execution_metrics.get("execution_cost", 0.0)),
            "slippage": float(execution_metrics.get("slippage", 0.0)),
            "drawdown": float(risk_metrics.get("drawdown", 0.0)),
            "exposure": float(risk_metrics.get("exposure", 0.0)),
            "treasury_penalty": float(treasury_metrics.get("treasury_penalty", 0.0)),
        }

    def _revalue_portfolio(self, env: Any) -> None:
        state = env.state_data
        portfolio = state["portfolio"]
        market_prices = state["market"]["prices"]

        holdings_value = 0.0
        for instrument, qty in portfolio["positions"].items():
            holdings_value += float(qty) * float(market_prices.get(instrument, 0.0))

        portfolio["unrealized_pnl"] = holdings_value
        portfolio["total_value"] = float(portfolio["cash"]) + holdings_value

    def _estimate_fill_quantity(self, env: Any, order: Order, mid_price: float) -> float:
        liquidity = float(env.state_data["market"].get("liquidity", {}).get(order.instrument, 1.0))
        base_fill = max(0.0, float(order.quantity) * min(1.0, liquidity))

        if order.order_type == OrderType.LIMIT and order.price is not None:
            if order.side == OrderSide.BUY and order.price >= mid_price:
                return base_fill
            if order.side == OrderSide.SELL and order.price <= mid_price:
                return base_fill
            return 0.0

        return base_fill

    def _compute_fill_price(self, mid_price: float, spread: float, side: OrderSide, order_type: OrderType, limit_price: float | None) -> float:
        if order_type == OrderType.LIMIT and limit_price is not None:
            return float(limit_price)

        half_spread = mid_price * spread * 0.5
        if side == OrderSide.BUY:
            return mid_price + half_spread
        return max(0.01, mid_price - half_spread)

    def _normalize_risk_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "drawdown": float(result.get("drawdown", 0.0)),
            "exposure": float(result.get("exposure", 0.0)),
            "compliance_status": result.get("compliance_status", ComplianceStatus.OK),
            "risk_level": result.get("risk_level", RiskLevel.LOW),
            "violations": list(result.get("violations", [])),
        }

    def _risk_penalty(self, risk_metrics: Dict[str, Any]) -> float:
        penalty = 0.0
        penalty += 2.0 if risk_metrics.get("compliance_status") == ComplianceStatus.VIOLATION else 0.0
        penalty += float(risk_metrics.get("drawdown", 0.0)) * 2.0
        penalty += float(risk_metrics.get("exposure", 0.0)) * 0.5
        return penalty

    def _normalize_reward(self, value: float) -> float:
        # Soft normalization into [0,1] for diagnostics only
        return max(0.0, min(1.0, 0.5 + value / 100000.0))

    def _aggregate_news_signal(self, news_items: Iterable[Any]) -> float:
        total = 0.0
        count = 0
        for item in news_items:
            if isinstance(item, dict):
                total += float(item.get("sentiment", 0.0)) * float(item.get("severity", 0.0))
                count += 1
        return total / count if count else 0.0

    def _call_engine_method(self, method: Any, env: Any, *args: Any) -> Any:
        try:
            return method(env.state_data, *args)
        except TypeError:
            try:
                return method(env, *args)
            except TypeError:
                return method(*args)

    def _merge_state_from_dict(self, env: Any, data: Dict[str, Any]) -> None:
        if not isinstance(data, dict):
            return

        for key in ("market", "portfolio", "treasury", "risk", "task_context", "metrics"):
            if key in data and isinstance(data[key], dict):
                env.state_data.setdefault(key, {}).update(deepcopy(data[key]))

        for key in ("news", "pending_orders", "violations"):
            if key in data and isinstance(data[key], list):
                env.state_data[key] = deepcopy(data[key])

        for key in ("hidden_regime", "last_action_error", "done"):
            if key in data:
                env.state_data[key] = deepcopy(data[key])
