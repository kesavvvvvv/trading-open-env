"""Episode reset logic for AITEA."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from ..config import AITEAConfig, get_config
from ..schemas import MarketRegime, NewsSignal, OrderType, PendingOrder
from .state_manager import AITEAState, StateManager, Transition, build_observation, update_derived_state


TASK_PROFILES: Dict[str, Dict[str, Any]] = {
    "execution_easy": {
        "kind": "execution",
        "horizon": 25,
        "liquidity_scale": 1.0,
        "volatility": 0.008,
        "news_probability": 0.02,
        "regime_flip_probability": 0.01,
        "target_symbol": "AAPL",
        "target_quantity": 2000,
        "target_side": "buy",
    },
    "liquidity_medium": {
        "kind": "liquidity",
        "horizon": 35,
        "liquidity_scale": 0.45,
        "volatility": 0.010,
        "news_probability": 0.04,
        "regime_flip_probability": 0.02,
        "target_symbol": "TSLA",
        "target_quantity": 2500,
        "target_side": "buy",
    },
    "fx_hedge_medium": {
        "kind": "hedge",
        "horizon": 35,
        "liquidity_scale": 0.70,
        "volatility": 0.009,
        "news_probability": 0.03,
        "regime_flip_probability": 0.02,
        "fx_exposure": 500_000.0,
        "hedge_symbol": "MSFT",
    },
    "rebalance_hard": {
        "kind": "rebalance",
        "horizon": 50,
        "liquidity_scale": 0.35,
        "volatility": 0.012,
        "news_probability": 0.08,
        "regime_flip_probability": 0.05,
        "target_weights": {
            "AAPL": 0.22,
            "MSFT": 0.24,
            "GOOG": 0.20,
            "TSLA": 0.14,
            "AMZN": 0.20,
        },
    },
    "news_adapt_hard": {
        "kind": "news",
        "horizon": 45,
        "liquidity_scale": 0.55,
        "volatility": 0.014,
        "news_probability": 0.25,
        "regime_flip_probability": 0.08,
    },
    "regime_challenge_hard": {
        "kind": "regime",
        "horizon": 60,
        "liquidity_scale": 0.50,
        "volatility": 0.018,
        "news_probability": 0.12,
        "regime_flip_probability": 0.18,
    },
}


def _stable_task_offset(task_name: str) -> int:
    return sum((i + 1) * ord(ch) for i, ch in enumerate(task_name)) % 10_000


def _profile_for(task_name: str, config: AITEAConfig) -> Dict[str, Any]:
    profile = dict(TASK_PROFILES.get(task_name, {}))
    if not profile:
        profile = {
            "kind": "generic",
            "horizon": config.episode_length,
            "liquidity_scale": 0.75,
            "volatility": 0.01,
            "news_probability": 0.03,
            "regime_flip_probability": 0.02,
        }
    profile.setdefault("horizon", config.episode_length)
    profile.setdefault("liquidity_scale", 0.75)
    profile.setdefault("volatility", 0.01)
    profile.setdefault("news_probability", 0.03)
    profile.setdefault("regime_flip_probability", 0.02)
    profile.setdefault("kind", "generic")
    return profile


class ResetManager:
    """Create fresh state for a new episode."""

    def __init__(self, config: AITEAConfig | None = None) -> None:
        self.config = config or get_config()
        self.state_manager = StateManager(self.config)

    def reset(self, task_name: str, episode_id: str | None = None) -> Tuple[AITEAState, Transition]:
        profile = _profile_for(task_name, self.config)
        seed = self.config.seed + _stable_task_offset(task_name)
        rng = random.Random(seed)

        episode_id = episode_id or f"{task_name}-{self.config.seed}-{seed}"
        assets = list(self.config.assets)

        base_prices = list(self.config.initial_prices)
        if not base_prices:
            base_prices = [100.0 for _ in assets]

        prices: Dict[str, float] = {}
        previous_prices: Dict[str, float] = {}
        positions: Dict[str, float] = {symbol: 0.0 for symbol in assets}
        avg_cost: Dict[str, float] = {symbol: 0.0 for symbol in assets}
        realized_by_symbol: Dict[str, float] = {symbol: 0.0 for symbol in assets}

        for i, symbol in enumerate(assets):
            base_price = float(base_prices[i % len(base_prices)])
            jitter = 1.0 + rng.uniform(-0.015, 0.015)
            prices[symbol] = max(1.0, base_price * jitter)
            previous_prices[symbol] = prices[symbol]

        cash = float(self.config.starting_cash)
        starting_cash = cash

        state = AITEAState(
            episode_id=episode_id,
            task_name=task_name,
            task_profile=profile,
            step=0,
            cash=cash,
            starting_cash=starting_cash,
            equity=starting_cash,
            peak_equity=starting_cash,
            prices=prices,
            previous_prices=previous_prices,
            positions=positions,
            avg_cost=avg_cost,
            realized_pnl_by_symbol=realized_by_symbol,
            realized_pnl=0.0,
            unrealized_pnl=0.0,
            gross_exposure=0.0,
            net_exposure=0.0,
            drawdown_pct=0.0,
            regime=MarketRegime.NORMAL,
            regime_confidence=0.65,
            hidden_volatility=float(profile.get("volatility", 0.01)),
            news_queue=[],
            pending_orders=[],
            recent_actions=[],
            recent_rewards=[],
            task_metrics={},
            done=False,
            last_error=None,
            rng=rng,
        )

        # Task-specific initial metrics
        kind = profile.get("kind", "generic")
        if kind in {"execution", "liquidity"}:
            target_qty = float(profile.get("target_quantity", 0.0))
            state.task_metrics.update(
                {
                    "kind": kind,
                    "target_quantity": target_qty,
                    "target_remaining": target_qty,
                    "target_symbol": str(profile.get("target_symbol", assets[0])),
                    "progress": 0.0,
                }
            )
        elif kind == "hedge":
            fx_exposure = float(profile.get("fx_exposure", 0.0))
            state.task_metrics.update(
                {
                    "kind": kind,
                    "fx_exposure": fx_exposure,
                    "hedge_error": abs(fx_exposure),
                    "hedge_symbol": str(profile.get("hedge_symbol", assets[1 % len(assets)])),
                }
            )
        elif kind == "rebalance":
            state.task_metrics.update(
                {
                    "kind": kind,
                    "target_weights": dict(profile.get("target_weights", {})),
                    "tracking_error": 1.0,
                }
            )
        elif kind in {"news", "regime"}:
            state.task_metrics.update(
                {
                    "kind": kind,
                    "stress_score": 0.0,
                    "equity_peak": starting_cash,
                }
            )
        else:
            state.task_metrics["kind"] = kind

        update_derived_state(state)
        obs = build_observation(state, self.config)
        info = self.state_manager.info(state, fill_ratio=1.0, violations=[])
        transition = Transition(observation=obs, reward=0.0, done=False, info=info, error=None)
        return state, transition
