"""
Reset manager for AITEA.

Creates a fresh episode state and applies task-specific initialization hooks.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict
import random

from aitea.config import AITEAConfig, set_global_seed
from aitea.schemas.common import ComplianceStatus, RiskLevel


class ResetManager:
    def reset_episode(self, env: Any, seed: int | None = None) -> Dict[str, Any]:
        config: AITEAConfig = env.config
        active_seed = config.seed if seed is None else seed

        set_global_seed(active_seed)
        env.rng = random.Random(active_seed)
        env.seed = active_seed

        asset_list = list(config.assets)
        initial_prices = deepcopy(config.initial_prices)

        market_state = {
            "prices": deepcopy(initial_prices),
            "volatility": config.volatility,
            "spread": config.spread,
            "liquidity": {asset: 1.0 for asset in asset_list},
            "regime_hint": "bull",
        }

        portfolio_state = {
            "cash": config.initial_cash,
            "positions": {asset: 0.0 for asset in asset_list},
            "unrealized_pnl": 0.0,
            "realized_pnl": 0.0,
            "total_value": config.initial_cash,
        }

        treasury_state = {
            "upcoming_outflows": {},
            "upcoming_inflows": {},
            "minimum_cash_required": config.initial_cash * 0.10,
            "current_liquidity_buffer": config.initial_cash * 0.15,
        }

        risk_state = {
            "drawdown": 0.0,
            "exposure": 0.0,
            "risk_level": RiskLevel.LOW,
            "compliance_status": ComplianceStatus.OK,
        }

        state: Dict[str, Any] = {
            "timestamp": 0,
            "step_index": 0,
            "episode_reward": 0.0,
            "done": False,
            "task_name": getattr(env.task, "name", "default"),
            "market": market_state,
            "portfolio": portfolio_state,
            "treasury": treasury_state,
            "risk": risk_state,
            "news": [],
            "pending_orders": [],
            "recent_history": [],
            "recent_history_summary": {},
            "task_context": {},
            "hidden_regime": "bull",
            "metrics": {},
            "last_action_error": None,
            "violations": [],
            "peak_portfolio_value": config.initial_cash,
        }

        task = env.task
        if task is not None:
            # Task-specific initialization hooks (duck-typed and tolerant)
            for hook_name in (
                "initialize_state",
                "reset_hook",
                "on_reset",
                "build_initial_state",
            ):
                hook = getattr(task, hook_name, None)
                if callable(hook):
                    try:
                        returned = hook(state, config)
                    except TypeError:
                        try:
                            returned = hook(state)
                        except TypeError:
                            returned = hook()
                    if isinstance(returned, dict):
                        state.update(returned)
                    break

            for context_name in (
                "get_task_context",
                "build_context",
                "task_context",
            ):
                ctx = getattr(task, context_name, None)
                if callable(ctx):
                    try:
                        task_ctx = ctx(state, config)
                    except TypeError:
                        try:
                            task_ctx = ctx(state)
                        except TypeError:
                            task_ctx = ctx()
                    if isinstance(task_ctx, dict):
                        state["task_context"] = task_ctx
                    break
                if isinstance(ctx, dict):
                    state["task_context"] = ctx
                    break

        env.state_data = state
        env.step_count = 0
        env.episode_reward = 0.0
        env.done = False
        env.last_info = None
        env.current_observation = None

        return state
