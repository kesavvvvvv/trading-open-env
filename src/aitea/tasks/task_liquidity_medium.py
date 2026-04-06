"""Medium liquidity task for AITEA."""

from __future__ import annotations

from ..env.state_manager import AITEAState
from ..registry import register_task
from .task_base import TaskBase


@register_task("liquidity_medium")
class LiquidityMediumTask(TaskBase):
    def __init__(self) -> None:
        super().__init__(
            task_name="liquidity_medium",
            difficulty="medium",
            kind="liquidity",
            description="Execute or rebalance under low-liquidity conditions.",
            grader_name="grader_liquidity",
            horizon=35,
        )

    def profile(self):
        profile = super().profile()
        profile.update(
            {
                "liquidity_scale": 0.45,
                "volatility": 0.010,
                "news_probability": 0.04,
                "regime_flip_probability": 0.02,
                "target_symbol": "TSLA",
                "target_quantity": 2500,
                "target_side": "buy",
            }
        )
        return profile

    def initialize_metrics(self):
        return {
            "kind": self.kind,
            "target_symbol": "TSLA",
            "target_quantity": 2500.0,
            "target_remaining": 2500.0,
            "progress": 0.0,
        }

    def success(self, state: AITEAState) -> bool:
        remaining = float(state.task_metrics.get("target_remaining", 10**9))
        return remaining <= 1.0 or state.drawdown_pct <= 0.08 or state.step >= self.horizon
