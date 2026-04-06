"""Hard portfolio rebalance task for AITEA."""

from __future__ import annotations

from ..env.state_manager import AITEAState
from ..registry import register_task
from .task_base import TaskBase


@register_task("rebalance_hard")
class RebalanceHardTask(TaskBase):
    def __init__(self) -> None:
        super().__init__(
            task_name="rebalance_hard",
            difficulty="hard",
            kind="rebalance",
            description="Rebalance a multi-asset portfolio under strict liquidity and risk constraints.",
            grader_name="grader_rebalance",
            horizon=50,
        )

    def profile(self):
        profile = super().profile()
        profile.update(
            {
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
            }
        )
        return profile

    def initialize_metrics(self):
        return {
            "kind": self.kind,
            "target_weights": {
                "AAPL": 0.22,
                "MSFT": 0.24,
                "GOOG": 0.20,
                "TSLA": 0.14,
                "AMZN": 0.20,
            },
            "tracking_error": 1.0,
        }

    def success(self, state: AITEAState) -> bool:
        tracking_error = float(state.task_metrics.get("tracking_error", 1.0))
        return tracking_error <= 0.08 or state.step >= self.horizon
