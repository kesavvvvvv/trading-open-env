"""Easy execution task for AITEA."""

from __future__ import annotations

from ..env.state_manager import AITEAState
from ..registry import register_task
from .task_base import TaskBase


@register_task("execution_easy")
class ExecutionEasyTask(TaskBase):
    def __init__(self) -> None:
        super().__init__(
            task_name="execution_easy",
            difficulty="easy",
            kind="execution",
            description="Execute a target order efficiently in a relatively stable market.",
            grader_name="grader_execution",
            horizon=25,
        )

    def profile(self):
        profile = super().profile()
        profile.update(
            {
                "liquidity_scale": 1.0,
                "volatility": 0.008,
                "news_probability": 0.02,
                "regime_flip_probability": 0.01,
                "target_symbol": "AAPL",
                "target_quantity": 2000,
                "target_side": "buy",
            }
        )
        return profile

    def initialize_metrics(self):
        return {
            "kind": self.kind,
            "target_symbol": "AAPL",
            "target_quantity": 2000.0,
            "target_remaining": 2000.0,
            "progress": 0.0,
        }

    def success(self, state: AITEAState) -> bool:
        target_remaining = float(state.task_metrics.get("target_remaining", 10**9))
        return target_remaining <= 1.0 or state.step >= self.horizon
