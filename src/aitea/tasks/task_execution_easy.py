"""Easy execution task."""

from __future__ import annotations

from dataclasses import dataclass

from .task_base import TaskBase


@dataclass(frozen=True)
class TaskExecutionEasy(TaskBase):
    name: str = "execution_easy"
    kind: str = "execution"
    difficulty: str = "easy"
    horizon: int = 25
    description: str = "Execute a target order efficiently in a relatively stable market."
    target_symbol: str = "AAPL"
    target_quantity: int = 2000
    target_side: str = "buy"

    def task_profile(self):
        profile = super().task_profile()
        profile.update(
            {
                "target_symbol": self.target_symbol,
                "target_quantity": float(self.target_quantity),
                "target_side": self.target_side,
                "liquidity_scale": 1.0,
                "volatility": 0.008,
                "news_probability": 0.02,
                "regime_flip_probability": 0.01,
            }
        )
        return profile

    def initial_metrics(self):
        metrics = super().initial_metrics()
        metrics.update(
            {
                "target_symbol": self.target_symbol,
                "target_quantity": float(self.target_quantity),
                "target_remaining": float(self.target_quantity),
                "progress": 0.0,
            }
        )
        return metrics


def create_task() -> TaskExecutionEasy:
    return TaskExecutionEasy()
