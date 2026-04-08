"""Medium liquidity-constrained task."""

from __future__ import annotations

from dataclasses import dataclass

from .task_base import TaskBase


@dataclass(frozen=True)
class TaskLiquidityMedium(TaskBase):
    name: str = "liquidity_medium"
    kind: str = "liquidity"
    difficulty: str = "medium"
    horizon: int = 35
    description: str = "Execute or rebalance under low-liquidity conditions."
    target_symbol: str = "TSLA"
    target_quantity: int = 2500
    target_side: str = "buy"

    def task_profile(self):
        profile = super().task_profile()
        profile.update(
            {
                "target_symbol": self.target_symbol,
                "target_quantity": float(self.target_quantity),
                "target_side": self.target_side,
                "liquidity_scale": 0.45,
                "volatility": 0.010,
                "news_probability": 0.04,
                "regime_flip_probability": 0.02,
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


def create_task() -> TaskLiquidityMedium:
    return TaskLiquidityMedium()
