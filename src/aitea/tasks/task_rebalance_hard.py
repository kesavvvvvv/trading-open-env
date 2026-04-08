"""Hard portfolio rebalance task."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

from .task_base import TaskBase


@dataclass(frozen=True)
class TaskRebalanceHard(TaskBase):
    name: str = "rebalance_hard"
    kind: str = "rebalance"
    difficulty: str = "hard"
    horizon: int = 50
    description: str = "Rebalance a multi-asset portfolio under strict constraints."
    target_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "AAPL": 0.22,
            "MSFT": 0.24,
            "GOOG": 0.20,
            "TSLA": 0.14,
            "AMZN": 0.20,
        }
    )

    def task_profile(self):
        profile = super().task_profile()
        profile.update(
            {
                "target_weights": dict(self.target_weights),
                "liquidity_scale": 0.35,
                "volatility": 0.012,
                "news_probability": 0.08,
                "regime_flip_probability": 0.05,
            }
        )
        return profile

    def initial_metrics(self):
        metrics = super().initial_metrics()
        metrics.update(
            {
                "tracking_error": 1.0,
                "turnover": 0.0,
            }
        )
        return metrics


def create_task() -> TaskRebalanceHard:
    return TaskRebalanceHard()
