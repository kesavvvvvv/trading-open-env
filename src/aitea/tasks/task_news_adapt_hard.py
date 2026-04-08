"""Hard news adaptation task."""

from __future__ import annotations

from dataclasses import dataclass

from .task_base import TaskBase


@dataclass(frozen=True)
class TaskNewsAdaptHard(TaskBase):
    name: str = "news_adapt_hard"
    kind: str = "news"
    difficulty: str = "hard"
    horizon: int = 45
    description: str = "Adapt strategy during structured news shocks."

    def task_profile(self):
        profile = super().task_profile()
        profile.update(
            {
                "liquidity_scale": 0.55,
                "volatility": 0.014,
                "news_probability": 0.25,
                "regime_flip_probability": 0.08,
            }
        )
        return profile

    def initial_metrics(self):
        metrics = super().initial_metrics()
        metrics.update(
            {
                "stress_score": 0.0,
                "equity_peak": 0.0,
            }
        )
        return metrics


def create_task() -> TaskNewsAdaptHard:
    return TaskNewsAdaptHard()
