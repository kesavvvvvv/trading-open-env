"""Hard hidden-regime challenge task."""

from __future__ import annotations

from dataclasses import dataclass

from .task_base import TaskBase


@dataclass(frozen=True)
class TaskRegimeChallengeHard(TaskBase):
    name: str = "regime_challenge_hard"
    kind: str = "regime"
    difficulty: str = "hard"
    horizon: int = 60
    description: str = "Operate under hidden market regime changes."

    def task_profile(self):
        profile = super().task_profile()
        profile.update(
            {
                "liquidity_scale": 0.50,
                "volatility": 0.018,
                "news_probability": 0.12,
                "regime_flip_probability": 0.18,
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


def create_task() -> TaskRegimeChallengeHard:
    return TaskRegimeChallengeHard()
