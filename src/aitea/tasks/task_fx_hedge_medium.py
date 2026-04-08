"""Medium FX hedge task."""

from __future__ import annotations

from dataclasses import dataclass

from .task_base import TaskBase


@dataclass(frozen=True)
class TaskFXHedgeMedium(TaskBase):
    name: str = "fx_hedge_medium"
    kind: str = "hedge"
    difficulty: str = "medium"
    horizon: int = 35
    description: str = "Reduce FX exposure efficiently while controlling cost."
    fx_exposure: float = 500_000.0
    hedge_symbol: str = "MSFT"

    def task_profile(self):
        profile = super().task_profile()
        profile.update(
            {
                "fx_exposure": float(self.fx_exposure),
                "hedge_symbol": self.hedge_symbol,
                "liquidity_scale": 0.70,
                "volatility": 0.009,
                "news_probability": 0.03,
                "regime_flip_probability": 0.02,
            }
        )
        return profile

    def initial_metrics(self):
        metrics = super().initial_metrics()
        metrics.update(
            {
                "fx_exposure": float(self.fx_exposure),
                "hedge_error": float(self.fx_exposure),
                "hedge_symbol": self.hedge_symbol,
            }
        )
        return metrics


def create_task() -> TaskFXHedgeMedium:
    return TaskFXHedgeMedium()
