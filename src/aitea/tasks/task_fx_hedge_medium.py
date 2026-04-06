"""Medium FX hedge task for AITEA."""

from __future__ import annotations

from ..env.state_manager import AITEAState
from ..registry import register_task
from .task_base import TaskBase


@register_task("fx_hedge_medium")
class FXHedgeMediumTask(TaskBase):
    def __init__(self) -> None:
        super().__init__(
            task_name="fx_hedge_medium",
            difficulty="medium",
            kind="hedge",
            description="Reduce FX exposure efficiently with limited transaction cost.",
            grader_name="grader_fx_hedge",
            horizon=35,
        )

    def profile(self):
        profile = super().profile()
        profile.update(
            {
                "liquidity_scale": 0.70,
                "volatility": 0.009,
                "news_probability": 0.03,
                "regime_flip_probability": 0.02,
                "fx_exposure": 500_000.0,
                "hedge_symbol": "MSFT",
            }
        )
        return profile

    def initialize_metrics(self):
        return {
            "kind": self.kind,
            "fx_exposure": 500_000.0,
            "hedge_error": 500_000.0,
            "hedge_symbol": "MSFT",
        }

    def success(self, state: AITEAState) -> bool:
        fx_exposure = float(state.task_metrics.get("fx_exposure", 10**9))
        return fx_exposure <= 1_000.0 or state.step >= self.horizon
