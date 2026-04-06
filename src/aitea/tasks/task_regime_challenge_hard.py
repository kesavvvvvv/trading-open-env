"""Hard hidden-regime challenge task for AITEA."""

from __future__ import annotations

from ..env.state_manager import AITEAState
from ..registry import register_task
from .task_base import TaskBase


@register_task("regime_challenge_hard")
class RegimeChallengeHardTask(TaskBase):
    def __init__(self) -> None:
        super().__init__(
            task_name="regime_challenge_hard",
            difficulty="hard",
            kind="regime",
            description="Operate under hidden market regime shifts and adversarial volatility.",
            grader_name="grader_regime_adaptation",
            horizon=60,
        )

    def profile(self):
        profile = super().profile()
        profile.update(
            {
                "liquidity_scale": 0.50,
                "volatility": 0.018,
                "news_probability": 0.12,
                "regime_flip_probability": 0.18,
            }
        )
        return profile

    def initialize_metrics(self):
        return {
            "kind": self.kind,
            "stress_score": 0.0,
            "equity_peak": self.config.starting_cash,
        }

    def success(self, state: AITEAState) -> bool:
        return state.drawdown_pct <= (self.config.max_drawdown_pct * 0.75) or state.step >= self.horizon
