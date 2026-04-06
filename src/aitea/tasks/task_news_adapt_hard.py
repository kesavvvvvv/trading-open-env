"""Hard news-adaptation task for AITEA."""

from __future__ import annotations

from ..env.state_manager import AITEAState
from ..registry import register_task
from .task_base import TaskBase


@register_task("news_adapt_hard")
class NewsAdaptHardTask(TaskBase):
    def __init__(self) -> None:
        super().__init__(
            task_name="news_adapt_hard",
            difficulty="hard",
            kind="news",
            description="Adapt strategy during structured news shocks with drawdown control.",
            grader_name="grader_news_response",
            horizon=45,
        )

    def profile(self):
        profile = super().profile()
        profile.update(
            {
                "liquidity_scale": 0.55,
                "volatility": 0.014,
                "news_probability": 0.25,
                "regime_flip_probability": 0.08,
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
