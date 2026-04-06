"""Hidden regime adaptation grader."""

from __future__ import annotations

from .grader_base import GraderBase
from ..env.state_manager import AITEAState


class RegimeAdaptationGrader(GraderBase):
    """Score consistency, return quality, robustness, and drawdown control."""

    def score(self, state: AITEAState) -> float:
        kind = str(state.task_metrics.get("kind", "regime"))
        if kind != "regime":
            return 0.0

        stability = self._reward_stability(state)
        drawdown_score = self._drawdown_score(state.drawdown_pct, soft_limit=0.10)

        realized = self._safe_float(state.realized_pnl, 0.0)
        unrealized = self._safe_float(state.unrealized_pnl, 0.0)
        total_return = realized + unrealized
        return_score = self._clip01((total_return / max(1.0, self.config.starting_cash * 0.03) + 1.0) / 2.0)

        volatility_penalty = self._clip01(self._safe_float(state.hidden_volatility, 0.0) / 0.05)
        consistency = self._clip01(1.0 - volatility_penalty)

        score = (
            0.30 * stability
            + 0.25 * drawdown_score
            + 0.25 * return_score
            + 0.20 * consistency
        )
        return self._clip01(score)

    def detail(self, state: AITEAState):
        total_return = self._safe_float(state.realized_pnl + state.unrealized_pnl, 0.0)
        return {
            "stability": self._reward_stability(state),
            "drawdown_score": self._drawdown_score(state.drawdown_pct, soft_limit=0.10),
            "return_score": self._clip01((total_return / max(1.0, self.config.starting_cash * 0.03) + 1.0) / 2.0),
            "consistency": self._clip01(1.0 - self._safe_float(state.hidden_volatility, 0.0) / 0.05),
        }
