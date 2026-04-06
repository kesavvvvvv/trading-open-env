"""News response grader."""

from __future__ import annotations

from .grader_base import GraderBase
from ..env.state_manager import AITEAState


class NewsResponseGrader(GraderBase):
    """Score response speed, drawdown protection, and recovery quality."""

    def score(self, state: AITEAState) -> float:
        kind = str(state.task_metrics.get("kind", "news"))
        if kind != "news":
            return 0.0

        stress = self._safe_float(state.task_metrics.get("stress_score", 0.0), 0.0)
        stress_score = self._clip01(1.0 - min(1.0, stress))

        drawdown_score = self._drawdown_score(state.drawdown_pct, soft_limit=0.10)
        pnl = self._safe_float(state.realized_pnl + state.unrealized_pnl, 0.0)
        pnl_score = self._clip01((pnl / max(1.0, self.config.starting_cash * 0.02) + 1.0) / 2.0)

        recovery_score = self._reward_stability(state)
        consistency = self._clip01(1.0 - min(1.0, abs(self._safe_float(state.task_metrics.get("equity_peak", state.equity), state.equity) - state.equity) / max(1.0, self.config.starting_cash * 0.05)))

        score = (
            0.30 * stress_score
            + 0.25 * drawdown_score
            + 0.20 * pnl_score
            + 0.15 * recovery_score
            + 0.10 * consistency
        )
        return self._clip01(score)

    def detail(self, state: AITEAState):
        return {
            "stress_score": self._clip01(1.0 - min(1.0, self._safe_float(state.task_metrics.get("stress_score", 0.0), 0.0))),
            "drawdown_score": self._drawdown_score(state.drawdown_pct, soft_limit=0.10),
            "pnl_score": self._clip01((self._safe_float(state.realized_pnl + state.unrealized_pnl, 0.0) / max(1.0, self.config.starting_cash * 0.02) + 1.0) / 2.0),
            "recovery_score": self._reward_stability(state),
        }
