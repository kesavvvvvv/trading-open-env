"""Portfolio rebalance grader."""

from __future__ import annotations

from .grader_base import GraderBase
from ..env.state_manager import AITEAState


class RebalanceGrader(GraderBase):
    """Score tracking error, turnover, risk violations, and final allocation quality."""

    def score(self, state: AITEAState) -> float:
        kind = str(state.task_metrics.get("kind", "rebalance"))
        if kind != "rebalance":
            return 0.0

        tracking_error = self._safe_float(state.task_metrics.get("tracking_error", 1.0), 1.0)
        tracking_score = self._clip01(1.0 - min(1.0, tracking_error / 0.25))

        turnover = self._safe_float(state.task_metrics.get("turnover", 0.0), 0.0)
        turnover_score = self._clip01(1.0 - min(1.0, turnover))

        violations = self._safe_float(state.task_metrics.get("violation_count", 0.0), 0.0)
        violation_score = self._clip01(1.0 - min(1.0, violations / 10.0))

        allocation_quality = 1.0
        target_weights = state.task_profile.get("target_weights", {})
        if target_weights:
            equity = max(1.0, state.equity)
            error = 0.0
            for symbol, target_w in target_weights.items():
                current_w = (state.positions.get(symbol, 0.0) * state.prices.get(symbol, 0.0)) / equity
                error += abs(float(current_w) - float(target_w))
            allocation_quality = self._clip01(1.0 - error / max(1, len(target_weights)))

        drawdown_score = self._drawdown_score(state.drawdown_pct, soft_limit=0.12)
        stability = self._reward_stability(state)

        score = (
            0.35 * tracking_score
            + 0.20 * turnover_score
            + 0.20 * violation_score
            + 0.15 * allocation_quality
            + 0.10 * drawdown_score
        )
        score = 0.85 * score + 0.15 * stability
        return self._clip01(score)

    def detail(self, state: AITEAState):
        tracking_error = self._safe_float(state.task_metrics.get("tracking_error", 1.0), 1.0)
        return {
            "tracking_score": self._clip01(1.0 - min(1.0, tracking_error / 0.25)),
            "turnover_score": self._clip01(1.0 - min(1.0, self._safe_float(state.task_metrics.get("turnover", 0.0), 0.0))),
            "violation_score": self._clip01(1.0 - min(1.0, self._safe_float(state.task_metrics.get("violation_count", 0.0), 0.0) / 10.0)),
            "drawdown_score": self._drawdown_score(state.drawdown_pct, soft_limit=0.12),
            "stability": self._reward_stability(state),
        }
