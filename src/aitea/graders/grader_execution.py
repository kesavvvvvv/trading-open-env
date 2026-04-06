"""Execution quality grader."""

from __future__ import annotations

from .grader_base import GraderBase
from ..env.state_manager import AITEAState


class ExecutionGrader(GraderBase):
    """Score implementation shortfall, completion, and execution smoothness."""

    def score(self, state: AITEAState) -> float:
        kind = str(state.task_metrics.get("kind", "execution"))
        if kind not in {"execution", "liquidity"}:
            return 0.0

        target_qty = self._safe_float(state.task_metrics.get("target_quantity", 0.0), 0.0)
        remaining = self._safe_float(state.task_metrics.get("target_remaining", target_qty), target_qty)
        progress = 1.0
        if target_qty > 0:
            progress = self._clip01(1.0 - remaining / target_qty)

        # Smoothness/reliability from recent rewards.
        stability = self._reward_stability(state)

        # Drawdown penalty.
        drawdown_score = self._drawdown_score(state.drawdown_pct, soft_limit=0.12)

        # Prefer completion strongly, then stability and risk control.
        score = 0.60 * progress + 0.20 * stability + 0.20 * drawdown_score
        return self._clip01(score)

    def detail(self, state: AITEAState):
        target_qty = self._safe_float(state.task_metrics.get("target_quantity", 0.0), 0.0)
        remaining = self._safe_float(state.task_metrics.get("target_remaining", target_qty), target_qty)
        progress = 0.0 if target_qty <= 0 else self._clip01(1.0 - remaining / target_qty)
        return {
            "progress": progress,
            "remaining_ratio": 0.0 if target_qty <= 0 else self._clip01(remaining / target_qty),
            "stability": self._reward_stability(state),
            "drawdown_score": self._drawdown_score(state.drawdown_pct, soft_limit=0.12),
        }
