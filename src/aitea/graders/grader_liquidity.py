"""Liquidity-constrained execution grader."""

from __future__ import annotations

from .grader_base import GraderBase
from ..env.state_manager import AITEAState


class LiquidityGrader(GraderBase):
    """Score partial fill handling, overtrading control, and cost discipline."""

    def score(self, state: AITEAState) -> float:
        kind = str(state.task_metrics.get("kind", "liquidity"))
        if kind != "liquidity":
            return 0.0

        target_qty = self._safe_float(state.task_metrics.get("target_quantity", 0.0), 0.0)
        remaining = self._safe_float(state.task_metrics.get("target_remaining", target_qty), target_qty)
        progress = 1.0 if target_qty <= 0 else self._clip01(1.0 - remaining / target_qty)

        background_drag = self._safe_float(state.task_metrics.get("market_drag", 0.0), 0.0)
        drag_penalty = self._clip01(background_drag / max(1.0, self.config.starting_cash * 0.001))

        turnover = self._safe_float(state.task_metrics.get("turnover", 0.0), 0.0)
        turnover_penalty = self._clip01(turnover)

        drawdown_score = self._drawdown_score(state.drawdown_pct, soft_limit=0.10)
        stability = self._reward_stability(state)

        score = (
            0.42 * progress
            + 0.18 * drawdown_score
            + 0.15 * stability
            + 0.25 * (1.0 - 0.5 * drag_penalty - 0.5 * turnover_penalty)
        )
        return self._clip01(score)

    def detail(self, state: AITEAState):
        target_qty = self._safe_float(state.task_metrics.get("target_quantity", 0.0), 0.0)
        remaining = self._safe_float(state.task_metrics.get("target_remaining", target_qty), target_qty)
        return {
            "completion": 0.0 if target_qty <= 0 else self._clip01(1.0 - remaining / target_qty),
            "drawdown_score": self._drawdown_score(state.drawdown_pct, soft_limit=0.10),
            "stability": self._reward_stability(state),
            "turnover": self._clip01(self._safe_float(state.task_metrics.get("turnover", 0.0), 0.0)),
            "market_drag": self._safe_float(state.task_metrics.get("market_drag", 0.0), 0.0),
        }
