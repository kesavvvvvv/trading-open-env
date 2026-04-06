"""FX hedge grader."""

from __future__ import annotations

from .grader_base import GraderBase
from ..env.state_manager import AITEAState


class FXHedgeGrader(GraderBase):
    """Score residual FX exposure reduction, cost control, and stability."""

    def score(self, state: AITEAState) -> float:
        kind = str(state.task_metrics.get("kind", "hedge"))
        if kind != "hedge":
            return 0.0

        initial_exposure = self._safe_float(state.task_profile.get("fx_exposure", 500_000.0), 500_000.0)
        current_exposure = self._safe_float(state.task_metrics.get("fx_exposure", initial_exposure), initial_exposure)
        exposure_score = 1.0 if initial_exposure <= 0 else self._clip01(1.0 - current_exposure / initial_exposure)

        hedge_error = self._safe_float(state.task_metrics.get("hedge_error", current_exposure), current_exposure)
        hedge_error_score = 1.0 if initial_exposure <= 0 else self._clip01(1.0 - hedge_error / initial_exposure)

        drawdown_score = self._drawdown_score(state.drawdown_pct, soft_limit=0.10)
        stability = self._reward_stability(state)

        score = (
            0.45 * exposure_score
            + 0.20 * hedge_error_score
            + 0.20 * drawdown_score
            + 0.15 * stability
        )
        return self._clip01(score)

    def detail(self, state: AITEAState):
        initial_exposure = self._safe_float(state.task_profile.get("fx_exposure", 500_000.0), 500_000.0)
        current_exposure = self._safe_float(state.task_metrics.get("fx_exposure", initial_exposure), initial_exposure)
        return {
            "exposure_score": 1.0 if initial_exposure <= 0 else self._clip01(1.0 - current_exposure / initial_exposure),
            "hedge_error_score": 1.0 if initial_exposure <= 0 else self._clip01(1.0 - self._safe_float(state.task_metrics.get("hedge_error", current_exposure), current_exposure) / initial_exposure),
            "drawdown_score": self._drawdown_score(state.drawdown_pct, soft_limit=0.10),
            "stability": self._reward_stability(state),
        }
