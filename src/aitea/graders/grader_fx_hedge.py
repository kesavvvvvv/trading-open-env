"""FX hedge quality grader."""

from __future__ import annotations

from typing import Any

from .grader_base import GraderBase, _clip01


class FXHedgeGrader(GraderBase):
    name = "fx_hedge"

    def score(self, source: Any = None) -> float:
        m = self.metrics(source)
        fx_exposure = abs(m.get("fx_exposure", m.get("hedge_error", 0.0)))
        hedge_error = abs(m.get("hedge_error", fx_exposure))
        execution_cost = max(0.0, m.get("execution_cost", 0.0))
        fill_ratio = _clip01(m.get("fill_ratio", 1.0))
        drawdown = max(0.0, m.get("drawdown_pct", m.get("drawdown", 0.0)))
        target = max(1.0, m.get("target_fx_exposure", m.get("initial_fx_exposure", 500000.0)))

        exposure_term = _clip01(1.0 - min(1.0, hedge_error / target))
        cost_term = _clip01(1.0 - min(1.0, execution_cost / max(1.0, target * 0.01)))
        risk_term = _clip01(1.0 - min(1.0, drawdown * 5.0))

        base = 0.45 * exposure_term + 0.20 * cost_term + 0.20 * fill_ratio + 0.15 * risk_term
        if fx_exposure <= 1000.0:
            base += 0.05
        return _clip01(base)
