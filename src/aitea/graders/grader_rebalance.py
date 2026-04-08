"""Portfolio rebalance grader."""

from __future__ import annotations

from typing import Any

from .grader_base import GraderBase, _clip01


class RebalanceGrader(GraderBase):
    name = "rebalance"

    def score(self, source: Any = None) -> float:
        m = self.metrics(source)
        tracking_error = abs(m.get("tracking_error", m.get("target_error", 1.0)))
        turnover = abs(m.get("turnover", 0.0))
        drawdown = max(0.0, m.get("drawdown_pct", m.get("drawdown", 0.0)))
        fill_ratio = _clip01(m.get("fill_ratio", 1.0))
        violation_count = max(0.0, m.get("violation_count", 0.0))

        tracking_term = _clip01(1.0 - min(1.0, tracking_error / 0.2))
        turnover_term = _clip01(1.0 - min(1.0, turnover * 6.0))
        risk_term = _clip01(1.0 - min(1.0, drawdown * 5.0))
        compliance_term = _clip01(1.0 - min(1.0, violation_count / 8.0))

        base = (
            0.36 * tracking_term
            + 0.18 * turnover_term
            + 0.18 * fill_ratio
            + 0.16 * risk_term
            + 0.12 * compliance_term
        )
        return _clip01(base)
