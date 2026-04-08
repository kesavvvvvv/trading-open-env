"""News shock response grader."""

from __future__ import annotations

from typing import Any

from .grader_base import GraderBase, _clip01


class NewsResponseGrader(GraderBase):
    name = "news_response"

    def score(self, source: Any = None) -> float:
        m = self.metrics(source)
        drawdown = max(0.0, m.get("drawdown_pct", m.get("drawdown", 0.0)))
        pnl_delta = m.get("pnl_delta", 0.0)
        recovery = _clip01(m.get("progress", 0.0))
        reward_mean = m.get("reward_mean_recent", 0.0)
        reward_std = abs(m.get("reward_std_recent", 0.0))
        violation_count = max(0.0, m.get("violation_count", 0.0))

        drawdown_term = _clip01(1.0 - min(1.0, drawdown * 7.0))
        pnl_term = _clip01(0.5 + 0.5 * (pnl_delta / max(1.0, abs(pnl_delta) + 1000.0)))
        stability_term = _clip01(1.0 - min(1.0, reward_std * 1.5))
        compliance_term = _clip01(1.0 - min(1.0, violation_count / 6.0))
        recovery_term = _clip01(0.4 * recovery + 0.6 * drawdown_term)

        base = (
            0.30 * drawdown_term
            + 0.20 * pnl_term
            + 0.20 * recovery_term
            + 0.15 * stability_term
            + 0.15 * compliance_term
        )
        if reward_mean > 0.0:
            base += 0.02
        return _clip01(base)
