"""Hidden regime adaptation grader."""

from __future__ import annotations

from typing import Any

from .grader_base import GraderBase, _clip01


class RegimeAdaptationGrader(GraderBase):
    name = "regime_adaptation"

    def score(self, source: Any = None) -> float:
        m = self.metrics(source)
        drawdown = max(0.0, m.get("drawdown_pct", m.get("drawdown", 0.0)))
        reward_mean = m.get("reward_mean_recent", 0.0)
        reward_std = abs(m.get("reward_std_recent", 0.0))
        gross_exposure = abs(m.get("gross_exposure", 0.0))
        net_exposure = abs(m.get("net_exposure", 0.0))
        equity = max(1.0, m.get("equity", 1.0))
        violation_count = max(0.0, m.get("violation_count", 0.0))
        progress = _clip01(m.get("progress", 0.0))

        drawdown_term = _clip01(1.0 - min(1.0, drawdown * 6.0))
        stability_term = _clip01(1.0 - min(1.0, reward_std * 1.4))
        exposure_term = _clip01(1.0 - min(1.0, (gross_exposure + net_exposure) / max(1.0, equity) * 0.35))
        compliance_term = _clip01(1.0 - min(1.0, violation_count / 8.0))
        return_term = _clip01(0.5 + 0.5 * (reward_mean / max(1.0, abs(reward_mean) + 1.0)))
        progress_term = _clip01(progress)

        base = (
            0.24 * drawdown_term
            + 0.18 * stability_term
            + 0.18 * exposure_term
            + 0.16 * compliance_term
            + 0.14 * return_term
            + 0.10 * progress_term
        )
        return _clip01(base)
