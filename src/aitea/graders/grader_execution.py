"""Execution quality grader."""

from __future__ import annotations

from typing import Any

from .grader_base import GraderBase, _clip01


class ExecutionGrader(GraderBase):
    name = "execution"

    def score(self, source: Any = None) -> float:
        m = self.metrics(source)
        target_remaining = max(0.0, m.get("target_remaining", 0.0))
        target_quantity = max(1.0, m.get("target_quantity", max(target_remaining, 1.0)))
        fill_ratio = _clip01(m.get("fill_ratio", 1.0))
        execution_cost = max(0.0, m.get("execution_cost", 0.0))
        slippage_cost = max(0.0, m.get("slippage_cost", m.get("slippage", 0.0)))
        pnl_delta = m.get("pnl_delta", 0.0)
        drawdown = max(0.0, m.get("drawdown_pct", m.get("drawdown", 0.0)))
        progress = _clip01(m.get("progress", 1.0 - target_remaining / target_quantity))

        cost_penalty = min(1.0, (execution_cost + slippage_cost) / max(1.0, target_quantity * 4.0))
        pnl_term = _clip01(0.5 + 0.5 * (pnl_delta / max(1.0, target_quantity * 10.0)))
        risk_term = _clip01(1.0 - min(1.0, drawdown * 6.0))

        base = (
            0.34 * progress
            + 0.22 * fill_ratio
            + 0.18 * pnl_term
            + 0.16 * risk_term
            + 0.10 * _clip01(1.0 - cost_penalty)
        )
        if target_remaining <= 0.0:
            base += 0.08
        return _clip01(base)
