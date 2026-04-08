"""Liquidity-constrained behavior grader."""

from __future__ import annotations

from typing import Any

from .grader_base import GraderBase, _clip01


class LiquidityGrader(GraderBase):
    name = "liquidity"

    def score(self, source: Any = None) -> float:
        m = self.metrics(source)
        fill_ratio = _clip01(m.get("fill_ratio", 0.0))
        turnover = abs(m.get("turnover", 0.0))
        slippage_cost = max(0.0, m.get("slippage_cost", m.get("slippage", 0.0)))
        pending_order_count = max(0.0, m.get("pending_order_count", 0.0))
        target_remaining = max(0.0, m.get("target_remaining", 0.0))
        drawdown = max(0.0, m.get("drawdown_pct", m.get("drawdown", 0.0)))

        liquidity_efficiency = _clip01(1.0 - min(1.0, slippage_cost / max(1.0, 5000.0 + turnover * 1e6)))
        pending_penalty = _clip01(1.0 - min(1.0, pending_order_count / 12.0))
        progress = _clip01(1.0 - min(1.0, target_remaining / max(1.0, m.get("target_quantity", 1.0))))
        risk_term = _clip01(1.0 - min(1.0, drawdown * 5.0))

        base = (
            0.30 * fill_ratio
            + 0.20 * liquidity_efficiency
            + 0.20 * pending_penalty
            + 0.20 * progress
            + 0.10 * risk_term
        )
        return _clip01(base)
