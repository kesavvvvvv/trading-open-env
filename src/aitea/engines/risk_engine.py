"""Risk control engine for AITEA."""

from __future__ import annotations

from typing import Dict, List

from ..config import AITEAConfig, get_config
from ..env.state_manager import AITEAState


class RiskEngine:
    """Check exposure, drawdown, leverage-style constraints, and termination risk."""

    def __init__(self, config: AITEAConfig | None = None) -> None:
        self.config = config or get_config()

    def evaluate(self, state: AITEAState) -> Dict[str, object]:
        violations: List[str] = []

        equity = max(1.0, float(state.equity))
        gross_pct = float(state.gross_exposure) / equity
        drawdown = float(state.drawdown_pct)

        if gross_pct > self.config.max_gross_exposure_pct:
            violations.append("gross_exposure_breach")
        if drawdown > self.config.max_drawdown_pct:
            violations.append("drawdown_breach")

        for symbol, qty in state.positions.items():
            price = float(state.prices.get(symbol, 0.0))
            pos_pct = abs(qty * price) / equity if equity > 0 else 0.0
            if pos_pct > self.config.max_position_pct:
                violations.append(f"position_limit_breach:{symbol}")

        severe = drawdown > (self.config.max_drawdown_pct * 1.25) or gross_pct > (self.config.max_gross_exposure_pct * 1.5)
        return {
            "violations": violations,
            "severe_breach": severe,
            "risk_score": max(0.0, 1.0 - min(1.0, drawdown + max(0.0, gross_pct - 1.0))),
            "gross_exposure_pct": gross_pct,
            "drawdown_pct": drawdown,
        }
