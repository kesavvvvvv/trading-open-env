"""Treasury and funding engine for AITEA."""

from __future__ import annotations

from typing import Dict

from ..config import AITEAConfig, get_config
from ..env.state_manager import AITEAState


class TreasuryEngine:
    """Manage cash balances, funding pressure, and capital sufficiency."""

    def __init__(self, config: AITEAConfig | None = None) -> None:
        self.config = config or get_config()

    def update(self, state: AITEAState) -> Dict[str, float]:
        """
        Recompute treasury-related diagnostics.
        This is intentionally lightweight but realistic enough for the environment.
        """
        equity_before = float(state.equity)
        state.cash = float(state.cash)

        funding_pressure = 0.0
        if state.cash < 0:
            funding_pressure = min(1.0, abs(state.cash) / max(1.0, self.config.starting_cash * 0.1))
        elif state.cash < self.config.starting_cash * 0.05:
            funding_pressure = 0.25

        capital_sufficiency = 1.0 if state.cash > 0 else max(0.0, 1.0 - abs(state.cash) / max(1.0, self.config.starting_cash))
        liquidity_buffer = max(0.0, state.cash / max(1.0, self.config.starting_cash))

        return {
            "equity_before": equity_before,
            "cash": state.cash,
            "funding_pressure": funding_pressure,
            "capital_sufficiency": capital_sufficiency,
            "liquidity_buffer": liquidity_buffer,
        }
