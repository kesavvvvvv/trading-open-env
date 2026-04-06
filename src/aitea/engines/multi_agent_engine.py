"""Internal multi-agent market participant engine for AITEA."""

from __future__ import annotations

import math
from typing import Dict

from ..config import AITEAConfig, get_config
from ..env.state_manager import AITEAState


class MultiAgentEngine:
    """
    Simulate background market participants:
    noise traders, liquidity providers, momentum traders, and adversarial flow.
    """

    def __init__(self, config: AITEAConfig | None = None) -> None:
        self.config = config or get_config()

    def flow(self, state: AITEAState) -> Dict[str, float]:
        liquidity_scale = float(state.task_profile.get("liquidity_scale", 0.75))
        kind = str(state.task_profile.get("kind", "generic"))

        noise_intensity = 0.15 + 0.35 * state.hidden_volatility
        momentum_intensity = 0.10 + 0.20 * state.hidden_volatility
        adversarial_intensity = 0.05 if kind in {"execution", "rebalance"} else 0.08

        # Produce a small hidden pressure metric that can later be used by the market/execution engines.
        pressure = 0.0
        for idx, symbol in enumerate(state.prices.keys()):
            phase = math.sin((state.step + 1) * 0.37 + idx)
            pressure += phase * (noise_intensity + momentum_intensity) * liquidity_scale

        pressure += adversarial_intensity * (1.0 if state.regime.value in {"volatile", "crisis"} else 0.5)

        # Background flow slightly changes cash pressure by simulating competition for liquidity.
        market_drag = max(0.0, abs(pressure)) * 0.0001 * self.config.starting_cash
        state.task_metrics["background_flow_pressure"] = float(pressure)
        state.task_metrics["market_drag"] = float(market_drag)

        return {
            "background_flow_pressure": pressure,
            "market_drag": market_drag,
            "noise_intensity": noise_intensity,
            "momentum_intensity": momentum_intensity,
            "adversarial_intensity": adversarial_intensity,
        }
