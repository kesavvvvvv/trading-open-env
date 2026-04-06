"""Hidden regime engine for AITEA."""

from __future__ import annotations

from typing import Dict

from ..config import AITEAConfig, get_config
from ..schemas import MarketRegime
from ..env.state_manager import AITEAState


class RegimeEngine:
    """Maintain latent market regime state and transitions."""

    def __init__(self, config: AITEAConfig | None = None) -> None:
        self.config = config or get_config()

    def step(self, state: AITEAState) -> Dict[str, object]:
        p = float(state.task_profile.get("regime_flip_probability", 0.02))
        changed = False

        if state.rng.random() < p:
            changed = True
            regimes = [
                MarketRegime.CALM,
                MarketRegime.NORMAL,
                MarketRegime.VOLATILE,
                MarketRegime.CRISIS,
                MarketRegime.RECOVERY,
            ]
            new_regime = regimes[state.rng.randrange(len(regimes))]
            state.regime = new_regime

            if new_regime == MarketRegime.CALM:
                state.hidden_volatility = max(0.003, state.hidden_volatility * 0.85)
                state.regime_confidence = 0.80
            elif new_regime == MarketRegime.NORMAL:
                state.hidden_volatility = max(0.006, state.hidden_volatility * 1.00)
                state.regime_confidence = 0.70
            elif new_regime == MarketRegime.VOLATILE:
                state.hidden_volatility = max(0.010, state.hidden_volatility * 1.50)
                state.regime_confidence = 0.52
            elif new_regime == MarketRegime.CRISIS:
                state.hidden_volatility = max(0.015, state.hidden_volatility * 2.00)
                state.regime_confidence = 0.40
            elif new_regime == MarketRegime.RECOVERY:
                state.hidden_volatility = max(0.008, state.hidden_volatility * 1.10)
                state.regime_confidence = 0.60

        return {
            "changed": changed,
            "regime": state.regime.value,
            "confidence": state.regime_confidence,
            "volatility": state.hidden_volatility,
        }
