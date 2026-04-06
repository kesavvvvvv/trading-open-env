"""Market dynamics engine for AITEA."""

from __future__ import annotations

import math
from typing import Dict, Tuple

from ..config import AITEAConfig, get_config
from ..schemas import MarketRegime, NewsSignal
from ..env.state_manager import AITEAState


class MarketEngine:
    """Simulate prices, spreads, volatility, liquidity, and correlations."""

    def __init__(self, config: AITEAConfig | None = None) -> None:
        self.config = config or get_config()

    def _regime_multiplier(self, regime: MarketRegime) -> float:
        if regime == MarketRegime.CALM:
            return 0.65
        if regime == MarketRegime.NORMAL:
            return 1.00
        if regime == MarketRegime.VOLATILE:
            return 1.60
        if regime == MarketRegime.CRISIS:
            return 2.25
        if regime == MarketRegime.RECOVERY:
            return 0.90
        return 1.0

    def _news_effect(self, state: AITEAState) -> Tuple[float, float]:
        if not state.news_queue:
            return 0.0, 1.0
        latest: NewsSignal = state.news_queue[-1]
        drift = latest.sentiment * latest.severity * 0.010
        vol_mult = 1.0 + latest.severity * 1.25
        return drift, vol_mult

    def advance(self, state: AITEAState) -> Dict[str, float]:
        """
        Advance market state by one step.

        Returns a small metrics dict with aggregate diagnostics.
        """
        prev_prices = dict(state.prices)
        state.previous_prices = prev_prices

        base_vol = float(state.task_profile.get("volatility", state.hidden_volatility))
        regime_mult = self._regime_multiplier(state.regime)
        news_drift, news_vol_mult = self._news_effect(state)

        liquidity_scale = float(state.task_profile.get("liquidity_scale", 0.75))
        total_abs_return = 0.0

        for idx, (symbol, price) in enumerate(state.prices.items()):
            seasonal = 0.00015 * math.sin((state.step + 1) / 5.0 + idx)
            idio_noise = state.rng.gauss(0.0, base_vol * regime_mult * news_vol_mult)
            market_corr = 0.35 * idio_noise
            drift = 0.00005 + seasonal + news_drift

            if symbol in (state.news_queue[-1].affected_symbols if state.news_queue else []):
                drift += news_drift * 1.5

            new_price = max(1.0, price * (1.0 + drift + idio_noise + market_corr))
            state.prices[symbol] = new_price

            if price > 0:
                total_abs_return += abs((new_price - price) / price)

        # Hidden volatility follows realized market stress gently.
        realized_stress = total_abs_return / max(1, len(state.prices))
        state.hidden_volatility = max(
            0.003,
            0.92 * state.hidden_volatility + 0.08 * max(base_vol, realized_stress),
        )

        avg_spread = 0.0
        avg_liquidity = 0.0
        for symbol, price in state.prices.items():
            spread_pct = 0.0012 + 0.7 * state.hidden_volatility * regime_mult
            if state.news_queue:
                spread_pct *= news_vol_mult
            spread = max(0.01, price * spread_pct)
            avg_spread += spread
            avg_liquidity += 10_000.0 * liquidity_scale / max(0.25, regime_mult)

        count = max(1, len(state.prices))
        return {
            "avg_spread": avg_spread / count,
            "avg_liquidity": avg_liquidity / count,
            "realized_stress": realized_stress,
            "hidden_volatility": state.hidden_volatility,
        }
