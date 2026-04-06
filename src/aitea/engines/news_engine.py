"""News shock engine for AITEA."""

from __future__ import annotations

from typing import Dict, List

from ..config import AITEAConfig, get_config
from ..schemas import NewsSignal
from ..env.state_manager import AITEAState


class NewsEngine:
    """Generate structured news events that affect market dynamics."""

    def __init__(self, config: AITEAConfig | None = None) -> None:
        self.config = config or get_config()

    def maybe_emit(self, state: AITEAState) -> Dict[str, object]:
        p = float(state.task_profile.get("news_probability", 0.03))
        emitted = False

        if state.rng.random() < p:
            emitted = True
            severity = max(0.05, min(1.0, abs(state.rng.gauss(0.45, 0.20))))
            sentiment = max(-1.0, min(1.0, state.rng.uniform(-1.0, 1.0)))
            symbols = list(state.prices.keys())[:2]
            headline = f"{state.task_name} news event at step {state.step}"
            state.news_queue.append(
                NewsSignal(
                    headline=headline,
                    severity=severity,
                    sentiment=sentiment,
                    affected_symbols=symbols,
                )
            )
            state.news_queue = state.news_queue[-3:]

        return {
            "emitted": emitted,
            "active_news_count": len(state.news_queue),
            "latest_severity": state.news_queue[-1].severity if state.news_queue else 0.0,
        }
