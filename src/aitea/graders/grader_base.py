"""Base grader for AITEA."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List

from ..config import AITEAConfig, get_config
from ..env.state_manager import AITEAState


class GraderBase(ABC):
    """Deterministic scorer that returns values in [0, 1]."""

    def __init__(self, config: AITEAConfig | None = None) -> None:
        self.config = config or get_config()

    @staticmethod
    def _clip01(value: float) -> float:
        return max(0.0, min(1.0, float(value)))

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except Exception:
            return default

    def _metric(self, state: AITEAState, key: str, default: float = 0.0) -> float:
        return self._safe_float(state.task_metrics.get(key, default), default)

    def _reward_stability(self, state: AITEAState) -> float:
        rewards = [self._safe_float(v, 0.0) for v in state.recent_rewards[-5:]]
        if len(rewards) < 2:
            return 1.0
        mean = sum(rewards) / len(rewards)
        variance = sum((r - mean) ** 2 for r in rewards) / len(rewards)
        # Convert variability into a stability score.
        return self._clip01(1.0 / (1.0 + variance))

    def _drawdown_score(self, drawdown_pct: float, soft_limit: float = 0.15) -> float:
        return self._clip01(1.0 - min(1.0, drawdown_pct / max(1e-9, soft_limit)))

    def _linear_score(self, value: float, best: float, worst: float, invert: bool = False) -> float:
        if best == worst:
            return 1.0
        x = (value - worst) / (best - worst)
        x = self._clip01(x)
        return 1.0 - x if invert else x

    @abstractmethod
    def score(self, state: AITEAState) -> float:
        """Return a deterministic normalized score in [0, 1]."""
        raise NotImplementedError

    def detail(self, state: AITEAState) -> Dict[str, float]:
        """Optional breakdown for debugging."""
        return {
            "step": float(state.step),
            "drawdown_pct": float(state.drawdown_pct),
            "equity": float(state.equity),
        }
