"""Base deterministic grader interface."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Any, Dict, Mapping


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def _extract_mapping(source: Any) -> Dict[str, float]:
    if source is None:
        return {}
    if isinstance(source, Mapping):
        return {str(k): _safe_float(v, 0.0) for k, v in source.items() if isinstance(v, (int, float))}
    out: Dict[str, float] = {}
    for attr in ("task_metrics", "metrics", "info_metrics"):
        candidate = getattr(source, attr, None)
        if isinstance(candidate, Mapping):
            for k, v in candidate.items():
                if isinstance(v, (int, float)) and math.isfinite(float(v)):
                    out[str(k)] = float(v)
    return out


class GraderBase(ABC):
    """Base deterministic grader."""

    name: str = "base"

    def __call__(self, source: Any = None) -> float:
        return self.score(source)

    @abstractmethod
    def score(self, source: Any = None) -> float:
        raise NotImplementedError

    def metrics(self, source: Any = None) -> Dict[str, float]:
        return _extract_mapping(source)

    def _from_error(self, error: float, scale: float = 1.0) -> float:
        return _clip01(1.0 - min(1.0, abs(error) / max(scale, 1e-8)))

    def _from_ratio(self, ratio: float, target: float = 1.0, tolerance: float = 0.1) -> float:
        return _clip01(1.0 - min(1.0, abs(ratio - target) / max(tolerance, 1e-8)))

    def _from_drawdown(self, drawdown: float, limit: float = 0.2) -> float:
        return _clip01(1.0 - min(1.0, max(0.0, drawdown) / max(limit, 1e-8)))

    def _penalize(self, base: float, penalty: float) -> float:
        return _clip01(base - max(0.0, penalty))
