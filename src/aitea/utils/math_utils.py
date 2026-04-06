"""Mathematical utilities for AITEA."""

from __future__ import annotations

from typing import Iterable, List


def clip(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(value)))


def safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
    if abs(denominator) < 1e-9:
        return default
    return numerator / denominator


def compute_return(current: float, previous: float) -> float:
    if previous <= 0:
        return 0.0
    return (current - previous) / previous


def compute_drawdown(equity_curve: Iterable[float]) -> float:
    peak = None
    max_drawdown = 0.0
    for value in equity_curve:
        v = float(value)
        if peak is None or v > peak:
            peak = v
        if peak is None or peak <= 0:
            continue
        drawdown = (peak - v) / peak
        if drawdown > max_drawdown:
            max_drawdown = drawdown
    return max_drawdown


def normalize(value: float, minimum: float, maximum: float) -> float:
    if maximum <= minimum:
        return 0.0
    return clip((value - minimum) / (maximum - minimum), 0.0, 1.0)


def zscore(value: float, mean: float, std: float) -> float:
    if abs(std) < 1e-9:
        return 0.0
    return (value - mean) / std


def rolling_mean(values: Iterable[float], window: int) -> float:
    series = [float(v) for v in values]
    if not series or window <= 0:
        return 0.0
    subset = series[-window:]
    return sum(subset) / len(subset)


def rolling_std(values: Iterable[float], window: int) -> float:
    series = [float(v) for v in values]
    if not series or window <= 1:
        return 0.0
    subset = series[-window:]
    mean = sum(subset) / len(subset)
    var = sum((x - mean) ** 2 for x in subset) / len(subset)
    return var ** 0.5


def cumulative_sum(values: Iterable[float]) -> List[float]:
    out: List[float] = []
    total = 0.0
    for v in values:
        total += float(v)
        out.append(total)
    return out
