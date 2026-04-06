"""Interpretable reward components for AITEA."""

from __future__ import annotations

from typing import Dict, Iterable

from ..config import AITEAConfig, get_config
from ..env.state_manager import AITEAState


def _clip(value: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(value)))


def pnl_component(state: AITEAState, config: AITEAConfig | None = None, pnl_delta: float | None = None) -> float:
    """
    Reward incremental PnL, normalized by starting capital.

    Positive PnL is rewarded; negative PnL is penalized.
    """
    cfg = config or get_config()
    delta = float(pnl_delta if pnl_delta is not None else (state.realized_pnl + state.unrealized_pnl))
    scale = max(1.0, cfg.starting_cash * 0.02)
    return _clip(delta / scale, -1.0, 1.0)


def execution_component(fill_ratio: float, execution_cost: float = 0.0, slippage: float = 0.0, config: AITEAConfig | None = None) -> float:
    """
    Reward efficient execution.

    High fill ratio helps; cost/slippage hurt.
    """
    cfg = config or get_config()
    fill_score = max(0.0, min(1.0, float(fill_ratio)))
    cost_scale = max(1.0, cfg.starting_cash * 0.005)
    cost_penalty = min(1.0, (abs(execution_cost) + abs(slippage)) / cost_scale)
    return _clip(0.8 * fill_score - 0.8 * cost_penalty, -1.0, 1.0)


def liquidity_component(turnover: float, market_drag: float = 0.0, config: AITEAConfig | None = None) -> float:
    """
    Reward conservative and liquidity-aware trading.

    Lower turnover and lower drag are better.
    """
    cfg = config or get_config()
    turnover_penalty = max(0.0, min(1.0, float(turnover)))
    drag_scale = max(1.0, cfg.starting_cash * 0.001)
    drag_penalty = max(0.0, min(1.0, abs(market_drag) / drag_scale))
    return _clip(1.0 - 0.6 * turnover_penalty - 0.4 * drag_penalty, -1.0, 1.0)


def risk_component(drawdown_pct: float, gross_exposure_pct: float = 0.0, violation_count: int = 0, config: AITEAConfig | None = None) -> float:
    """
    Reward staying within risk limits.

    Lower drawdown, lower exposure, and fewer violations are better.
    """
    cfg = config or get_config()
    dd_penalty = max(0.0, min(1.0, float(drawdown_pct) / max(1e-9, cfg.max_drawdown_pct)))
    exposure_penalty = max(0.0, min(1.0, float(gross_exposure_pct) / max(1e-9, cfg.max_gross_exposure_pct)))
    violations_penalty = max(0.0, min(1.0, float(violation_count) / 10.0))
    return _clip(1.0 - 0.45 * dd_penalty - 0.35 * exposure_penalty - 0.20 * violations_penalty, -1.0, 1.0)


def compliance_component(violations: Iterable[str] | None = None) -> float:
    """
    Reward rule compliance.

    No violations => 1.0. More violations => lower score.
    """
    count = len(list(violations or []))
    return _clip(1.0 - min(1.0, count / 8.0), 0.0, 1.0)


def stability_component(recent_rewards: Iterable[float] | None = None) -> float:
    """
    Reward stable learning behavior.

    Smooth reward trajectories are better than volatile ones.
    """
    rewards = [float(r) for r in (recent_rewards or [])]
    if len(rewards) < 2:
        return 1.0

    mean = sum(rewards) / len(rewards)
    variance = sum((r - mean) ** 2 for r in rewards) / len(rewards)
    return max(0.0, min(1.0, 1.0 / (1.0 + variance)))


def portfolio_component(
    target_error: float = 0.0,
    completion_progress: float = 0.0,
    config: AITEAConfig | None = None,
) -> float:
    """
    Generic portfolio/task progress component.

    Lower target error and higher completion progress are better.
    """
    _ = config or get_config()
    error_penalty = max(0.0, min(1.0, float(target_error)))
    progress_score = max(0.0, min(1.0, float(completion_progress)))
    return _clip(0.65 * progress_score + 0.35 * (1.0 - error_penalty), -1.0, 1.0)
