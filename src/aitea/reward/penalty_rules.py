"""Penalty rules for AITEA reward shaping."""

from __future__ import annotations

from typing import Iterable, List

from ..config import AITEAConfig, get_config
from ..env.state_manager import AITEAState


def _clip(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(value)))


def invalid_action_penalty(violations: Iterable[str] | None = None) -> float:
    """
    Penalize malformed or unsafe actions.
    """
    count = len(list(violations or []))
    return _clip(count / 10.0, 0.0, 1.0)


def excessive_churn_penalty(state: AITEAState, threshold: int = 5) -> float:
    """
    Penalize frequent action switching and overtrading.
    """
    recent = [str(a) for a in state.recent_actions[-threshold:]]
    if len(recent) < threshold:
        return 0.0

    unique_actions = len(set(recent))
    repetition_ratio = 1.0 - (unique_actions / max(1, len(recent)))
    return _clip(repetition_ratio, 0.0, 1.0)


def no_op_abuse_penalty(state: AITEAState, threshold: int = 4) -> float:
    """
    Penalize repeated holding/no-op behavior when the task needs progress.
    """
    recent = [str(a).lower() for a in state.recent_actions[-threshold:]]
    if not recent:
        return 0.0

    hold_count = sum(1 for a in recent if "hold_position" in a or "hold" == a or "noop" in a or "no-op" in a)
    return _clip(hold_count / max(1, threshold), 0.0, 1.0)


def risk_breach_penalty(state: AITEAState, violations: Iterable[str] | None = None, config: AITEAConfig | None = None) -> float:
    """
    Penalize drawdown and exposure violations.
    """
    cfg = config or get_config()
    v = list(violations or [])
    drawdown_term = _clip(float(state.drawdown_pct) / max(1e-9, cfg.max_drawdown_pct), 0.0, 1.0)
    exposure_term = _clip(float(state.gross_exposure) / max(1.0, cfg.max_gross_exposure_pct * max(1.0, state.starting_cash)), 0.0, 1.0)
    violation_term = _clip(len(v) / 8.0, 0.0, 1.0)
    return _clip(0.45 * drawdown_term + 0.35 * exposure_term + 0.20 * violation_term, 0.0, 1.0)


def destructive_action_penalty(state: AITEAState) -> float:
    """
    Penalize actions that severely damage the portfolio or cash position.
    """
    cash_stress = 0.0
    if state.cash < 0:
        cash_stress = _clip(abs(state.cash) / max(1.0, state.starting_cash * 0.25), 0.0, 1.0)

    drawdown_stress = _clip(float(state.drawdown_pct) / 0.25, 0.0, 1.0)
    return _clip(0.6 * cash_stress + 0.4 * drawdown_stress, 0.0, 1.0)


def repeated_harmful_behavior_penalty(state: AITEAState, recent_reward_threshold: float = 0.0) -> float:
    """
    Penalize repeated negative-step behavior.
    """
    recent = [float(r) for r in state.recent_rewards[-6:]]
    if len(recent) < 3:
        return 0.0

    harmful = sum(1 for r in recent if r < recent_reward_threshold)
    return _clip(harmful / len(recent), 0.0, 1.0)


def total_penalty(
    state: AITEAState,
    violations: Iterable[str] | None = None,
    config: AITEAConfig | None = None,
) -> dict:
    """
    Return a full penalty breakdown.
    """
    cfg = config or get_config()
    v = list(violations or [])

    invalid = invalid_action_penalty(v)
    churn = excessive_churn_penalty(state)
    noop = no_op_abuse_penalty(state)
    risk = risk_breach_penalty(state, v, cfg)
    destructive = destructive_action_penalty(state)
    repeated = repeated_harmful_behavior_penalty(state)

    return {
        "invalid_action": invalid,
        "excessive_churn": churn,
        "no_op_abuse": noop,
        "risk_breach": risk,
        "destructive_action": destructive,
        "repeated_harmful_behavior": repeated,
        "total": _clip(
            0.30 * invalid
            + 0.15 * churn
            + 0.10 * noop
            + 0.20 * risk
            + 0.15 * destructive
            + 0.10 * repeated,
            0.0,
            1.0,
        ),
    }
