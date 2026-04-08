"""Dense reward computation for AITEA."""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, Mapping

from ..config import AITEAConfig, get_config
from ..schemas import Reward


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def _safe_metrics(source: Any) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    if source is None:
        return metrics
    if isinstance(source, Mapping):
        for k, v in source.items():
            if isinstance(v, (int, float)) and math.isfinite(float(v)):
                metrics[str(k)] = float(v)
        return metrics
    task_metrics = getattr(source, "task_metrics", None)
    if isinstance(task_metrics, Mapping):
        for k, v in task_metrics.items():
            if isinstance(v, (int, float)) and math.isfinite(float(v)):
                metrics[str(k)] = float(v)
    return metrics


def _has_any(violations: Iterable[str], *keys: str) -> bool:
    violation_set = set(violations)
    return any(key in violation_set for key in keys)


class RewardModel:
    """Compute bounded dense reward with interpretable components."""

    def __init__(self, config: AITEAConfig | None = None) -> None:
        self.config = config or get_config()

    def compute(
        self,
        state: Any,
        *,
        pnl_delta: float = 0.0,
        execution_cost: float = 0.0,
        slippage: float = 0.0,
        fill_ratio: float = 1.0,
        turnover: float = 0.0,
        market_drag: float = 0.0,
        target_error: float = 0.0,
        completion_progress: float = 0.0,
        violations: Iterable[str] | None = None,
    ) -> Reward:
        metrics = _safe_metrics(state)
        task_profile = getattr(state, "task_profile", {}) or {}
        task_kind = str(task_profile.get("kind", "generic"))
        violations_list = list(violations or [])

        equity = max(1.0, _safe_float(metrics.get("equity", getattr(state, "equity", 1.0)), 1.0))
        starting_cash = max(1.0, _safe_float(metrics.get("starting_cash", getattr(state, "starting_cash", equity)), equity))
        drawdown = _clip01(_safe_float(metrics.get("drawdown_pct", getattr(state, "drawdown_pct", 0.0)), 0.0))
        gross_exposure = _safe_float(metrics.get("gross_exposure", getattr(state, "gross_exposure", 0.0)), 0.0)
        net_exposure = abs(_safe_float(metrics.get("net_exposure", getattr(state, "net_exposure", 0.0)), 0.0))
        target_remaining = _safe_float(metrics.get("target_remaining", 0.0), 0.0)
        progress_metric = _clip01(_safe_float(metrics.get("progress", completion_progress), completion_progress))
        recent_reward_std = abs(_safe_float(metrics.get("reward_std_recent", 0.0), 0.0))

        pnl_ratio = pnl_delta / equity
        execution_ratio = execution_cost / equity
        slippage_ratio = slippage / equity
        drag_ratio = market_drag / equity
        turnover_ratio = abs(turnover)

        target_error = _clip01(target_error)
        fill_ratio = _clip01(fill_ratio)
        completion_progress = _clip01(completion_progress)

        if task_kind in {"execution", "liquidity"}:
            task_progress = _clip01(1.0 - target_error)
            portfolio_progress = _clip01(completion_progress)
        elif task_kind == "hedge":
            hedge_error = _clip01(_safe_float(metrics.get("hedge_error", target_remaining), target_error))
            task_progress = _clip01(1.0 - hedge_error)
            portfolio_progress = _clip01(completion_progress)
        elif task_kind == "rebalance":
            tracking_error = _clip01(_safe_float(metrics.get("tracking_error", target_error), target_error))
            task_progress = _clip01(1.0 - tracking_error)
            portfolio_progress = _clip01(completion_progress)
        else:
            task_progress = _clip01(1.0 - target_error)
            portfolio_progress = _clip01(completion_progress)

        pnl_component = _clip01(0.5 + 0.5 * math.tanh(pnl_ratio * 12.0))
        execution_component = _clip01(1.0 - min(1.0, execution_ratio * 45.0 + slippage_ratio * 22.0 + drag_ratio * 8.0))
        liquidity_component = _clip01(fill_ratio * (0.5 + 0.5 * task_progress))
        risk_component = _clip01(1.0 - min(1.0, drawdown * 6.0 + (gross_exposure / starting_cash) * 0.04 + (net_exposure / starting_cash) * 0.03))
        compliance_component = 1.0
        stability_component = _clip01(1.0 - min(1.0, abs(recent_reward_std) * 0.8 + abs(turnover_ratio) * 0.12))
        portfolio_component = _clip01(0.6 * portfolio_progress + 0.4 * task_progress)

        invalid_symbol = _has_any(violations_list, "invalid_symbol") or any(v.startswith("unknown_symbol:") for v in violations_list)
        invalid_action = _has_any(violations_list, "invalid_action")
        no_op_abuse = _has_any(violations_list, "no_op_abuse")
        risk_breach = _has_any(violations_list, "risk_breach", "gross_exposure_breach", "drawdown_breach", "position_limit_breach")

        penalty_map: Dict[str, float] = {
            "invalid_action": 0.40 if invalid_action else 0.0,
            "invalid_symbol": 0.45 if invalid_symbol else 0.0,
            "excessive_churn": 0.10 if _has_any(violations_list, "excessive_churn") else 0.0,
            "no_op_abuse": 0.35 if no_op_abuse else 0.0,
            "risk_breach": 0.20 if _has_any(violations_list, "risk_breach") else 0.0,
            "gross_exposure_breach": 0.18 if _has_any(violations_list, "gross_exposure_breach") else 0.0,
            "drawdown_breach": 0.30 if _has_any(violations_list, "drawdown_breach") else 0.0,
            "position_limit_breach": 0.20 if _has_any(violations_list, "position_limit_breach") else 0.0,
            "destructive_action": 0.28 if _has_any(violations_list, "destructive_action") else 0.0,
            "repeated_harmful_behavior": 0.16 if _has_any(violations_list, "repeated_harmful_behavior") else 0.0,
            "liquidity_constraint": 0.12 if _has_any(violations_list, "liquidity_constraint") else 0.0,
            "partial_fill": 0.06 if _has_any(violations_list, "partial_fill_position_cap", "partial_fill") else 0.0,
            "cancel_id_not_found": 0.05 if _has_any(violations_list, "cancel_id_not_found") else 0.0,
        }

        penalty_total = float(sum(penalty_map.values()))

        raw_total = (
            0.22 * pnl_component
            + 0.18 * execution_component
            + 0.12 * liquidity_component
            + 0.18 * risk_component
            + 0.12 * compliance_component
            + 0.08 * stability_component
            + 0.10 * portfolio_component
            + 0.10 * progress_metric
        )

        if invalid_symbol:
            raw_total *= 0.45
        if invalid_action:
            raw_total *= 0.55
        if no_op_abuse:
            raw_total *= 0.20
        if risk_breach:
            raw_total *= 0.65

        raw_total = raw_total - penalty_total
        # FORCE NEGATIVE REWARD FOR NO-OP
        if state.task_metrics.get("force_negative_reward", 0.0) == 1.0:
            raw_total = -0.3
        clipped_total = max(-1.0, min(1.0, raw_total))
        normalized_score = _clip01((math.tanh(clipped_total * 2.2 - 1.0) + 1.0) / 2.0)

        components = {
            "pnl": pnl_component,
            "execution": execution_component,
            "liquidity": liquidity_component,
            "risk": risk_component,
            "compliance": compliance_component,
            "stability": stability_component,
            "portfolio": portfolio_component,
        }

        penalty_map["total"] = penalty_total

        return Reward(
            total=clipped_total,
            normalized_score=normalized_score,
            components=components,
            penalties=penalty_map,
            raw_total=raw_total,
            clipped_total=clipped_total,
        )
