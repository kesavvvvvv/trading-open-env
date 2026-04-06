"""Main dense reward model for AITEA."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from ..config import AITEAConfig, get_config
from ..env.state_manager import AITEAState
from ..schemas import Action, Reward
from .penalty_rules import total_penalty
from .reward_components import (
    compliance_component,
    execution_component,
    liquidity_component,
    portfolio_component,
    pnl_component,
    risk_component,
    stability_component,
)


def _clip(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(value)))


class RewardModel:
    """
    Dense reward model for institutional trading tasks.

    The model rewards:
    - PnL improvement
    - better execution
    - liquidity-aware behavior
    - risk discipline
    - compliance
    - stability

    And penalizes:
    - invalid actions
    - churn
    - no-op abuse
    - risk breaches
    - destructive behavior
    - repeated harmful behavior
    """

    def __init__(self, config: AITEAConfig | None = None) -> None:
        self.config = config or get_config()

    def compute(
        self,
        state: AITEAState,
        action: Optional[Any] = None,
        *,
        pnl_delta: float = 0.0,
        execution_cost: float = 0.0,
        slippage: float = 0.0,
        fill_ratio: float = 1.0,
        turnover: float = 0.0,
        market_drag: float = 0.0,
        target_error: float = 0.0,
        completion_progress: float = 0.0,
        violations: Optional[list[str]] = None,
    ) -> Reward:
        """
        Compute the reward for the current step.

        Returns a Reward schema with a breakdown that is easy to inspect.
        """
        violation_list = list(violations or [])

        # Core components
        components: Dict[str, float] = {
            "pnl": pnl_component(state, self.config, pnl_delta=pnl_delta),
            "execution": execution_component(fill_ratio=fill_ratio, execution_cost=execution_cost, slippage=slippage, config=self.config),
            "liquidity": liquidity_component(turnover=turnover, market_drag=market_drag, config=self.config),
            "risk": risk_component(
                drawdown_pct=state.drawdown_pct,
                gross_exposure_pct=(state.gross_exposure / max(1.0, state.equity)) if state.equity > 0 else 0.0,
                violation_count=len(violation_list),
                config=self.config,
            ),
            "compliance": compliance_component(violation_list),
            "stability": stability_component(state.recent_rewards),
            "portfolio": portfolio_component(target_error=target_error, completion_progress=completion_progress, config=self.config),
        }

        penalties = total_penalty(state, violation_list, self.config)

        # Weighted dense reward.
        # The coefficients are intentionally balanced so no single term dominates too early.
        raw_total = (
            0.28 * components["pnl"]
            + 0.18 * components["execution"]
            + 0.12 * components["liquidity"]
            + 0.16 * components["risk"]
            + 0.08 * components["compliance"]
            + 0.08 * components["stability"]
            + 0.10 * components["portfolio"]
            - 0.40 * penalties["total"]
        )

        clipped_total = _clip(raw_total, self.config.reward_clip_min, self.config.reward_clip_max)

        # Normalize to [0, 1] for grading and comparability.
        normalized_score = _clip((clipped_total - self.config.reward_clip_min) / (self.config.reward_clip_max - self.config.reward_clip_min), 0.0, 1.0)

        # Slight bonus for keeping the trajectory stable.
        if len(state.recent_rewards) >= 3:
            normalized_score = _clip(0.95 * normalized_score + 0.05 * components["stability"], 0.0, 1.0)

        return Reward(
            total=clipped_total,
            normalized_score=normalized_score,
            components=components,
            penalties=penalties,
            raw_total=raw_total,
            clipped_total=clipped_total,
        )
