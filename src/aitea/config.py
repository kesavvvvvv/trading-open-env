"""Central configuration for AITEA."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value is not None and value != "" else default


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return float(value) if value is not None and value != "" else default


@dataclass(frozen=True)
class AITEAConfig:
    # Core episode settings
    episode_length: int = _env_int("AITEA_EPISODE_LENGTH", 50)
    seed: int = _env_int("AITEA_SEED", 42)

    # Capital and market universe
    starting_cash: float = _env_float("AITEA_STARTING_CASH", 1_000_000.0)
    assets: List[str] = field(
        default_factory=lambda: ["AAPL", "MSFT", "GOOG", "TSLA", "AMZN"]
    )
    initial_prices: List[float] = field(
        default_factory=lambda: [180.0, 420.0, 2800.0, 700.0, 190.0]
    )

    # Trading and execution
    transaction_cost_pct: float = _env_float("AITEA_TRANSACTION_COST_PCT", 0.001)
    slippage_coefficient: float = _env_float("AITEA_SLIPPAGE_COEFFICIENT", 0.01)
    max_order_size_pct: float = _env_float("AITEA_MAX_ORDER_SIZE_PCT", 0.25)
    max_daily_turnover_pct: float = _env_float("AITEA_MAX_DAILY_TURNOVER_PCT", 1.0)

    # Risk settings
    max_gross_exposure_pct: float = _env_float("AITEA_MAX_GROSS_EXPOSURE_PCT", 2.0)
    max_drawdown_pct: float = _env_float("AITEA_MAX_DRAWDOWN_PCT", 0.15)
    max_position_pct: float = _env_float("AITEA_MAX_POSITION_PCT", 0.35)

    # Reward weights
    pnl_weight: float = _env_float("AITEA_PNL_WEIGHT", 1.0)
    cost_weight: float = _env_float("AITEA_COST_WEIGHT", 0.5)
    slippage_weight: float = _env_float("AITEA_SLIPPAGE_WEIGHT", 0.5)
    risk_weight: float = _env_float("AITEA_RISK_WEIGHT", 1.0)
    penalty_weight: float = _env_float("AITEA_PENALTY_WEIGHT", 1.0)

    # Normalization / clipping
    reward_clip_min: float = _env_float("AITEA_REWARD_CLIP_MIN", -1.0)
    reward_clip_max: float = _env_float("AITEA_REWARD_CLIP_MAX", 1.0)

    # Task defaults
    default_task: str = os.getenv("AITEA_DEFAULT_TASK", "execution_easy")


CONFIG = AITEAConfig()


def get_config() -> AITEAConfig:
    """Return the immutable global config instance."""
    return CONFIG
