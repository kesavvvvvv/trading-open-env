"""
Central configuration for AITEA.

Keep all global defaults here so the codebase does not rely on hardcoded values
spread across many files.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List
import os
import random


@dataclass(slots=True)
class AITEAConfig:
    # Episode settings
    episode_horizon: int = 50
    seed: int = 42

    # Market universe
    assets: List[str] = field(default_factory=lambda: [
        "AAPL",
        "MSFT",
        "GOOG",
        "EURUSD",
    ])

    initial_prices: Dict[str, float] = field(default_factory=lambda: {
        "AAPL": 180.0,
        "MSFT": 350.0,
        "GOOG": 140.0,
        "EURUSD": 1.10,
    })

    # Starting capital and portfolio constraints
    initial_cash: float = 1_000_000.0
    max_position_size: float = 100_000.0

    # Market microstructure
    volatility: float = 0.02
    spread: float = 0.001
    slippage_coeff: float = 0.0005

    # Costs
    transaction_cost_pct: float = 0.001

    # Risk limits
    max_drawdown: float = 0.20
    daily_loss_limit: float = 0.05

    # Feature switches
    enable_news: bool = True
    enable_regimes: bool = True
    enable_multi_agent: bool = True

    # Multi-agent pressure
    noise_trader_strength: float = 0.10

    # Reward weights
    reward_weights: Dict[str, float] = field(default_factory=lambda: {
        "pnl": 1.0,
        "cost": 0.5,
        "risk": 0.7,
        "drawdown": 0.8,
        "compliance": 1.0,
        "stability": 0.3,
    })

    # Penalty weights
    penalty_weights: Dict[str, float] = field(default_factory=lambda: {
        "invalid_action": 1.0,
        "overtrading": 0.5,
        "constraint_violation": 2.0,
        "missed_obligation": 3.0,
    })


def load_config() -> AITEAConfig:
    """
    Load config with environment variable overrides.
    """
    cfg = AITEAConfig()

    cfg.episode_horizon = int(os.getenv("EPISODE_HORIZON", str(cfg.episode_horizon)))
    cfg.seed = int(os.getenv("AITEA_SEED", str(cfg.seed)))
    cfg.initial_cash = float(os.getenv("INITIAL_CASH", str(cfg.initial_cash)))

    assets_env = os.getenv("AITEA_ASSETS")
    if assets_env:
        cfg.assets = [a.strip() for a in assets_env.split(",") if a.strip()]

    return cfg


CONFIG = load_config()


def get_config() -> AITEAConfig:
    """
    Return the active config object.
    """
    return CONFIG


def set_global_seed(seed: int) -> None:
    """
    Set Python's RNG seed for reproducibility.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
