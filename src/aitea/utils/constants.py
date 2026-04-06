"""Project-wide constants for AITEA."""

from __future__ import annotations

# Task names
TASK_EXECUTION = "execution_easy"
TASK_LIQUIDITY = "liquidity_medium"
TASK_FX_HEDGE = "fx_hedge_medium"
TASK_REBALANCE = "rebalance_hard"
TASK_NEWS = "news_adapt_hard"
TASK_REGIME = "regime_challenge_hard"

ALL_TASKS = [
    TASK_EXECUTION,
    TASK_LIQUIDITY,
    TASK_FX_HEDGE,
    TASK_REBALANCE,
    TASK_NEWS,
    TASK_REGIME,
]

# Status labels
STATUS_OK = "ok"
STATUS_ERROR = "error"
STATUS_DONE = "done"

# Numeric constants
EPS = 1e-9
MAX_FLOAT = 1e12

# Logging tags
LOG_START = "[START]"
LOG_STEP = "[STEP]"
LOG_END = "[END]"

# Default limits
MAX_STEPS = 1000
MAX_ACTIONS_BUFFER = 50
MAX_REWARD_BUFFER = 50
