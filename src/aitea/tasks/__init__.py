"""Task exports and factory helpers for AITEA."""

from __future__ import annotations

from ..registry import get_task, list_tasks, register_task

from .task_base import TaskBase
from .task_execution_easy import ExecutionEasyTask
from .task_fx_hedge_medium import FXHedgeMediumTask
from .task_liquidity_medium import LiquidityMediumTask
from .task_news_adapt_hard import NewsAdaptHardTask
from .task_rebalance_hard import RebalanceHardTask
from .task_regime_challenge_hard import RegimeChallengeHardTask


def create_task(task_name: str) -> TaskBase:
    """Create a task instance from the registry."""
    task_cls = get_task(task_name)
    return task_cls()


__all__ = [
    "TaskBase",
    "ExecutionEasyTask",
    "LiquidityMediumTask",
    "FXHedgeMediumTask",
    "RebalanceHardTask",
    "NewsAdaptHardTask",
    "RegimeChallengeHardTask",
    "create_task",
    "register_task",
    "get_task",
    "list_tasks",
]
