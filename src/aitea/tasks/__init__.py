"""Task exports and registry for AITEA."""

from .task_base import TaskBase
from .task_execution_easy import TaskExecutionEasy, create_task as create_execution_easy_task
from .task_fx_hedge_medium import TaskFXHedgeMedium, create_task as create_fx_hedge_medium_task
from .task_liquidity_medium import TaskLiquidityMedium, create_task as create_liquidity_medium_task
from .task_news_adapt_hard import TaskNewsAdaptHard, create_task as create_news_adapt_hard_task
from .task_rebalance_hard import TaskRebalanceHard, create_task as create_rebalance_hard_task
from .task_regime_challenge_hard import TaskRegimeChallengeHard, create_task as create_regime_challenge_hard_task

TASK_REGISTRY = {
    "execution_easy": TaskExecutionEasy,
    "liquidity_medium": TaskLiquidityMedium,
    "fx_hedge_medium": TaskFXHedgeMedium,
    "rebalance_hard": TaskRebalanceHard,
    "news_adapt_hard": TaskNewsAdaptHard,
    "regime_challenge_hard": TaskRegimeChallengeHard,
}


def get_task_class(name: str):
    return TASK_REGISTRY[name]


def create_task(name: str):
    return get_task_class(name)()


__all__ = [
    "TaskBase",
    "TaskExecutionEasy",
    "TaskLiquidityMedium",
    "TaskFXHedgeMedium",
    "TaskRebalanceHard",
    "TaskNewsAdaptHard",
    "TaskRegimeChallengeHard",
    "TASK_REGISTRY",
    "create_task",
    "get_task_class",
]
