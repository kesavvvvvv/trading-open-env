"""
Task and environment registry for AITEA.

This file keeps task registration centralized and gives the inference script
and API a clean way to construct environments.
"""

from __future__ import annotations

from typing import Any, Dict, List, Type


_TASK_REGISTRY: Dict[str, Type[Any]] = {}


def register_task(name: str, task_cls: Type[Any]) -> None:
    """
    Register a task class under a unique string name.
    """
    if not name or not name.strip():
        raise ValueError("Task name must be a non-empty string.")

    if name in _TASK_REGISTRY:
        raise ValueError(f"Task '{name}' is already registered.")

    _TASK_REGISTRY[name] = task_cls


def register_tasks() -> None:
    """
    Register all built-in tasks.

    Importing here keeps circular imports low while the rest of the project
    is still being built.
    """
    if _TASK_REGISTRY:
        return

    from aitea.tasks.task_execution_easy import TaskExecutionEasy
    from aitea.tasks.task_liquidity_medium import TaskLiquidityMedium
    from aitea.tasks.task_fx_hedge_medium import TaskFXHedgeMedium
    from aitea.tasks.task_rebalance_hard import TaskRebalanceHard
    from aitea.tasks.task_news_adapt_hard import TaskNewsAdaptHard
    from aitea.tasks.task_regime_challenge_hard import TaskRegimeChallengeHard

    register_task("execution_easy", TaskExecutionEasy)
    register_task("liquidity_medium", TaskLiquidityMedium)
    register_task("fx_hedge_medium", TaskFXHedgeMedium)
    register_task("rebalance_hard", TaskRebalanceHard)
    register_task("news_adapt_hard", TaskNewsAdaptHard)
    register_task("regime_challenge_hard", TaskRegimeChallengeHard)


def get_task(name: str) -> Type[Any]:
    """
    Fetch a registered task class by name.
    """
    if name not in _TASK_REGISTRY:
        available = ", ".join(sorted(_TASK_REGISTRY.keys())) or "<none>"
        raise ValueError(f"Task '{name}' not found. Available tasks: {available}")

    return _TASK_REGISTRY[name]


def list_tasks() -> List[str]:
    """
    Return all registered task names.
    """
    return sorted(_TASK_REGISTRY.keys())


def create_env(task_name: str, **kwargs: Any) -> Any:
    """
    Create an AITEA environment for the given task.
    """
    from aitea.env.aitea_env import AITEAEnv

    task_cls = get_task(task_name)
    task_instance = task_cls(**kwargs)
    return AITEAEnv(task=task_instance)
