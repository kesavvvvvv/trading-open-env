"""Task and environment registry."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Type, TypeVar

from .config import AITEAConfig, get_config

TASK_REGISTRY: Dict[str, Type[Any]] = {}
T = TypeVar("T")


def register_task(name: str, task_cls: Type[T] | None = None):
    """
    Register a task class under a string name.

    Usage:
        @register_task("execution_easy")
        class ExecutionEasyTask(...):
            ...
    """
    def decorator(cls: Type[T]) -> Type[T]:
        TASK_REGISTRY[name] = cls
        return cls

    if task_cls is not None:
        return decorator(task_cls)
    return decorator


def get_task(name: str) -> Type[Any]:
    """Return the registered task class for a task name."""
    if name not in TASK_REGISTRY:
        available = ", ".join(list_tasks()) or "none"
        raise KeyError(f"Unknown task '{name}'. Available tasks: {available}")
    return TASK_REGISTRY[name]


def list_tasks() -> List[str]:
    """Return all registered task names in sorted order."""
    return sorted(TASK_REGISTRY.keys())


def create_env(task_name: str | None = None, config: AITEAConfig | None = None, **kwargs: Any):
    """
    Create the main environment instance for a given task.

    The environment class is imported lazily so this module stays lightweight
    and does not create circular imports during startup.
    """
    from .env.aitea_env import AITEAEnv

    cfg = config or get_config()
    chosen_task = task_name or cfg.default_task
    return AITEAEnv(task_name=chosen_task, config=cfg, **kwargs)


__all__ = [
    "TASK_REGISTRY",
    "register_task",
    "get_task",
    "list_tasks",
    "create_env",
]
