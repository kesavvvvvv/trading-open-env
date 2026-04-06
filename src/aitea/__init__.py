"""AITEA package exports."""

from __future__ import annotations

__version__ = "0.1.0"

from .config import AITEAConfig, get_config
from .registry import create_env, get_task, list_tasks, register_task

try:
    from .env.aitea_env import AITEAEnv
except Exception:
    AITEAEnv = None  # type: ignore[assignment]

try:
    from .schemas.action import Action
    from .schemas.info import Info
    from .schemas.observation import Observation
    from .schemas.reward import Reward
except Exception:
    Action = None  # type: ignore[assignment]
    Info = None  # type: ignore[assignment]
    Observation = None  # type: ignore[assignment]
    Reward = None  # type: ignore[assignment]

__all__ = [
    "__version__",
    "AITEAConfig",
    "get_config",
    "AITEAEnv",
    "register_task",
    "get_task",
    "list_tasks",
    "create_env",
    "Observation",
    "Action",
    "Reward",
    "Info",
]
