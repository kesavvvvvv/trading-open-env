"""
AITEA - AI Institutional Treasury & Execution Arena

Top-level package exports.
"""

from __future__ import annotations

__version__ = "0.1.0"

# Keep imports defensive so the package can still be imported while
# other modules are being built.
try:
    from aitea.env.aitea_env import AITEAEnv
except Exception:
    AITEAEnv = None  # type: ignore[assignment]

try:
    from aitea.registry import register_task, register_tasks, get_task, list_tasks, create_env
except Exception:
    register_task = None  # type: ignore[assignment]
    register_tasks = None  # type: ignore[assignment]
    get_task = None  # type: ignore[assignment]
    list_tasks = None  # type: ignore[assignment]
    create_env = None  # type: ignore[assignment]

try:
    from aitea.schemas.observation import Observation
    from aitea.schemas.action import Action
    from aitea.schemas.reward import Reward
    from aitea.schemas.info import Info
except Exception:
    Observation = None  # type: ignore[assignment]
    Action = None  # type: ignore[assignment]
    Reward = None  # type: ignore[assignment]
    Info = None  # type: ignore[assignment]

__all__ = [
    "__version__",
    "AITEAEnv",
    "register_task",
    "register_tasks",
    "get_task",
    "list_tasks",
    "create_env",
    "Observation",
    "Action",
    "Reward",
    "Info",
]
