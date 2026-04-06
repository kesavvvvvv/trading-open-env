"""Structured logging helpers for AITEA."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional

from .constants import LOG_END, LOG_START, LOG_STEP


def _timestamp() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def format_kv(data: Dict[str, Any]) -> str:
    parts = []
    for key, value in data.items():
        if value is None:
            rendered = "null"
        elif isinstance(value, bool):
            rendered = str(value).lower()
        else:
            rendered = str(value)
        parts.append(f"{key}={rendered}")
    return " ".join(parts)


def format_start(task: str, env: str, model: str) -> str:
    return f"{LOG_START} {format_kv({'task': task, 'env': env, 'model': model})}"


def format_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> str:
    return f"{LOG_STEP} {format_kv({'step': step, 'action': action, 'reward': f'{reward:.2f}', 'done': str(done).lower(), 'error': error if error else 'null'})}"


def format_end(success: bool, steps: int, rewards: list[float]) -> str:
    rewards_str = ",".join(f"{float(r):.2f}" for r in rewards)
    return f"{LOG_END} {format_kv({'success': str(success).lower(), 'steps': steps, 'rewards': rewards_str})}"


def debug(message: str, **fields: Any) -> str:
    payload = {"ts": _timestamp(), "msg": message, **fields}
    return format_kv(payload)


def step_summary(step: int, reward: float, equity: float, drawdown: float) -> str:
    return format_kv(
        {
            "step": step,
            "reward": f"{reward:.4f}",
            "equity": f"{equity:.2f}",
            "drawdown": f"{drawdown:.4f}",
        }
    )
