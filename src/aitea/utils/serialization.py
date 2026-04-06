"""Safe serialization helpers for AITEA."""

from __future__ import annotations

import json
from typing import Any, Dict, List


def model_to_dict(obj: Any) -> Dict[str, Any]:
    if obj is None:
        return {}
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "dict"):
        return obj.dict()
    if isinstance(obj, dict):
        return obj
    return {"value": obj}


def model_to_json(obj: Any, *, indent: int | None = None) -> str:
    return json.dumps(model_to_dict(obj), ensure_ascii=False, indent=indent, default=str)


def json_to_dict(data: str) -> Dict[str, Any]:
    if not data:
        return {}
    parsed = json.loads(data)
    if not isinstance(parsed, dict):
        raise ValueError("Expected a JSON object.")
    return parsed


def trajectory_to_json(trajectory: List[Dict[str, Any]], *, indent: int | None = None) -> str:
    return json.dumps(trajectory, ensure_ascii=False, indent=indent, default=str)


def safe_snapshot(state: Any) -> Dict[str, Any]:
    """
    Convert state-like objects into a JSON-safe snapshot.
    """
    raw = model_to_dict(state)
    snapshot: Dict[str, Any] = {}

    for key, value in raw.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            snapshot[key] = value
        elif isinstance(value, dict):
            snapshot[key] = value
        elif isinstance(value, list):
            snapshot[key] = value
        else:
            snapshot[key] = str(value)
    return snapshot
