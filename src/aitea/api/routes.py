"""HTTP routes for AITEA."""

from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from ..agents.baseline_rules import baseline_action

router = APIRouter()


class StepRequest(BaseModel):
    action: Dict[str, Any] = Field(default_factory=dict)


class ResetRequest(BaseModel):
    task_name: Optional[str] = None
    episode_id: Optional[str] = None


def _get_env(request: Request):
    env = getattr(request.app.state, "env", None)
    if env is None:
        raise HTTPException(status_code=503, detail="Environment not initialized")
    return env


def _to_dict(obj: Any) -> Dict[str, Any]:
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "dict"):
        return obj.dict()
    if isinstance(obj, dict):
        return obj
    return {"value": str(obj)}


@router.get("/health")
async def health():
    return {"status": "ok"}


@router.get("/state")
async def state(request: Request):
    env = _get_env(request)
    obs = env.state()
    return {"status": "ok", "state": _to_dict(obs)}


@router.post("/reset")
async def reset(payload: ResetRequest, request: Request):
    env = _get_env(request)
    result = env.reset(task_name=payload.task_name, episode_id=payload.episode_id)
    return {
        "status": "ok",
        "transition": _to_dict(result),
    }


@router.post("/step")
async def step(payload: StepRequest, request: Request):
    env = _get_env(request)
    action = payload.action
    if not action:
        observation = env.state()
        action = _to_dict(baseline_action(observation))
    result = env.step(action)
    return {
        "status": "ok",
        "transition": _to_dict(result),
    }
