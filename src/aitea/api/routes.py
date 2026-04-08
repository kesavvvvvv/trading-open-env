"""HTTP routes for AITEA."""

from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter, HTTPException, Request

from ..agents.baseline_rules import baseline_action

router = APIRouter()


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


def _transition_payload(result: Any) -> Dict[str, Any]:
    return {
        "observation": _to_dict(getattr(result, "observation", None)),
        "reward": float(getattr(result, "reward", 0.0)),
        "done": bool(getattr(result, "done", False)),
        "info": _to_dict(getattr(result, "info", {})),
        "error": getattr(result, "error", None),
        "reward_detail": _to_dict(getattr(result, "reward_detail", {})),
    }


async def _json_body(request: Request) -> Dict[str, Any]:
    raw = await request.body()
    if not raw:
        return {}
    try:
        body = await request.json()
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid JSON body") from exc
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="Request body must be a JSON object")
    return body


@router.get("/health")
async def health():
    return {"status": "ok"}


@router.get("/state")
async def state(request: Request):
    env = _get_env(request)
    obs = env.state()
    return {"status": "ok", "state": _to_dict(obs)}


@router.post("/reset")
async def reset(request: Request):
    env = _get_env(request)
    body = await _json_body(request)

    task_name = body.get("task_name") or body.get("task")
    episode_id = body.get("episode_id")

    result = env.reset(task_name=task_name, episode_id=episode_id)
    return {
        "status": "ok",
        "transition": _transition_payload(result),
    }


@router.post("/step")
async def step(request: Request):
    env = _get_env(request)
    body = await _json_body(request)

    if "action" in body:
        action = body["action"]
        if not isinstance(action, dict):
            raise HTTPException(status_code=400, detail="Action must be a JSON object")
    else:
        action = body

    if not action:
        observation = env.state()
        action = _to_dict(baseline_action(observation))

    try:
        result = env.step(action)
    except Exception as exc:
        return {
            "status": "error",
            "message": str(exc),
        }

    return {
        "status": "ok",
        "transition": _transition_payload(result),
    }
