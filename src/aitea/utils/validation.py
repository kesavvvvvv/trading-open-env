"""Runtime validation helpers for AITEA."""

from __future__ import annotations

from typing import Any, Iterable

from ..schemas import Action, Observation, Reward


def validate_action(action: Any) -> Action:
    if isinstance(action, Action):
        return action
    validator = getattr(Action, "model_validate", None)
    if callable(validator):
        return validator(action)
    return Action.parse_obj(action)


def validate_observation(observation: Any) -> Observation:
    if isinstance(observation, Observation):
        return observation
    validator = getattr(Observation, "model_validate", None)
    if callable(validator):
        return validator(observation)
    return Observation.parse_obj(observation)


def validate_reward(reward: Any) -> Reward:
    if isinstance(reward, Reward):
        return reward
    validator = getattr(Reward, "model_validate", None)
    if callable(validator):
        return validator(reward)
    return Reward.parse_obj(reward)


def validate_reward_range(value: float) -> float:
    v = float(value)
    if not -1.0 <= v <= 1.0:
        raise ValueError(f"Reward out of expected range [-1, 1]: {v}")
    return v


def validate_task_name(task_name: str, allowed: Iterable[str]) -> str:
    allowed_set = set(allowed)
    if task_name not in allowed_set:
        raise ValueError(f"Unknown task '{task_name}'. Allowed: {sorted(allowed_set)}")
    return task_name


def validate_non_empty_mapping(mapping: Any, name: str = "mapping") -> None:
    if mapping is None:
        raise ValueError(f"{name} cannot be None.")
    if isinstance(mapping, dict) and len(mapping) == 0:
        raise ValueError(f"{name} cannot be empty.")
