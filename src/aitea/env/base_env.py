"""
Base environment contract for AITEA.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

from aitea.schemas.action import Action
from aitea.schemas.info import Info
from aitea.schemas.observation import Observation
from aitea.schemas.reward import Reward


class BaseEnv(ABC):
    """
    Abstract environment contract.
    Every environment must expose reset, step, state, and close.
    """

    def __init__(self) -> None:
        self.closed: bool = False
        self.done: bool = False
        self.step_count: int = 0
        self.episode_reward: float = 0.0

    @abstractmethod
    def reset(self, *args: Any, **kwargs: Any) -> Observation:
        raise NotImplementedError

    @abstractmethod
    def step(self, action: Action) -> tuple[Observation, Reward, bool, Info]:
        raise NotImplementedError

    @abstractmethod
    def state(self) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        raise NotImplementedError

    def on_reset(self) -> None:
        """
        Optional lifecycle hook.
        """
        return None

    def on_step(self) -> None:
        """
        Optional lifecycle hook.
        """
        return None

    def on_close(self) -> None:
        """
        Optional lifecycle hook.
        """
        return None

    def validate_action(self, action: Any) -> Action:
        """
        Validate and coerce a raw action into the typed Action schema.
        """
        if isinstance(action, Action):
            return action

        if isinstance(action, dict):
            if hasattr(Action, "model_validate"):
                return Action.model_validate(action)  # pydantic v2
            return Action.parse_obj(action)  # pydantic v1

        raise TypeError(f"Invalid action type: {type(action)!r}")

    def validate_observation(self, observation: Any) -> Observation:
        if isinstance(observation, Observation):
            return observation

        if isinstance(observation, dict):
            if hasattr(Observation, "model_validate"):
                return Observation.model_validate(observation)
            return Observation.parse_obj(observation)

        raise TypeError(f"Invalid observation type: {type(observation)!r}")

    def validate_reward(self, reward: Any) -> Reward:
        if isinstance(reward, Reward):
            return reward

        if isinstance(reward, dict):
            if hasattr(Reward, "model_validate"):
                return Reward.model_validate(reward)
            return Reward.parse_obj(reward)

        raise TypeError(f"Invalid reward type: {type(reward)!r}")

    def validate_info(self, info: Any) -> Info:
        if isinstance(info, Info):
            return info

        if isinstance(info, dict):
            if hasattr(Info, "model_validate"):
                return Info.model_validate(info)
            return Info.parse_obj(info)

        raise TypeError(f"Invalid info type: {type(info)!r}")

    def ensure_open(self) -> None:
        if self.closed:
            raise RuntimeError("Environment is closed.")

    def ensure_not_done(self) -> None:
        if self.done:
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")
