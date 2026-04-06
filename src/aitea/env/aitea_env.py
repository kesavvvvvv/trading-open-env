"""Main AITEA environment implementation."""

from __future__ import annotations

from typing import Any, Optional

from ..config import AITEAConfig, get_config
from ..schemas import Observation
from .base_env import BaseEnv
from .reset_manager import ResetManager
from .state_manager import AITEAState, StateManager, Transition
from .step_manager import StepManager


class AITEAEnv(BaseEnv):
    """OpenEnv-compatible institutional trading simulation."""

    def __init__(
        self,
        task_name: Optional[str] = None,
        config: AITEAConfig | None = None,
        episode_id: Optional[str] = None,
        auto_reset: bool = True,
        **_: Any,
    ) -> None:
        super().__init__()
        self.config = config or get_config()
        self.task_name = task_name or self.config.default_task
        self.reset_manager = ResetManager(self.config)
        self.state_manager = StateManager(self.config)
        self.step_manager = StepManager(self.config)
        self.state_data: AITEAState | None = None
        self.current_transition: Transition | None = None

        if auto_reset:
            self.reset(task_name=self.task_name, episode_id=episode_id)

    @classmethod
    async def from_docker_image(cls, image_name: str | None = None, **kwargs: Any) -> "AITEAEnv":
        """Compatibility helper for baseline runners that expect an async constructor."""
        _ = image_name
        return cls(**kwargs)

    def reset(self, task_name: Optional[str] = None, episode_id: Optional[str] = None) -> Transition:
        self._ensure_open()
        chosen_task = task_name or self.task_name or self.config.default_task
        self.task_name = chosen_task
        self.state_data, transition = self.reset_manager.reset(chosen_task, episode_id=episode_id)
        self.current_transition = transition
        return transition

    def step(self, action: Any) -> Transition:
        self._ensure_open()
        if self.state_data is None:
            self.reset(task_name=self.task_name)
        assert self.state_data is not None
        transition = self.step_manager.step(self.state_data, action)
        self.current_transition = transition
        return transition

    def state(self) -> Observation:
        self._ensure_open()
        if self.state_data is None:
            self.reset(task_name=self.task_name)
        assert self.state_data is not None
        return self.state_manager.observation(self.state_data)

    def close(self) -> None:
        self._closed = True

    async def areset(self, task_name: Optional[str] = None, episode_id: Optional[str] = None) -> Transition:
        return self.reset(task_name=task_name, episode_id=episode_id)

    async def astep(self, action: Any) -> Transition:
        return self.step(action)

    async def astate(self) -> Observation:
        return self.state()

    async def aclose(self) -> None:
        self.close()
