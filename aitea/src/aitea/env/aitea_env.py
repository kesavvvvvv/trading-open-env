"""
Main public OpenEnv-compatible environment for AITEA.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Optional

from aitea.config import AITEAConfig, get_config
from aitea.env.base_env import BaseEnv
from aitea.env.reset_manager import ResetManager
from aitea.env.state_manager import StateManager
from aitea.env.step_manager import StepManager
from aitea.schemas.action import Action
from aitea.schemas.info import Info
from aitea.schemas.observation import Observation
from aitea.schemas.reward import Reward


class AITEAEnv(BaseEnv):
    def __init__(self, task: Any | None = None, config: AITEAConfig | None = None, seed: int | None = None) -> None:
        super().__init__()
        self.config: AITEAConfig = config or get_config()
        self.seed: int = self.config.seed if seed is None else seed
        self.task: Any | None = task

        self.reset_manager = ResetManager()
        self.state_manager = StateManager()
        self.step_manager = StepManager()

        # Engines are optional here; they can be attached later or remain None.
        self.market_engine: Any | None = None
        self.treasury_engine: Any | None = None
        self.execution_engine: Any | None = None
        self.risk_engine: Any | None = None
        self.news_engine: Any | None = None
        self.regime_engine: Any | None = None
        self.multi_agent_engine: Any | None = None
        self.reward_model: Any | None = None

        self.rng: Any = None
        self.state_data: Dict[str, Any] = {}
        self.current_observation: Observation | None = None
        self.last_info: Info | None = None
        self.last_action_error: str | None = None

    def attach_engines(
        self,
        *,
        market_engine: Any | None = None,
        treasury_engine: Any | None = None,
        execution_engine: Any | None = None,
        risk_engine: Any | None = None,
        news_engine: Any | None = None,
        regime_engine: Any | None = None,
        multi_agent_engine: Any | None = None,
        reward_model: Any | None = None,
    ) -> "AITEAEnv":
        self.market_engine = market_engine
        self.treasury_engine = treasury_engine
        self.execution_engine = execution_engine
        self.risk_engine = risk_engine
        self.news_engine = news_engine
        self.regime_engine = regime_engine
        self.multi_agent_engine = multi_agent_engine
        self.reward_model = reward_model
        return self

    def reset(self, seed: int | None = None) -> Observation:
        self.ensure_open()
        state = self.reset_manager.reset_episode(self, seed=seed if seed is not None else self.seed)
        observation = self.state_manager.build_observation(self)
        self.current_observation = observation
        self.done = False
        return observation

    def step(self, action: Any) -> tuple[Observation, Reward, bool, Info]:
        self.ensure_open()
        self.ensure_not_done()
        try:
            return self.step_manager.step(self, action)
        except Exception as exc:
            self.last_action_error = str(exc)
            raise

    def state(self) -> Dict[str, Any]:
        return self.state_manager.snapshot(self)

    def close(self) -> None:
        if self.closed:
            return

        self.on_close()
        self.closed = True

    def render(self) -> Dict[str, Any]:
        """
        Lightweight debug render.
        """
        return deepcopy(self.state_data)

    @property
    def task_name(self) -> str:
        return getattr(self.task, "name", "default")
