"""Base task definitions for AITEA."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from ..config import AITEAConfig, get_config
from ..env.state_manager import AITEAState


@dataclass
class TaskBase(ABC):
    """
    Base class for AITEA tasks.

    Tasks define the scenario profile used by reset logic and the success
    conditions used by graders or reporting layers.
    """
    task_name: str
    difficulty: str
    kind: str
    description: str
    grader_name: str
    horizon: int
    config: AITEAConfig = field(default_factory=get_config)

    def profile(self) -> Dict[str, Any]:
        """Return the task profile consumed by the environment reset logic."""
        return {
            "name": self.task_name,
            "kind": self.kind,
            "difficulty": self.difficulty,
            "description": self.description,
            "grader_name": self.grader_name,
            "horizon": self.horizon,
        }

    def metadata(self) -> Dict[str, Any]:
        """Return human-readable task metadata."""
        return {
            "task_name": self.task_name,
            "difficulty": self.difficulty,
            "kind": self.kind,
            "description": self.description,
            "grader_name": self.grader_name,
            "horizon": self.horizon,
        }

    def initialize_metrics(self) -> Dict[str, Any]:
        """Return task-specific initial metrics."""
        return {"kind": self.kind}

    @abstractmethod
    def success(self, state: AITEAState) -> bool:
        """Determine whether the task has been successfully completed."""
        raise NotImplementedError

    def summarize(self) -> str:
        return f"{self.task_name} ({self.difficulty})"

    def bind(self, config: Optional[AITEAConfig] = None) -> "TaskBase":
        """Return a copy-like task bound to a config if needed."""
        if config is not None:
            self.config = config
        return self
