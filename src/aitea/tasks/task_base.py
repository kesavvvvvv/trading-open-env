"""Base task definition for AITEA."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class TaskBase:
    name: str = "generic_task"
    kind: str = "generic"
    difficulty: str = "easy"
    horizon: int = 25
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def task_profile(self) -> Dict[str, Any]:
        profile = {
            "name": self.name,
            "kind": self.kind,
            "difficulty": self.difficulty,
            "horizon": self.horizon,
        }
        profile.update(self.metadata)
        return profile

    def initial_metrics(self) -> Dict[str, float]:
        return {
            "horizon": float(self.horizon),
            "progress": 0.0,
        }

    def success_threshold(self) -> float:
        return 0.85

    def failure_threshold(self) -> float:
        return 0.25

    def build(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "profile": self.task_profile(),
            "metrics": self.initial_metrics(),
            "description": self.description,
        }
