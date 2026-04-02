"""
Reward schema for AITEA.
"""

from __future__ import annotations

from typing import Dict, Optional

from pydantic import BaseModel, Field


class Reward(BaseModel):
    value: float

    # Component breakdown of the reward signal
    components: Dict[str, float] = Field(default_factory=dict)

    # Penalties applied on this step
    penalties: Dict[str, float] = Field(default_factory=dict)

    # Optional normalized score in [0, 1]
    normalized_score: Optional[float] = None

    class Config:
        extra = "forbid"
