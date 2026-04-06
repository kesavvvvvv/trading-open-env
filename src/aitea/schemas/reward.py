"""Reward schema for AITEA."""

from __future__ import annotations

from typing import Dict

from pydantic import Field

from .common import _BaseSchema


class Reward(_BaseSchema):
    total: float = Field(default=0.0)
    normalized_score: float = Field(default=0.0, ge=0.0, le=1.0)
    components: Dict[str, float] = Field(default_factory=dict)
    penalties: Dict[str, float] = Field(default_factory=dict)
    raw_total: float = Field(default=0.0)
    clipped_total: float = Field(default=0.0)
