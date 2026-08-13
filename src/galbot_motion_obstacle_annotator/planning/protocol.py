from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from .models import PlanRequest, PlanResult


@dataclass(frozen=True)
class PlannerMetadata:
    planner_id: str
    display_name: str
    description: str
    supports_environment_obstacles: bool = False
    supports_cartesian_targets: bool = True
    returns_timestamps: bool = False


@runtime_checkable
class MotionPlannerPlugin(Protocol):
    @property
    def metadata(self) -> PlannerMetadata:
        ...

    def is_available(self) -> tuple[bool, str]:
        ...

    def plan(self, request: PlanRequest) -> PlanResult:
        ...
