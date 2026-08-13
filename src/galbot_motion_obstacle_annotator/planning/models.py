from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from ..models import Obstacle

FloatArray = NDArray[np.float64]


def _finite_vector(values: Sequence[float], length: int, name: str) -> FloatArray:
    result = np.asarray(values, dtype=float)
    if result.shape != (length,):
        raise ValueError(f"{name} must contain {length} values")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain only finite values")
    return result


@dataclass(frozen=True)
class PoseTarget:
    chain_name: str
    position: FloatArray
    orientation_xyzw: FloatArray
    reference_frame: str = "world"
    frame_id: str = "EndEffector"

    def __post_init__(self) -> None:
        if not self.chain_name.strip():
            raise ValueError("chain_name must be non-empty")
        if not self.reference_frame.strip():
            raise ValueError("reference_frame must be non-empty")
        if not self.frame_id.strip():
            raise ValueError("frame_id must be non-empty")
        position = _finite_vector(self.position, 3, "position")
        orientation = _finite_vector(self.orientation_xyzw, 4, "orientation_xyzw")
        norm = float(np.linalg.norm(orientation))
        if norm < 1e-12:
            raise ValueError("orientation_xyzw must have non-zero norm")
        object.__setattr__(self, "position", position)
        object.__setattr__(self, "orientation_xyzw", orientation / norm)


@dataclass(frozen=True)
class JointTrajectory:
    chain_name: str
    positions: FloatArray
    timestamps: FloatArray | None = None

    def __post_init__(self) -> None:
        positions = np.asarray(self.positions, dtype=float)
        if positions.ndim != 2 or positions.shape[0] == 0 or positions.shape[1] == 0:
            raise ValueError("positions must be a non-empty two-dimensional array")
        if not np.isfinite(positions).all():
            raise ValueError("positions must contain only finite values")
        object.__setattr__(self, "positions", positions)
        if self.timestamps is None:
            return
        timestamps = np.asarray(self.timestamps, dtype=float)
        if timestamps.shape != (positions.shape[0],):
            raise ValueError("timestamps must match the trajectory frame count")
        if not np.isfinite(timestamps).all() or np.any(np.diff(timestamps) < 0.0):
            raise ValueError("timestamps must be finite and non-decreasing")
        object.__setattr__(self, "timestamps", timestamps)


@dataclass(frozen=True)
class PlanRequest:
    target: PoseTarget
    start_joint_positions: Mapping[str, Sequence[float]]
    obstacles: Sequence[Obstacle] = ()
    collision_check: bool = True
    environment_collision_check: bool = True
    timeout_seconds: float = 20.0
    options: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.start_joint_positions:
            raise ValueError("start_joint_positions is required for visualization-only planning")
        normalized: dict[str, tuple[float, ...]] = {}
        for chain_name, values in self.start_joint_positions.items():
            if not chain_name.strip():
                raise ValueError("start_joint_positions contains an empty chain name")
            array = np.asarray(values, dtype=float)
            if array.ndim != 1 or not len(array) or not np.isfinite(array).all():
                raise ValueError(f"Invalid start joint positions for {chain_name}")
            normalized[chain_name] = tuple(float(value) for value in array)
        if not np.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0.0:
            raise ValueError("timeout_seconds must be greater than zero")
        object.__setattr__(self, "start_joint_positions", MappingProxyType(normalized))
        object.__setattr__(self, "obstacles", tuple(self.obstacles))
        object.__setattr__(self, "options", MappingProxyType(dict(self.options)))


@dataclass(frozen=True)
class PlanResult:
    planner_id: str
    success: bool
    status: str
    trajectories: Mapping[str, JointTrajectory] = field(default_factory=dict)
    message: str = ""
    planning_seconds: float | None = None
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "trajectories", MappingProxyType(dict(self.trajectories)))
        object.__setattr__(self, "diagnostics", MappingProxyType(dict(self.diagnostics)))
