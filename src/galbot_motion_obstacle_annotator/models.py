from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from .geometry import compose_box_transform, decompose_box_transform, matrix_to_quaternion, rpy_to_matrix

FloatArray = NDArray[np.float64]
SUPPORTED_OBSTACLE_TYPES = ("box", "sphere", "cylinder")


def _vector(values, length: int) -> FloatArray:
    result = np.asarray(values, dtype=float)
    if result.shape != (length,):
        raise ValueError(f"Expected {length} values, got {result.shape}")
    return result


@dataclass
class Obstacle:
    obstacle_id: str
    obstacle_type: str = "box"
    center: FloatArray = field(default_factory=lambda: np.zeros(3, dtype=float))
    rpy: FloatArray = field(default_factory=lambda: np.zeros(3, dtype=float))
    scale: FloatArray = field(default_factory=lambda: np.ones(3, dtype=float))
    target_frame: str = "world"

    def __post_init__(self) -> None:
        if self.obstacle_type not in SUPPORTED_OBSTACLE_TYPES:
            raise ValueError(f"Unsupported obstacle type: {self.obstacle_type}")
        self.center = _vector(self.center, 3)
        self.rpy = _vector(self.rpy, 3)
        self.scale = _vector(self.scale, 3)
        if np.any(self.scale < 0.0):
            raise ValueError("Obstacle scale values cannot be negative")

    @property
    def transform(self) -> FloatArray:
        return compose_box_transform(self.center, self.rpy, self.display_size)

    @property
    def display_size(self) -> FloatArray:
        if self.obstacle_type == "sphere":
            diameter = max(float(self.scale[0]) * 2.0, 0.01)
            return np.array([diameter, diameter, diameter])
        if self.obstacle_type == "cylinder":
            diameter = max(float(self.scale[0]) * 2.0, 0.01)
            return np.array([diameter, diameter, max(float(self.scale[1]), 0.01)])
        return np.maximum(self.scale, 0.01)

    def update_box_from_transform(self, matrix: FloatArray) -> None:
        center, rpy, size, _ = decompose_box_transform(matrix)
        if self.obstacle_type == "sphere":
            radius = max(float(np.max(size)) / 2.0, 0.01)
            self.center = center
            self.rpy = np.zeros(3, dtype=float)
            self.scale = np.array([radius, 0.0, 0.0], dtype=float)
            return
        if self.obstacle_type == "cylinder":
            radius = max(float(np.max(size[:2])) / 2.0, 0.01)
            height = max(float(size[2]), 0.01)
            self.center = center
            self.rpy = rpy
            self.scale = np.array([radius, height, 0.0], dtype=float)
            return
        self.center = center
        self.rpy = rpy
        self.scale = size

    def to_motion_dict(self) -> dict[str, object]:
        quaternion = matrix_to_quaternion(rpy_to_matrix(*self.rpy))
        return {
            "obstacle_id": self.obstacle_id,
            "obstacle_type": self.obstacle_type,
            "target_frame": self.target_frame,
            "pose": [*self.center.tolist(), *quaternion.tolist()],
            "scale": self.scale.tolist(),
        }


ObstacleBox = Obstacle
