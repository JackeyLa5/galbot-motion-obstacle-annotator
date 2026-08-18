from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np

LEG_JOINT_NAMES = tuple(f"leg_joint{index}" for index in range(1, 6))
HEAD_JOINT_NAMES = tuple(f"head_joint{index}" for index in range(1, 3))
LEFT_ARM_JOINT_NAMES = tuple(f"left_arm_joint{index}" for index in range(1, 8))
RIGHT_ARM_JOINT_NAMES = tuple(f"right_arm_joint{index}" for index in range(1, 8))
MOTION_PLANNING_JOINT_NAMES = LEG_JOINT_NAMES + LEFT_ARM_JOINT_NAMES + RIGHT_ARM_JOINT_NAMES
SINGLE_ARM_PLANNING_JOINT_NAMES = {
    "left_arm": LEFT_ARM_JOINT_NAMES,
    "right_arm": RIGHT_ARM_JOINT_NAMES,
}
GALBOT_REFERENCE_JOINT_NAMES = (
    LEG_JOINT_NAMES + HEAD_JOINT_NAMES + LEFT_ARM_JOINT_NAMES + RIGHT_ARM_JOINT_NAMES
)
ARM_JOINT_NAMES = {
    "left_arm": LEFT_ARM_JOINT_NAMES,
    "right_arm": RIGHT_ARM_JOINT_NAMES,
}

INITIAL_ROBOT_JOINT_POSITIONS = {
    "leg_joint1": 0.6,
    "leg_joint2": 1.8,
    "leg_joint3": 1.2,
    "leg_joint4": 0.0,
    "leg_joint5": 0.0,
    "head_joint1": 0.0,
    "head_joint2": 0.0,
    "left_arm_joint1": 1.9,
    "left_arm_joint2": -1.5,
    "left_arm_joint3": -0.6,
    "left_arm_joint4": -2.1,
    "left_arm_joint5": 0.0,
    "left_arm_joint6": -0.25,
    "left_arm_joint7": 0.1,
    "right_arm_joint1": -1.9,
    "right_arm_joint2": 1.5,
    "right_arm_joint3": 0.6,
    "right_arm_joint4": 2.1,
    "right_arm_joint5": 0.0,
    "right_arm_joint6": 0.25,
    "right_arm_joint7": -0.1,
}


@dataclass
class RobotEnvironmentState:
    joint_positions: dict[str, float] = field(default_factory=dict)

    def replace(self, values: Mapping[str, float]) -> None:
        normalized: dict[str, float] = {}
        for name, value in values.items():
            numeric = float(value)
            if not name.strip() or not np.isfinite(numeric):
                raise ValueError(f"Invalid robot joint state: {name!r}={value!r}")
            normalized[str(name)] = numeric
        self.joint_positions = normalized

    def update(self, values: Mapping[str, float]) -> None:
        updated = dict(self.joint_positions)
        for name, value in values.items():
            numeric = float(value)
            if not name.strip() or not np.isfinite(numeric):
                raise ValueError(f"Invalid robot joint state: {name!r}={value!r}")
            updated[str(name)] = numeric
        self.joint_positions = updated

    def initialize_missing(self, joint_names: tuple[str, ...], value: float = 0.0) -> None:
        numeric = float(value)
        if not np.isfinite(numeric):
            raise ValueError("Initial robot joint value must be finite")
        updated = dict(self.joint_positions)
        for name in joint_names:
            updated.setdefault(name, numeric)
        self.joint_positions = updated

    def positions_for(self, joint_names: tuple[str, ...]) -> list[float]:
        missing = [name for name in joint_names if name not in self.joint_positions]
        if missing:
            raise ValueError(f"Current environment is missing robot joints: {missing}")
        return [self.joint_positions[name] for name in joint_names]

    def mapping_for(self, joint_names: tuple[str, ...]) -> dict[str, float]:
        return dict(zip(joint_names, self.positions_for(joint_names)))
