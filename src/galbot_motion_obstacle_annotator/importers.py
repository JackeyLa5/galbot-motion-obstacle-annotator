from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .geometry import matrix_to_rpy, quaternion_to_matrix
from .models import Obstacle


def load_json(path: str | Path) -> tuple[list[Obstacle], str]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Invalid JSON: root must be an object")
    obstacles_data = payload.get("obstacles", [])
    if not isinstance(obstacles_data, list):
        raise ValueError("Invalid JSON: obstacles must be a list")

    obstacles: list[Obstacle] = []
    for index, item in enumerate(obstacles_data, start=1):
        if not isinstance(item, dict):
            raise ValueError("Invalid JSON: each obstacle must be an object")
        pose = item.get("pose")
        scale = item.get("scale")
        if not isinstance(pose, list) or len(pose) != 7:
            raise ValueError(f"Invalid obstacle pose at index {index}")
        if not isinstance(scale, list) or len(scale) != 3:
            raise ValueError(f"Invalid obstacle scale at index {index}")

        obstacle_id = item.get("obstacle_id", f"obstacle_{index:03d}")
        obstacle_type = item.get("obstacle_type", "box")
        target_frame = item.get("target_frame", "world")
        if not isinstance(obstacle_id, str):
            raise ValueError(f"Invalid obstacle ID at index {index}: expected a string")
        if not isinstance(obstacle_type, str):
            raise ValueError(f"Invalid obstacle type at index {index}: expected a string")
        if not isinstance(target_frame, str):
            raise ValueError(f"Invalid target frame at index {index}: expected a string")

        quaternion = np.asarray(pose[3:], dtype=float)
        rotation = quaternion_to_matrix(quaternion)
        obstacle = Obstacle(
            obstacle_id=obstacle_id,
            obstacle_type=obstacle_type,
            target_frame=target_frame,
            center=np.asarray(pose[:3], dtype=float),
            rpy=matrix_to_rpy(rotation),
            scale=np.asarray(scale, dtype=float),
        )
        obstacle.validate_for_motion()
        obstacles.append(obstacle)

    source_point_cloud = payload.get("source_point_cloud", "")
    if source_point_cloud is None:
        source_point_cloud = ""
    if not isinstance(source_point_cloud, str):
        raise ValueError("Invalid JSON: source_point_cloud must be a string or null")
    return obstacles, source_point_cloud
