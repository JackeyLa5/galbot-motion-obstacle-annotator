from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .geometry import matrix_to_rpy, quaternion_to_matrix
from .models import Obstacle


def load_json(path: str | Path) -> tuple[list[Obstacle], str]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    obstacles_data = payload.get("obstacles", [])
    if not isinstance(obstacles_data, list):
        raise ValueError("Invalid JSON: obstacles must be a list")

    obstacles: list[Obstacle] = []
    for item in obstacles_data:
        if not isinstance(item, dict):
            raise ValueError("Invalid JSON: each obstacle must be an object")
        pose = item.get("pose")
        scale = item.get("scale")
        if not isinstance(pose, list) or len(pose) != 7:
            raise ValueError(f"Invalid obstacle pose for {item.get('obstacle_id', '<unknown>')}")
        if not isinstance(scale, list) or len(scale) != 3:
            raise ValueError(f"Invalid obstacle scale for {item.get('obstacle_id', '<unknown>')}")

        quaternion = np.asarray(pose[3:], dtype=float)
        rotation = quaternion_to_matrix(quaternion)
        obstacles.append(
            Obstacle(
                obstacle_id=str(item.get("obstacle_id", f"obstacle_{len(obstacles) + 1:03d}")),
                obstacle_type=str(item.get("obstacle_type", "box")),
                target_frame=str(item.get("target_frame", "world")),
                center=np.asarray(pose[:3], dtype=float),
                rpy=matrix_to_rpy(rotation),
                scale=np.asarray(scale, dtype=float),
            )
        )

    source_point_cloud = payload.get("source_point_cloud", "")
    if source_point_cloud is None:
        source_point_cloud = ""
    return obstacles, str(source_point_cloud)
