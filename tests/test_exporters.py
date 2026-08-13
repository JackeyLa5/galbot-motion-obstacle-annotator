import json
import re

import numpy as np
import pytest

from galbot_motion_obstacle_annotator.exporters import export_json, export_python
from galbot_motion_obstacle_annotator.importers import load_json
from galbot_motion_obstacle_annotator.models import Obstacle


def test_export_motion_files(tmp_path):
    obstacle = Obstacle(
        obstacle_id="table",
        center=np.array([1.0, 2.0, 3.0]),
        scale=np.array([1.2, 0.8, 0.7]),
    )

    json_path = tmp_path / "obstacles.json"
    python_path = tmp_path / "obstacles.py"
    export_json(json_path, [obstacle], "global_cloud_cleaned.pcd")
    export_python(python_path, [obstacle])

    payload = json.loads(json_path.read_text())
    assert payload["obstacles"][0]["pose"] == [1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0]
    assert payload["obstacles"][0]["obstacle_type"] == "box"
    assert "motion.add_obstacle" in python_path.read_text()
    assert 'target_frame=obstacle["target_frame"]' in python_path.read_text()
    compile(python_path.read_text(), str(python_path), "exec")


def test_all_sdk_obstacle_types_serialize():
    for obstacle_type in ("box", "sphere", "cylinder"):
        obstacle = Obstacle(obstacle_id=obstacle_type, obstacle_type=obstacle_type)
        assert obstacle.to_motion_dict()["obstacle_type"] == obstacle_type


def test_load_json_round_trip(tmp_path):
    json_path = tmp_path / "obstacles.json"
    original = Obstacle(
        obstacle_id="bin",
        obstacle_type="cylinder",
        center=np.array([1.0, 2.0, 3.0]),
        rpy=np.array([0.1, 0.2, 0.3]),
        scale=np.array([0.4, 0.8, 0.0]),
        target_frame="world",
    )
    export_json(json_path, [original], "cloud.pcd")

    obstacles, source_point_cloud = load_json(json_path)
    assert source_point_cloud == "cloud.pcd"
    assert len(obstacles) == 1
    loaded = obstacles[0]
    assert loaded.obstacle_id == "bin"
    assert loaded.obstacle_type == "cylinder"
    np.testing.assert_allclose(loaded.center, original.center)
    np.testing.assert_allclose(loaded.scale, original.scale)


@pytest.mark.parametrize(
    ("obstacle", "message"),
    [
        (
            Obstacle(obstacle_id="flat_box", scale=np.array([1.0, 0.0, 1.0])),
            "Box scale values must be greater than zero",
        ),
        (
            Obstacle(
                obstacle_id="sphere",
                obstacle_type="sphere",
                scale=np.array([0.5, 0.1, 0.0]),
            ),
            "Sphere scale must be [radius, 0, 0] with a positive radius",
        ),
        (
            Obstacle(
                obstacle_id="cylinder",
                obstacle_type="cylinder",
                scale=np.array([0.5, 1.0, 0.1]),
            ),
            "Cylinder scale must be [radius, height, 0] with positive radius and height",
        ),
    ],
)
def test_export_rejects_invalid_motion_scale(tmp_path, obstacle, message):
    with pytest.raises(ValueError, match=re.escape(message)):
        export_json(tmp_path / "obstacles.json", [obstacle])


@pytest.mark.parametrize(
    "payload",
    [
        [],
        {"obstacles": "not a list"},
        {
            "obstacles": [
                {
                    "obstacle_id": "table",
                    "obstacle_type": "box",
                    "target_frame": "world",
                    "pose": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                    "scale": [1.0, 0.0, 1.0],
                }
            ]
        },
        {
            "obstacles": [
                {
                    "obstacle_id": 1,
                    "obstacle_type": "box",
                    "target_frame": "world",
                    "pose": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                    "scale": [1.0, 1.0, 1.0],
                }
            ]
        },
    ],
)
def test_load_json_rejects_invalid_payload(tmp_path, payload):
    json_path = tmp_path / "invalid.json"
    json_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError):
        load_json(json_path)
