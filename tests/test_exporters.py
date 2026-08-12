import json

import numpy as np

from galbot_motion_obstacle_annotator.exporters import export_json, export_python
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
