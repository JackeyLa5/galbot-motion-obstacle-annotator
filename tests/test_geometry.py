import math

import numpy as np

from galbot_motion_obstacle_annotator.geometry import (
    compose_box_transform,
    compose_pose,
    decompose_box_transform,
)


def test_box_transform_round_trip():
    center = np.array([1.0, -2.0, 0.75])
    rpy = np.array([0.2, -0.3, 0.7])
    size = np.array([1.2, 0.8, 2.1])

    matrix = compose_box_transform(center, rpy, size)
    actual_center, actual_rpy, actual_size, quaternion = decompose_box_transform(matrix)

    np.testing.assert_allclose(actual_center, center)
    np.testing.assert_allclose(actual_rpy, rpy)
    np.testing.assert_allclose(actual_size, size)
    assert math.isclose(float(np.linalg.norm(quaternion)), 1.0)


def test_compose_identity_pose():
    matrix = compose_pose(np.array([1.0, 2.0, 3.0]), np.array([0.0, 0.0, 0.0, 1.0]))

    np.testing.assert_allclose(matrix[:3, :3], np.eye(3))
    np.testing.assert_allclose(matrix[:3, 3], [1.0, 2.0, 3.0])
