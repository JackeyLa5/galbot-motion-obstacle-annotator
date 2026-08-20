import math

import numpy as np
import pytest

from galbot_motion_obstacle_annotator.geometry import (
    compose_box_transform,
    compose_pose,
    decompose_box_transform,
    orbit_camera_frame,
    rotate_vector_around_axis,
)
from galbot_motion_obstacle_annotator.models import Obstacle


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


@pytest.mark.parametrize(
    "kwargs",
    [
        {"obstacle_id": "", "scale": np.ones(3)},
        {"obstacle_id": "box", "target_frame": "", "scale": np.ones(3)},
        {"obstacle_id": "box", "center": np.array([0.0, np.nan, 0.0])},
    ],
)
def test_obstacle_rejects_invalid_metadata_and_values(kwargs):
    with pytest.raises(ValueError):
        Obstacle(**kwargs)


def test_rotate_vector_around_axis_quarter_turn():
    rotated = rotate_vector_around_axis(np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0]), 90.0)
    np.testing.assert_allclose(rotated, [0.0, 1.0, 0.0], atol=1e-9)


def test_rotate_vector_around_axis_zero_angle_is_identity():
    vector = np.array([0.3, -1.2, 5.0])
    rotated = rotate_vector_around_axis(vector, np.array([0.0, 0.0, 1.0]), 0.0)
    np.testing.assert_allclose(rotated, vector)


def test_orbit_camera_frame_zero_angles_is_noop():
    position = np.array([5.0, 0.0, 2.0])
    focal_point = np.array([0.0, 0.0, 1.0])
    up = np.array([0.0, 0.0, 1.0])
    pivot = np.array([1.0, 1.0, 0.5])

    new_position, new_focal, new_up = orbit_camera_frame(pivot, position, focal_point, up, 0.0, 0.0)

    np.testing.assert_allclose(new_position, position, atol=1e-9)
    np.testing.assert_allclose(new_focal, focal_point, atol=1e-9)
    np.testing.assert_allclose(new_up, up, atol=1e-9)


def test_orbit_camera_frame_preserves_pivot_distance():
    position = np.array([4.0, -3.0, 1.5])
    focal_point = np.array([0.0, 0.0, 0.5])
    up = np.array([0.0, 0.0, 1.0])
    pivot = np.array([0.5, 0.5, 0.0])
    original_distance = np.linalg.norm(position - pivot)

    new_position, _new_focal, new_up = orbit_camera_frame(pivot, position, focal_point, up, 37.0, -22.0)

    assert math.isclose(float(np.linalg.norm(new_position - pivot)), float(original_distance), rel_tol=1e-9)
    assert math.isclose(float(np.linalg.norm(new_up)), 1.0, rel_tol=1e-9)


def test_orbit_camera_frame_pure_azimuth_rotates_around_world_up():
    position = np.array([2.0, 0.0, 0.0])
    focal_point = np.array([0.0, 0.0, 0.0])
    up = np.array([0.0, 0.0, 1.0])
    pivot = np.array([0.0, 0.0, 0.0])

    new_position, _new_focal, _new_up = orbit_camera_frame(pivot, position, focal_point, up, 90.0, 0.0)

    np.testing.assert_allclose(new_position, [0.0, 2.0, 0.0], atol=1e-9)
