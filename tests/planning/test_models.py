import numpy as np
import pytest

from galbot_motion_obstacle_annotator.planning.models import JointTrajectory, PlanRequest, PoseTarget


def target() -> PoseTarget:
    return PoseTarget(
        chain_name="left_arm",
        position=np.array([0.5, 0.1, 1.0]),
        orientation_xyzw=np.array([0.0, 0.0, 0.0, 2.0]),
    )


def test_pose_target_normalizes_quaternion():
    np.testing.assert_allclose(target().orientation_xyzw, [0.0, 0.0, 0.0, 1.0])


def test_plan_request_requires_explicit_start_state():
    with pytest.raises(ValueError, match="start_joint_positions is required"):
        PlanRequest(target=target(), start_joint_positions={})


def test_joint_trajectory_validates_timestamps():
    with pytest.raises(ValueError, match="non-decreasing"):
        JointTrajectory(
            chain_name="left_arm",
            positions=np.zeros((2, 7)),
            timestamps=np.array([1.0, 0.5]),
        )
