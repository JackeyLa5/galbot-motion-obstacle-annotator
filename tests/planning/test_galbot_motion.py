from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from galbot_motion_obstacle_annotator.models import Obstacle
from galbot_motion_obstacle_annotator.planning.galbot_motion import GalbotMotionPlanner
from galbot_motion_obstacle_annotator.planning.models import PlanRequest, PoseTarget


class FakeStatus:
    SUCCESS = "SUCCESS"


class FakeParameter:
    def __init__(self):
        self.is_direct_execute = True
        self.is_tool_pose = False
        self.is_blocking = False
        self.is_check_collision = False
        self.timeout_second = 0.0
        self.reference_frame = ""
        self.enable_env_collision_check = False


class LegacyParameter:
    def __init__(self):
        self.is_direct_execute = True
        self.is_blocking = False
        self.is_check_collision = False
        self.timeout_second = 0.0
        self.reference_frame = ""


class FakePose:
    def __init__(self):
        self.position = SimpleNamespace(x=0.0, y=0.0, z=0.0)
        self.orientation = SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0)


class FakePoseState:
    def __init__(self):
        self.chain_name = ""
        self.frame_id = ""
        self.reference_frame = ""
        self.pose = FakePose()


class FakeJointStates:
    def __init__(self):
        self.chain_name = ""
        self.joint_positions = []

    def set_joint_positions(self, positions):
        self.joint_positions = list(positions)


class FakeRobotStates:
    def __init__(self):
        self.whole_body_joint = []
        self.base_state = None

    def set_whole_body_joint(self, positions):
        self.whole_body_joint = list(positions)

    def set_base_state(self, pose):
        self.base_state = pose


class FakeMotion:
    def __init__(self):
        self.calls = []
        self.removed = []

    def init(self):
        self.calls.append(("init",))
        return True

    def add_obstacle(self, **kwargs):
        self.calls.append(("add_obstacle", kwargs))
        assert kwargs["reference_joint_positions"] == [0.0, 0.1, 0.2, 0.3]
        assert kwargs["reference_base_pose"] == [1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        return FakeStatus.SUCCESS

    def remove_obstacle(self, obstacle_id):
        self.removed.append(obstacle_id)
        return FakeStatus.SUCCESS

    def motion_plan(self, target, start, reference, collision_check, params):
        self.calls.append(("motion_plan", target, start, reference, collision_check, params))
        assert params.is_direct_execute is False
        assert params.is_tool_pose is True
        assert target.frame_id == "TCP"
        assert start is None
        assert reference.whole_body_joint == [0.0, 0.1, 0.2, 0.3]
        assert reference.base_state.position.x == 1.0
        assert reference.base_state.orientation.w == 1.0
        return FakeStatus.SUCCESS, {"left_arm": [[0.1, 0.2], [0.3, 0.4]]}


class FailingObstacleMotion(FakeMotion):
    def add_obstacle(self, **kwargs):
        self.calls.append(("add_obstacle", kwargs))
        return FakeStatus.SUCCESS if len(self.calls) == 2 else "FAULT"


FAKE_SDK = SimpleNamespace(
    GalbotMotion=lambda: None,
    Parameter=FakeParameter,
    PoseState=FakePoseState,
    JointStates=FakeJointStates,
    RobotStates=FakeRobotStates,
    Pose=FakePose,
    MotionStatus=FakeStatus,
)


def request() -> PlanRequest:
    return PlanRequest(
        target=PoseTarget(
            chain_name="left_arm",
            position=np.array([0.5, 0.0, 1.0]),
            orientation_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
        ),
        start_joint_positions={"left_arm": [0.1, 0.2]},
        obstacles=[Obstacle("box", scale=np.ones(3))],
        options={
            "galbot_whole_body_joint_positions": [0.0, 0.1, 0.2, 0.3],
            "galbot_base_pose": [1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        },
    )


def test_galbot_plugin_forces_plan_only_and_returns_trajectory():
    motion = FakeMotion()
    planner = GalbotMotionPlanner(sdk_module=FAKE_SDK, motion=motion)

    result = planner.plan(request())

    assert result.success
    assert result.diagnostics["direct_execute"] is False
    assert result.diagnostics["is_tool_pose"] is True
    np.testing.assert_allclose(result.trajectories["left_arm"].positions, [[0.1, 0.2], [0.3, 0.4]])
    assert len(motion.removed) == 1
    assert motion.removed[0].startswith("visualization_")


def test_galbot_plugin_refuses_missing_whole_body_reference():
    planner = GalbotMotionPlanner(sdk_module=FAKE_SDK, motion=FakeMotion())
    invalid_request = PlanRequest(
        target=request().target,
        start_joint_positions={"left_arm": [0.1, 0.2]},
    )

    result = planner.plan(invalid_request)

    assert not result.success
    assert "galbot_whole_body_joint_positions" in result.message


def test_galbot_plugin_cleans_partial_obstacle_loads():
    motion = FailingObstacleMotion()
    planner = GalbotMotionPlanner(sdk_module=FAKE_SDK, motion=motion)
    base_request = request()
    two_obstacles = PlanRequest(
        target=base_request.target,
        start_joint_positions=base_request.start_joint_positions,
        obstacles=[
            Obstacle("box_a", scale=np.ones(3)),
            Obstacle("box_b", scale=np.ones(3)),
        ],
        options=base_request.options,
    )

    result = planner.plan(two_obstacles)

    assert not result.success
    assert len(motion.removed) == 1


def test_galbot_plugin_rejects_sdk_without_environment_collision_option():
    legacy_sdk = SimpleNamespace(**vars(FAKE_SDK))
    legacy_sdk.Parameter = LegacyParameter
    planner = GalbotMotionPlanner(sdk_module=legacy_sdk, motion=FakeMotion())

    result = planner.plan(request())

    assert not result.success
    assert "does not expose enable_env_collision_check" in result.message
