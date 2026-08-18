from types import SimpleNamespace

import numpy as np

from galbot_motion_obstacle_annotator.models import Obstacle
from galbot_motion_obstacle_annotator.planning.models import PlanRequest, PoseTarget
from galbot_motion_obstacle_annotator.planning.pyroki import (
    PyrokiPlanner,
    express_obstacles_in_base_frame,
    express_world_scene_in_base_frame,
    resolve_pyroki_urdf_path,
    sample_reachable_positions_collision_aware,
)
from galbot_motion_obstacle_annotator.planning.registry import default_registry


def request(tmp_path, **option_overrides) -> PlanRequest:
    urdf_path = tmp_path / "robot.urdf"
    urdf_path.write_text("<robot name='test'/>", encoding="utf-8")
    options = {
        "pyroki_urdf_path": urdf_path,
        "pyroki_joint_names": ["joint_a", "joint_b"],
        "pyroki_start_joint_positions": {"joint_a": 0.1, "joint_b": -0.2},
        "pyroki_active_joint_names": ["joint_a"],
        "pyroki_target_link": "tool_link",
        "pyroki_timesteps": 4,
        "pyroki_dt": 0.1,
    }
    options.update(option_overrides)
    return PlanRequest(
        target=PoseTarget(
            chain_name="left_arm",
            position=np.array([0.5, 0.0, 0.8]),
            orientation_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
            reference_frame="base_link",
        ),
        start_joint_positions={"left_arm": [0.1, -0.2]},
        options=options,
    )


def test_default_registry_contains_pyroki():
    assert default_registry().get("pyroki").metadata.display_name == "PyRoki"


def test_world_scene_is_expressed_relative_to_moved_robot_base():
    base_transform = np.array(
        [
            [0.0, -1.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 2.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    target = PoseTarget(
        chain_name="left_arm",
        position=np.array([1.0, 3.0, 0.8]),
        orientation_xyzw=np.array([0.0, 0.0, np.sqrt(0.5), np.sqrt(0.5)]),
        reference_frame="world",
        frame_id="left_gripper_tcp",
    )
    obstacle = Obstacle(
        "table",
        "box",
        center=[2.0, 2.0, 0.4],
        rpy=[0.0, 0.0, np.pi / 2.0],
        scale=[1.0, 0.6, 0.8],
    )

    local_target, local_obstacles = express_world_scene_in_base_frame(
        target,
        [obstacle],
        base_transform,
    )

    assert local_target.reference_frame == "base_link"
    assert local_target.frame_id == "left_gripper_tcp"
    np.testing.assert_allclose(local_target.position, [1.0, 0.0, 0.8], atol=1e-8)
    np.testing.assert_allclose(local_target.orientation_xyzw, [0.0, 0.0, 0.0, 1.0], atol=1e-8)
    assert local_obstacles[0].target_frame == "base_link"
    np.testing.assert_allclose(local_obstacles[0].center, [0.0, -1.0, 0.4], atol=1e-8)
    np.testing.assert_allclose(local_obstacles[0].rpy, [0.0, 0.0, 0.0], atol=1e-8)
    np.testing.assert_allclose(local_obstacles[0].scale, obstacle.scale)


def test_express_obstacles_in_base_frame_matches_world_scene_conversion():
    base_transform = np.array(
        [
            [0.0, -1.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 2.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    obstacle = Obstacle(
        "table",
        "box",
        center=[2.0, 2.0, 0.4],
        rpy=[0.0, 0.0, np.pi / 2.0],
        scale=[1.0, 0.6, 0.8],
    )

    local_obstacles = express_obstacles_in_base_frame([obstacle], base_transform)

    assert local_obstacles[0].target_frame == "base_link"
    np.testing.assert_allclose(local_obstacles[0].center, [0.0, -1.0, 0.4], atol=1e-8)
    np.testing.assert_allclose(local_obstacles[0].rpy, [0.0, 0.0, 0.0], atol=1e-8)
    np.testing.assert_allclose(local_obstacles[0].scale, obstacle.scale)


def test_pyroki_reports_missing_optional_dependencies():
    def missing_dependencies():
        raise ModuleNotFoundError("No module named 'pyroki'")

    available, message = PyrokiPlanner(dependency_loader=missing_dependencies).is_available()

    assert not available
    assert "PyRoki dependencies are unavailable" in message


def test_sample_reachable_positions_collision_aware_reports_missing_dependencies():
    def missing_dependencies():
        raise ModuleNotFoundError("No module named 'pyroki'")

    try:
        sample_reachable_positions_collision_aware(
            "robot.urdf",
            tip_link="tip",
            active_joint_names=("joint1",),
            joint_limits={},
            fixed_joint_positions={},
            obstacles_base_frame=[],
            sample_count=10,
            rng=np.random.default_rng(0),
            dependency_loader=missing_dependencies,
        )
    except RuntimeError as error:
        assert "PyRoki dependencies are unavailable" in str(error)
    else:
        raise AssertionError("Expected missing PyRoki dependencies to raise RuntimeError")


def test_pyroki_requires_an_existing_urdf(tmp_path):
    planner = PyrokiPlanner(solver=lambda **kwargs: np.zeros((4, 2)), dependency_loader=dict)
    invalid_request = request(tmp_path, pyroki_urdf_path=tmp_path / "missing.urdf")

    result = planner.plan(invalid_request)

    assert not result.success
    assert "URDF does not exist" in result.message


def test_resolve_pyroki_urdf_path_prefers_fixed_base_sibling(tmp_path):
    mobile = tmp_path / "robot.urdf"
    fixed = tmp_path / "robot_fixed_base.urdf"
    mobile.write_text("<robot name='mobile'/>", encoding="utf-8")
    fixed.write_text("<robot name='fixed'/>", encoding="utf-8")

    assert resolve_pyroki_urdf_path(mobile) == fixed


def test_resolve_pyroki_urdf_path_keeps_fixed_base_input(tmp_path):
    fixed = tmp_path / "robot_fixed_base.urdf"
    fixed.write_text("<robot name='fixed'/>", encoding="utf-8")

    assert resolve_pyroki_urdf_path(fixed) == fixed


def test_pyroki_requires_explicit_joint_mapping(tmp_path):
    planner = PyrokiPlanner(solver=lambda **kwargs: np.zeros((4, 2)), dependency_loader=dict)
    invalid_request = request(tmp_path, pyroki_start_joint_positions=None)

    result = planner.plan(invalid_request)

    assert not result.success
    assert "pyroki_start_joint_positions" in result.message


def test_pyroki_returns_trajectory_timestamps_and_diagnostics(tmp_path):
    captured = {}

    def solver(**kwargs):
        captured.update(kwargs)
        return np.array(
            [[0.1, -0.2], [0.2, -0.1], [0.3, 0.0], [0.4, 0.1]],
            dtype=float,
        ), {"position_error_m": 0.004}

    planner = PyrokiPlanner(solver=solver, dependency_loader=dict)

    result = planner.plan(request(tmp_path))

    assert result.success
    assert captured["joint_names"] == ("joint_a", "joint_b")
    np.testing.assert_allclose(captured["start_positions"], [0.1, -0.2])
    np.testing.assert_allclose(result.trajectories["left_arm"].timestamps, [0.0, 0.1, 0.2, 0.3])
    assert result.diagnostics["joint_names"] == ("joint_a", "joint_b")
    assert result.diagnostics["position_error_m"] == 0.004
    assert result.diagnostics["execution_enabled"] is False


def test_pyroki_rejects_invalid_solver_shape(tmp_path):
    planner = PyrokiPlanner(solver=lambda **kwargs: np.zeros((3, 2)), dependency_loader=dict)

    result = planner.plan(request(tmp_path))

    assert not result.success
    assert "expected (4, 2)" in result.message


def test_pyroki_validates_active_joint_names(tmp_path):
    invalid_request = request(tmp_path, pyroki_active_joint_names=["missing_joint"])

    try:
        PyrokiPlanner._active_joint_names(invalid_request, ("joint_a", "joint_b"))
    except ValueError as error:
        assert "Unknown PyRoki active joints" in str(error)
    else:
        raise AssertionError("Expected unknown active joints to fail")


def test_pyroki_accepts_single_arm_or_arm_plus_leg_active_set(tmp_path):
    base = request(tmp_path)

    assert PyrokiPlanner._active_joint_names(base, ("joint_a", "joint_b")) == ("joint_a",)
    leg_request = request(tmp_path, pyroki_active_joint_names=["joint_a", "joint_b"])
    assert PyrokiPlanner._active_joint_names(leg_request, ("joint_a", "joint_b")) == (
        "joint_a",
        "joint_b",
    )


class FakeCollision:
    class Box:
        @staticmethod
        def from_extent(**kwargs):
            return "box", kwargs

    class Sphere:
        @staticmethod
        def from_center_and_radius(**kwargs):
            return "sphere", kwargs

    class Capsule:
        @staticmethod
        def from_radius_height(**kwargs):
            return "capsule", kwargs


def test_pyroki_maps_supported_obstacles_to_collision_geometry(tmp_path):
    base_request = request(tmp_path)
    obstacle_request = PlanRequest(
        target=base_request.target,
        start_joint_positions=base_request.start_joint_positions,
        obstacles=[
            Obstacle("box", "box", center=[1.0, 0.0, 0.5], scale=[0.5, 0.4, 1.0], target_frame="base_link"),
            Obstacle("sphere", "sphere", center=[0.0, 1.0, 0.5], scale=[0.2, 0.0, 0.0], target_frame="base_link"),
            Obstacle("cylinder", "cylinder", center=[0.0, 0.0, 0.5], scale=[0.1, 0.8, 0.0], target_frame="base_link"),
        ],
        options=base_request.options,
    )

    collisions = PyrokiPlanner._world_collision(
        SimpleNamespace(collision=FakeCollision), obstacle_request
    )

    assert [geometry[0] for geometry in collisions] == ["box", "sphere", "capsule"]
    np.testing.assert_allclose(collisions[0][1]["extent"], [0.5, 0.4, 1.0])
    assert collisions[1][1]["radius"] == 0.2
    assert collisions[2][1]["height"] == 0.8
