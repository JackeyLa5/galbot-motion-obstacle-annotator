from pathlib import Path

import numpy as np

from galbot_motion_obstacle_annotator.robot_model import (
    load_tcp_gripper_visuals,
    load_urdf_actuated_joint_names,
    load_urdf_joint_limits,
    load_urdf_link_transforms,
    load_urdf_visuals,
    sample_reachable_positions,
    sample_until_valid,
)


def test_revolute_joint_position_is_applied(tmp_path: Path):
    urdf_dir = tmp_path / "description" / "urdf"
    urdf_dir.mkdir(parents=True)
    urdf_path = urdf_dir / "robot.urdf"
    urdf_path.write_text(
        """
<robot name="test">
  <link name="base"/>
  <joint name="joint1" type="revolute">
    <origin xyz="1 0 0"/>
    <parent link="base"/>
    <child link="child"/>
    <axis xyz="0 0 1"/>
  </joint>
  <link name="child">
    <visual>
      <origin xyz="1 0 0"/>
      <geometry><box size="1 1 1"/></geometry>
    </visual>
  </link>
</robot>
""",
        encoding="utf-8",
    )

    visual = load_urdf_visuals(urdf_path, {"joint1": np.pi / 2.0})[0]

    np.testing.assert_allclose(visual.transform[:3, 3], [1.0, 1.0, 0.0], atol=1e-10)


def test_gripper_visuals_are_expressed_in_tcp_frame(tmp_path: Path):
    urdf_dir = tmp_path / "description" / "urdf"
    urdf_dir.mkdir(parents=True)
    urdf_path = urdf_dir / "robot.urdf"
    urdf_path.write_text(
        """
<robot name="test">
  <link name="base"/>
  <joint name="mount_joint" type="fixed">
    <origin xyz="1 0 0"/>
    <parent link="base"/>
    <child link="mount"/>
  </joint>
  <link name="mount"/>
  <joint name="gripper_joint" type="fixed">
    <origin xyz="0.2 0 0"/>
    <parent link="mount"/>
    <child link="gripper"/>
  </joint>
  <link name="gripper">
    <visual>
      <origin xyz="0.1 0 0"/>
      <geometry><box size="0.1 0.1 0.1"/></geometry>
    </visual>
  </link>
  <joint name="tcp_joint" type="fixed">
    <origin xyz="0.4 0 0"/>
    <parent link="mount"/>
    <child link="tcp"/>
  </joint>
  <link name="tcp"/>
</robot>
""",
        encoding="utf-8",
    )

    visuals = load_tcp_gripper_visuals(urdf_path, "mount", "tcp")

    assert len(visuals) == 1
    np.testing.assert_allclose(visuals[0].transform[:3, 3], [-0.1, 0.0, 0.0])


def test_tcp_preview_can_include_shared_wrist_flange_subtree(tmp_path: Path):
    urdf_dir = tmp_path / "description" / "urdf"
    urdf_dir.mkdir(parents=True)
    urdf_path = urdf_dir / "robot.urdf"
    urdf_path.write_text(
        """
<robot name="test">
  <link name="base"/>
  <joint name="mount_joint" type="fixed"><parent link="base"/><child link="mount"/></joint>
  <link name="mount"><visual><geometry><box size="1 1 1"/></geometry></visual></link>
  <joint name="tcp_joint" type="fixed"><origin xyz="1 0 0"/><parent link="mount"/><child link="tcp"/></joint>
  <link name="tcp"/>
  <joint name="wrist_joint" type="fixed"><origin xyz="0 1 0"/><parent link="base"/><child link="wrist"/></joint>
  <link name="wrist"><visual><geometry><box size="1 1 1"/></geometry></visual></link>
  <joint name="camera_joint" type="fixed"><origin xyz="0 1 0"/><parent link="wrist"/><child link="camera"/></joint>
  <link name="camera"><visual><geometry><box size="1 1 1"/></geometry></visual></link>
</robot>
""",
        encoding="utf-8",
    )

    visuals = load_tcp_gripper_visuals(
        urdf_path, "mount", "tcp", auxiliary_root_links=("wrist",)
    )

    assert {visual.link_name for visual in visuals} == {"mount", "wrist", "camera"}
    wrist = next(visual for visual in visuals if visual.link_name == "wrist")
    camera = next(visual for visual in visuals if visual.link_name == "camera")
    np.testing.assert_allclose(wrist.transform[:3, 3], [-1.0, 1.0, 0.0])
    np.testing.assert_allclose(camera.transform[:3, 3], [-1.0, 2.0, 0.0])


def test_load_actuated_joint_names_excludes_fixed_and_mimic(tmp_path: Path):
    urdf_path = tmp_path / "robot.urdf"
    urdf_path.write_text(
        """
<robot name="test">
  <joint name="arm" type="revolute"/>
  <joint name="fixed" type="fixed"/>
  <joint name="finger" type="revolute"><mimic joint="arm"/></joint>
  <joint name="slide" type="prismatic"/>
</robot>
""",
        encoding="utf-8",
    )

    assert load_urdf_actuated_joint_names(urdf_path) == ("arm", "slide")


def test_load_urdf_joint_limits_uses_explicit_and_safe_fallbacks(tmp_path: Path):
    urdf_path = tmp_path / "robot.urdf"
    urdf_path.write_text(
        """
<robot name="test">
  <joint name="arm" type="revolute"><limit lower="-1.2" upper="2.3"/></joint>
  <joint name="wheel" type="continuous"/>
  <joint name="slide" type="prismatic"/>
  <joint name="mimic_arm" type="revolute"><mimic joint="arm"/></joint>
</robot>
""",
        encoding="utf-8",
    )

    limits = load_urdf_joint_limits(urdf_path)

    assert limits["arm"] == (-1.2, 2.3)
    assert limits["wheel"] == (-np.pi, np.pi)
    assert limits["slide"] == (-1.0, 1.0)
    assert "mimic_arm" not in limits


def test_gripper_tcp_must_be_below_mount(tmp_path: Path):
    urdf_dir = tmp_path / "description" / "urdf"
    urdf_dir.mkdir(parents=True)
    urdf_path = urdf_dir / "robot.urdf"
    urdf_path.write_text(
        """
<robot name="test">
  <link name="base"/>
  <joint name="mount_joint" type="fixed">
    <parent link="base"/><child link="mount"/>
  </joint>
  <link name="mount"/>
  <joint name="tcp_joint" type="fixed">
    <parent link="base"/><child link="tcp"/>
  </joint>
  <link name="tcp"/>
</robot>
""",
        encoding="utf-8",
    )

    try:
        load_tcp_gripper_visuals(urdf_path, "mount", "tcp")
    except ValueError as error:
        assert "is not below mount link" in str(error)
    else:
        raise AssertionError("Expected unrelated TCP link to be rejected")


def test_sample_reachable_positions_bounds_and_variation(tmp_path: Path):
    urdf_dir = tmp_path / "description" / "urdf"
    urdf_dir.mkdir(parents=True)
    urdf_path = urdf_dir / "robot.urdf"
    urdf_path.write_text(TWO_LINK_ARM_URDF, encoding="utf-8")

    points, active_joint_values = sample_reachable_positions(
        urdf_path,
        tip_link="tip",
        active_joint_names=("joint1", "joint2"),
        joint_limits={"joint1": (-np.pi, np.pi), "joint2": (-np.pi, np.pi)},
        fixed_joint_positions={},
        sample_count=300,
        rng=np.random.default_rng(1),
    )

    assert points.shape == (300, 3)
    assert active_joint_values.shape == (300, 2)
    radii = np.linalg.norm(points[:, :2], axis=1)
    # Two 1m links: max reach is bounded by the triangle inequality (L1 + L2).
    assert np.all(radii <= 2.0 + 1e-9)
    assert np.all(points[:, 2] == 0.0)
    # Both joints are actually varied, not stuck at a single configuration.
    assert radii.std() > 0.1
    assert np.all(active_joint_values >= -np.pi) and np.all(active_joint_values <= np.pi)
    assert active_joint_values[:, 0].std() > 0.1 and active_joint_values[:, 1].std() > 0.1


TWO_LINK_ARM_URDF = """
<robot name="test">
  <link name="base"/>
  <joint name="joint1" type="revolute">
    <origin xyz="0 0 0"/>
    <parent link="base"/>
    <child link="shoulder"/>
    <axis xyz="0 0 1"/>
  </joint>
  <link name="shoulder"/>
  <joint name="upper_arm" type="fixed">
    <origin xyz="1 0 0"/>
    <parent link="shoulder"/>
    <child link="elbow"/>
  </joint>
  <link name="elbow"/>
  <joint name="joint2" type="revolute">
    <origin xyz="0 0 0"/>
    <parent link="elbow"/>
    <child link="forearm"/>
    <axis xyz="0 0 1"/>
  </joint>
  <link name="forearm"/>
  <joint name="wrist_to_tip" type="fixed">
    <origin xyz="1 0 0"/>
    <parent link="forearm"/>
    <child link="tip"/>
  </joint>
  <link name="tip"/>
</robot>
"""


def test_sample_reachable_positions_joint_values_reproduce_positions(tmp_path: Path):
    urdf_dir = tmp_path / "description" / "urdf"
    urdf_dir.mkdir(parents=True)
    urdf_path = urdf_dir / "robot.urdf"
    urdf_path.write_text(TWO_LINK_ARM_URDF, encoding="utf-8")

    active_joint_names = ("joint1", "joint2")
    points, active_joint_values = sample_reachable_positions(
        urdf_path,
        tip_link="tip",
        active_joint_names=active_joint_names,
        joint_limits={"joint1": (-np.pi, np.pi), "joint2": (-np.pi, np.pi)},
        fixed_joint_positions={},
        sample_count=25,
        rng=np.random.default_rng(2),
    )

    # This round-trip is exactly what the "click a reachable point -> preview
    # the arm pose" UI feature depends on: the returned joint values must
    # reproduce the returned position via plain FK.
    for index in range(len(points)):
        joint_positions = dict(zip(active_joint_names, active_joint_values[index]))
        transforms = load_urdf_link_transforms(urdf_path, joint_positions)
        np.testing.assert_allclose(transforms["tip"][:3, 3], points[index], atol=1e-9)


def test_sample_reachable_positions_respects_fixed_joint_positions(tmp_path: Path):
    urdf_dir = tmp_path / "description" / "urdf"
    urdf_dir.mkdir(parents=True)
    urdf_path = urdf_dir / "robot.urdf"
    urdf_path.write_text(
        """
<robot name="test">
  <link name="base"/>
  <joint name="yaw" type="revolute">
    <origin xyz="0 0 0"/>
    <parent link="base"/>
    <child link="mount"/>
    <axis xyz="0 0 1"/>
  </joint>
  <link name="mount"/>
  <joint name="offset" type="fixed">
    <origin xyz="1 0 0"/>
    <parent link="mount"/>
    <child link="tip"/>
  </joint>
  <link name="tip"/>
</robot>
""",
        encoding="utf-8",
    )

    points, active_joint_values = sample_reachable_positions(
        urdf_path,
        tip_link="tip",
        active_joint_names=(),
        joint_limits={},
        fixed_joint_positions={"yaw": np.pi / 2.0},
        sample_count=4,
        rng=np.random.default_rng(0),
    )

    np.testing.assert_allclose(points, np.tile([0.0, 1.0, 0.0], (4, 1)), atol=1e-9)
    assert active_joint_values.shape == (4, 0)


def test_sample_reachable_positions_rejects_unknown_tip_link(tmp_path: Path):
    urdf_dir = tmp_path / "description" / "urdf"
    urdf_dir.mkdir(parents=True)
    urdf_path = urdf_dir / "robot.urdf"
    urdf_path.write_text(
        """
<robot name="test">
  <link name="base"/>
</robot>
""",
        encoding="utf-8",
    )

    try:
        sample_reachable_positions(
            urdf_path,
            tip_link="does_not_exist",
            active_joint_names=(),
            joint_limits={},
            fixed_joint_positions={},
            sample_count=1,
            rng=np.random.default_rng(0),
        )
    except ValueError as error:
        assert "does_not_exist" in str(error)
    else:
        raise AssertionError("Expected unknown tip link to be rejected")


def test_sample_until_valid_discards_invalid_rows_and_accumulates_across_batches():
    calls = []

    def draw_batch(n):
        calls.append(n)
        start = sum(calls[:-1])
        positions = np.arange(start, start + n, dtype=float).reshape(-1, 1) * np.ones((1, 3))
        joint_values = positions[:, :1]
        # Every 3rd candidate (by absolute index) is invalid.
        indices = np.arange(start, start + n)
        valid_mask = indices % 3 != 0
        return positions, joint_values, valid_mask

    positions, joint_values, batches_drawn, candidates_drawn = sample_until_valid(
        draw_batch, sample_count=10, batch_size=4, max_attempts=20
    )

    assert positions.shape == (10, 3)
    assert joint_values.shape == (10, 1)
    # 2 out of every 3 candidates are valid, so it takes multiple batches of 4.
    assert batches_drawn > 1
    assert candidates_drawn == batches_drawn * 4
    # No invalid (index % 3 == 0) candidate leaked through.
    assert np.all(np.asarray(joint_values, dtype=int).flatten() % 3 != 0)


def test_sample_until_valid_stops_at_max_attempts_when_always_invalid():
    def draw_batch(n):
        positions = np.zeros((n, 3), dtype=float)
        joint_values = np.zeros((n, 2), dtype=float)
        valid_mask = np.zeros(n, dtype=bool)
        return positions, joint_values, valid_mask

    positions, joint_values, batches_drawn, candidates_drawn = sample_until_valid(
        draw_batch, sample_count=10, batch_size=5, max_attempts=3
    )

    assert positions.shape == (0, 3)
    assert joint_values.shape == (0, 0)
    assert batches_drawn == 3
    assert candidates_drawn == 15


def test_sample_until_valid_truncates_to_sample_count():
    def draw_batch(n):
        positions = np.ones((n, 3), dtype=float)
        joint_values = np.ones((n, 1), dtype=float)
        valid_mask = np.ones(n, dtype=bool)
        return positions, joint_values, valid_mask

    positions, joint_values, batches_drawn, candidates_drawn = sample_until_valid(
        draw_batch, sample_count=3, batch_size=10, max_attempts=5
    )

    assert positions.shape == (3, 3)
    assert joint_values.shape == (3, 1)
    assert batches_drawn == 1
    assert candidates_drawn == 10
