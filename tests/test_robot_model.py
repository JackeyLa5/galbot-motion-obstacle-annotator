from pathlib import Path

import numpy as np

from galbot_motion_obstacle_annotator.robot_model import (
    load_tcp_gripper_visuals,
    load_urdf_actuated_joint_names,
    load_urdf_joint_limits,
    load_urdf_visuals,
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
