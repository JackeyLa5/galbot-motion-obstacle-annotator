from pathlib import Path

import numpy as np

from galbot_motion_obstacle_annotator.robot_model import load_urdf_visuals


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
