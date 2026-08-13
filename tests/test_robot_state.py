import pytest

from galbot_motion_obstacle_annotator.robot_state import (
    GALBOT_REFERENCE_JOINT_NAMES,
    INITIAL_ROBOT_JOINT_POSITIONS,
    LEG_JOINT_NAMES,
    LEFT_ARM_JOINT_NAMES,
    MOTION_PLANNING_JOINT_NAMES,
    RIGHT_ARM_JOINT_NAMES,
    SINGLE_ARM_PLANNING_JOINT_NAMES,
    RobotEnvironmentState,
)


def test_environment_state_supplies_current_planning_positions():
    state = RobotEnvironmentState({name: float(index) for index, name in enumerate(MOTION_PLANNING_JOINT_NAMES)})

    state.update({"left_arm_joint1": 42.0})

    assert state.positions_for(MOTION_PLANNING_JOINT_NAMES)[5] == 42.0


def test_environment_state_rejects_missing_planning_joint():
    state = RobotEnvironmentState({"leg_joint1": 0.0})

    with pytest.raises(ValueError, match="missing robot joints"):
        state.positions_for(MOTION_PLANNING_JOINT_NAMES)


def test_motion_joints_exclude_head_and_wheels_but_galbot_reference_keeps_head():
    assert len(MOTION_PLANNING_JOINT_NAMES) == 19
    assert len(GALBOT_REFERENCE_JOINT_NAMES) == 21
    assert not any(name.startswith("wheel") for name in MOTION_PLANNING_JOINT_NAMES)
    assert not any(name.startswith("head") for name in MOTION_PLANNING_JOINT_NAMES)
    assert GALBOT_REFERENCE_JOINT_NAMES[5:7] == ("head_joint1", "head_joint2")


def test_default_single_arm_planning_keeps_other_arm_and_legs_fixed():
    assert SINGLE_ARM_PLANNING_JOINT_NAMES["left_arm"] == LEFT_ARM_JOINT_NAMES
    assert SINGLE_ARM_PLANNING_JOINT_NAMES["right_arm"] == RIGHT_ARM_JOINT_NAMES
    assert len(LEG_JOINT_NAMES + SINGLE_ARM_PLANNING_JOINT_NAMES["left_arm"]) == 12


def test_environment_initializes_only_missing_urdf_joints():
    state = RobotEnvironmentState({"left_arm_joint1": 1.25})

    state.initialize_missing(("left_arm_joint1", "left_arm_joint2"))

    assert state.joint_positions == {"left_arm_joint1": 1.25, "left_arm_joint2": 0.0}


def test_initial_robot_pose_matches_the_established_preview_pose():
    assert INITIAL_ROBOT_JOINT_POSITIONS["leg_joint2"] == 1.8
    assert INITIAL_ROBOT_JOINT_POSITIONS["left_arm_joint4"] == -2.1
    assert INITIAL_ROBOT_JOINT_POSITIONS["right_arm_joint4"] == 2.1
