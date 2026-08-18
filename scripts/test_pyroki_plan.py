#!/usr/bin/env python3
"""Standalone PyRoki planning probe for a single TCP target."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from galbot_motion_obstacle_annotator.planning.models import PlanRequest, PoseTarget
from galbot_motion_obstacle_annotator.planning.pyroki import PyrokiPlanner, resolve_pyroki_urdf_path
from galbot_motion_obstacle_annotator.robot_model import load_urdf_actuated_joint_names
from galbot_motion_obstacle_annotator.robot_state import (
    ARM_JOINT_NAMES,
    INITIAL_ROBOT_JOINT_POSITIONS,
    LEG_JOINT_NAMES,
    RobotEnvironmentState,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a standalone PyRoki planning check.")
    parser.add_argument(
        "--robot-urdf",
        type=Path,
        default=Path("galbot_one_golf_description/urdf/galbot_one_golf.urdf"),
        help="Robot URDF path. If a sibling *_fixed_base.urdf exists, PyRoki will use it.",
    )
    parser.add_argument("--arm", choices=("left_arm", "right_arm"), default="left_arm")
    parser.add_argument("--with-leg", action="store_true", help="Allow leg joints to move during planning.")
    parser.add_argument("--x", type=float, default=0.65, help="TCP target x in base_link.")
    parser.add_argument("--y", type=float, default=0.20, help="TCP target y in base_link.")
    parser.add_argument("--z", type=float, default=0.85, help="TCP target z in base_link.")
    parser.add_argument("--qx", type=float, default=0.0, help="TCP target quaternion x in base_link.")
    parser.add_argument("--qy", type=float, default=0.0, help="TCP target quaternion y in base_link.")
    parser.add_argument("--qz", type=float, default=0.0, help="TCP target quaternion z in base_link.")
    parser.add_argument("--qw", type=float, default=1.0, help="TCP target quaternion w in base_link.")
    parser.add_argument("--timesteps", type=int, default=32)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--self-collision", action="store_true", help="Enable self collision checking.")
    parser.add_argument(
        "--env-collision",
        action="store_true",
        help="Enable environment collision checking. No obstacles are loaded by this script.",
    )
    return parser


def default_robot_state(joint_names: tuple[str, ...]) -> RobotEnvironmentState:
    state = RobotEnvironmentState(dict(INITIAL_ROBOT_JOINT_POSITIONS))
    state.initialize_missing(joint_names, value=0.0)
    return state


def target_link_for_arm(arm_name: str) -> str:
    prefix = "left" if arm_name == "left_arm" else "right"
    return f"{prefix}_gripper_tcp_link"


def main() -> int:
    args = build_parser().parse_args()
    input_urdf = args.robot_urdf.expanduser().resolve()
    resolved_urdf = resolve_pyroki_urdf_path(input_urdf)
    joint_names = load_urdf_actuated_joint_names(resolved_urdf)
    robot_state = default_robot_state(joint_names)
    active_joint_names = ARM_JOINT_NAMES[args.arm]
    if args.with_leg:
        active_joint_names = LEG_JOINT_NAMES + active_joint_names

    request = PlanRequest(
        target=PoseTarget(
            chain_name=args.arm,
            position=np.array([args.x, args.y, args.z], dtype=float),
            orientation_xyzw=np.array([args.qx, args.qy, args.qz, args.qw], dtype=float),
            reference_frame="base_link",
            frame_id=target_link_for_arm(args.arm),
        ),
        start_joint_positions={
            "leg": robot_state.positions_for(LEG_JOINT_NAMES),
            "left_arm": robot_state.positions_for(ARM_JOINT_NAMES["left_arm"]),
            "right_arm": robot_state.positions_for(ARM_JOINT_NAMES["right_arm"]),
        },
        collision_check=args.self_collision,
        environment_collision_check=args.env_collision,
        options={
            "pyroki_urdf_path": str(input_urdf),
            "pyroki_joint_names": joint_names,
            "pyroki_start_joint_positions": {
                name: robot_state.joint_positions.get(name, 0.0) for name in joint_names
            },
            "pyroki_active_joint_names": active_joint_names,
            "pyroki_target_link": target_link_for_arm(args.arm),
            "pyroki_timesteps": args.timesteps,
            "pyroki_dt": args.dt,
        },
    )

    planner = PyrokiPlanner()
    print(f"Input URDF   : {input_urdf}")
    print(f"PyRoki URDF  : {resolved_urdf}")
    print(f"Arm          : {args.arm}")
    print(f"Active joints: {active_joint_names}")
    print(f"Target link  : {target_link_for_arm(args.arm)}")
    print(
        "Target pose  : "
        f"pos=({args.x:.3f}, {args.y:.3f}, {args.z:.3f}) "
        f"quat=({args.qx:.3f}, {args.qy:.3f}, {args.qz:.3f}, {args.qw:.3f}) "
        "[base_link]"
    )
    print(
        "Collision    : "
        f"self={args.self_collision} env={args.env_collision}"
    )

    result = planner.plan(request)
    print(f"\nResult       : success={result.success} status={result.status}")
    if result.message:
        print(f"Message      : {result.message}")
    if result.planning_seconds is not None:
        print(f"Time         : {result.planning_seconds:.3f}s")
    if result.diagnostics:
        print("Diagnostics  :")
        for key, value in sorted(result.diagnostics.items()):
            print(f"  - {key}: {value}")
    trajectory = result.trajectories.get(args.arm)
    if trajectory is not None:
        print(f"Trajectory   : {trajectory.positions.shape[0]} frames x {trajectory.positions.shape[1]} joints")
        print(f"Final joints : {trajectory.positions[-1].tolist()}")
    return 0 if result.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
