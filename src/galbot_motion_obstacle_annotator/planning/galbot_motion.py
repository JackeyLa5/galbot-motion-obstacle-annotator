from __future__ import annotations

import time
from importlib import import_module
from typing import Any
from uuid import uuid4

import numpy as np

from .models import JointTrajectory, PlanRequest, PlanResult
from .protocol import PlannerMetadata


class GalbotMotionPlanner:
    """Galbot Motion adapter that always requests planning without execution."""

    metadata = PlannerMetadata(
        planner_id="galbot-motion",
        display_name="Galbot Motion",
        description="Plan with Galbot Motion and return joint trajectories without commanding robot motion.",
        supports_environment_obstacles=True,
        supports_cartesian_targets=True,
        returns_timestamps=False,
    )

    def __init__(self, sdk_module: Any | None = None, motion: Any | None = None) -> None:
        self._sdk_module = sdk_module
        self._motion = motion
        self._initialized = False

    def is_available(self) -> tuple[bool, str]:
        try:
            self._sdk()
        except (ImportError, AttributeError) as error:
            return False, str(error)
        return True, "Galbot SDK is available"

    def plan(self, request: PlanRequest) -> PlanResult:
        started = time.monotonic()
        loaded_obstacle_ids: list[str] = []
        try:
            sdk = self._sdk()
            motion = self._motion_instance(sdk)
            self._initialize(motion)
            if request.environment_collision_check:
                loaded_obstacle_ids = self._load_temporary_obstacles(motion, request)
            target = self._pose_target(sdk, request)
            reference_state = self._reference_state(sdk, request)
            params = self._planning_parameters(sdk, request)
            status, raw_trajectories = motion.motion_plan(
                target,
                None,
                reference_state,
                request.collision_check,
                params,
            )
            success = status == sdk.MotionStatus.SUCCESS
            trajectories = {
                chain_name: JointTrajectory(chain_name, np.asarray(positions, dtype=float))
                for chain_name, positions in raw_trajectories.items()
                if positions
            }
            return PlanResult(
                planner_id=self.metadata.planner_id,
                success=success,
                status=getattr(status, "name", str(status)),
                trajectories=trajectories,
                message="" if success else "Galbot Motion did not produce a successful plan",
                planning_seconds=time.monotonic() - started,
                diagnostics={"direct_execute": False, "is_tool_pose": True},
            )
        except Exception as error:
            return PlanResult(
                planner_id=self.metadata.planner_id,
                success=False,
                status="ERROR",
                message=str(error),
                planning_seconds=time.monotonic() - started,
                diagnostics={"direct_execute": False, "is_tool_pose": True},
            )
        finally:
            self._remove_temporary_obstacles(loaded_obstacle_ids)

    def _sdk(self) -> Any:
        if self._sdk_module is None:
            self._sdk_module = import_module("galbot_sdk.g1")
        return self._sdk_module

    def _motion_instance(self, sdk: Any) -> Any:
        if self._motion is None:
            self._motion = sdk.GalbotMotion()
        return self._motion

    def _initialize(self, motion: Any) -> None:
        if self._initialized:
            return
        if not motion.init():
            raise RuntimeError("Galbot Motion initialization failed")
        self._initialized = True

    def _planning_parameters(self, sdk: Any, request: PlanRequest) -> Any:
        params = sdk.Parameter()
        params.is_direct_execute = False
        params.is_tool_pose = True
        params.is_blocking = True
        params.is_check_collision = request.collision_check
        params.timeout_second = request.timeout_seconds
        params.reference_frame = request.target.reference_frame
        if hasattr(params, "enable_env_collision_check"):
            params.enable_env_collision_check = request.environment_collision_check
        elif request.environment_collision_check:
            raise RuntimeError(
                "The installed Galbot SDK does not expose enable_env_collision_check; "
                "environment-aware comparison requires a newer SDK build"
            )
        if bool(params.is_direct_execute):
            raise RuntimeError("Galbot Motion direct execution must remain disabled")
        return params

    @staticmethod
    def _pose_target(sdk: Any, request: PlanRequest) -> Any:
        target = sdk.PoseState()
        target.chain_name = request.target.chain_name
        target.frame_id = "TCP"
        target.reference_frame = request.target.reference_frame
        target.pose.position.x, target.pose.position.y, target.pose.position.z = request.target.position
        (
            target.pose.orientation.x,
            target.pose.orientation.y,
            target.pose.orientation.z,
            target.pose.orientation.w,
        ) = request.target.orientation_xyzw
        return target

    @staticmethod
    def _reference_state(sdk: Any, request: PlanRequest) -> Any:
        if request.target.chain_name not in request.start_joint_positions:
            raise ValueError(
                f"Missing start joint positions for target chain {request.target.chain_name}"
            )
        positions = request.options.get("galbot_whole_body_joint_positions")
        if positions is None:
            raise ValueError(
                "Galbot Motion requires options['galbot_whole_body_joint_positions'] "
                "to avoid reading the real robot state"
            )
        positions = np.asarray(positions, dtype=float)
        if positions.ndim != 1 or not len(positions) or not np.isfinite(positions).all():
            raise ValueError(
                "options['galbot_whole_body_joint_positions'] must contain finite joint values"
            )
        base_pose = request.options.get("galbot_base_pose")
        if base_pose is None:
            raise ValueError(
                "Galbot Motion requires options['galbot_base_pose'] "
                "to avoid reading the real robot state"
            )
        base_pose = np.asarray(base_pose, dtype=float)
        if base_pose.shape != (7,) or not np.isfinite(base_pose).all():
            raise ValueError("options['galbot_base_pose'] must contain 7 finite values")
        quaternion_norm = float(np.linalg.norm(base_pose[3:]))
        if quaternion_norm < 1e-12:
            raise ValueError("options['galbot_base_pose'] quaternion must have non-zero norm")
        base_pose = base_pose.copy()
        base_pose[3:] /= quaternion_norm
        state = sdk.RobotStates()
        state.set_whole_body_joint(positions)
        pose = sdk.Pose()
        pose.position.x, pose.position.y, pose.position.z = base_pose[:3]
        pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = base_pose[3:]
        state.set_base_state(pose)
        return state

    def _load_temporary_obstacles(self, motion: Any, request: PlanRequest) -> list[str]:
        loaded_ids: list[str] = []
        reference_joint_positions = request.options["galbot_whole_body_joint_positions"]
        reference_base_pose = request.options["galbot_base_pose"]
        try:
            for obstacle in request.obstacles:
                obstacle.validate_for_motion()
                payload = obstacle.to_motion_dict()
                temporary_id = f"visualization_{uuid4().hex}_{obstacle.obstacle_id}"
                status = motion.add_obstacle(
                    obstacle_id=temporary_id,
                    obstacle_type=payload["obstacle_type"],
                    pose=payload["pose"],
                    scale=payload["scale"],
                    target_frame=payload["target_frame"],
                    reference_joint_positions=reference_joint_positions,
                    reference_base_pose=reference_base_pose,
                )
                if not self._status_is_success(status):
                    raise RuntimeError(f"Failed to load obstacle {obstacle.obstacle_id}: {status}")
                loaded_ids.append(temporary_id)
        except Exception:
            self._remove_temporary_obstacles(loaded_ids)
            raise
        return loaded_ids

    def _remove_temporary_obstacles(self, obstacle_ids: list[str]) -> None:
        if self._motion is None:
            return
        for obstacle_id in obstacle_ids:
            try:
                self._motion.remove_obstacle(obstacle_id)
            except Exception:
                pass

    def _status_is_success(self, status: Any) -> bool:
        return status == self._sdk().MotionStatus.SUCCESS
