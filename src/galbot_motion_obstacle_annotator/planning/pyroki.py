from __future__ import annotations

import time
from importlib import import_module
from importlib.util import find_spec
from pathlib import Path
from typing import Any, Callable, Mapping, Tuple, Union

import numpy as np

from ..geometry import matrix_to_quaternion, matrix_to_rpy, quaternion_to_matrix, rpy_to_matrix
from ..models import Obstacle
from .models import JointTrajectory, PlanRequest, PlanResult, PoseTarget
from .protocol import PlannerMetadata

Solver = Callable[..., Union[np.ndarray, Tuple[np.ndarray, Mapping[str, Any]]]]


def resolve_pyroki_urdf_path(path_like: str | Path) -> Path:
    """Prefer a sibling fixed-base URDF for PyRoki when available."""
    path = Path(path_like).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"PyRoki URDF does not exist: {path}")
    stem = path.stem
    if stem.endswith("_fixed_base"):
        return path
    candidate = path.with_name(f"{stem}_fixed_base{path.suffix}")
    if candidate.is_file():
        return candidate
    return path


def express_world_scene_in_base_frame(
    target: PoseTarget,
    obstacles: list[Obstacle],
    base_transform: np.ndarray,
) -> tuple[PoseTarget, list[Obstacle]]:
    """Express a world-frame TCP target and obstacles in ``base_link``."""
    base_transform = np.asarray(base_transform, dtype=float)
    if base_transform.shape != (4, 4) or not np.isfinite(base_transform).all():
        raise ValueError("Robot base transform must be a finite 4x4 matrix")
    if target.reference_frame not in {"world", "base_link"}:
        raise ValueError(
            "PyRoki target conversion requires reference_frame='world' or 'base_link', "
            f"got {target.reference_frame!r}"
        )

    world_to_base = np.linalg.inv(base_transform)
    if target.reference_frame == "world":
        target_world = np.eye(4, dtype=float)
        target_world[:3, :3] = quaternion_to_matrix(target.orientation_xyzw)
        target_world[:3, 3] = target.position
        target_base = world_to_base @ target_world
        target_position = target_base[:3, 3].copy()
        target_orientation = matrix_to_quaternion(target_base[:3, :3])
    else:
        target_position = target.position.copy()
        target_orientation = target.orientation_xyzw.copy()
    local_target = PoseTarget(
        chain_name=target.chain_name,
        position=target_position,
        orientation_xyzw=target_orientation,
        reference_frame="base_link",
        frame_id=target.frame_id,
    )

    local_obstacles: list[Obstacle] = []
    for obstacle in obstacles:
        if obstacle.target_frame != "world":
            raise ValueError(
                f"PyRoki obstacle conversion requires target_frame='world', got {obstacle.target_frame!r}"
            )
        obstacle_world = np.eye(4, dtype=float)
        obstacle_world[:3, :3] = rpy_to_matrix(*obstacle.rpy)
        obstacle_world[:3, 3] = obstacle.center
        obstacle_base = world_to_base @ obstacle_world
        local_obstacles.append(
            Obstacle(
                obstacle_id=obstacle.obstacle_id,
                obstacle_type=obstacle.obstacle_type,
                center=obstacle_base[:3, 3],
                rpy=matrix_to_rpy(obstacle_base[:3, :3]),
                scale=obstacle.scale.copy(),
                target_frame="base_link",
            )
        )
    return local_target, local_obstacles


class PyrokiPlanner:
    """PyRoki trajectory-optimization adapter with lazy optional dependencies."""

    metadata = PlannerMetadata(
        planner_id="pyroki",
        display_name="PyRoki",
        description="Optimize a collision-aware joint trajectory from a URDF using JAX and PyRoki.",
        supports_environment_obstacles=True,
        supports_cartesian_targets=True,
        returns_timestamps=True,
    )

    def __init__(
        self,
        solver: Solver | None = None,
        dependency_loader: Callable[[], dict[str, Any]] | None = None,
    ) -> None:
        self._solver = solver
        self._dependency_loader = dependency_loader or self._load_dependencies
        self._uses_default_dependency_loader = dependency_loader is None

    def is_available(self) -> tuple[bool, str]:
        if self._uses_default_dependency_loader:
            missing = [
                name
                for name in ("jax", "jaxlie", "jaxls", "pyroki", "yourdfpy")
                if find_spec(name) is None
            ]
            if missing:
                return False, f"PyRoki dependencies are unavailable: {', '.join(missing)}"
            return True, "PyRoki dependencies are available"
        try:
            self._dependency_loader()
        except (ImportError, ModuleNotFoundError) as error:
            return False, f"PyRoki dependencies are unavailable: {error}"
        return True, "PyRoki dependencies are available"

    def plan(self, request: PlanRequest) -> PlanResult:
        started = time.monotonic()
        try:
            urdf_path = self._urdf_path(request)
            joint_names = self._joint_names(request)
            start_positions = self._ordered_start_positions(request, joint_names)
            target_link_name = str(request.options.get("pyroki_target_link", request.target.frame_id))
            timesteps = int(request.options.get("pyroki_timesteps", 32))
            dt = float(request.options.get("pyroki_dt", 0.05))
            if request.target.reference_frame != "base_link":
                raise ValueError("PyRoki requires target.reference_frame='base_link'")
            if timesteps < 2:
                raise ValueError("pyroki_timesteps must be at least 2")
            if not np.isfinite(dt) or dt <= 0.0:
                raise ValueError("pyroki_dt must be greater than zero")

            dependencies = self._dependency_loader()
            solver = self._solver or self._solve_trajectory
            solver_result = solver(
                dependencies=dependencies,
                urdf_path=urdf_path,
                joint_names=joint_names,
                start_positions=start_positions,
                target_link_name=target_link_name,
                request=request,
                timesteps=timesteps,
                dt=dt,
            )
            trajectory, solver_diagnostics = self._solver_result(solver_result)
            expected_shape = (timesteps, len(joint_names))
            if trajectory.shape != expected_shape:
                raise ValueError(
                    f"PyRoki returned trajectory shape {trajectory.shape}; expected {expected_shape}"
                )
            if not np.isfinite(trajectory).all():
                raise ValueError("PyRoki returned non-finite joint positions")

            diagnostics = {
                "urdf_path": str(urdf_path),
                "target_link_name": target_link_name,
                "joint_names": joint_names,
                "timesteps": timesteps,
                "dt": dt,
                "optimizer": "jaxls",
                "collision_check": request.collision_check,
                "environment_collision_check": request.environment_collision_check,
                "execution_enabled": False,
            }
            diagnostics.update(solver_diagnostics)
            return PlanResult(
                planner_id=self.metadata.planner_id,
                success=True,
                status="SUCCESS",
                trajectories={
                    request.target.chain_name: JointTrajectory(
                        chain_name=request.target.chain_name,
                        positions=trajectory,
                        timestamps=np.arange(timesteps, dtype=float) * dt,
                    )
                },
                planning_seconds=time.monotonic() - started,
                diagnostics=diagnostics,
            )
        except Exception as error:
            return PlanResult(
                planner_id=self.metadata.planner_id,
                success=False,
                status="ERROR",
                message=str(error),
                planning_seconds=time.monotonic() - started,
            )

    @staticmethod
    def _urdf_path(request: PlanRequest) -> Path:
        value = request.options.get("pyroki_urdf_path")
        if not value:
            raise ValueError("PyRoki requires options['pyroki_urdf_path']")
        return resolve_pyroki_urdf_path(value)

    @staticmethod
    def _load_dependencies() -> dict[str, Any]:
        return {
            "jax": import_module("jax"),
            "jnp": import_module("jax.numpy"),
            "jaxlie": import_module("jaxlie"),
            "jaxls": import_module("jaxls"),
            "pyroki": import_module("pyroki"),
            "yourdfpy": import_module("yourdfpy"),
        }

    @staticmethod
    def _joint_names(request: PlanRequest) -> tuple[str, ...]:
        configured = request.options.get("pyroki_joint_names")
        if not configured:
            raise ValueError("PyRoki requires options['pyroki_joint_names'] in URDF actuated order")
        names = tuple(str(name) for name in configured)
        if any(not name.strip() for name in names):
            raise ValueError("PyRoki joint names must be non-empty")
        if len(set(names)) != len(names):
            raise ValueError("PyRoki joint names must be unique")
        return names

    @staticmethod
    def _ordered_start_positions(request: PlanRequest, joint_names: tuple[str, ...]) -> np.ndarray:
        values_by_name = request.options.get("pyroki_start_joint_positions")
        if not isinstance(values_by_name, Mapping):
            raise ValueError("PyRoki requires options['pyroki_start_joint_positions'] as a mapping")
        try:
            values = np.asarray([values_by_name[name] for name in joint_names], dtype=float)
        except KeyError as error:
            raise ValueError(f"Missing PyRoki start joint: {error.args[0]}") from error
        if values.shape != (len(joint_names),) or not np.isfinite(values).all():
            raise ValueError("PyRoki start positions must be finite scalar values")
        return values

    @staticmethod
    def _locked_joint_targets(
        request: PlanRequest,
        joint_names: tuple[str, ...],
        start_positions: np.ndarray,
    ) -> np.ndarray:
        targets = np.asarray(start_positions, dtype=float).copy()
        configured = request.options.get("pyroki_locked_joint_positions", {})
        if not isinstance(configured, Mapping):
            raise ValueError("PyRoki options['pyroki_locked_joint_positions'] must be a mapping")
        index_by_name = {name: index for index, name in enumerate(joint_names)}
        for name, value in configured.items():
            if name not in index_by_name:
                continue
            numeric = float(value)
            if not np.isfinite(numeric):
                raise ValueError(f"Invalid locked joint target for {name!r}: {value!r}")
            targets[index_by_name[name]] = numeric
        return targets

    @staticmethod
    def _solver_result(
        result: np.ndarray | tuple[np.ndarray, Mapping[str, Any]],
    ) -> tuple[np.ndarray, dict[str, Any]]:
        diagnostics: dict[str, Any] = {}
        trajectory: Any = result
        if isinstance(result, tuple):
            if len(result) != 2 or not isinstance(result[1], Mapping):
                raise ValueError("PyRoki solver tuple must contain trajectory and diagnostics")
            trajectory, raw_diagnostics = result
            diagnostics.update(raw_diagnostics)
        return np.asarray(trajectory, dtype=float), diagnostics

    @classmethod
    def _solve_trajectory(
        cls,
        *,
        dependencies: dict[str, Any],
        urdf_path: Path,
        joint_names: tuple[str, ...],
        start_positions: np.ndarray,
        target_link_name: str,
        request: PlanRequest,
        timesteps: int,
        dt: float,
    ) -> tuple[np.ndarray, Mapping[str, Any]]:
        jax = dependencies["jax"]
        jnp = dependencies["jnp"]
        jaxlie = dependencies["jaxlie"]
        jaxls = dependencies["jaxls"]
        pyroki = dependencies["pyroki"]
        yourdfpy = dependencies["yourdfpy"]

        urdf = yourdfpy.URDF.load(str(urdf_path))
        robot = pyroki.Robot.from_urdf(urdf, default_joint_cfg=jnp.asarray(start_positions))
        urdf_joint_names = tuple(str(name) for name in robot.joints.actuated_names)
        if urdf_joint_names != joint_names:
            raise ValueError(
                "pyroki_joint_names must exactly match the URDF actuated joint order: "
                f"{urdf_joint_names}"
            )
        if target_link_name not in robot.links.names:
            raise ValueError(f"PyRoki target link does not exist in the URDF: {target_link_name}")
        active_joint_names = cls._active_joint_names(request, urdf_joint_names)
        active_joint_mask = jnp.asarray(
            [1.0 if name in active_joint_names else 0.0 for name in urdf_joint_names]
        )
        locked_joint_mask = 1.0 - active_joint_mask
        locked_joint_targets = cls._locked_joint_targets(request, urdf_joint_names, start_positions)

        needs_collision_model = request.collision_check or request.environment_collision_check
        robot_collision = (
            pyroki.collision.RobotCollision.from_urdf(urdf) if needs_collision_model else None
        )
        world_collision = (
            cls._world_collision(pyroki, request) if request.environment_collision_check else []
        )
        target_link_index = robot.links.names.index(target_link_name)
        target_wxyz = jnp.asarray(cls._xyzw_to_wxyz(request.target.orientation_xyzw))
        target_pose = jaxlie.SE3.from_rotation_and_translation(
            jaxlie.SO3(target_wxyz), jnp.asarray(request.target.position)
        )
        trajectory_var = robot.joint_var_cls(jnp.arange(timesteps))
        initial = jnp.repeat(jnp.asarray(start_positions)[None], timesteps, axis=0)
        costs = [
            pyroki.costs.rest_cost(
                trajectory_var,
                trajectory_var.default_factory()[None],
                jnp.array([0.01])[None],
            ),
            pyroki.costs.smoothness_cost(
                robot.joint_var_cls(jnp.arange(1, timesteps)),
                robot.joint_var_cls(jnp.arange(0, timesteps - 1)),
                jnp.array([1.0])[None],
            ),
            pyroki.costs.pose_cost_analytic_jac(
                robot,
                robot.joint_var_cls(timesteps - 1),
                target_pose,
                jnp.asarray(target_link_index),
                pos_weight=10.0,
                ori_weight=1.0,
                joint_mask=active_joint_mask,
            ),
        ]
        batched_robot = jax.tree.map(lambda value: value[None], robot)
        costs.append(pyroki.costs.limit_constraint(batched_robot, trajectory_var))
        costs.append(
            pyroki.costs.limit_velocity_constraint(
                batched_robot,
                robot.joint_var_cls(jnp.arange(1, timesteps)),
                robot.joint_var_cls(jnp.arange(0, timesteps - 1)),
                dt,
            )
        )
        if timesteps >= 5:
            costs.append(
                pyroki.costs.five_point_acceleration_cost(
                    robot.joint_var_cls(jnp.arange(2, timesteps - 2)),
                    robot.joint_var_cls(jnp.arange(4, timesteps)),
                    robot.joint_var_cls(jnp.arange(3, timesteps - 1)),
                    robot.joint_var_cls(jnp.arange(1, timesteps - 3)),
                    robot.joint_var_cls(jnp.arange(0, timesteps - 4)),
                    dt,
                    jnp.array([0.1])[None],
                )
            )
        if request.collision_check:
            batched_collision = jax.tree.map(lambda value: value[None], robot_collision)
            costs.append(
                pyroki.costs.self_collision_cost(
                    batched_robot,
                    batched_collision,
                    trajectory_var,
                    0.02,
                    5.0,
                )
            )

        @jaxls.Cost.factory(kind="constraint_eq_zero", name="pyroki_start_configuration")
        def start_constraint(values, variable):
            return (values[variable] - jnp.asarray(start_positions)).flatten()

        costs.append(start_constraint(robot.joint_var_cls(0)))

        @jaxls.Cost.factory(kind="constraint_eq_zero", name="pyroki_locked_joints")
        def locked_joint_constraint(values, variable):
            return (
                (values[variable] - jnp.asarray(locked_joint_targets)) * locked_joint_mask
            ).flatten()

        costs.append(locked_joint_constraint(trajectory_var))
        if bool(request.options.get("pyroki_keep_leg_upright", False)):
            cls._append_leg_sum_constraints(
                costs=costs,
                dependencies=dependencies,
                robot=robot,
                joint_names=urdf_joint_names,
                timesteps=timesteps,
            )
            cls._append_leg_upright_band_constraints(
                costs=costs,
                dependencies=dependencies,
                request=request,
                robot=robot,
                joint_names=urdf_joint_names,
                timesteps=timesteps,
            )
        if request.environment_collision_check:
            cls._append_world_collision_constraints(
                costs=costs,
                dependencies=dependencies,
                robot=batched_robot,
                robot_collision=jax.tree.map(lambda value: value[None], robot_collision),
                world_collision=world_collision,
                timesteps=timesteps,
            )

        solution = (
            jaxls.LeastSquaresProblem(costs=costs, variables=[trajectory_var])
            .analyze()
            .solve(initial_vals=jaxls.VarValues.make((trajectory_var.with_value(initial),)))
        )
        trajectory = np.asarray(solution[trajectory_var], dtype=float)
        position_error, orientation_error = cls._target_errors(
            robot=robot,
            trajectory=trajectory,
            target_link_index=target_link_index,
            target_position=request.target.position,
            target_xyzw=request.target.orientation_xyzw,
        )
        position_tolerance = float(request.options.get("pyroki_position_tolerance", 0.02))
        orientation_tolerance = float(request.options.get("pyroki_orientation_tolerance", 0.15))
        if position_error > position_tolerance or orientation_error > orientation_tolerance:
            raise ValueError(
                "PyRoki solution did not reach the target within tolerance: "
                f"position_error={position_error:.6f} m, "
                f"orientation_error={orientation_error:.6f} rad"
            )
        return trajectory, {
            "position_error_m": position_error,
            "orientation_error_rad": orientation_error,
            "position_tolerance_m": position_tolerance,
            "orientation_tolerance_rad": orientation_tolerance,
            "world_obstacle_count": len(world_collision),
            "cylinder_approximation": "capsule",
            "active_joint_names": active_joint_names,
            "locked_joint_count": len(urdf_joint_names) - len(active_joint_names),
        }

    @staticmethod
    def _active_joint_names(
        request: PlanRequest, urdf_joint_names: tuple[str, ...]
    ) -> tuple[str, ...]:
        configured = request.options.get("pyroki_active_joint_names")
        if configured is None:
            configured = urdf_joint_names
        names = tuple(str(name) for name in configured)
        if not names:
            raise ValueError("PyRoki requires at least one active joint")
        unknown = sorted(set(names) - set(urdf_joint_names))
        if unknown:
            raise ValueError(f"Unknown PyRoki active joints: {unknown}")
        if len(set(names)) != len(names):
            raise ValueError("PyRoki active joint names must be unique")
        return names

    @staticmethod
    def _append_leg_upright_band_constraints(
        *,
        costs: list[Any],
        dependencies: dict[str, Any],
        request: PlanRequest,
        robot: Any,
        joint_names: tuple[str, ...],
        timesteps: int,
    ) -> None:
        jnp = dependencies["jnp"]
        jaxls = dependencies["jaxls"]

        required = ("leg_joint1", "leg_joint2", "leg_joint3")
        missing = [name for name in required if name not in joint_names]
        if missing:
            raise ValueError(f"PyRoki upright-leg constraint requires joints: {missing}")
        leg_joint1_index = joint_names.index("leg_joint1")
        leg_joint2_index = joint_names.index("leg_joint2")
        leg_joint3_index = joint_names.index("leg_joint3")
        leg_joint2_tolerance = float(request.options.get("pyroki_leg_joint2_ratio_tolerance", 0.35))
        leg_joint3_tolerance = float(request.options.get("pyroki_leg_joint3_ratio_tolerance", 0.35))
        for timestep in range(timesteps):
            @jaxls.Cost.factory(
                kind="constraint_geq_zero",
                name=f"pyroki_leg_upright_band_t{timestep}",
            )
            def upright_leg_band_constraint(
                values,
                variable,
                q1=leg_joint1_index,
                q2=leg_joint2_index,
                q3=leg_joint3_index,
                tol2=leg_joint2_tolerance,
                tol3=leg_joint3_tolerance,
            ):
                q = values[variable]
                joint2_error = q[q2] - 3.0 * q[q1]
                joint3_error = q[q3] - 2.0 * q[q1]
                return jnp.asarray(
                    [
                        tol2 + joint2_error,
                        tol2 - joint2_error,
                        tol3 + joint3_error,
                        tol3 - joint3_error,
                    ]
                )

            costs.append(upright_leg_band_constraint(robot.joint_var_cls(timestep)))

    @staticmethod
    def _append_leg_sum_constraints(
        *,
        costs: list[Any],
        dependencies: dict[str, Any],
        robot: Any,
        joint_names: tuple[str, ...],
        timesteps: int,
    ) -> None:
        jnp = dependencies["jnp"]
        jaxls = dependencies["jaxls"]

        required = ("leg_joint1", "leg_joint2", "leg_joint3")
        missing = [name for name in required if name not in joint_names]
        if missing:
            raise ValueError(f"PyRoki leg-sum constraint requires joints: {missing}")
        leg_joint1_index = joint_names.index("leg_joint1")
        leg_joint2_index = joint_names.index("leg_joint2")
        leg_joint3_index = joint_names.index("leg_joint3")
        for timestep in range(timesteps):
            @jaxls.Cost.factory(
                kind="constraint_eq_zero",
                name=f"pyroki_leg_sum_t{timestep}",
            )
            def leg_sum_constraint(values, variable, q1=leg_joint1_index, q2=leg_joint2_index, q3=leg_joint3_index):
                q = values[variable]
                return jnp.asarray([q[q1] + q[q3] - q[q2]])

            costs.append(leg_sum_constraint(robot.joint_var_cls(timestep)))

    @staticmethod
    def _append_world_collision_constraints(
        *,
        costs: list[Any],
        dependencies: dict[str, Any],
        robot: Any,
        robot_collision: Any,
        world_collision: list[Any],
        timesteps: int,
    ) -> None:
        jax = dependencies["jax"]
        jnp = dependencies["jnp"]
        jaxls = dependencies["jaxls"]
        pyroki = dependencies["pyroki"]

        def world_collision_residual(values, robot_model, collision_model, obstacle, previous, current):
            swept = collision_model.get_swept_capsules(
                robot_model, values[previous], values[current]
            )
            distances = pyroki.collision.collide(
                swept.reshape((-1, 1)), obstacle.reshape((1, -1))
            )
            return distances.flatten() - 0.05

        for obstacle in world_collision:
            costs.append(
                jaxls.Cost(
                    world_collision_residual,
                    (
                        robot,
                        robot_collision,
                        jax.tree.map(lambda value: value[None], obstacle),
                        robot.joint_var_cls(jnp.arange(0, timesteps - 1)),
                        robot.joint_var_cls(jnp.arange(1, timesteps)),
                    ),
                    kind="constraint_geq_zero",
                    name="pyroki_world_collision_sweep",
                )
            )

    @staticmethod
    def _world_collision(pyroki: Any, request: PlanRequest) -> list[Any]:
        collisions = []
        for obstacle in request.obstacles:
            obstacle.validate_for_motion()
            if obstacle.target_frame != "base_link":
                raise ValueError(
                    f"PyRoki obstacle {obstacle.obstacle_id!r} must use target_frame='base_link'"
                )
            center = np.asarray(obstacle.center, dtype=float)
            wxyz = PyrokiPlanner._xyzw_to_wxyz(
                matrix_to_quaternion(rpy_to_matrix(*obstacle.rpy))
            )
            if obstacle.obstacle_type == "box":
                collisions.append(
                    pyroki.collision.Box.from_extent(
                        extent=np.asarray(obstacle.scale, dtype=float),
                        position=center,
                        wxyz=wxyz,
                    )
                )
            elif obstacle.obstacle_type == "sphere":
                collisions.append(
                    pyroki.collision.Sphere.from_center_and_radius(
                        center=center,
                        radius=np.asarray(obstacle.scale[0]),
                    )
                )
            else:
                collisions.append(
                    pyroki.collision.Capsule.from_radius_height(
                        radius=np.asarray(obstacle.scale[0]),
                        height=np.asarray(obstacle.scale[1]),
                        position=center,
                        wxyz=wxyz,
                    )
                )
        return collisions

    @staticmethod
    def _target_errors(
        *,
        robot: Any,
        trajectory: np.ndarray,
        target_link_index: int,
        target_position: np.ndarray,
        target_xyzw: np.ndarray,
    ) -> tuple[float, float]:
        final_poses = np.asarray(robot.forward_kinematics(trajectory[-1]), dtype=float)
        final_wxyz = final_poses[target_link_index, :4]
        final_position = final_poses[target_link_index, 4:]
        position_error = float(np.linalg.norm(final_position - target_position))
        target_wxyz = PyrokiPlanner._xyzw_to_wxyz(target_xyzw)
        dot = float(np.clip(abs(np.dot(final_wxyz, target_wxyz)), 0.0, 1.0))
        orientation_error = float(2.0 * np.arccos(dot))
        return position_error, orientation_error

    @staticmethod
    def _xyzw_to_wxyz(quaternion: np.ndarray) -> np.ndarray:
        quaternion = np.asarray(quaternion, dtype=float)
        return quaternion[[3, 0, 1, 2]]
