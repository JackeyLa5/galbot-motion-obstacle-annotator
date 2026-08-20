from __future__ import annotations

from pathlib import Path

import numpy as np
import pyvista as pv
from PySide6.QtWidgets import QMessageBox

from ..geometry import matrix_to_quaternion
from ..planning.models import PlanRequest, PlanResult
from ..planning.pyroki import express_world_scene_in_base_frame, resolve_pyroki_urdf_path
from ..robot_model import load_urdf_actuated_joint_names, load_urdf_link_transforms
from ..robot_state import ARM_JOINT_NAMES, GALBOT_REFERENCE_JOINT_NAMES, LEG_JOINT_NAMES
from .widgets import PlannerWorker


class PlanningMixin:
    def _clear_plan_result(self) -> None:
        self.playback_timer.stop()
        self.plan_result = None
        self.playback_frames.clear()
        self.playback_tcp_points = np.empty((0, 3), dtype=float)
        self.playback_joint_names = ()
        self.playback_joint_positions = np.empty((0, 0), dtype=float)
        self.playback_index = 0
        if hasattr(self, "play_button"):
            self.play_button.setText("播放")
            self.play_button.setEnabled(False)
            self.replay_button.setEnabled(False)
            self.plan_status_label.setText("目标已变化，请重新规划")
        self.plotter.remove_actor("planned_tcp_path", render=False)
        self._clear_compare_paths()
        if hasattr(self, "workspace_toggle_button") and self.workspace_points is not None:
            self._clear_reachable_workspace()

    def plan_tcp_target(self) -> None:
        if self.planner_worker is not None and self.planner_worker.isRunning():
            return
        if not self.tcp_target_selected:
            QMessageBox.warning(self, "没有 TCP 目标", "请先选择抓取点或应用 TCP 姿态。")
            return
        planner_id = self.planner_combo.currentData()
        planner = self.planner_registry.get(planner_id)
        available, reason = planner.is_available()
        if not available:
            QMessageBox.warning(self, "规划器不可用", reason)
            return
        try:
            request = self._build_plan_request(planner_id)
        except ValueError as error:
            QMessageBox.warning(self, "规划参数无效", str(error))
            return
        self._print_plan_request_debug(planner_id, request)
        self._clear_plan_result()
        self.plan_button.setEnabled(False)
        self.plan_status_label.setText(f"正在使用 {planner.metadata.display_name} 规划……")
        self.planner_worker = PlannerWorker(planner, request)
        self.planner_worker.finished_with_result.connect(self._planning_finished)
        self.planner_worker.finished.connect(self._planner_thread_finished)
        self.planner_worker.start()

    def _build_plan_request(self, planner_id: str) -> PlanRequest:
        robot_path = Path(self.robot_path_edit.text()).expanduser()
        if not robot_path.is_file():
            raise ValueError("请先加载有效的机器人 URDF")
        planning_start_positions = {
            "leg": self.robot_state.positions_for(LEG_JOINT_NAMES),
            "left_arm": self.robot_state.positions_for(ARM_JOINT_NAMES["left_arm"]),
            "right_arm": self.robot_state.positions_for(ARM_JOINT_NAMES["right_arm"]),
        }
        request_target = self.tcp_pose_target()
        request_obstacles = list(self.obstacles)
        options: dict[str, object] = {}
        if planner_id == "galbot-motion":
            options["galbot_whole_body_joint_positions"] = self.robot_state.positions_for(
                GALBOT_REFERENCE_JOINT_NAMES
            )
            options["galbot_base_pose"] = self._robot_base_pose_values()
        elif planner_id == "pyroki":
            request_target, request_obstacles = express_world_scene_in_base_frame(
                request_target,
                request_obstacles,
                self.robot_base_transform,
            )
            pyroki_robot_path = resolve_pyroki_urdf_path(robot_path)
            joint_names = load_urdf_actuated_joint_names(pyroki_robot_path)
            active_joint_names = ARM_JOINT_NAMES[self.tcp_arm_combo.currentText()]
            if self.pyroki_leg_planning_check.isChecked():
                active_joint_names = LEG_JOINT_NAMES[:3] + active_joint_names
            options.update(
                {
                    "pyroki_urdf_path": pyroki_robot_path,
                    "pyroki_joint_names": joint_names,
                    "pyroki_start_joint_positions": {
                        name: self.robot_state.joint_positions.get(name, 0.0) for name in joint_names
                    },
                    "pyroki_active_joint_names": active_joint_names,
                    "pyroki_target_link": self.tcp_link_edit.text().strip(),
                    "pyroki_keep_leg_upright": self.pyroki_leg_planning_check.isChecked(),
                    "pyroki_locked_joint_positions": {
                        "leg_joint4": 0.0,
                        "leg_joint5": 0.0,
                    },
                }
            )
        return PlanRequest(
            target=request_target,
            start_joint_positions=planning_start_positions,
            obstacles=request_obstacles,
            collision_check=False,
            environment_collision_check=self.environment_collision_check.isChecked(),
            options=options,
        )

    def _robot_base_pose_values(self) -> list[float]:
        position = self.robot_base_transform[:3, 3]
        quaternion = matrix_to_quaternion(self.robot_base_transform[:3, :3])
        return [*position.tolist(), *quaternion.tolist()]

    def _print_plan_request_debug(self, planner_id: str, request: PlanRequest) -> None:
        print("\n=== Planning Request ===")
        print(f"planner_id                  : {planner_id}")
        print(f"chain_name                  : {request.target.chain_name}")
        print(f"target_frame                : {request.target.reference_frame}")
        print(f"target_link                 : {request.options.get('pyroki_target_link', request.target.frame_id)}")
        print(f"target_position             : {request.target.position.tolist()}")
        print(f"target_orientation_xyzw     : {request.target.orientation_xyzw.tolist()}")
        print(f"collision_check             : {request.collision_check}")
        print(f"environment_collision_check : {request.environment_collision_check}")
        print(f"timeout_seconds             : {request.timeout_seconds}")
        print(f"robot_base_position_world   : {self.robot_base_transform[:3, 3].tolist()}")
        print(
            "robot_base_orientation_xyzw : "
            f"{matrix_to_quaternion(self.robot_base_transform[:3, :3]).tolist()}"
        )
        print(f"obstacle_count              : {len(request.obstacles)}")
        for obstacle in request.obstacles:
            print(
                "  obstacle                  : "
                f"id={obstacle.obstacle_id} type={obstacle.obstacle_type} "
                f"frame={obstacle.target_frame} center={obstacle.center.tolist()} "
                f"rpy={obstacle.rpy.tolist()} scale={obstacle.scale.tolist()}"
            )

        if planner_id != "pyroki":
            return
        urdf_path = request.options.get("pyroki_urdf_path")
        joint_names = tuple(request.options.get("pyroki_joint_names", ()))
        active_joint_names = tuple(request.options.get("pyroki_active_joint_names", ()))
        start_joint_positions = request.options.get("pyroki_start_joint_positions", {})
        print(f"pyroki_urdf_path            : {urdf_path}")
        print(f"pyroki_joint_count          : {len(joint_names)}")
        print(f"pyroki_joint_names          : {joint_names}")
        print(f"pyroki_active_joint_names   : {active_joint_names}")
        print(f"pyroki_locked_joint_count   : {len(joint_names) - len(active_joint_names)}")
        print("pyroki_start_joint_positions:")
        for name in joint_names:
            print(f"  - {name}: {start_joint_positions.get(name)}")

    def _planning_finished(self, result: PlanResult) -> None:
        self._print_plan_result_debug(result)
        self.plan_result = result
        self.plan_button.setEnabled(True)
        if not result.success:
            self.plan_status_label.setText(
                f"不可达 / 规划失败：{result.status}\n{result.message or '规划器未返回可用轨迹'}"
            )
            self.play_button.setEnabled(False)
            self.replay_button.setEnabled(False)
            return
        trajectory = result.trajectories.get(self.tcp_arm_combo.currentText())
        if trajectory is None:
            self.plan_status_label.setText("规划器返回成功，但没有当前手臂的轨迹")
            return
        try:
            self._prepare_playback(trajectory.positions, result.diagnostics)
        except (OSError, RuntimeError, ValueError) as error:
            self.plan_status_label.setText(f"轨迹可达，但可视化准备失败：{error}")
            return
        elapsed = "" if result.planning_seconds is None else f"，耗时 {result.planning_seconds:.3f}s"
        self.plan_status_label.setText(
            f"可达：{result.planner_id}，{len(self.playback_frames)} 帧{elapsed}"
        )
        self.play_button.setEnabled(True)
        self.replay_button.setEnabled(True)
        self._render_planned_tcp_path()

    def _print_plan_result_debug(self, result: PlanResult) -> None:
        print("\n=== Planning Result ===")
        print(f"planner_id   : {result.planner_id}")
        print(f"success      : {result.success}")
        print(f"status       : {result.status}")
        print(f"message      : {result.message}")
        print(f"time_seconds : {result.planning_seconds}")
        if result.diagnostics:
            print("diagnostics  :")
            for key, value in result.diagnostics.items():
                print(f"  - {key}: {value}")
        if result.trajectories:
            print("trajectories :")
            for chain_name, trajectory in result.trajectories.items():
                print(
                    f"  - {chain_name}: positions_shape={trajectory.positions.shape} "
                    f"timestamps_shape={None if trajectory.timestamps is None else trajectory.timestamps.shape}"
                )

    def _planner_thread_finished(self) -> None:
        if self.planner_worker is not None:
            self.planner_worker.deleteLater()
        self.planner_worker = None
        if not self.compare_active:
            self.plan_button.setEnabled(True)

    def compare_planners(self) -> None:
        """Plan the same TCP target with both PyRoki and Galbot Motion and overlay both paths.

        Lets the user check whether Galbot Motion's own planner also finds the
        target reachable, alongside PyRoki - not just visualize one planner's result.
        """
        if self.planner_worker is not None and self.planner_worker.isRunning():
            return
        if not self.tcp_target_selected:
            QMessageBox.warning(self, "没有 TCP 目标", "请先选择抓取点或应用 TCP 姿态。")
            return
        self._clear_plan_result()
        self.compare_requests = {}
        self.compare_results = {}
        self.compare_unavailable = {}
        for planner_id in ("pyroki", "galbot-motion"):
            planner = self.planner_registry.get(planner_id)
            available, reason = planner.is_available()
            if not available:
                self.compare_unavailable[planner_id] = reason
                continue
            try:
                self.compare_requests[planner_id] = self._build_plan_request(planner_id)
            except ValueError as error:
                self.compare_unavailable[planner_id] = str(error)
        if not self.compare_requests:
            QMessageBox.warning(self, "规划器不可用", "PyRoki 和 Galbot Motion 均不可用，无法对比。")
            return
        self.compare_queue = list(self.compare_requests)
        self.compare_active = True
        self.compare_button.setEnabled(False)
        self.plan_button.setEnabled(False)
        self.plan_status_label.setText("正在对比 PyRoki / Galbot Motion 规划……")
        self._run_next_compare_plan()

    def _run_next_compare_plan(self) -> None:
        if not self.compare_queue:
            self.compare_active = False
            self.compare_button.setEnabled(True)
            self.plan_button.setEnabled(True)
            self._show_compare_results()
            return
        planner_id = self.compare_queue.pop(0)
        planner = self.planner_registry.get(planner_id)
        request = self.compare_requests[planner_id]
        self._print_plan_request_debug(planner_id, request)
        self.planner_worker = PlannerWorker(planner, request)
        self.planner_worker.finished_with_result.connect(
            lambda result, planner_id=planner_id: self.compare_results.__setitem__(planner_id, result)
        )
        self.planner_worker.finished.connect(self._planner_thread_finished)
        self.planner_worker.finished.connect(self._run_next_compare_plan)
        self.planner_worker.start()

    def _show_compare_results(self) -> None:
        self._clear_compare_paths()
        colors = {"pyroki": "#00e5ff", "galbot-motion": "#ffb703"}
        lines = []
        for planner_id in ("pyroki", "galbot-motion"):
            planner = self.planner_registry.get(planner_id)
            name = planner.metadata.display_name
            if planner_id in self.compare_unavailable:
                lines.append(f"{name}: 不可用（{self.compare_unavailable[planner_id]}）")
                continue
            result = self.compare_results.get(planner_id)
            if result is None:
                lines.append(f"{name}: 未返回结果")
                continue
            self._print_plan_result_debug(result)
            if not result.success:
                lines.append(f"{name}: 不可达（{result.status}：{result.message or '规划器未返回可用轨迹'}）")
                continue
            trajectory = result.trajectories.get(self.tcp_arm_combo.currentText())
            if trajectory is None:
                lines.append(f"{name}: 规划成功，但没有当前手臂的轨迹")
                continue
            try:
                tcp_points = self._compute_tcp_path_points(
                    trajectory.positions, result.diagnostics, self.tcp_arm_combo.currentText()
                )
            except (OSError, RuntimeError, ValueError) as error:
                lines.append(f"{name}: 可达，但可视化失败（{error}）")
                continue
            self._render_compare_path(planner_id, tcp_points, colors[planner_id])
            elapsed = "" if result.planning_seconds is None else f"，耗时 {result.planning_seconds:.3f}s"
            lines.append(f"{name}: 可达，{len(tcp_points)} 帧{elapsed}")
        self.plan_status_label.setText("\n".join(lines))

    def _compute_tcp_path_points(self, positions: np.ndarray, diagnostics, arm_name: str) -> np.ndarray:
        robot_path = Path(self.robot_path_edit.text()).expanduser()
        joint_names = diagnostics.get("joint_names")
        if joint_names is None:
            joint_names = ARM_JOINT_NAMES[arm_name]
        joint_names = tuple(str(name) for name in joint_names)
        if positions.shape[1] != len(joint_names):
            raise ValueError("轨迹关节维度与关节名称数量不一致")
        tcp_link = self.tcp_link_edit.text().strip()
        tcp_points = []
        for values in positions:
            joint_positions = dict(self.robot_state.joint_positions)
            joint_positions.update(zip(joint_names, values))
            transforms = load_urdf_link_transforms(robot_path, joint_positions)
            tcp_points.append((self.robot_base_transform @ transforms[tcp_link])[:3, 3])
        return np.asarray(tcp_points, dtype=float)

    def _render_compare_path(self, planner_id: str, tcp_points: np.ndarray, color: str) -> None:
        if len(tcp_points) < 2:
            return
        actor_name = f"compare_tcp_path_{planner_id}"
        self.plotter.add_mesh(
            pv.lines_from_points(tcp_points),
            name=actor_name,
            color=color,
            line_width=5,
            pickable=False,
            reset_camera=False,
        )
        self.compare_path_actor_names.add(actor_name)
        self.plotter.render()

    def _clear_compare_paths(self) -> None:
        for actor_name in self.compare_path_actor_names:
            self.plotter.remove_actor(actor_name, render=False)
        self.compare_path_actor_names.clear()

    def _prepare_playback(self, positions: np.ndarray, diagnostics) -> None:
        robot_path = Path(self.robot_path_edit.text()).expanduser()
        arm_name = self.tcp_arm_combo.currentText()
        joint_names = diagnostics.get("joint_names")
        if joint_names is None:
            joint_names = ARM_JOINT_NAMES[arm_name]
        joint_names = tuple(str(name) for name in joint_names)
        if positions.shape[1] != len(joint_names):
            raise ValueError("轨迹关节维度与关节名称数量不一致")
        frames: list[dict[str, np.ndarray]] = []
        tcp_points: list[np.ndarray] = []
        for values in positions:
            joint_positions = dict(self.robot_state.joint_positions)
            joint_positions.update(zip(joint_names, values))
            transforms = load_urdf_link_transforms(robot_path, joint_positions)
            frames.append(transforms)
            tcp_points.append(
                (self.robot_base_transform @ transforms[self.tcp_link_edit.text().strip()])[:3, 3]
            )
        self.playback_frames = frames
        self.playback_tcp_points = np.asarray(tcp_points, dtype=float)
        self.playback_joint_names = joint_names
        self.playback_joint_positions = np.asarray(positions, dtype=float)
        self.playback_index = 0

    def _render_planned_tcp_path(self) -> None:
        self.plotter.remove_actor("planned_tcp_path", render=False)
        if len(self.playback_tcp_points) >= 2:
            self.plotter.add_mesh(
                pv.lines_from_points(self.playback_tcp_points),
                name="planned_tcp_path",
                color="#00e5ff",
                line_width=5,
                pickable=False,
                reset_camera=False,
            )
        self.plotter.render()

    def toggle_playback(self) -> None:
        if not self.playback_frames:
            return
        if self.playback_timer.isActive():
            self.playback_timer.stop()
            self.play_button.setText("播放")
            return
        if self.playback_index >= len(self.playback_frames) - 1:
            self.playback_index = 0
        self.playback_timer.start(80)
        self.play_button.setText("暂停")

    def replay_trajectory(self) -> None:
        if not self.playback_frames:
            return
        self.playback_timer.stop()
        self.playback_index = 0
        self._show_playback_frame(0)
        self.playback_timer.start(80)
        self.play_button.setText("暂停")

    def _advance_playback(self) -> None:
        if not self.playback_frames:
            self.playback_timer.stop()
            return
        self._show_playback_frame(self.playback_index)
        self.playback_index += 1
        if self.playback_index >= len(self.playback_frames):
            self.playback_timer.stop()
            self.play_button.setText("播放")
            if self.keep_final_pose_check.isChecked():
                self._apply_playback_final_pose()

    def _show_playback_frame(self, frame_index: int) -> None:
        transforms = self.playback_frames[frame_index]
        for visual in self.robot_visuals:
            actor = self.robot_actors.get(visual.name)
            if actor is None or visual.link_name not in transforms:
                continue
            actor.user_matrix = (
                self.robot_base_transform @ transforms[visual.link_name] @ visual.local_transform
            )
        self.plotter.render()

    def _apply_playback_final_pose(self) -> None:
        if not len(self.playback_frames) or not len(self.playback_joint_names):
            return
        if self.playback_joint_positions.shape[0] == 0:
            return
        final_positions = self.playback_joint_positions[-1]
        self.robot_state.update(dict(zip(self.playback_joint_names, final_positions, strict=False)))
        self._sync_joint_state_controls()
        self.robot_joint_plan_invalidated = False
        if self.tcp_target_selected:
            self._refresh_tcp_target_label()
        self.plan_status_label.setText("播放结束：已将末帧姿态写回当前机器人状态")
