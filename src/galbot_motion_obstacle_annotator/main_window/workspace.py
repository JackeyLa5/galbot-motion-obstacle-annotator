from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pyvista as pv
from PySide6.QtWidgets import QGroupBox, QHBoxLayout, QLabel, QMessageBox, QPushButton, QVBoxLayout
from vtkmodules.vtkRenderingCore import vtkPointPicker

from ..planning.pyroki import express_obstacles_in_base_frame, resolve_pyroki_urdf_path
from ..robot_model import load_urdf_link_transforms
from ..robot_state import ARM_JOINT_NAMES
from .widgets import WorkspaceSamplerWorker


class WorkspaceMixin:
    def _build_workspace_group(self) -> QGroupBox:
        group = QGroupBox("机械臂工作空间")
        layout = QVBoxLayout(group)
        hint = QLabel(
            "按当前“运动链”的关节限位随机采样，显示末端大致可达范围；仅表示位置可达，不代表任意抓取朝向都可行。"
        )
        hint.setWordWrap(True)
        hint.setObjectName("sectionHint")
        layout.addWidget(hint)

        row = QHBoxLayout()
        self.workspace_sample_count_spin = self._spin(200.0, 5000.0, 100.0)
        self.workspace_sample_count_spin.setDecimals(0)
        self.workspace_sample_count_spin.setValue(1500.0)
        row.addWidget(self.workspace_sample_count_spin)
        self.workspace_compute_button = QPushButton("计算工作空间")
        self.workspace_compute_button.clicked.connect(self.compute_reachable_workspace)
        row.addWidget(self.workspace_compute_button)
        self.workspace_toggle_button = QPushButton("显示工作空间")
        self.workspace_toggle_button.setEnabled(False)
        self.workspace_toggle_button.clicked.connect(self._toggle_workspace_visibility)
        row.addWidget(self.workspace_toggle_button)
        layout.addLayout(row)

        self.workspace_pick_button = QPushButton("点击可达点查看姿态")
        self.workspace_pick_button.setEnabled(False)
        self.workspace_pick_button.clicked.connect(self._toggle_workspace_pick_mode)
        layout.addWidget(self.workspace_pick_button)

        self.workspace_status_label = QLabel("尚未计算")
        self.workspace_status_label.setWordWrap(True)
        layout.addWidget(self.workspace_status_label)
        self.workspace_pose_label = QLabel("尚未预览姿态")
        self.workspace_pose_label.setWordWrap(True)
        layout.addWidget(self.workspace_pose_label)
        return group

    def compute_reachable_workspace(self) -> None:
        if self.workspace_worker is not None and self.workspace_worker.isRunning():
            return
        if not self.robot_visuals:
            QMessageBox.warning(self, "没有机器人模型", "请先加载机器人模型。")
            return
        path = Path(self.robot_path_edit.text()).expanduser()
        if not path.is_file():
            QMessageBox.warning(self, "没有机器人模型", "请先加载有效的机器人 URDF。")
            return
        tip_link = self.tcp_link_edit.text().strip()
        if not tip_link:
            QMessageBox.warning(self, "没有目标 Link", "请先在“工具 TCP 抓取姿态”中填写工具 TCP Link。")
            return
        arm_name = self.tcp_arm_combo.currentText()
        active_joint_names = ARM_JOINT_NAMES[arm_name]
        sample_count = int(self.workspace_sample_count_spin.value())

        pyroki_available, _reason = self.planner_registry.get("pyroki").is_available()
        obstacles_base_frame = []
        use_obstacles = pyroki_available and self.environment_collision_check.isChecked() and self.obstacles
        if use_obstacles:
            try:
                obstacles_base_frame = express_obstacles_in_base_frame(
                    self.obstacles, self.robot_base_transform
                )
            except ValueError as error:
                QMessageBox.warning(self, "障碍物坐标转换失败", str(error))
                return

        self._stop_workspace_pick_mode()
        self.workspace_active_joint_names = active_joint_names
        self.workspace_requested_sample_count = sample_count
        self.workspace_used_pyroki = pyroki_available
        self.workspace_used_obstacles = bool(obstacles_base_frame)
        self.workspace_compute_button.setEnabled(False)
        self.workspace_toggle_button.setEnabled(False)
        self.workspace_pick_button.setEnabled(False)
        mode_note = "（PyRoki 碰撞过滤中…）" if pyroki_available else ""
        self.workspace_status_label.setText(f"正在计算 {sample_count} 个采样点{mode_note}……")
        self.workspace_started_at = time.perf_counter()
        worker_urdf_path = resolve_pyroki_urdf_path(path) if pyroki_available else path
        self.workspace_worker = WorkspaceSamplerWorker(
            worker_urdf_path,
            tip_link,
            active_joint_names,
            dict(self.robot_joint_limits),
            dict(self.robot_state.joint_positions),
            sample_count,
            use_collision_awareness=pyroki_available,
            obstacles_base_frame=obstacles_base_frame,
        )
        self.workspace_worker.finished_with_points.connect(self._workspace_sampling_finished)
        self.workspace_worker.finished.connect(self._workspace_worker_finished)
        self.workspace_worker.start()

    def _workspace_worker_finished(self) -> None:
        if self.workspace_worker is not None:
            self.workspace_worker.deleteLater()
        self.workspace_worker = None
        self.workspace_compute_button.setEnabled(True)

    def _workspace_sampling_finished(self, result) -> None:
        elapsed = time.perf_counter() - self.workspace_started_at
        points, joint_values = result
        if points is None:
            message = self.workspace_worker.error_message if self.workspace_worker is not None else "未知错误"
            QMessageBox.warning(self, "工作空间计算失败", message)
            self.workspace_status_label.setText("计算失败，请检查“工具 TCP Link”是否正确")
            return
        self.workspace_points = np.asarray(points, dtype=float)
        self.workspace_joint_values = np.asarray(joint_values, dtype=float)
        diagnostics = self.workspace_worker.diagnostics if self.workspace_worker is not None else {}
        found = len(self.workspace_points)
        if diagnostics:
            obstacle_note = "和障碍物碰撞" if self.workspace_used_obstacles else ""
            short_note = (
                f"，仅找到 {found}/{self.workspace_requested_sample_count} 个（可行空间较小）"
                if found < self.workspace_requested_sample_count
                else ""
            )
            self.workspace_status_label.setText(
                f"{found} 个无碰撞采样点（已用 PyRoki 过滤自碰撞{obstacle_note}）"
                f"{short_note}，尝试 {diagnostics['candidates_drawn']} 次，用时 {elapsed:.1f}s"
            )
        else:
            caveat = "" if self.workspace_used_pyroki else "（未安装 PyRoki，未做碰撞检查，可能包含穿模姿态）"
            self.workspace_status_label.setText(f"{found} 个采样点{caveat}，用时 {elapsed:.1f}s")
        self.workspace_toggle_button.setEnabled(True)
        self.workspace_toggle_button.setText("隐藏工作空间")
        self.workspace_pick_button.setEnabled(True)
        self.workspace_visible = True
        self._render_reachable_workspace()

    def _toggle_workspace_visibility(self) -> None:
        if self.workspace_points is None:
            return
        self.workspace_visible = not self.workspace_visible
        if not self.workspace_visible:
            self._stop_workspace_pick_mode()
        self.workspace_toggle_button.setText("隐藏工作空间" if self.workspace_visible else "显示工作空间")
        self.workspace_pick_button.setEnabled(self.workspace_visible)
        self._render_reachable_workspace()

    def _render_reachable_workspace(self) -> None:
        self.plotter.remove_actor("reachable_workspace", render=False)
        self.workspace_actor = None
        if self.workspace_points is None or not len(self.workspace_points) or not self.workspace_visible:
            self.plotter.render()
            return
        base_rotation = self.robot_base_transform[:3, :3]
        base_translation = self.robot_base_transform[:3, 3]
        world_points = self.workspace_points @ base_rotation.T + base_translation
        self.workspace_actor = self.plotter.add_mesh(
            pv.PolyData(world_points),
            name="reachable_workspace",
            style="points",
            color="#ffd166",
            point_size=4.0,
            opacity=0.35,
            render_points_as_spheres=False,
            pickable=self.workspace_pick_mode,
            reset_camera=False,
        )
        self.plotter.render()

    def _clear_reachable_workspace(self) -> None:
        self._stop_workspace_pick_mode()
        self.plotter.remove_actor("reachable_workspace", render=False)
        self.workspace_actor = None
        self.workspace_points = None
        self.workspace_joint_values = None
        self.workspace_active_joint_names = ()
        self.workspace_visible = False
        self.workspace_toggle_button.setEnabled(False)
        self.workspace_toggle_button.setText("显示工作空间")
        self.workspace_pick_button.setEnabled(False)
        self.workspace_pose_label.setText("尚未预览姿态")
        self.workspace_status_label.setText("机器人状态已变化，工作空间需要重新计算")

    def _toggle_workspace_pick_mode(self) -> None:
        if self.workspace_pick_mode:
            self._stop_workspace_pick_mode()
            self.statusBar().showMessage("已停止查看工作空间姿态")
            return
        if self.workspace_points is None or not self.workspace_visible or self.workspace_actor is None:
            QMessageBox.warning(self, "没有工作空间", "请先计算并显示工作空间。")
            return
        self._stop_selection_mode()
        self.tcp_selection_mode = False
        self.workspace_pick_mode = True
        self.workspace_actor.SetPickable(True)
        self.workspace_pick_button.setText("停止查看姿态")
        self.statusBar().showMessage("工作空间点选模式：右键点击可达点云中的点，预览对应的机械臂姿态")

    def _stop_workspace_pick_mode(self) -> None:
        if not self.workspace_pick_mode:
            return
        self.workspace_pick_mode = False
        if self.workspace_actor is not None:
            self.workspace_actor.SetPickable(False)
        self.workspace_pick_button.setText("点击可达点查看姿态")

    def _pick_workspace_point_from_qt_position(self, qt_x: float, qt_y: float) -> bool:
        if self.workspace_actor is None or self.workspace_joint_values is None:
            return False
        widget = self.plotter.interactor
        render_width, render_height = self.plotter.render_window.GetSize()
        if widget.width() <= 0 or widget.height() <= 0 or render_width <= 0 or render_height <= 0:
            return False
        display_x = qt_x * render_width / widget.width()
        display_y = (widget.height() - qt_y - 1.0) * render_height / widget.height()
        picker = vtkPointPicker()
        picker.SetTolerance(0.02)
        picker.AddPickList(self.workspace_actor)
        picker.PickFromListOn()
        if picker.Pick(display_x, display_y, 0.0, self.plotter.renderer) != 1:
            self.statusBar().showMessage("没有选中工作空间中的点，请在黄色点云上重新右键")
            return False
        if picker.GetDataSet() is None or picker.GetPointId() < 0:
            return False
        point_index = picker.GetPointId()
        if not 0 <= point_index < len(self.workspace_joint_values):
            return False
        self._preview_workspace_pose(point_index)
        return True

    def _restore_workspace_pick_camera(self) -> None:
        if self.pending_workspace_pick_camera_state is not None:
            self._restore_camera_state(self.pending_workspace_pick_camera_state)
            self.pending_workspace_pick_camera_state = None
            self.plotter.render()

    def _preview_workspace_pose(self, point_index: int) -> None:
        path = Path(self.robot_path_edit.text()).expanduser()
        if not path.is_file():
            return
        sampled_values = self.workspace_joint_values[point_index]
        overrides = dict(zip(self.workspace_active_joint_names, sampled_values))
        joint_positions = dict(self.robot_state.joint_positions)
        joint_positions.update(overrides)
        transforms = load_urdf_link_transforms(path, joint_positions)
        for visual in self.robot_visuals:
            actor = self.robot_actors.get(visual.name)
            link_transform = transforms.get(visual.link_name)
            if actor is None or link_transform is None or visual.local_transform is None:
                continue
            actor.user_matrix = self.robot_base_transform @ link_transform @ visual.local_transform
        self.plotter.render()
        formatted = ", ".join(
            f"{name}={value:.3f}"
            for name, value in zip(self.workspace_active_joint_names, sampled_values)
        )
        self.workspace_pose_label.setText(f"预览姿态（未写回当前机器人状态）：{formatted}")
