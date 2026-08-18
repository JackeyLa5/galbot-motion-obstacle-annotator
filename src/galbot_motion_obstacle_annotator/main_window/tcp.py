from __future__ import annotations

from pathlib import Path

import numpy as np
import pyvista as pv
from PySide6.QtCore import QSignalBlocker, Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
)
from vtkmodules.vtkRenderingCore import vtkPointPicker

from ..geometry import matrix_to_quaternion, matrix_to_rpy, rpy_to_matrix
from ..planning.models import PoseTarget
from ..robot_model import load_tcp_gripper_visuals
from .widgets import FocusedWheelComboBox


class TcpMixin:
    def _build_tcp_group(self) -> QGroupBox:
        group = QGroupBox("工具 TCP 抓取姿态")
        layout = QVBoxLayout(group)
        hint = QLabel(
            "固定使用 is_tool_pose=true：TCP 表单和规划目标使用机器人 base_link 坐标系，场景里按当前机器人位姿渲染。"
        )
        hint.setWordWrap(True)
        hint.setObjectName("sectionHint")
        layout.addWidget(hint)

        form = QFormLayout()
        self.tcp_arm_combo = FocusedWheelComboBox()
        self.tcp_arm_combo.addItems(["left_arm", "right_arm"])
        self.tcp_arm_combo.currentTextChanged.connect(self._reload_tcp_gripper)
        form.addRow("运动链", self.tcp_arm_combo)
        self.tcp_mount_edit = QLineEdit("left_arm_end_effector_mount_link")
        self.tcp_link_edit = QLineEdit("left_gripper_tcp_link")
        form.addRow("安装 Link", self.tcp_mount_edit)
        form.addRow("工具 TCP Link", self.tcp_link_edit)
        self.tcp_position_spins = self._add_vector_row(form, "TCP 位置 XYZ（base_link）", -1000.0, 1000.0, 0.01)
        self.tcp_rpy_spins = self._add_vector_row(form, "TCP 旋转 RPY°（base_link）", -360.0, 360.0, 1.0)
        layout.addLayout(form)

        row = QHBoxLayout()
        self.tcp_select_button = QPushButton("选择抓取姿态")
        self.tcp_select_button.clicked.connect(self._toggle_tcp_pose_edit_mode)
        self.tcp_apply_button = QPushButton("应用 TCP 姿态")
        self.tcp_apply_button.setProperty("cssClass", "primary")
        self.tcp_apply_button.clicked.connect(self._apply_tcp_form)
        row.addWidget(self.tcp_select_button)
        row.addWidget(self.tcp_apply_button)
        layout.addLayout(row)

        planning_form = QFormLayout()
        self.planner_combo = FocusedWheelComboBox()
        for planner in self.planner_registry.all():
            available, reason = planner.is_available()
            label = planner.metadata.display_name if available else f"{planner.metadata.display_name}（不可用）"
            self.planner_combo.addItem(label, planner.metadata.planner_id)
            self.planner_combo.setItemData(self.planner_combo.count() - 1, reason, Qt.ToolTipRole)
        pyroki_index = self.planner_combo.findData("pyroki")
        if pyroki_index >= 0:
            self.planner_combo.setCurrentIndex(pyroki_index)
        planning_form.addRow("规划器", self.planner_combo)
        self.environment_collision_check = QCheckBox("启用场景障碍物检查")
        self.environment_collision_check.setChecked(False)
        planning_form.addRow(self.environment_collision_check)
        self.pyroki_leg_planning_check = QCheckBox("PyRoki：允许腿部参与规划")
        self.pyroki_leg_planning_check.setChecked(False)
        self.pyroki_leg_planning_check.setToolTip(
            "默认只规划当前目标手臂；勾选后额外放开腿部 5 个关节。Galbot Motion 不使用此选项。"
        )
        planning_form.addRow(self.pyroki_leg_planning_check)
        layout.addLayout(planning_form)

        plan_row = QHBoxLayout()
        self.plan_button = QPushButton("检查是否可达")
        self.plan_button.setProperty("cssClass", "primary")
        self.plan_button.clicked.connect(self.plan_tcp_target)
        self.play_button = QPushButton("播放")
        self.play_button.clicked.connect(self.toggle_playback)
        self.replay_button = QPushButton("重播")
        self.replay_button.clicked.connect(self.replay_trajectory)
        self.play_button.setEnabled(False)
        self.replay_button.setEnabled(False)
        plan_row.addWidget(self.plan_button)
        plan_row.addWidget(self.play_button)
        plan_row.addWidget(self.replay_button)
        layout.addLayout(plan_row)
        self.keep_final_pose_check = QCheckBox("播放结束后保留末帧姿态")
        self.keep_final_pose_check.setChecked(False)
        self.keep_final_pose_check.setToolTip("不勾选时播放结束仅为预览；勾选后会把末帧写回当前机器人关节状态。")
        layout.addWidget(self.keep_final_pose_check)
        self.plan_status_label = QLabel("尚未规划")
        self.plan_status_label.setWordWrap(True)
        layout.addWidget(self.plan_status_label)
        self.tcp_target_label = QLabel("尚未选择 TCP 目标")
        self.tcp_target_label.setWordWrap(True)
        layout.addWidget(self.tcp_target_label)
        return group

    def _tcp_gizmo_actors(self) -> tuple[object, ...]:
        if self.tcp_transform_widget is None:
            return ()
        return (
            *self.tcp_transform_widget._arrows,
            *self.tcp_transform_widget._circles,
        )

    def _tcp_visual_actors(self) -> tuple[object, ...]:
        actors: list[object] = []
        for actor_name in self.tcp_actor_names:
            actor = self.plotter.renderer.actors.get(actor_name)
            if actor is not None:
                actors.append(actor)
        return tuple(actors)

    def _toggle_tcp_pose_edit_mode(self) -> None:
        self.tcp_pose_edit_mode = not self.tcp_pose_edit_mode
        if self.tcp_pose_edit_mode:
            self._stop_selection_mode()
            self.tcp_selection_mode = True
            self.tcp_transform_active = True
            self._render_tcp_gripper()
            self.tcp_select_button.setText("结束抓取姿态编辑")
            self.statusBar().showMessage("抓取姿态编辑模式：右键选点，左键拖拽抓手，点击“应用 TCP 姿态”完成")
            return
        self.tcp_selection_mode = False
        self.tcp_transform_active = False
        self.tcp_select_button.setText("选择抓取姿态")
        self._render_tcp_gripper()
        self.statusBar().showMessage("已结束抓取姿态编辑")

    def _toggle_tcp_selection_mode(self) -> None:
        self.tcp_selection_mode = not self.tcp_selection_mode
        if self.tcp_selection_mode:
            self._stop_selection_mode()
            self.tcp_transform_active = False
            self._destroy_tcp_transform_widget()
            self.tcp_select_button.setText("停止选择 TCP")
            self.statusBar().showMessage("TCP 点选模式：在场景中右键选择抓取点")
        else:
            self.tcp_select_button.setText("选择抓取点")
            self._render_tcp_gripper()
            self.statusBar().showMessage("已停止 TCP 点选模式")

    def _pick_tcp_point_from_qt_position(self, qt_x: float, qt_y: float) -> bool:
        widget = self.plotter.interactor
        render_width, render_height = self.plotter.render_window.GetSize()
        if widget.width() <= 0 or widget.height() <= 0 or render_width <= 0 or render_height <= 0:
            return False
        display_x = qt_x * render_width / widget.width()
        display_y = (widget.height() - qt_y - 1.0) * render_height / widget.height()
        picker = vtkPointPicker()
        picker.SetTolerance(0.02)
        if picker.Pick(display_x, display_y, 0.0, self.plotter.renderer) != 1:
            self.statusBar().showMessage("没有选中点，请在可见点云上重新右键")
            return False
        if picker.GetDataSet() is None or picker.GetPointId() < 0:
            self.statusBar().showMessage("没有选中有效点云点，请重新右键")
            return False
        point = np.asarray(picker.GetPickPosition(), dtype=float)
        if point.shape != (3,) or not np.isfinite(point).all():
            return False
        self.tcp_target_selected = True
        self.tcp_transform_active = True
        self._set_tcp_pose(self._world_point_to_base(point), self.tcp_pose_matrix[:3, :3])
        self._clear_plan_result()
        return True

    def _restore_tcp_selection_camera(self) -> None:
        if self.pending_tcp_camera_state is not None:
            self._restore_camera_state(self.pending_tcp_camera_state)
            self.pending_tcp_camera_state = None
            self.plotter.render()

    def _tcp_pose_matrix_from_form(self) -> tuple[np.ndarray, np.ndarray]:
        position = np.array([spin.value() for spin in self.tcp_position_spins], dtype=float)
        rpy = np.deg2rad([spin.value() for spin in self.tcp_rpy_spins])
        matrix = np.eye(4, dtype=float)
        matrix[:3, :3] = rpy_to_matrix(*rpy)
        matrix[:3, 3] = position
        return matrix, position

    def _tcp_world_matrix(self) -> np.ndarray:
        return self.robot_base_transform @ self.tcp_pose_matrix

    def _world_point_to_base(self, point: np.ndarray) -> np.ndarray:
        world_point = np.ones(4, dtype=float)
        world_point[:3] = np.asarray(point, dtype=float)
        return (np.linalg.inv(self.robot_base_transform) @ world_point)[:3]

    def _refresh_tcp_target_label(self) -> None:
        position = self.tcp_pose_matrix[:3, 3]
        quaternion = matrix_to_quaternion(self.tcp_pose_matrix[:3, :3])
        world_position = self._tcp_world_matrix()[:3, 3]
        self.tcp_target_label.setText(
            f"TCP 目标：{self.tcp_link_edit.text()} | is_tool_pose=true | reference_frame=base_link\n"
            f"Base 位置：({position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f}) m\n"
            f"Base 四元数 XYZW：({quaternion[0]:.3f}, {quaternion[1]:.3f}, "
            f"{quaternion[2]:.3f}, {quaternion[3]:.3f})\n"
            f"World 预览：({world_position[0]:.3f}, {world_position[1]:.3f}, {world_position[2]:.3f}) m"
        )

    def _set_tcp_pose(self, position: np.ndarray, rotation: np.ndarray) -> None:
        matrix = np.eye(4, dtype=float)
        matrix[:3, :3] = rotation
        matrix[:3, 3] = np.asarray(position, dtype=float)
        self.tcp_pose_matrix = matrix
        rpy = np.rad2deg(matrix_to_rpy(rotation))
        blockers = [
            QSignalBlocker(widget)
            for widget in (*self.tcp_position_spins, *self.tcp_rpy_spins)
        ]
        for spin, value in zip(self.tcp_position_spins, position):
            spin.setValue(float(value))
        for spin, value in zip(self.tcp_rpy_spins, rpy):
            spin.setValue(float(value))
        del blockers
        self._render_tcp_gripper()
        self._refresh_tcp_target_label()

    def _apply_tcp_form(self) -> None:
        matrix, _position = self._tcp_pose_matrix_from_form()
        self.tcp_pose_matrix = matrix
        self.tcp_target_selected = True
        self.tcp_transform_active = False
        self.tcp_pose_edit_mode = False
        self.tcp_selection_mode = False
        self._clear_plan_result()
        self._render_tcp_gripper()
        self.tcp_select_button.setText("选择抓取姿态")
        self._refresh_tcp_target_label()

    def tcp_pose_target(self) -> PoseTarget:
        """Return the currently edited TCP pose for a visualization-only plan request."""
        return PoseTarget(
            chain_name=self.tcp_arm_combo.currentText(),
            position=self.tcp_pose_matrix[:3, 3].copy(),
            orientation_xyzw=matrix_to_quaternion(self.tcp_pose_matrix[:3, :3]),
            reference_frame="base_link",
            frame_id=self.tcp_link_edit.text().strip(),
        )

    def _destroy_tcp_transform_widget(self) -> None:
        self.tcp_last_widget_matrix = None
        if self.tcp_transform_widget is None:
            if self.tcp_gizmo_actor is not None:
                self.plotter.remove_actor("tcp_pose_gizmo", render=False)
                self.tcp_gizmo_actor = None
            return
        self.tcp_transform_widget.disable()
        self.tcp_transform_widget.remove()
        self.tcp_transform_widget = None
        if self.tcp_gizmo_actor is not None:
            self.plotter.remove_actor("tcp_pose_gizmo", render=False)
            self.tcp_gizmo_actor = None

    @staticmethod
    def _tcp_gizmo_geometry() -> pv.PolyData:
        return pv.Box(bounds=(-0.2, 0.2, -0.2, 0.2, -0.2, 0.2))

    def _sync_tcp_gizmo_visuals(self, transform: np.ndarray) -> None:
        if self.tcp_transform_widget is None:
            return
        for actor in (*self.tcp_transform_widget._arrows, *self.tcp_transform_widget._circles):
            actor.user_matrix = transform

    def _transform_tcp_widget_changed(self, matrix: np.ndarray) -> None:
        if self.updating_tcp_widget:
            return
        matrix = np.asarray(matrix, dtype=float)
        if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
            return
        current_world_matrix = self._tcp_world_matrix()
        # The gizmo widget always drags/rotates along its own fixed world axes;
        # re-interpret those raw deltas as being along the TCP's own current
        # axes (matching the visually re-oriented arrows/rings), not the
        # robot base's or the world's axes.
        current_rotation = current_world_matrix[:3, :3]

        raw_world_matrix = matrix.copy()
        raw_rotation = raw_world_matrix[:3, :3]
        u, _, vh = np.linalg.svd(raw_rotation)
        raw_world_matrix[:3, :3] = u @ vh
        if self.tcp_last_widget_matrix is None:
            self.tcp_last_widget_matrix = raw_world_matrix.copy()
            return

        previous_widget_matrix = self.tcp_last_widget_matrix
        previous_widget_rotation = previous_widget_matrix[:3, :3]
        raw_translation_delta = raw_world_matrix[:3, 3] - previous_widget_matrix[:3, 3]
        mapped_world_translation = current_world_matrix[:3, 3] + current_rotation @ raw_translation_delta

        raw_rotation_delta = raw_world_matrix[:3, :3] @ previous_widget_rotation.T
        world_rotation_delta = current_rotation @ raw_rotation_delta @ current_rotation.T
        mapped_world_rotation = world_rotation_delta @ current_world_matrix[:3, :3]
        u, _, vh = np.linalg.svd(mapped_world_rotation)
        mapped_world_rotation = u @ vh

        world_matrix = np.eye(4, dtype=float)
        world_matrix[:3, :3] = mapped_world_rotation
        world_matrix[:3, 3] = mapped_world_translation
        base_matrix = np.linalg.inv(self.robot_base_transform) @ world_matrix
        rotation = base_matrix[:3, :3]
        u, _, vh = np.linalg.svd(rotation)
        rotation = u @ vh
        self.tcp_pose_matrix[:3, :3] = rotation
        self.tcp_pose_matrix[:3, 3] = base_matrix[:3, 3]
        self.tcp_target_selected = True
        self._clear_plan_result()
        rpy = np.rad2deg(matrix_to_rpy(rotation))
        blockers = [
            QSignalBlocker(widget)
            for widget in (*self.tcp_position_spins, *self.tcp_rpy_spins)
        ]
        for spin, value in zip(self.tcp_position_spins, base_matrix[:3, 3]):
            spin.setValue(float(value))
        for spin, value in zip(self.tcp_rpy_spins, rpy):
            spin.setValue(float(value))
        del blockers
        for actor_name in self.tcp_actor_names:
            actor = self.plotter.renderer.actors.get(actor_name)
            if actor is not None:
                actor.user_matrix = self._tcp_world_matrix()
        if self.tcp_handle_actor is not None:
            self.tcp_handle_actor.user_matrix = self._tcp_world_matrix()
        if self.tcp_gizmo_actor is not None:
            self.updating_tcp_widget = True
            self.tcp_gizmo_actor.user_matrix = self._tcp_world_matrix()
            self._sync_tcp_gizmo_visuals(self._tcp_world_matrix())
            self.updating_tcp_widget = False
        self.tcp_last_widget_matrix = raw_world_matrix.copy()
        self._refresh_tcp_target_label()

    def _render_tcp_gripper(self) -> None:
        self._destroy_tcp_transform_widget()
        for actor_name in self.tcp_actor_names:
            self.plotter.remove_actor(actor_name, render=False)
        self.tcp_actor_names.clear()
        if self.tcp_handle_actor is not None:
            self.plotter.remove_actor("tcp_pose_handle", render=False)
            self.tcp_handle_actor = None
        if not self.tcp_gripper_visuals or not self.tcp_target_selected:
            self.plotter.render()
            return
        for index, visual in enumerate(self.tcp_gripper_visuals):
            mesh = visual.mesh.copy(deep=True)
            mesh.transform(visual.transform, inplace=True)
            actor_name = f"tcp_gripper_visual_{index}"
            options = {
                "name": actor_name,
                "opacity": 0.92,
                "smooth_shading": True,
                "pickable": False,
                "color": "#ff9f43" if visual.texture is None else None,
                "reset_camera": False,
            }
            if visual.texture is not None:
                options.pop("color")
                options["texture"] = visual.texture
            self.plotter.add_mesh(mesh, **options).user_matrix = self._tcp_world_matrix()
            self.tcp_actor_names.append(actor_name)
        if not self.tcp_transform_active:
            self.plotter.render()
            return
        handle = pv.Sphere(radius=0.035)
        self.tcp_handle_actor = self.plotter.add_mesh(
            handle,
            name="tcp_pose_handle",
            color="#ff5a5f",
            opacity=0.35,
            pickable=False,
            reset_camera=False,
        )
        self.tcp_handle_actor.user_matrix = self._tcp_world_matrix()
        self.tcp_gizmo_actor = self.plotter.add_mesh(
            self._tcp_gizmo_geometry(),
            name="tcp_pose_gizmo",
            color="#ff5a5f",
            opacity=0.0,
            pickable=False,
            reset_camera=False,
            show_edges=False,
        )
        self.tcp_gizmo_actor.user_matrix = self._tcp_world_matrix()
        self.tcp_transform_widget = self.plotter.add_affine_transform_widget(
            self.tcp_gizmo_actor,
            origin=tuple(float(value) for value in self._tcp_world_matrix()[:3, 3]),
            scale=0.32,
            axes_colors=("#ef476f", "#06d6a0", "#118ab2"),
            release_callback=self._transform_tcp_widget_changed,
            interact_callback=self._transform_tcp_widget_changed,
        )
        self.updating_tcp_widget = True
        self._sync_tcp_gizmo_visuals(self._tcp_world_matrix())
        self.updating_tcp_widget = False
        self.tcp_last_widget_matrix = self._tcp_world_matrix().copy()
        self.tcp_transform_widget._actor_length *= 3.0
        self.plotter.render()

    def _reload_tcp_gripper(self, arm_name: str) -> None:
        if not hasattr(self, "robot_path_edit") or not self.robot_path_edit.text():
            return
        prefix = "left" if arm_name == "left_arm" else "right"
        self.tcp_mount_edit.setText(f"{prefix}_arm_end_effector_mount_link")
        self.tcp_link_edit.setText(f"{prefix}_gripper_tcp_link")
        path = Path(self.robot_path_edit.text()).expanduser()
        try:
            self.tcp_gripper_visuals = load_tcp_gripper_visuals(
                path,
                self.tcp_mount_edit.text(),
                self.tcp_link_edit.text(),
                self.robot_state.joint_positions,
                auxiliary_root_links=(f"{prefix}_arm_link7",),
                excluded_links=(
                    self.tcp_mount_edit.text(),
                    f"{prefix}_gripper_flange_link",
                ),
            )
        except (OSError, RuntimeError, ValueError):
            self.tcp_gripper_visuals = []
        self._render_tcp_gripper()
        if hasattr(self, "workspace_toggle_button") and self.workspace_points is not None:
            self._clear_reachable_workspace()
