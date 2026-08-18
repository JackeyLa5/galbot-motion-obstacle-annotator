from __future__ import annotations

from pathlib import Path

import numpy as np
import pyvista as pv
from PySide6.QtCore import QSignalBlocker, Qt
from PySide6.QtWidgets import (
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from ..geometry import compose_pose, matrix_to_quaternion, matrix_to_rpy, rpy_to_matrix
from ..robot_model import (
    load_urdf_actuated_joint_names,
    load_urdf_joint_limits,
    load_urdf_link_transforms,
    load_urdf_visuals,
)
from ..robot_state import GALBOT_REFERENCE_JOINT_NAMES


class RobotMixin:
    def _build_robot_group(self) -> QGroupBox:
        group = QGroupBox("机器人模型")
        layout = QVBoxLayout(group)

        path_row = QHBoxLayout()
        self.robot_path_edit = QLineEdit()
        browse_button = QPushButton("选择")
        browse_button.clicked.connect(self.browse_robot_model)
        path_row.addWidget(self.robot_path_edit)
        path_row.addWidget(browse_button)
        layout.addLayout(path_row)

        form = QFormLayout()
        self.robot_position_spins = self._add_vector_row(form, "位置 XYZ", -1000.0, 1000.0, 0.01)
        self.robot_quaternion_spins = self._add_vector_row(form, "四元数 XYZ", -1.0, 1.0, 0.01)
        self.robot_qw_spin = self._spin(-1.0, 1.0, 0.01)
        self.robot_qw_spin.setValue(1.0)
        form.addRow("四元数 W", self.robot_qw_spin)
        layout.addLayout(form)

        row = QHBoxLayout()
        load_button = QPushButton("加载模型")
        load_button.clicked.connect(self.load_robot_model)
        apply_button = QPushButton("应用机器人位姿")
        apply_button.setProperty("cssClass", "primary")
        apply_button.clicked.connect(self.apply_robot_pose)
        row.addWidget(load_button)
        row.addWidget(apply_button)
        layout.addLayout(row)
        hint = QLabel("左键点击机器人显示底盘控制器；点击其他场景对象或空白处结束移动。")
        hint.setWordWrap(True)
        hint.setObjectName("sectionHint")
        layout.addWidget(hint)
        return group

    def _build_joint_state_group(self) -> QWidget:
        group = QFrame()
        group.setObjectName("jointStateCard")
        layout = QVBoxLayout(group)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(7)
        self.robot_joint_state_layout = layout
        self._rebuild_joint_state_controls()
        return group

    def _toggle_joint_state_panel(self) -> None:
        self.robot_joint_panel_expanded = not self.robot_joint_panel_expanded
        self.joint_scroll.setVisible(self.robot_joint_panel_expanded)
        self.robot_joint_panel_toggle.setArrowType(
            Qt.LeftArrow if self.robot_joint_panel_expanded else Qt.RightArrow
        )
        self.robot_joint_panel_toggle.setToolTip(
            "折叠关节状态栏" if self.robot_joint_panel_expanded else "展开关节状态栏"
        )

    def _rebuild_joint_state_controls(self) -> None:
        if not hasattr(self, "robot_joint_state_layout"):
            return
        while self.robot_joint_state_layout.count():
            item = self.robot_joint_state_layout.takeAt(0)
            self._delete_layout_item(item)
        self.robot_joint_sliders.clear()
        self.robot_joint_value_labels.clear()
        for index, joint_name in enumerate(GALBOT_REFERENCE_JOINT_NAMES):
            if index in {5, 7, 14}:
                separator = QFrame()
                separator.setFrameShape(QFrame.HLine)
                separator.setStyleSheet("color: #3b4654; background: #3b4654; max-height: 1px;")
                self.robot_joint_state_layout.addWidget(separator)
            joint_block = QVBoxLayout()
            joint_block.setSpacing(1)
            header_row = QHBoxLayout()
            header_row.setSpacing(4)
            label = QLabel(joint_name)
            label.setObjectName("jointName")
            label.setFixedHeight(16)
            value_label = QLabel()
            value_label.setFixedSize(58, 20)
            value_label.setAlignment(Qt.AlignCenter)
            value_label.setObjectName("jointValue")
            header_row.addWidget(label)
            header_row.addStretch(1)
            header_row.addWidget(value_label)

            slider_row = QHBoxLayout()
            slider_row.setSpacing(10)
            slider = QSlider(Qt.Horizontal)
            slider.setTracking(True)
            slider.setMinimumWidth(260)
            slider.setFixedHeight(20)
            lower, upper = self.robot_joint_limits.get(joint_name, (-np.pi, np.pi))
            slider.setRange(
                int(np.ceil(lower * self.robot_joint_slider_scale)),
                int(np.floor(upper * self.robot_joint_slider_scale)),
            )
            minimum_label = QLabel(f"{lower:.3f}")
            minimum_label.setFixedSize(42, 20)
            minimum_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            minimum_label.setObjectName("jointLimit")
            maximum_label = QLabel(f"{upper:.3f}")
            maximum_label.setFixedSize(42, 20)
            maximum_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            maximum_label.setObjectName("jointLimit")
            slider.valueChanged.connect(
                lambda value, name=joint_name: self._robot_joint_slider_changed(name, value)
            )
            slider_row.addWidget(minimum_label)
            slider_row.addWidget(slider, 1)
            slider_row.addWidget(maximum_label)
            joint_block.addLayout(header_row)
            joint_block.addLayout(slider_row)
            self.robot_joint_state_layout.addLayout(joint_block)
            self.robot_joint_sliders[joint_name] = slider
            self.robot_joint_value_labels[joint_name] = value_label
        self._sync_joint_state_controls()

    @classmethod
    def _delete_layout_item(cls, item) -> None:
        widget = item.widget()
        if widget is not None:
            # deleteLater() alone only frees the widget once Qt's event loop goes
            # idle, which a render-heavy app may never do promptly; reparenting
            # immediately makes it stop rendering/laying out synchronously.
            widget.setParent(None)
            widget.deleteLater()
            return
        layout = item.layout()
        if layout is None:
            return
        while layout.count():
            cls._delete_layout_item(layout.takeAt(0))
        layout.deleteLater()

    def _sync_joint_state_controls(self) -> None:
        constrained_positions: dict[str, float] = {}
        for joint_name, slider in self.robot_joint_sliders.items():
            value = float(self.robot_state.joint_positions.get(joint_name, 0.0))
            lower, upper = self.robot_joint_limits.get(joint_name, (-np.pi, np.pi))
            value = float(np.clip(value, lower, upper))
            constrained_positions[joint_name] = value
            blocker = QSignalBlocker(slider)
            slider.setValue(round(value * self.robot_joint_slider_scale))
            del blocker
            self.robot_joint_value_labels[joint_name].setText(f"{value:.3f}")
        self.robot_state.update(constrained_positions)

    def _robot_joint_slider_changed(self, joint_name: str, raw_value: int) -> None:
        value = raw_value / self.robot_joint_slider_scale
        self.robot_state.update({joint_name: value})
        self.robot_joint_value_labels[joint_name].setText(f"{value:.3f}")
        self.robot_joint_plan_invalidated = True
        if not self.robot_joint_update_timer.isActive():
            self.robot_joint_update_timer.start()

    def _apply_robot_joint_sliders(self) -> None:
        path = Path(self.robot_path_edit.text()).expanduser()
        if not path.is_file():
            return
        link_transforms = load_urdf_link_transforms(path, self.robot_state.joint_positions)
        for visual in self.robot_visuals:
            link_transform = link_transforms.get(visual.link_name)
            actor = self.robot_actors.get(visual.name)
            if link_transform is None or actor is None or visual.local_transform is None:
                continue
            visual.transform = link_transform @ visual.local_transform
            actor.user_matrix = self.robot_base_transform @ visual.transform
        if self.robot_joint_plan_invalidated:
            self._clear_plan_result()
            self.robot_joint_plan_invalidated = False
        self.plotter.render()

    def _robot_gizmo_actors(self) -> tuple[object, ...]:
        if self.robot_base_gizmo_widget is None:
            return ()
        return (
            *self.robot_base_gizmo_widget._arrows,
            *self.robot_base_gizmo_widget._circles,
        )

    def _robot_drag_mode_for_actor(self, actor: object) -> str | None:
        if self.robot_base_gizmo_widget is None or actor is None:
            return None
        if actor is self.robot_base_gizmo_widget._arrows[0]:
            return "translate_x"
        if actor is self.robot_base_gizmo_widget._arrows[1]:
            return "translate_y"
        return None

    def _drag_robot_base_in_local_frame(self, qt_x: float, qt_y: float) -> None:
        if self.robot_drag_mode is None:
            return
        current_display_pos = self._qt_to_display_position(qt_x, qt_y)
        if self.robot_drag_last_display_pos is None:
            self.robot_drag_last_display_pos = current_display_pos
            return
        display_delta = current_display_pos - self.robot_drag_last_display_pos
        if np.linalg.norm(display_delta) < 1e-9:
            return

        current_transform = self.robot_base_transform.copy()
        current_rotation = current_transform[:3, :3]
        axis_index = 0 if self.robot_drag_mode == "translate_x" else 1
        local_axis = np.zeros(3, dtype=float)
        local_axis[axis_index] = 1.0
        world_axis = current_rotation @ local_axis
        origin_display = self._world_to_display_position(current_transform[:3, 3])
        axis_display = self._world_to_display_position(current_transform[:3, 3] + world_axis)
        screen_axis = axis_display - origin_display
        axis_screen_norm = float(np.linalg.norm(screen_axis))
        if axis_screen_norm < 1e-9:
            return

        movement_scalar = float(np.dot(display_delta, screen_axis / axis_screen_norm) / axis_screen_norm)
        new_position = current_transform[:3, 3] + movement_scalar * world_axis
        constrained = compose_pose(
            np.array([new_position[0], new_position[1], current_transform[2, 3]], dtype=float),
            matrix_to_quaternion(current_rotation),
        )
        self.robot_base_transform = constrained
        self.updating_robot_base_gizmo = True
        if self.robot_base_gizmo_actor is not None:
            self.robot_base_gizmo_actor.user_matrix = constrained
        self._sync_robot_base_gizmo_visuals(constrained)
        self.updating_robot_base_gizmo = False
        self.robot_base_last_widget_matrix = constrained.copy()
        self.robot_base_pending_transform = constrained
        self.robot_base_move_dirty = True
        self.robot_drag_last_display_pos = current_display_pos
        if not self.robot_base_update_timer.isActive():
            self.robot_base_update_timer.start()

    def _select_robot_from_qt_position(self, x: float, y: float) -> None:
        camera_state = self._camera_state()
        actor = self._pick_scene_actor(x, y)
        if actor in self.robot_actors.values():
            self.start_robot_base_move()
            self._restore_camera_state(camera_state)
            self.plotter.render()
            return
        if actor in self._robot_gizmo_actors():
            self._restore_camera_state(camera_state)
            return
        self.finish_robot_base_move()
        self._restore_camera_state(camera_state)

    def _load_default_robot_if_available(self) -> None:
        description_dir = Path(__file__).resolve().parents[3] / "galbot_one_golf_description"
        urdf_path = description_dir / "urdf" / "galbot_one_golf.urdf"
        if urdf_path.exists():
            self.robot_path_edit.setText(str(urdf_path))
            self.load_robot_model()
        else:
            self.statusBar().showMessage("未找到机器人资产仓库，请先 clone galbot_one_golf_description")

    def browse_robot_model(self) -> None:
        initial = self.robot_path_edit.text() or str(Path.cwd())
        path = QFileDialog.getExistingDirectory(self, "选择机器人模型目录", initial)
        if path:
            self.robot_path_edit.setText(path)
            self.load_robot_model()

    def load_robot_model(self) -> None:
        path = Path(self.robot_path_edit.text()).expanduser()
        if path.is_dir():
            candidates = sorted((path / "urdf").glob("*.urdf")) or sorted(path.glob("*.urdf"))
            if not candidates:
                QMessageBox.warning(self, "没有 URDF", f"目录中没有找到 URDF：{path}")
                return
            path = candidates[0]
            self.robot_path_edit.setText(str(path))
        try:
            actuated_joint_names = load_urdf_actuated_joint_names(path)
            self.robot_joint_limits = load_urdf_joint_limits(path)
            # Keep the established G1 preview pose for the known joints, while
            # allowing arbitrary URDF-specific joints to start at zero.
            self.robot_state.initialize_missing(
                actuated_joint_names,
                value=0.0,
            )
            self.robot_visuals = load_urdf_visuals(path, self.robot_state.joint_positions)
            self._rebuild_joint_state_controls()
            self.robot_path_edit.setText(str(path))
        except (OSError, RuntimeError, ValueError) as error:
            QMessageBox.critical(self, "机器人加载失败", str(error))
            return
        self.apply_robot_pose()
        self._reload_tcp_gripper(self.tcp_arm_combo.currentText())
        self.statusBar().showMessage(
            f"已加载机器人模型：{path.name}，共 {len(self.robot_visuals)} 个视觉网格"
        )

    def update_environment_joint_positions(self, joint_positions: dict[str, float]) -> None:
        """Update the visual environment state, ready for future real-robot synchronization."""
        self.robot_state.update(joint_positions)
        self._sync_joint_state_controls()
        path = Path(self.robot_path_edit.text()).expanduser()
        if not path.is_file():
            return
        self.robot_visuals = load_urdf_visuals(path, self.robot_state.joint_positions)
        self._render_robot(self.robot_base_transform)
        self._reload_tcp_gripper(self.tcp_arm_combo.currentText())
        self._clear_plan_result()

    def apply_robot_pose(self) -> None:
        if not self.robot_visuals:
            return
        position = np.array([spin.value() for spin in self.robot_position_spins])
        quaternion = np.array(
            [*[spin.value() for spin in self.robot_quaternion_spins], self.robot_qw_spin.value()]
        )
        norm = np.linalg.norm(quaternion)
        if norm < 1e-9:
            QMessageBox.warning(self, "四元数无效", "机器人位姿四元数不能全为 0。")
            return
        quaternion /= norm
        for spin, value in zip([*self.robot_quaternion_spins, self.robot_qw_spin], quaternion):
            spin.setValue(float(value))
        self.robot_base_transform = compose_pose(position, quaternion)
        self._render_robot(self.robot_base_transform)
        self._render_robot_base_gizmo()
        if self.full_point_cloud is not None:
            self.apply_point_cloud_filter(reset_camera=False)

    def _render_robot(self, base_transform: np.ndarray) -> None:
        for actor_name in self.robot_actor_names:
            self.plotter.remove_actor(actor_name, render=False)
        self.robot_actor_names.clear()
        self.robot_actors.clear()

        for index, visual in enumerate(self.robot_visuals):
            mesh = visual.mesh.copy(deep=True)
            actor_name = f"robot_visual_{index}"
            mesh_options = {
                "name": actor_name,
                "opacity": 1.0,
                "smooth_shading": True,
                "pickable": True,
            }
            if visual.texture is not None:
                mesh_options["texture"] = visual.texture
            else:
                mesh_options["color"] = "#ffffff"
            actor = self.plotter.add_mesh(
                mesh,
                **mesh_options,
            )
            actor.user_matrix = base_transform @ visual.transform
            self.robot_actor_names.append(actor_name)
            self.robot_actors[visual.name] = actor
        self.plotter.render()

    def _update_robot_actor_matrices(self, base_transform: np.ndarray) -> None:
        """Move existing actors without rebuilding meshes or filtering the cloud."""
        for visual in self.robot_visuals:
            actor = self.robot_actors.get(visual.name)
            if actor is not None:
                actor.user_matrix = base_transform @ visual.transform

    def _destroy_robot_base_gizmo(self) -> None:
        self.robot_base_last_widget_matrix = None
        if self.robot_base_gizmo_widget is not None:
            self.robot_base_gizmo_widget.disable()
            self.robot_base_gizmo_widget.remove()
            self.robot_base_gizmo_widget = None
        if self.robot_base_gizmo_actor is not None:
            self.plotter.remove_actor("robot_base_gizmo", render=False)
            self.robot_base_gizmo_actor = None

    @staticmethod
    def _robot_base_gizmo_geometry() -> pv.PolyData:
        # The affine widget supplies the visible arrows and rotation rings.
        # Keep the invisible anchor at a normal visual size; drag sensitivity
        # is adjusted separately below.
        return pv.Box(bounds=(-0.25, 0.25, -0.25, 0.25, -0.25, 0.25))

    def _sync_robot_base_gizmo_visuals(self, transform: np.ndarray) -> None:
        if self.robot_base_gizmo_widget is None:
            return
        for actor in self._robot_gizmo_actors():
            actor.user_matrix = transform

    def _render_robot_base_gizmo(self) -> None:
        self._destroy_robot_base_gizmo()
        if not self.robot_visuals or not self.robot_base_move_mode:
            return
        self.robot_base_gizmo_actor = self.plotter.add_mesh(
            self._robot_base_gizmo_geometry(),
            name="robot_base_gizmo",
            color="#ef476f",
            opacity=0.0,
            pickable=False,
            reset_camera=False,
            show_edges=False,
        )
        self.robot_base_gizmo_actor.user_matrix = self.robot_base_transform
        self.robot_base_gizmo_widget = self.plotter.add_affine_transform_widget(
            self.robot_base_gizmo_actor,
            origin=tuple(float(value) for value in self.robot_base_transform[:3, 3]),
            scale=0.45,
            axes_colors=("#ef476f", "#06d6a0", "#118ab2"),
            release_callback=self._robot_base_gizmo_released,
            interact_callback=self._robot_base_gizmo_changed,
        )
        # Only X/Y translation and Z rotation are valid for the mobile base.
        # AffineWidget3D creates all three arrows and rings by default.
        for actor in self.robot_base_gizmo_widget._arrows[2:]:
            actor.SetVisibility(False)
        for actor in self.robot_base_gizmo_widget._circles[:2]:
            actor.SetVisibility(False)
        # PyVista uses actor length only for its mouse-to-world translation
        # conversion after the widget actors have already been created. Raising
        # it here improves drag response without enlarging arrows or rings.
        self.robot_base_gizmo_widget._actor_length *= 3.0
        self.robot_base_last_widget_matrix = self.robot_base_transform.copy()

    def start_robot_base_move(self) -> None:
        if not self.robot_visuals:
            self.statusBar().showMessage("请先加载机器人模型")
            return
        if self.robot_base_move_mode:
            return
        camera_state = self._camera_state()
        self.robot_base_move_mode = True
        self._render_robot_base_gizmo()
        self._restore_camera_state(camera_state)
        self.plotter.render()
        self.statusBar().showMessage("移动模式已开启：可沿 X/Y 平移或绕 Z 轴旋转机器人")

    def finish_robot_base_move(self) -> None:
        if not self.robot_base_move_mode:
            return
        self.robot_base_move_mode = False
        self.robot_base_update_timer.stop()
        self._apply_pending_robot_base_transform()
        self._destroy_robot_base_gizmo()
        if self.robot_base_move_dirty:
            self._clear_plan_result()
            self.robot_base_move_dirty = False
        if self.full_point_cloud is not None:
            self.apply_point_cloud_filter(reset_camera=False)
        else:
            self.plotter.render()
        self.statusBar().showMessage("移动模式已结束，已根据机器人新位置刷新附近点云")

    def _robot_base_gizmo_changed(self, matrix: np.ndarray) -> None:
        if self.updating_robot_base_gizmo:
            return
        raw_world_matrix = np.asarray(matrix, dtype=float)
        if raw_world_matrix.shape != (4, 4) or not np.isfinite(raw_world_matrix).all():
            return
        raw_rotation = raw_world_matrix[:3, :3]
        u, _, vh = np.linalg.svd(raw_rotation)
        raw_world_matrix[:3, :3] = u @ vh
        if self.robot_base_last_widget_matrix is None:
            self.robot_base_last_widget_matrix = raw_world_matrix.copy()
            return

        previous_widget_matrix = self.robot_base_last_widget_matrix
        current_transform = self.robot_base_transform.copy()
        current_rotation = current_transform[:3, :3]
        current_yaw = matrix_to_rpy(current_rotation)[2]

        world_translation_delta = raw_world_matrix[:3, 3] - previous_widget_matrix[:3, 3]
        local_translation_delta = current_rotation.T @ world_translation_delta
        mapped_translation = current_transform[:3, 3] + current_rotation @ np.array(
            [local_translation_delta[0], local_translation_delta[1], 0.0],
            dtype=float,
        )

        raw_rotation_delta = raw_world_matrix[:3, :3] @ previous_widget_matrix[:3, :3].T
        local_rotation_delta = current_rotation.T @ raw_rotation_delta @ current_rotation
        delta_yaw = matrix_to_rpy(local_rotation_delta)[2]
        constrained_rpy = np.array([0.0, 0.0, current_yaw + delta_yaw], dtype=float)
        constrained = compose_pose(
            np.array([mapped_translation[0], mapped_translation[1], current_transform[2, 3]], dtype=float),
            matrix_to_quaternion(rpy_to_matrix(*constrained_rpy)),
        )
        self.robot_base_transform = constrained
        self.updating_robot_base_gizmo = True
        self.robot_base_gizmo_actor.user_matrix = constrained
        self._sync_robot_base_gizmo_visuals(constrained)
        self.updating_robot_base_gizmo = False
        self.robot_base_last_widget_matrix = constrained.copy()
        self.robot_base_pending_transform = constrained
        self.robot_base_move_dirty = True
        if not self.robot_base_update_timer.isActive():
            self.robot_base_update_timer.start()

    def _robot_base_gizmo_released(self, matrix: np.ndarray) -> None:
        self._robot_base_gizmo_changed(matrix)
        self.robot_base_update_timer.stop()
        self._apply_pending_robot_base_transform()

    def _apply_pending_robot_base_transform(self) -> None:
        if self.robot_base_pending_transform is None:
            return
        transform = self.robot_base_pending_transform
        self.robot_base_pending_transform = None
        self._sync_robot_base_form(transform)
        self._update_robot_actor_matrices(transform)
        if self.tcp_target_selected:
            self._render_tcp_gripper()
            self._refresh_tcp_target_label()
        self.plotter.render()

    def _sync_robot_base_form(self, transform: np.ndarray) -> None:
        position = transform[:3, 3]
        quaternion = matrix_to_quaternion(transform[:3, :3])
        widgets = [
            *self.robot_position_spins,
            *self.robot_quaternion_spins,
            self.robot_qw_spin,
        ]
        blockers = [QSignalBlocker(widget) for widget in widgets]
        for spin, value in zip(self.robot_position_spins, position):
            spin.setValue(float(value))
        for spin, value in zip((*self.robot_quaternion_spins, self.robot_qw_spin), quaternion):
            spin.setValue(float(value))
        del blockers
