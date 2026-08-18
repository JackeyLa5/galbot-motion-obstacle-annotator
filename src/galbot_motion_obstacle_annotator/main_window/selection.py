from __future__ import annotations

import numpy as np
import pyvista as pv
from PySide6.QtWidgets import QGroupBox, QHBoxLayout, QLabel, QMessageBox, QPushButton, QVBoxLayout
from vtkmodules.vtkRenderingCore import vtkPointPicker

from ..geometry import matrix_to_rpy
from ..models import SUPPORTED_OBSTACLE_TYPES, Obstacle
from .widgets import FocusedWheelComboBox


class SelectionMixin:
    def _build_selection_group(self) -> QGroupBox:
        group = QGroupBox("低开销点选标注")
        layout = QVBoxLayout(group)
        hint = QLabel("开始后在点云上右键逐点采样；完成后根据碰撞体类型自动拟合。")
        hint.setWordWrap(True)
        hint.setObjectName("sectionHint")
        layout.addWidget(hint)

        row = QHBoxLayout()
        self.fit_mode_combo = FocusedWheelComboBox()
        self.fit_mode_combo.addItems(["AABB（稳定）", "OBB（贴合旋转）"])
        self.selection_type_combo = FocusedWheelComboBox()
        self.selection_type_combo.addItems(SUPPORTED_OBSTACLE_TYPES)
        self.selection_button = QPushButton("开始逐点选择")
        self.selection_button.clicked.connect(self.toggle_selection_mode)
        row.addWidget(self.selection_type_combo)
        row.addWidget(self.fit_mode_combo)
        row.addWidget(self.selection_button)
        layout.addLayout(row)

        self.picked_count_label = QLabel("已选择 0 个点")
        layout.addWidget(self.picked_count_label)
        action_row = QHBoxLayout()
        undo_button = QPushButton("撤销一点")
        undo_button.clicked.connect(self.undo_picked_point)
        clear_button = QPushButton("清空点")
        clear_button.clicked.connect(self.clear_picked_points)
        create_button = QPushButton("生成碰撞体")
        create_button.setProperty("cssClass", "primary")
        create_button.clicked.connect(self.create_obstacle_from_points)
        action_row.addWidget(undo_button)
        action_row.addWidget(clear_button)
        action_row.addWidget(create_button)
        layout.addLayout(action_row)
        return group

    def toggle_selection_mode(self) -> None:
        if self.selection_mode:
            self._stop_selection_mode()
            return
        if self.point_cloud is None:
            QMessageBox.warning(self, "没有点云", "请先打开 PCD 文件。")
            return

        self.deselect_obstacle()
        self.selection_mode = True
        self._prime_selection_mode()
        self.selection_button.setText("停止选择")
        self.statusBar().showMessage("点选模式：在点云上右键采样，完成后点击“生成碰撞体”")

    def _stop_selection_mode(self) -> None:
        self.selection_mode = False
        self.pending_selection_camera_state = None
        self.selection_primer_point = None
        self.selection_button.setText("开始逐点选择")

    def _pick_point_from_qt_position(self, qt_x: float, qt_y: float) -> bool:
        widget = self.plotter.interactor
        render_width, render_height = self.plotter.render_window.GetSize()
        if widget.width() <= 0 or widget.height() <= 0 or render_width <= 0 or render_height <= 0:
            return False
        display_x = qt_x * render_width / widget.width()
        display_y = (widget.height() - qt_y - 1.0) * render_height / widget.height()
        picker = vtkPointPicker()
        picker.SetTolerance(0.02)
        if picker.Pick(display_x, display_y, 0.0, self.plotter.renderer) != 1:
            return False
        if picker.GetDataSet() is None or picker.GetPointId() < 0:
            return False
        self._point_picked(np.asarray(picker.GetPickPosition(), dtype=float))
        return True

    def _prime_selection_mode(self) -> None:
        if self.point_cloud is None or not self.point_cloud.n_points:
            return
        camera_state = self._camera_state()
        focal_point = np.asarray(self.plotter.camera.focal_point, dtype=float)
        points = np.asarray(self.point_cloud.points)
        index = int(np.argmin(np.linalg.norm(points - focal_point, axis=1)))
        self.selection_primer_point = points[index].copy()
        self.picked_points_mesh.shallow_copy(pv.PolyData(np.asarray([self.selection_primer_point])))
        self.picked_points_actor.prop.opacity = 0.0
        self.plotter.render()
        self._restore_camera_state(camera_state)
        self.plotter.render()

    def _restore_selection_camera(self) -> None:
        if self.selection_mode and self.pending_selection_camera_state is not None:
            self._restore_camera_state(self.pending_selection_camera_state)
            self.pending_selection_camera_state = None
            self.plotter.render()

    def _point_picked(self, point) -> None:
        point = np.asarray(point, dtype=float)
        if point.shape != (3,) or not np.isfinite(point).all():
            return
        if self.picked_points and np.linalg.norm(point - self.picked_points[-1]) < 1e-6:
            return
        self.picked_points.append(point)
        self._render_picked_points()

    def _render_picked_points(self) -> None:
        camera_state = self._camera_state()
        if self.picked_points:
            self.picked_points_mesh.shallow_copy(pv.PolyData(np.asarray(self.picked_points)))
            self.picked_points_actor.prop.opacity = 1.0
        else:
            self.picked_points_actor.prop.opacity = 0.0
        self._restore_camera_state(camera_state)
        self.picked_count_label.setText(f"已选择 {len(self.picked_points)} 个点")
        self.plotter.render()

    def undo_picked_point(self) -> None:
        if self.picked_points:
            self.picked_points.pop()
            self._render_picked_points()

    def clear_picked_points(self) -> None:
        self.picked_points.clear()
        self._render_picked_points()

    def create_obstacle_from_points(self) -> None:
        obstacle_type = self.selection_type_combo.currentText()
        minimum_points = {"box": 2, "sphere": 2, "cylinder": 3}.get(obstacle_type, 0)
        if len(self.picked_points) < minimum_points:
            QMessageBox.warning(
                self,
                "选点过少",
                f"{obstacle_type} 至少需要 {minimum_points} 个点；当前为 {len(self.picked_points)} 个。",
            )
            return

        points = np.asarray(self.picked_points)
        obstacle = self._fit_selected_points(points, obstacle_type)
        self.obstacles.append(obstacle)
        self._refresh_list()
        self._stop_selection_mode()
        self.clear_picked_points()
        self.statusBar().showMessage(f"已用 {len(points)} 个采样点生成 {obstacle.obstacle_id}")

    def _fit_selected_points(self, points: np.ndarray, obstacle_type: str) -> Obstacle:
        minimum_size = 0.01
        if obstacle_type == "box" and self.fit_mode_combo.currentIndex() == 0:
            lower = points.min(axis=0)
            upper = points.max(axis=0)
            center = (lower + upper) / 2.0
            scale = np.maximum(upper - lower, minimum_size)
            rpy = np.zeros(3)
        elif obstacle_type == "box":
            mean = points.mean(axis=0)
            covariance = np.cov(points - mean, rowvar=False)
            eigenvalues, axes = np.linalg.eigh(covariance)
            axes = axes[:, np.argsort(eigenvalues)[::-1]]
            if np.linalg.det(axes) < 0.0:
                axes[:, -1] *= -1.0
            projected = (points - mean) @ axes
            lower = projected.min(axis=0)
            upper = projected.max(axis=0)
            center = mean + axes @ ((lower + upper) / 2.0)
            scale = np.maximum(upper - lower, minimum_size)
            rpy = matrix_to_rpy(axes)
        elif obstacle_type == "sphere":
            lower = points.min(axis=0)
            upper = points.max(axis=0)
            center = (lower + upper) / 2.0
            radius = max(float(np.linalg.norm(points - center, axis=1).max()), minimum_size)
            rpy = np.zeros(3)
            scale = np.array([radius, 0.0, 0.0])
        elif obstacle_type == "cylinder":
            mean = points.mean(axis=0)
            covariance = np.cov(points - mean, rowvar=False)
            eigenvalues, eigenvectors = np.linalg.eigh(covariance)
            axis = eigenvectors[:, np.argmax(eigenvalues)]
            if axis[2] < 0.0:
                axis *= -1.0
            helper = np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
            local_x = np.cross(helper, axis)
            local_x /= np.linalg.norm(local_x)
            local_y = np.cross(axis, local_x)
            rotation = np.column_stack((local_x, local_y, axis))
            axial = (points - mean) @ axis
            axial_min, axial_max = axial.min(), axial.max()
            center = mean + axis * ((axial_min + axial_max) / 2.0)
            radial = points - center - np.outer((points - center) @ axis, axis)
            radius = max(float(np.linalg.norm(radial, axis=1).max()), minimum_size)
            height = max(float(axial_max - axial_min), minimum_size)
            rpy = matrix_to_rpy(rotation)
            scale = np.array([radius, height, 0.0])
        else:
            center = points.mean(axis=0) if len(points) else np.zeros(3)
            rpy = np.zeros(3)
            scale = np.ones(3)

        return Obstacle(
            obstacle_id=f"obstacle_{len(self.obstacles) + 1:03d}",
            obstacle_type=obstacle_type,
            center=center,
            rpy=rpy,
            scale=scale,
        )
