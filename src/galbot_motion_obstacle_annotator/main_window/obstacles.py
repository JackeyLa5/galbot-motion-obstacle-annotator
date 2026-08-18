from __future__ import annotations

import numpy as np
import pyvista as pv
from PySide6.QtCore import QSignalBlocker
from PySide6.QtWidgets import (
    QAbstractItemView,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLineEdit,
    QListWidget,
    QPushButton,
    QVBoxLayout,
)

from ..models import SUPPORTED_OBSTACLE_TYPES, Obstacle
from .widgets import FocusedWheelComboBox


class ObstacleMixin:
    def _build_obstacle_group(self) -> QGroupBox:
        group = QGroupBox("碰撞体")
        layout = QVBoxLayout(group)
        self.obstacle_list = QListWidget()
        self.obstacle_list.setSelectionMode(QAbstractItemView.SingleSelection)
        self.obstacle_list.currentRowChanged.connect(self.select_obstacle)
        layout.addWidget(self.obstacle_list)

        row = QHBoxLayout()
        add_button = QPushButton("手动新建")
        add_button.clicked.connect(self.add_obstacle)
        delete_button = QPushButton("删除")
        delete_button.clicked.connect(self.delete_obstacle)
        deselect_button = QPushButton("取消选中")
        deselect_button.clicked.connect(self.deselect_obstacle)
        row.addWidget(add_button)
        row.addWidget(delete_button)
        row.addWidget(deselect_button)
        layout.addLayout(row)

        form = QFormLayout()
        self.id_edit = QLineEdit()
        self.type_combo = FocusedWheelComboBox()
        self.type_combo.addItems(SUPPORTED_OBSTACLE_TYPES)
        self.frame_edit = QLineEdit("world")
        form.addRow("名称", self.id_edit)
        form.addRow("类型", self.type_combo)
        form.addRow("目标坐标系", self.frame_edit)
        self.center_spins = self._add_vector_row(form, "中心 XYZ", -1000.0, 1000.0, 0.001)
        self.rpy_spins = self._add_vector_row(form, "旋转 RPY°", -360.0, 360.0, 0.1)
        self.size_spins = self._add_vector_row(form, "Scale XYZ", 0.0, 1000.0, 0.001)
        layout.addLayout(form)

        apply_button = QPushButton("应用碰撞体参数")
        apply_button.setProperty("cssClass", "primary")
        apply_button.clicked.connect(self.apply_form)
        layout.addWidget(apply_button)
        self._connect_form_live_updates()
        return group

    def _default_obstacle(self) -> Obstacle:
        if self.point_cloud is None:
            center = np.zeros(3)
            scale = np.ones(3)
        else:
            bounds = np.asarray(self.point_cloud.bounds).reshape(3, 2)
            center = bounds.mean(axis=1)
            scale = np.maximum((bounds[:, 1] - bounds[:, 0]) * 0.2, 0.05)
        return Obstacle(
            obstacle_id=f"obstacle_{len(self.obstacles) + 1:03d}",
            obstacle_type=self.selection_type_combo.currentText(),
            center=center,
            scale=scale,
        )

    def add_obstacle(self) -> None:
        obstacle = self._default_obstacle()
        self.obstacles.append(obstacle)
        self._refresh_list(len(self.obstacles) - 1)

    def delete_obstacle(self) -> None:
        if self.active_index < 0:
            return
        del self.obstacles[self.active_index]
        next_index = min(self.active_index, len(self.obstacles) - 1)
        self._refresh_list(next_index)

    def deselect_obstacle(self) -> None:
        with QSignalBlocker(self.obstacle_list):
            self.obstacle_list.setCurrentRow(-1)
            self.obstacle_list.clearSelection()
        self.select_obstacle(-1)

    def _destroy_transform_widget(self) -> None:
        if self.transform_widget is None:
            return
        self.transform_widget.disable()
        self.transform_widget.remove()
        self.transform_widget = None

    def _refresh_list(self, selected_index: int = -1) -> None:
        self._destroy_transform_widget()
        with QSignalBlocker(self.obstacle_list):
            self.obstacle_list.clear()
            self.obstacle_list.addItems(
                [f"{obstacle.obstacle_id} [{obstacle.obstacle_type}]" for obstacle in self.obstacles]
            )
        self._render_obstacles()
        self.obstacle_list.setCurrentRow(selected_index)
        if selected_index < 0:
            self.select_obstacle(-1)

    def _set_list_item_text(self, index: int) -> None:
        if not 0 <= index < len(self.obstacles):
            return
        item = self.obstacle_list.item(index)
        if item is None:
            return
        obstacle = self.obstacles[index]
        item.setText(f"{obstacle.obstacle_id} [{obstacle.obstacle_type}]")

    def select_obstacle(self, index: int) -> None:
        self.active_index = index
        self._destroy_transform_widget()
        self._render_obstacles()
        if not 0 <= index < len(self.obstacles):
            return

        obstacle = self.obstacles[index]
        self._update_form(obstacle)
        actor = self.obstacle_actors.get(index)
        if actor is None:
            return
        self.updating_widget = True
        actor.user_matrix = obstacle.transform
        self.transform_widget = self.plotter.add_affine_transform_widget(
            actor,
            release_callback=self._transform_widget_changed,
            interact_callback=self._transform_widget_changed,
        )
        self.updating_widget = False
        self.plotter.render()

    def _transform_widget_changed(self, matrix: np.ndarray) -> None:
        if self.updating_widget or not 0 <= self.active_index < len(self.obstacles):
            return
        try:
            self.obstacles[self.active_index].update_box_from_transform(np.asarray(matrix))
        except ValueError:
            return
        self._update_form(self.obstacles[self.active_index])

    def _update_form(self, obstacle: Obstacle) -> None:
        widgets = [
            self.id_edit,
            self.type_combo,
            self.frame_edit,
            *self.center_spins,
            *self.rpy_spins,
            *self.size_spins,
        ]
        blockers = [QSignalBlocker(widget) for widget in widgets]
        self.id_edit.setText(obstacle.obstacle_id)
        self.type_combo.setCurrentText(obstacle.obstacle_type)
        self.frame_edit.setText(obstacle.target_frame)
        for spin, value in zip(self.center_spins, obstacle.center):
            spin.setValue(float(value))
        for spin, value in zip(self.rpy_spins, np.degrees(obstacle.rpy)):
            spin.setValue(float(value))
        for spin, value in zip(self.size_spins, obstacle.scale):
            spin.setValue(float(value))
        del blockers

    def apply_form(self) -> None:
        if not 0 <= self.active_index < len(self.obstacles):
            return
        obstacle = self.obstacles[self.active_index]
        old_type = obstacle.obstacle_type
        obstacle.obstacle_id = self.id_edit.text().strip() or f"obstacle_{self.active_index + 1:03d}"
        obstacle.obstacle_type = self.type_combo.currentText()
        obstacle.target_frame = self.frame_edit.text().strip() or "world"
        obstacle.center = np.array([spin.value() for spin in self.center_spins])
        obstacle.rpy = np.radians([spin.value() for spin in self.rpy_spins])
        obstacle.scale = np.array([spin.value() for spin in self.size_spins])

        self.updating_widget = True
        actor = self.obstacle_actors.get(self.active_index)
        if actor is not None:
            actor.user_matrix = obstacle.transform
        self.updating_widget = False

        type_changed = obstacle.obstacle_type != old_type
        if type_changed:
            self._refresh_list(self.active_index)
            return

        self._set_list_item_text(self.active_index)
        self.plotter.render()

    def _render_obstacles(self) -> None:
        previous_actor_names = [f"obstacle_box_{index}" for index in self.obstacle_actors]
        self.obstacle_actors = {}
        for actor_name in previous_actor_names:
            self.plotter.remove_actor(actor_name, render=False)
        type_colors = {
            "box": "#22b8cf",
            "sphere": "#b26be8",
            "cylinder": "#72c472",
        }
        for index, obstacle in enumerate(self.obstacles):
            geometry = self._widget_geometry(obstacle)
            selected = index == self.active_index
            color = "#ffcc00" if selected else type_colors[obstacle.obstacle_type]
            actor = self.plotter.add_mesh(
                geometry,
                name=f"obstacle_box_{index}",
                style="surface",
                color=color,
                opacity=0.42 if selected else 0.28,
                show_edges=True,
                edge_color=color,
                line_width=2,
                lighting=False,
                pickable=False,
                reset_camera=False,
            )
            actor.user_matrix = obstacle.transform
            self.obstacle_actors[index] = actor
        self.plotter.render()

    @staticmethod
    def _widget_geometry(obstacle: Obstacle) -> pv.PolyData:
        if obstacle.obstacle_type == "sphere":
            return pv.Sphere(radius=0.5)
        if obstacle.obstacle_type == "cylinder":
            return pv.Cylinder(radius=0.5, height=1.0, direction=(0.0, 0.0, 1.0))
        return pv.Box(bounds=(-0.5, 0.5, -0.5, 0.5, -0.5, 0.5))
