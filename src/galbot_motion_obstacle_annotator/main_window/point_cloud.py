from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyvista as pv
from PySide6.QtWidgets import QFileDialog, QFormLayout, QGroupBox, QMessageBox, QPushButton

from ..importers import load_json as load_obstacles_json
from ..pcd import load_pcd
from .widgets import FocusedWheelComboBox


class PointCloudMixin:
    def _build_filter_group(self) -> QGroupBox:
        group = QGroupBox("点云显示范围")
        form = QFormLayout(group)
        self.xy_half_range_spin = self._spin(0.1, 1000.0, 1.0)
        self.xy_half_range_spin.setValue(10.0)
        form.addRow("XY 半范围/m", self.xy_half_range_spin)
        self.z_min_spin = self._spin(-1000.0, 1000.0, 0.1)
        self.z_min_spin.setValue(-2.0)
        self.z_max_spin = self._spin(-1000.0, 1000.0, 0.1)
        self.z_max_spin.setValue(2.0)
        form.addRow("Z 最低/m", self.z_min_spin)
        form.addRow("Z 最高/m", self.z_max_spin)

        self.point_color_mode_combo = FocusedWheelComboBox()
        self.point_color_mode_combo.addItems(["按高度着色", "柔和单色"])
        self.point_color_mode_combo.currentIndexChanged.connect(self._render_point_cloud)
        form.addRow("点云颜色", self.point_color_mode_combo)

        self.point_size_spin = self._spin(1.0, 10.0, 0.5)
        self.point_size_spin.setDecimals(1)
        self.point_size_spin.setValue(2.0)
        self.point_size_spin.valueChanged.connect(self._render_point_cloud)
        form.addRow("点大小", self.point_size_spin)

        self.point_opacity_spin = self._spin(0.1, 1.0, 0.05)
        self.point_opacity_spin.setDecimals(2)
        self.point_opacity_spin.setValue(0.72)
        self.point_opacity_spin.valueChanged.connect(self._render_point_cloud)
        form.addRow("点云透明度", self.point_opacity_spin)

        apply_button = QPushButton("应用显示过滤")
        apply_button.clicked.connect(self.apply_point_cloud_filter)
        form.addRow(apply_button)
        return group

    def open_point_cloud(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "打开点云", "", "Point cloud (*.pcd *.ply *.vtk *.vtp)")
        if path:
            self.load_point_cloud(path)

    def open_json(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "打开 JSON", "", "JSON (*.json)")
        if path:
            self.load_json(path)

    def load_point_cloud(self, path: str) -> None:
        try:
            if Path(path).suffix.lower() == ".pcd":
                geometry = pv.PolyData(load_pcd(path))
            else:
                geometry = pv.read(path)
                if not isinstance(geometry, pv.PolyData):
                    geometry = geometry.extract_surface()
        except (OSError, RuntimeError, ValueError) as error:
            QMessageBox.critical(self, "点云加载失败", str(error))
            return

        self.point_cloud_path = str(Path(path).resolve())
        self.full_point_cloud = geometry
        self.apply_point_cloud_filter(reset_camera=True)

    def load_json(self, path: str) -> None:
        try:
            obstacles, source_point_cloud = load_obstacles_json(path)
        except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
            QMessageBox.critical(self, "JSON 加载失败", str(error))
            return

        self.obstacles = obstacles
        self.obstacles_json_path = str(Path(path).resolve())
        if source_point_cloud and not self.point_cloud_path:
            self.point_cloud_path = source_point_cloud
        self._refresh_list(0 if self.obstacles else -1)
        self.statusBar().showMessage(f"已加载 JSON：{Path(path).name}，共 {len(self.obstacles)} 个碰撞体")

    def apply_point_cloud_filter(self, reset_camera: bool = True) -> None:
        if self.full_point_cloud is None:
            return
        if self.z_min_spin.value() >= self.z_max_spin.value():
            QMessageBox.warning(self, "过滤范围无效", "Z 最低值必须小于 Z 最高值。")
            return

        center_x = self.robot_position_spins[0].value()
        center_y = self.robot_position_spins[1].value()
        half_range = self.xy_half_range_spin.value()
        points = self.full_point_cloud.points
        mask = (
            (points[:, 0] >= center_x - half_range)
            & (points[:, 0] <= center_x + half_range)
            & (points[:, 1] >= center_y - half_range)
            & (points[:, 1] <= center_y + half_range)
            & (points[:, 2] >= self.z_min_spin.value())
            & (points[:, 2] <= self.z_max_spin.value())
        )
        self.point_cloud = pv.PolyData(points[mask])
        self._render_point_cloud(reset_camera=reset_camera)
        self.statusBar().showMessage(
            f"显示 {self.point_cloud.n_points:,}/{self.full_point_cloud.n_points:,} 个点；"
            f"XY 中心=({center_x:.2f}, {center_y:.2f})，半范围={half_range:.1f}m，"
            f"Z=[{self.z_min_spin.value():.1f}, {self.z_max_spin.value():.1f}]m"
        )

    def _render_point_cloud(self, *_args, reset_camera: bool = False) -> None:
        if self.point_cloud is None:
            return
        self.plotter.remove_actor("point_cloud", render=False)
        if self.point_cloud.n_points:
            mesh_options = {
                "name": "point_cloud",
                "style": "points",
                "point_size": self.point_size_spin.value(),
                "opacity": self.point_opacity_spin.value(),
                "render_points_as_spheres": False,
                "pickable": True,
                "reset_camera": False,
                "show_scalar_bar": False,
            }
            if self.point_color_mode_combo.currentIndex() == 0:
                height = np.asarray(self.point_cloud.points[:, 2], dtype=float)
                self.point_cloud["height"] = height
                sample_step = max(1, len(height) // 100_000)
                sampled_height = height[::sample_step]
                lower, upper = np.percentile(sampled_height, (2.0, 98.0))
                if upper - lower < 1e-6:
                    lower, upper = float(height.min()), float(height.max() + 1e-6)
                mesh_options.update(
                    {
                        "scalars": "height",
                        "cmap": "viridis",
                        "clim": (float(lower), float(upper)),
                    }
                )
            else:
                mesh_options["color"] = "#8394a7"
            self.plotter.add_mesh(self.point_cloud, **mesh_options)
        if reset_camera:
            self.plotter.reset_camera()
        self.plotter.render()
