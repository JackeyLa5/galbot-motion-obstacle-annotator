from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyvista as pv
from PySide6.QtCore import QEvent, QSignalBlocker, QTimer, Qt
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSplitter,
    QVBoxLayout,
    QWidget,
)
from pyvistaqt import QtInteractor
from vtkmodules.vtkRenderingCore import vtkPointPicker

from .exporters import export_json, export_python
from .geometry import compose_pose, matrix_to_rpy
from .importers import load_json as load_obstacles_json
from .models import SUPPORTED_OBSTACLE_TYPES, Obstacle
from .pcd import load_pcd
from .robot_model import RobotVisual, load_urdf_visuals

DEFAULT_ROBOT_JOINT_POSITIONS = {
    "leg_joint1": 0.6,
    "leg_joint2": 1.8,
    "leg_joint3": 1.2,
    "leg_joint4": 0.0,
    "leg_joint5": 0.0,
    "head_joint1": 0.0,
    "head_joint2": 0.0,
    "left_arm_joint1": 1.9,
    "left_arm_joint2": -1.5,
    "left_arm_joint3": -0.6,
    "left_arm_joint4": -2.1,
    "left_arm_joint5": 0.0,
    "left_arm_joint6": -0.25,
    "left_arm_joint7": 0.1,
    "right_arm_joint1": -1.9,
    "right_arm_joint2": 1.5,
    "right_arm_joint3": 0.6,
    "right_arm_joint4": 2.1,
    "right_arm_joint5": 0.0,
    "right_arm_joint6": 0.25,
    "right_arm_joint7": -0.1,
}


class MainWindow(QMainWindow):
    def __init__(self, point_cloud_path: str | None = None) -> None:
        super().__init__()
        self.setWindowTitle("Galbot Motion Obstacle Annotator")
        self.resize(1550, 950)

        self.point_cloud_path = ""
        self.obstacles_json_path = ""
        self.full_point_cloud: pv.PolyData | None = None
        self.point_cloud: pv.PolyData | None = None
        self.obstacles: list[Obstacle] = []
        self.obstacle_actors: dict[int, object] = {}
        self.active_index = -1
        self.transform_widget = None
        self.updating_widget = False
        self.selection_mode = False
        self.pending_selection_camera_state: dict[str, object] | None = None
        self.selection_primer_point: np.ndarray | None = None
        self.picked_points: list[np.ndarray] = []
        self.robot_visuals: list[RobotVisual] = []
        self.robot_actor_names: list[str] = []

        self.plotter = QtInteractor(self)
        self.plotter.interactor.installEventFilter(self)
        self.plotter.set_background("#20242b")
        self.plotter.add_axes(line_width=3)
        self.picked_points_mesh = pv.PolyData(np.zeros((1, 3), dtype=float))
        self.picked_points_actor = self.plotter.add_mesh(
            self.picked_points_mesh,
            name="selected_points",
            style="points",
            color="#ff5a5f",
            point_size=12,
            render_points_as_spheres=True,
            pickable=False,
            opacity=0.0,
            reset_camera=False,
            render=False,
        )
        self._build_ui()
        self._build_menu()

        if point_cloud_path:
            self.load_point_cloud(point_cloud_path)
        self._load_default_robot_if_available()

    def _build_ui(self) -> None:
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self.plotter.interactor)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(520)
        scroll.setWidget(self._build_control_panel())
        splitter.addWidget(scroll)
        splitter.setStretchFactor(0, 1)
        splitter.setSizes([1030, 520])
        self.setCentralWidget(splitter)
        self.statusBar().showMessage("打开一个 PCD 文件开始标注")

    def _build_menu(self) -> None:
        file_menu = self.menuBar().addMenu("文件")
        actions = [
            ("打开点云", self.open_point_cloud),
            ("打开 JSON", self.open_json),
            ("加载机器人 URDF", self.browse_robot_model),
            ("导出 JSON", self.save_json),
            ("导出 Python", self.save_python),
        ]
        for title, callback in actions:
            action = QAction(title, self)
            action.triggered.connect(callback)
            file_menu.addAction(action)

    def _build_control_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)

        open_button = QPushButton("打开 PCD")
        open_button.clicked.connect(self.open_point_cloud)
        json_button = QPushButton("打开 JSON")
        json_button.clicked.connect(self.open_json)
        layout.addWidget(open_button)
        layout.addWidget(json_button)
        layout.addWidget(self._build_filter_group())
        layout.addWidget(self._build_selection_group())
        layout.addWidget(self._build_obstacle_group())
        layout.addWidget(self._build_robot_group())

        view_row = QHBoxLayout()
        for title, callback in (
            ("俯视", self.plotter.view_xy),
            ("前视", self.plotter.view_xz),
            ("侧视", self.plotter.view_yz),
            ("透视", self.plotter.view_isometric),
        ):
            button = QPushButton(title)
            button.clicked.connect(callback)
            view_row.addWidget(button)
        layout.addLayout(view_row)

        export_row = QHBoxLayout()
        json_button = QPushButton("导出 JSON")
        json_button.clicked.connect(self.save_json)
        python_button = QPushButton("导出 Python")
        python_button.clicked.connect(self.save_python)
        export_row.addWidget(json_button)
        export_row.addWidget(python_button)
        layout.addLayout(export_row)
        layout.addStretch()
        return panel

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

        self.point_color_mode_combo = QComboBox()
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

    def _build_selection_group(self) -> QGroupBox:
        group = QGroupBox("低开销点选标注")
        layout = QVBoxLayout(group)
        hint = QLabel("开始后在点云上右键逐点采样；完成后根据碰撞体类型自动拟合。")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        row = QHBoxLayout()
        self.fit_mode_combo = QComboBox()
        self.fit_mode_combo.addItems(["AABB（稳定）", "OBB（贴合旋转）"])
        self.selection_type_combo = QComboBox()
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
        create_button.clicked.connect(self.create_obstacle_from_points)
        action_row.addWidget(undo_button)
        action_row.addWidget(clear_button)
        action_row.addWidget(create_button)
        layout.addLayout(action_row)
        return group

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
        self.type_combo = QComboBox()
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
        apply_button.clicked.connect(self.apply_form)
        layout.addWidget(apply_button)
        self._connect_form_live_updates()
        return group

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
        apply_button.clicked.connect(self.apply_robot_pose)
        row.addWidget(load_button)
        row.addWidget(apply_button)
        layout.addLayout(row)
        return group

    def _add_vector_row(
        self,
        form: QFormLayout,
        title: str,
        minimum: float,
        maximum: float,
        step: float,
        count: int = 3,
    ) -> list[QDoubleSpinBox]:
        container = QWidget()
        row = QHBoxLayout(container)
        row.setContentsMargins(0, 0, 0, 0)
        spins = [self._spin(minimum, maximum, step) for _ in range(count)]
        for spin in spins:
            row.addWidget(spin)
        form.addRow(title, container)
        return spins

    @staticmethod
    def _spin(minimum: float, maximum: float, step: float) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(minimum, maximum)
        spin.setDecimals(4)
        spin.setSingleStep(step)
        return spin

    def _connect_form_live_updates(self) -> None:
        self.id_edit.editingFinished.connect(self._apply_form_live)
        self.frame_edit.editingFinished.connect(self._apply_form_live)
        self.type_combo.currentTextChanged.connect(self._apply_form_live)
        for spin in (*self.center_spins, *self.rpy_spins, *self.size_spins):
            spin.valueChanged.connect(self._apply_form_live)

    def _apply_form_live(self, *_args) -> None:
        if self.updating_widget or not 0 <= self.active_index < len(self.obstacles):
            return
        self.apply_form()

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

    def eventFilter(self, watched, event) -> bool:
        if watched is self.plotter.interactor and self.selection_mode:
            if event.type() == QEvent.MouseButtonPress and event.button() == Qt.RightButton:
                self.pending_selection_camera_state = self._camera_state()
                return True
            if event.type() == QEvent.MouseButtonRelease and event.button() == Qt.RightButton:
                self._pick_point_from_qt_position(event.position().x(), event.position().y())
                QTimer.singleShot(0, self._restore_selection_camera)
                return True
            if event.type() == QEvent.ContextMenu:
                return True
        return super().eventFilter(watched, event)

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

    def _camera_state(self) -> dict[str, object]:
        camera = self.plotter.camera
        return {
            "position": camera.position,
            "focal_point": camera.focal_point,
            "up": camera.up,
            "clipping_range": camera.clipping_range,
            "parallel_scale": camera.parallel_scale,
            "parallel_projection": camera.parallel_projection,
        }

    def _restore_camera_state(self, camera_state: dict[str, object]) -> None:
        camera = self.plotter.camera
        camera.position = camera_state["position"]
        camera.focal_point = camera_state["focal_point"]
        camera.up = camera_state["up"]
        camera.clipping_range = camera_state["clipping_range"]
        camera.parallel_scale = camera_state["parallel_scale"]
        camera.parallel_projection = camera_state["parallel_projection"]

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

    def _load_default_robot_if_available(self) -> None:
        description_dir = Path(__file__).resolve().parents[2] / "galbot_one_golf_description"
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
            self.robot_visuals = load_urdf_visuals(path, DEFAULT_ROBOT_JOINT_POSITIONS)
        except (OSError, RuntimeError, ValueError) as error:
            QMessageBox.critical(self, "机器人加载失败", str(error))
            return
        self.apply_robot_pose()
        self.statusBar().showMessage(
            f"已加载机器人模型：{path.name}，共 {len(self.robot_visuals)} 个视觉网格"
        )

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
        self._render_robot(compose_pose(position, quaternion))
        if self.full_point_cloud is not None:
            self.apply_point_cloud_filter(reset_camera=False)

    def _render_robot(self, base_transform: np.ndarray) -> None:
        for actor_name in self.robot_actor_names:
            self.plotter.remove_actor(actor_name, render=False)
        self.robot_actor_names.clear()

        for index, visual in enumerate(self.robot_visuals):
            mesh = visual.mesh.copy(deep=True)
            mesh.transform(base_transform @ visual.transform, inplace=True)
            actor_name = f"robot_visual_{index}"
            mesh_options = {
                "name": actor_name,
                "opacity": 1.0,
                "smooth_shading": True,
                "pickable": False,
            }
            if visual.texture is not None:
                mesh_options["texture"] = visual.texture
            else:
                mesh_options["color"] = "#ffffff"
            self.plotter.add_mesh(
                mesh,
                **mesh_options,
            )
            self.robot_actor_names.append(actor_name)
        self.plotter.render()

    def _validate_export(self) -> bool:
        if not self.obstacles:
            QMessageBox.warning(self, "没有碰撞体", "请至少创建一个碰撞体。")
            return False
        ids = [obstacle.obstacle_id for obstacle in self.obstacles]
        if len(ids) != len(set(ids)):
            QMessageBox.warning(self, "名称重复", "碰撞体 obstacle_id 必须唯一。")
            return False
        for obstacle in self.obstacles:
            try:
                obstacle.validate_for_motion()
            except ValueError as error:
                QMessageBox.warning(self, "碰撞体参数无效", f"{obstacle.obstacle_id}: {error}")
                return False
        return True

    def save_json(self) -> None:
        if not self._validate_export():
            return
        default_name = Path(self.obstacles_json_path).name if self.obstacles_json_path else "obstacles.json"
        initial = self.obstacles_json_path or default_name
        path, _ = QFileDialog.getSaveFileName(self, "导出 JSON", initial, "JSON (*.json)")
        if path:
            export_json(path, self.obstacles, self.point_cloud_path)
            self.obstacles_json_path = str(Path(path).resolve())
            self.statusBar().showMessage(f"已导出：{path}")

    def save_python(self) -> None:
        if not self._validate_export():
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "导出 Python", "add_annotated_obstacles.py", "Python (*.py)"
        )
        if path:
            export_python(path, self.obstacles)
            self.statusBar().showMessage(f"已导出：{path}")

    def closeEvent(self, event) -> None:
        self.plotter.close()
        super().closeEvent(event)
