from __future__ import annotations

import numpy as np
import pyvista as pv
from PySide6.QtCore import QEvent, Qt, QTimer
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QSlider,
    QSplitter,
    QToolButton,
    QVBoxLayout,
    QWidget,
)
from pyvistaqt import QtInteractor
from vtkmodules.vtkRenderingCore import vtkPropPicker

from ..models import Obstacle
from ..planning.models import PlanRequest, PlanResult
from ..planning.registry import default_registry
from ..robot_model import RobotVisual
from ..robot_state import INITIAL_ROBOT_JOINT_POSITIONS, RobotEnvironmentState
from .export import ExportMixin
from .obstacles import ObstacleMixin
from .plan_execution import PlanningMixin
from .point_cloud import PointCloudMixin
from .robot import RobotMixin
from .selection import SelectionMixin
from .tcp import TcpMixin
from .theme import STYLESHEET
from .widgets import FocusedWheelDoubleSpinBox, PlannerWorker, WorkspaceSamplerWorker
from .workspace import WorkspaceMixin


class MainWindow(
    QMainWindow,
    PointCloudMixin,
    SelectionMixin,
    ObstacleMixin,
    RobotMixin,
    TcpMixin,
    WorkspaceMixin,
    PlanningMixin,
    ExportMixin,
):
    def __init__(self, point_cloud_path: str | None = None) -> None:
        super().__init__()
        self.setWindowTitle("Galbot Motion Obstacle Annotator")
        self.resize(1550, 950)
        self.setStyleSheet(STYLESHEET)

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
        self.robot_state = RobotEnvironmentState(dict(INITIAL_ROBOT_JOINT_POSITIONS))
        self.robot_actor_names: list[str] = []
        self.robot_actors: dict[str, object] = {}
        self.robot_base_transform = np.eye(4, dtype=float)
        self.robot_base_gizmo_actor = None
        self.robot_base_gizmo_widget = None
        self.updating_robot_base_gizmo = False
        self.robot_base_last_widget_matrix: np.ndarray | None = None
        self.robot_base_move_mode = False
        self.robot_base_move_dirty = False
        self.robot_base_pending_transform: np.ndarray | None = None
        self.robot_base_update_timer = QTimer(self)
        self.robot_base_update_timer.setSingleShot(True)
        self.robot_base_update_timer.setInterval(50)
        self.robot_base_update_timer.timeout.connect(self._apply_pending_robot_base_transform)
        self.robot_pick_press_position: tuple[float, float] | None = None
        self.robot_mouse_capture = False
        self.robot_mouse_camera_state: dict[str, object] | None = None
        self.robot_drag_mode: str | None = None
        self.robot_drag_last_display_pos: np.ndarray | None = None
        self.robot_joint_limits: dict[str, tuple[float, float]] = {}
        self.robot_joint_sliders: dict[str, QSlider] = {}
        self.robot_joint_value_labels: dict[str, QLabel] = {}
        self.robot_joint_slider_scale = 1000
        self.robot_joint_update_timer = QTimer(self)
        self.robot_joint_update_timer.setSingleShot(True)
        self.robot_joint_update_timer.setInterval(33)
        self.robot_joint_update_timer.timeout.connect(self._apply_robot_joint_sliders)
        self.robot_joint_plan_invalidated = False
        self.tcp_gripper_visuals: list[RobotVisual] = []
        self.tcp_actor_names: list[str] = []
        self.tcp_handle_actor = None
        self.tcp_gizmo_actor = None
        self.tcp_transform_widget = None
        self.tcp_transform_active = False
        self.tcp_pose_matrix = np.eye(4, dtype=float)
        self.tcp_pose_edit_mode = False
        self.tcp_selection_mode = False
        self.pending_tcp_camera_state: dict[str, object] | None = None
        self.updating_tcp_widget = False
        self.tcp_target_selected = False
        self.workspace_points: np.ndarray | None = None
        self.workspace_joint_values: np.ndarray | None = None
        self.workspace_active_joint_names: tuple[str, ...] = ()
        self.workspace_actor = None
        self.workspace_visible = False
        self.workspace_pick_mode = False
        self.pending_workspace_pick_camera_state: dict[str, object] | None = None
        self.workspace_worker: WorkspaceSamplerWorker | None = None
        self.workspace_started_at = 0.0
        self.workspace_requested_sample_count = 0
        self.workspace_used_pyroki = False
        self.workspace_used_obstacles = False
        self.planner_registry = default_registry()
        self.planner_worker: PlannerWorker | None = None
        self.plan_result: PlanResult | None = None
        self.playback_frames: list[dict[str, np.ndarray]] = []
        self.playback_tcp_points = np.empty((0, 3), dtype=float)
        self.playback_joint_names: tuple[str, ...] = ()
        self.playback_joint_positions = np.empty((0, 0), dtype=float)
        self.playback_index = 0
        self.compare_path_actor_names: set[str] = set()
        self.compare_active = False
        self.compare_queue: list[str] = []
        self.compare_requests: dict[str, PlanRequest] = {}
        self.compare_results: dict[str, PlanResult | None] = {}
        self.compare_unavailable: dict[str, str] = {}
        self.playback_timer = QTimer(self)
        self.playback_timer.timeout.connect(self._advance_playback)

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
        central = QWidget()
        central_layout = QHBoxLayout(central)
        central_layout.setContentsMargins(0, 0, 0, 0)
        central_layout.setSpacing(0)

        self.joint_scroll = QScrollArea()
        self.joint_scroll.setWidgetResizable(True)
        self.joint_scroll.setMinimumWidth(430)
        self.joint_scroll.setMaximumWidth(540)
        self.joint_scroll.setFrameShape(QFrame.NoFrame)
        self.joint_scroll.setWidget(self._build_joint_state_group())
        central_layout.addWidget(self.joint_scroll)

        self.robot_joint_panel_expanded = True
        self.robot_joint_panel_toggle = QToolButton()
        self.robot_joint_panel_toggle.setArrowType(Qt.LeftArrow)
        self.robot_joint_panel_toggle.setFixedSize(18, 54)
        self.robot_joint_panel_toggle.setToolTip("折叠关节状态栏")
        self.robot_joint_panel_toggle.setStyleSheet(
            """
            QToolButton {
                background: rgba(54, 62, 72, 150);
                border: 0;
                border-radius: 0 6px 6px 0;
                padding: 0;
            }
            QToolButton:hover { background: rgba(79, 89, 101, 205); }
            QToolButton:pressed { background: rgba(91, 103, 116, 230); }
            """
        )
        self.robot_joint_panel_toggle.clicked.connect(self._toggle_joint_state_panel)
        toggle_column = QVBoxLayout()
        toggle_column.setContentsMargins(0, 0, 0, 0)
        toggle_column.addStretch()
        toggle_column.addWidget(self.robot_joint_panel_toggle)
        toggle_column.addStretch()
        central_layout.addLayout(toggle_column)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(0)
        splitter.addWidget(self.plotter.interactor)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(620)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setWidget(self._build_control_panel())
        splitter.addWidget(scroll)
        splitter.setStretchFactor(0, 1)
        splitter.setSizes([920, 620])
        central_layout.addWidget(splitter, 1)
        self.setCentralWidget(central)
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
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(12)

        open_row = QHBoxLayout()
        open_button = QPushButton("打开 PCD")
        open_button.clicked.connect(self.open_point_cloud)
        json_button = QPushButton("打开 JSON")
        json_button.clicked.connect(self.open_json)
        open_row.addWidget(open_button)
        open_row.addWidget(json_button)
        layout.addLayout(open_row)

        layout.addWidget(self._make_collapsible(self._build_filter_group(), expanded=False))
        layout.addWidget(self._make_collapsible(self._build_selection_group(), expanded=True))
        layout.addWidget(self._make_collapsible(self._build_obstacle_group(), expanded=True))
        layout.addWidget(self._make_collapsible(self._build_robot_group(), expanded=False))
        layout.addWidget(self._make_collapsible(self._build_tcp_group(), expanded=True))
        layout.addWidget(self._make_collapsible(self._build_workspace_group(), expanded=False))

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
        json_button.setProperty("cssClass", "primary")
        json_button.clicked.connect(self.save_json)
        python_button = QPushButton("导出 Python")
        python_button.setProperty("cssClass", "primary")
        python_button.clicked.connect(self.save_python)
        export_row.addWidget(json_button)
        export_row.addWidget(python_button)
        layout.addLayout(export_row)
        layout.addStretch()
        return panel

    @staticmethod
    def _make_collapsible(group: QGroupBox, *, expanded: bool = True) -> QGroupBox:
        """Turn a QGroupBox into a collapsible section (click the header to fold it)."""
        base_title = group.title()
        content_layout = group.layout()

        def apply(checked: bool) -> None:
            group.setTitle(("▾ " if checked else "▸ ") + base_title)
            if content_layout is None:
                return
            for index in range(content_layout.count()):
                item = content_layout.itemAt(index)
                widget = item.widget()
                if widget is not None:
                    widget.setVisible(checked)
                    continue
                sub_layout = item.layout()
                if sub_layout is None:
                    continue
                for sub_index in range(sub_layout.count()):
                    sub_widget = sub_layout.itemAt(sub_index).widget()
                    if sub_widget is not None:
                        sub_widget.setVisible(checked)

        group.setCheckable(True)
        group.setChecked(expanded)
        group.toggled.connect(apply)
        apply(expanded)
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
        spin = FocusedWheelDoubleSpinBox()
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

    def eventFilter(self, watched, event) -> bool:
        if watched is self.plotter.interactor and self.tcp_selection_mode:
            if event.type() == QEvent.MouseButtonPress and event.button() == Qt.RightButton:
                self.pending_tcp_camera_state = self._camera_state()
                return True
            if event.type() == QEvent.MouseButtonRelease and event.button() == Qt.RightButton:
                self._pick_tcp_point_from_qt_position(event.position().x(), event.position().y())
                QTimer.singleShot(0, self._restore_tcp_selection_camera)
                return True
            if event.type() == QEvent.ContextMenu:
                return True
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
        if watched is self.plotter.interactor and self.workspace_pick_mode:
            if event.type() == QEvent.MouseButtonPress and event.button() == Qt.RightButton:
                self.pending_workspace_pick_camera_state = self._camera_state()
                return True
            if event.type() == QEvent.MouseButtonRelease and event.button() == Qt.RightButton:
                self._pick_workspace_point_from_qt_position(event.position().x(), event.position().y())
                QTimer.singleShot(0, self._restore_workspace_pick_camera)
                return True
            if event.type() == QEvent.ContextMenu:
                return True
        if (
            watched is self.plotter.interactor
            and not self.tcp_selection_mode
            and not self.selection_mode
            and not self.workspace_pick_mode
        ):
            if event.type() == QEvent.MouseButtonPress and event.button() == Qt.LeftButton:
                self.robot_pick_press_position = (event.position().x(), event.position().y())
                actor = self._pick_scene_actor(
                    event.position().x(), event.position().y()
                )
                gizmo_actors = self._robot_gizmo_actors()
                tcp_gizmo_actors = self._tcp_gizmo_actors()
                robot_drag_mode = self._robot_drag_mode_for_actor(actor)
                if robot_drag_mode is not None:
                    self.robot_drag_mode = robot_drag_mode
                    self.robot_drag_last_display_pos = self._qt_to_display_position(
                        event.position().x(), event.position().y()
                    )
                    self.robot_mouse_camera_state = self._camera_state()
                    return True
                if actor in self.robot_actors.values():
                    self.robot_mouse_capture = True
                    self.robot_mouse_camera_state = self._camera_state()
                    if self.tcp_transform_active:
                        self.tcp_transform_active = False
                        self._render_tcp_gripper()
                    self.start_robot_base_move()
                    return True
                if actor in gizmo_actors:
                    self.robot_mouse_capture = False
                    return super().eventFilter(watched, event)
                if actor in tcp_gizmo_actors or actor is self.tcp_handle_actor:
                    self.robot_mouse_capture = False
                    return super().eventFilter(watched, event)
                if self.tcp_pose_edit_mode and actor in self._tcp_visual_actors():
                    camera_state = self._camera_state()
                    self.tcp_transform_active = True
                    self._render_tcp_gripper()
                    self._restore_camera_state(camera_state)
                    self.plotter.render()
                    return True
                if self.tcp_transform_active:
                    camera_state = self._camera_state()
                    self.tcp_transform_active = False
                    self._render_tcp_gripper()
                    self._restore_camera_state(camera_state)
                    self.plotter.render()
                if self.robot_base_move_mode:
                    self.finish_robot_base_move()
                self.robot_mouse_capture = False
                self.robot_mouse_camera_state = None
                return super().eventFilter(watched, event)
            elif event.type() == QEvent.MouseMove and self.robot_drag_mode is not None:
                self._drag_robot_base_in_local_frame(event.position().x(), event.position().y())
                return True
            elif event.type() == QEvent.MouseButtonRelease and event.button() == Qt.LeftButton:
                if self.robot_drag_mode is not None:
                    self.robot_drag_mode = None
                    self.robot_drag_last_display_pos = None
                    if self.robot_mouse_camera_state is not None:
                        self._restore_camera_state(self.robot_mouse_camera_state)
                        self.robot_mouse_camera_state = None
                        self.plotter.render()
                    self.robot_base_update_timer.stop()
                    self._apply_pending_robot_base_transform()
                    self.robot_pick_press_position = None
                    return True
                if self.robot_mouse_capture:
                    self.robot_mouse_capture = False
                    if self.robot_mouse_camera_state is not None:
                        self._restore_camera_state(self.robot_mouse_camera_state)
                        self.robot_mouse_camera_state = None
                        self.plotter.render()
                    self.robot_pick_press_position = None
                    return True
                self.robot_pick_press_position = None
                self.robot_mouse_camera_state = None
                return super().eventFilter(watched, event)
        return super().eventFilter(watched, event)

    def _pick_scene_actor(self, x: float, y: float):
        display_x, display_y = self._qt_to_display_position(x, y)
        picker = vtkPropPicker()
        picker.Pick(float(display_x), float(display_y), 0.0, self.plotter.renderer)
        return picker.GetActor()

    def _qt_to_display_position(self, x: float, y: float) -> np.ndarray:
        render_width, render_height = self.plotter.render_window.GetSize()
        widget_width = max(1, self.plotter.interactor.width())
        widget_height = max(1, self.plotter.interactor.height())
        return np.array(
            [
                x * render_width / widget_width,
                (widget_height - y - 1.0) * render_height / widget_height,
            ],
            dtype=float,
        )

    def _world_to_display_position(self, point: np.ndarray) -> np.ndarray:
        self.plotter.renderer.SetWorldPoint(float(point[0]), float(point[1]), float(point[2]), 1.0)
        self.plotter.renderer.WorldToDisplay()
        display = self.plotter.renderer.GetDisplayPoint()
        return np.array([display[0], display[1]], dtype=float)

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

    def closeEvent(self, event) -> None:
        self.plotter.close()
        super().closeEvent(event)
