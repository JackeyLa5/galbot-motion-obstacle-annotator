from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyvista as pv
from PySide6.QtCore import QEvent, QSignalBlocker, QThread, QTimer, Qt, Signal
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QCheckBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSlider,
    QSplitter,
    QToolButton,
    QVBoxLayout,
    QWidget,
)
from pyvistaqt import QtInteractor
from vtkmodules.vtkRenderingCore import vtkPointPicker, vtkPropPicker

from .exporters import export_json, export_python
from .geometry import (
    compose_pose,
    matrix_to_quaternion,
    matrix_to_rpy,
    rpy_to_matrix,
)
from .importers import load_json as load_obstacles_json
from .models import SUPPORTED_OBSTACLE_TYPES, Obstacle
from .pcd import load_pcd
from .planning.models import PlanRequest, PlanResult, PoseTarget
from .planning.pyroki import express_world_scene_in_base_frame, resolve_pyroki_urdf_path
from .planning.registry import default_registry
from .robot_state import (
    ARM_JOINT_NAMES,
    GALBOT_REFERENCE_JOINT_NAMES,
    INITIAL_ROBOT_JOINT_POSITIONS,
    LEG_JOINT_NAMES,
    RobotEnvironmentState,
)
from .robot_model import (
    RobotVisual,
    load_tcp_gripper_visuals,
    load_urdf_actuated_joint_names,
    load_urdf_joint_limits,
    load_urdf_link_transforms,
    load_urdf_visuals,
)

class PlannerWorker(QThread):
    finished_with_result = Signal(object)

    def __init__(self, planner, request: PlanRequest) -> None:
        super().__init__()
        self.planner = planner
        self.request = request

    def run(self) -> None:
        self.finished_with_result.emit(self.planner.plan(self.request))


class FocusedWheelDoubleSpinBox(QDoubleSpinBox):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFocusPolicy(Qt.StrongFocus)

    def wheelEvent(self, event) -> None:
        if not self.hasFocus():
            event.ignore()
            return
        super().wheelEvent(event)


class FocusedWheelComboBox(QComboBox):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFocusPolicy(Qt.StrongFocus)

    def wheelEvent(self, event) -> None:
        if not self.hasFocus():
            event.ignore()
            return
        super().wheelEvent(event)


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
        self.tcp_last_widget_matrix: np.ndarray | None = None
        self.tcp_pose_matrix = np.eye(4, dtype=float)
        self.tcp_pose_edit_mode = False
        self.tcp_selection_mode = False
        self.pending_tcp_camera_state: dict[str, object] | None = None
        self.updating_tcp_widget = False
        self.tcp_target_selected = False
        self.planner_registry = default_registry()
        self.planner_worker: PlannerWorker | None = None
        self.plan_result: PlanResult | None = None
        self.playback_frames: list[dict[str, np.ndarray]] = []
        self.playback_tcp_points = np.empty((0, 3), dtype=float)
        self.playback_joint_names: tuple[str, ...] = ()
        self.playback_joint_positions = np.empty((0, 0), dtype=float)
        self.playback_index = 0
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
        layout.addWidget(self._build_tcp_group())

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

    def _build_joint_state_group(self) -> QWidget:
        group = QFrame()
        group.setObjectName("jointStateCard")
        group.setStyleSheet(
            """
            QFrame#jointStateCard { background: #20252d; border: 0; }
            QLabel#jointName { color: #d9e2ec; font-size: 11px; }
            QLabel#jointLimit { color: #7f8b99; font-size: 10px; }
            QLabel#jointValue { color: #dce8e2; background: #303b38; border: 1px solid #53615c; border-radius: 4px; padding: 2px 5px; font-weight: 600; }
            QSlider::groove:horizontal { height: 5px; background: #3a4552; border-radius: 2px; }
            QSlider::sub-page:horizontal { background: #829b91; border-radius: 2px; }
            QSlider::add-page:horizontal { background: #303944; border-radius: 2px; }
            QSlider::handle:horizontal { width: 12px; margin: -4px 0; background: #e7ece9; border: 2px solid #829b91; border-radius: 7px; }
            """
        )
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
            value_label = QLabel()
            value_label.setFixedWidth(58)
            value_label.setAlignment(Qt.AlignCenter)
            value_label.setObjectName("jointValue")
            header_row.addWidget(label)
            header_row.addStretch(1)
            header_row.addWidget(value_label)

            slider_row = QHBoxLayout()
            slider_row.setSpacing(5)
            slider = QSlider(Qt.Horizontal)
            slider.setTracking(True)
            slider.setMinimumWidth(260)
            lower, upper = self.robot_joint_limits.get(joint_name, (-np.pi, np.pi))
            slider.setRange(
                int(np.ceil(lower * self.robot_joint_slider_scale)),
                int(np.floor(upper * self.robot_joint_slider_scale)),
            )
            minimum_label = QLabel(f"{lower:.3f}")
            minimum_label.setFixedWidth(42)
            minimum_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            minimum_label.setObjectName("jointLimit")
            maximum_label = QLabel(f"{upper:.3f}")
            maximum_label.setFixedWidth(42)
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
        if item.widget() is not None:
            item.widget().deleteLater()
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

    def _build_selection_group(self) -> QGroupBox:
        group = QGroupBox("低开销点选标注")
        layout = QVBoxLayout(group)
        hint = QLabel("开始后在点云上右键逐点采样；完成后根据碰撞体类型自动拟合。")
        hint.setWordWrap(True)
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
        hint = QLabel("左键点击机器人显示底盘控制器；点击其他场景对象或空白处结束移动。")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        return group

    def _build_tcp_group(self) -> QGroupBox:
        group = QGroupBox("工具 TCP 抓取姿态")
        layout = QVBoxLayout(group)
        hint = QLabel(
            "固定使用 is_tool_pose=true：TCP 表单和规划目标使用机器人 base_link 坐标系，场景里按当前机器人位姿渲染。"
        )
        hint.setWordWrap(True)
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
        if watched is self.plotter.interactor and not self.tcp_selection_mode and not self.selection_mode:
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

    def _world_to_display_position(self, point: np.ndarray) -> np.ndarray:
        self.plotter.renderer.SetWorldPoint(float(point[0]), float(point[1]), float(point[2]), 1.0)
        self.plotter.renderer.WorldToDisplay()
        display = self.plotter.renderer.GetDisplayPoint()
        return np.array([display[0], display[1]], dtype=float)

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
        matrix, position = self._tcp_pose_matrix_from_form()
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
        base_rotation = self.robot_base_transform[:3, :3]

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
        mapped_world_translation = current_world_matrix[:3, 3] + base_rotation @ raw_translation_delta

        raw_rotation_delta = raw_world_matrix[:3, :3] @ previous_widget_rotation.T
        mapped_world_rotation = base_rotation @ raw_rotation_delta @ base_rotation.T @ current_world_matrix[:3, :3]
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
        self.plan_button.setEnabled(True)

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
