from __future__ import annotations

from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import QComboBox, QDoubleSpinBox, QWidget

from ..models import Obstacle
from ..planning.models import PlanRequest
from ..planning.pyroki import sample_reachable_positions_collision_aware
from ..robot_model import sample_reachable_positions


class PlannerWorker(QThread):
    finished_with_result = Signal(object)

    def __init__(self, planner, request: PlanRequest) -> None:
        super().__init__()
        self.planner = planner
        self.request = request

    def run(self) -> None:
        self.finished_with_result.emit(self.planner.plan(self.request))


class WorkspaceSamplerWorker(QThread):
    finished_with_points = Signal(object)

    def __init__(
        self,
        urdf_path: Path,
        tip_link: str,
        active_joint_names: tuple[str, ...],
        joint_limits: dict[str, tuple[float, float]],
        fixed_joint_positions: dict[str, float],
        sample_count: int,
        use_collision_awareness: bool = False,
        obstacles_base_frame: list[Obstacle] | None = None,
    ) -> None:
        super().__init__()
        self.urdf_path = urdf_path
        self.tip_link = tip_link
        self.active_joint_names = active_joint_names
        self.joint_limits = joint_limits
        self.fixed_joint_positions = fixed_joint_positions
        self.sample_count = sample_count
        self.use_collision_awareness = use_collision_awareness
        self.obstacles_base_frame = obstacles_base_frame or []
        self.error_message = ""
        self.diagnostics: dict[str, int] = {}

    def run(self) -> None:
        try:
            if self.use_collision_awareness:
                points, active_joint_values, diagnostics = sample_reachable_positions_collision_aware(
                    self.urdf_path,
                    self.tip_link,
                    self.active_joint_names,
                    self.joint_limits,
                    self.fixed_joint_positions,
                    self.obstacles_base_frame,
                    self.sample_count,
                    np.random.default_rng(),
                )
                self.diagnostics = diagnostics
            else:
                points, active_joint_values = sample_reachable_positions(
                    self.urdf_path,
                    self.tip_link,
                    self.active_joint_names,
                    self.joint_limits,
                    self.fixed_joint_positions,
                    self.sample_count,
                    np.random.default_rng(),
                )
        except (OSError, RuntimeError, ValueError) as error:
            self.error_message = str(error)
            self.finished_with_points.emit((None, None))
            return
        self.finished_with_points.emit((points, active_joint_values))


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
