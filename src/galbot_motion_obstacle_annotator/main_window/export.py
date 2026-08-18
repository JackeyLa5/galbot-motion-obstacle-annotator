from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import QFileDialog, QMessageBox

from ..exporters import export_json, export_python


class ExportMixin:
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
