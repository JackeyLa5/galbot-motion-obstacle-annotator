from __future__ import annotations

import numpy as np
from vtkmodules.vtkRenderingCore import vtkCellPicker, vtkPointPicker, vtkPropPicker

from ..geometry import orbit_camera_frame

CAMERA_ORBIT_SENSITIVITY_DEG_PER_PIXEL = 0.3
CAMERA_ORBIT_ELEVATION_LIMIT_DEG = 85.0


class CameraMixin:
    """Alt+left-drag orbit around the point under the cursor (Isaac Sim / Maya style).

    Plain left-drag is reserved for picking/dragging obstacles, the robot,
    and TCP gizmos elsewhere in the eventFilter; middle-drag pan and scroll
    zoom already work through VTK's default trackball interactor style and
    are left untouched.
    """

    def _pick_orbit_pivot(self, x: float, y: float) -> np.ndarray:
        display_x, display_y = self._qt_to_display_position(x, y)
        renderer = self.plotter.renderer

        point_picker = vtkPointPicker()
        point_picker.SetTolerance(0.015)
        if point_picker.Pick(display_x, display_y, 0.0, renderer) and point_picker.GetPointId() >= 0:
            return np.asarray(point_picker.GetPickPosition(), dtype=float)

        cell_picker = vtkCellPicker()
        cell_picker.SetTolerance(0.0005)
        if cell_picker.Pick(display_x, display_y, 0.0, renderer) and cell_picker.GetCellId() >= 0:
            return np.asarray(cell_picker.GetPickPosition(), dtype=float)

        prop_picker = vtkPropPicker()
        if prop_picker.Pick(display_x, display_y, 0.0, renderer) and prop_picker.GetActor() is not None:
            return np.asarray(prop_picker.GetPickPosition(), dtype=float)

        return self._project_to_focal_plane(display_x, display_y)

    def _project_to_focal_plane(self, display_x: float, display_y: float) -> np.ndarray:
        """Fall back pivot for clicks that hit nothing: where the view ray crosses the focal plane."""
        camera = self.plotter.camera
        near = self._unproject_display_point(display_x, display_y, 0.0)
        far = self._unproject_display_point(display_x, display_y, 1.0)
        ray_direction = far - near
        ray_length = float(np.linalg.norm(ray_direction))
        focal_point = np.asarray(camera.focal_point, dtype=float)
        if ray_length < 1e-9:
            return focal_point
        ray_direction /= ray_length
        plane_normal = focal_point - np.asarray(camera.position, dtype=float)
        plane_normal_length = float(np.linalg.norm(plane_normal))
        if plane_normal_length < 1e-9:
            return focal_point
        plane_normal /= plane_normal_length
        denominator = float(np.dot(ray_direction, plane_normal))
        if abs(denominator) < 1e-9:
            return focal_point
        distance = float(np.dot(focal_point - near, plane_normal)) / denominator
        return near + ray_direction * distance

    def _unproject_display_point(self, display_x: float, display_y: float, z: float) -> np.ndarray:
        renderer = self.plotter.renderer
        renderer.SetDisplayPoint(display_x, display_y, z)
        renderer.DisplayToWorld()
        world = renderer.GetWorldPoint()
        w = world[3] if abs(world[3]) > 1e-9 else 1.0
        return np.array([world[0] / w, world[1] / w, world[2] / w], dtype=float)

    def _start_camera_orbit(self, x: float, y: float) -> None:
        camera = self.plotter.camera
        self.camera_orbit_active = True
        self.camera_orbit_pivot = self._pick_orbit_pivot(x, y)
        self.camera_orbit_start_position = np.asarray(camera.position, dtype=float)
        self.camera_orbit_start_focal = np.asarray(camera.focal_point, dtype=float)
        self.camera_orbit_start_up = np.asarray(camera.up, dtype=float)
        self.camera_orbit_start_mouse = (float(x), float(y))

    def _update_camera_orbit(self, x: float, y: float) -> None:
        if (
            not self.camera_orbit_active
            or self.camera_orbit_pivot is None
            or self.camera_orbit_start_mouse is None
        ):
            return
        start_x, start_y = self.camera_orbit_start_mouse
        delta_x = float(x) - start_x
        delta_y = float(y) - start_y
        azimuth_degrees = -delta_x * CAMERA_ORBIT_SENSITIVITY_DEG_PER_PIXEL
        elevation_degrees = float(
            np.clip(
                -delta_y * CAMERA_ORBIT_SENSITIVITY_DEG_PER_PIXEL,
                -CAMERA_ORBIT_ELEVATION_LIMIT_DEG,
                CAMERA_ORBIT_ELEVATION_LIMIT_DEG,
            )
        )
        new_position, new_focal, new_up = orbit_camera_frame(
            self.camera_orbit_pivot,
            self.camera_orbit_start_position,
            self.camera_orbit_start_focal,
            self.camera_orbit_start_up,
            azimuth_degrees,
            elevation_degrees,
        )
        camera = self.plotter.camera
        camera.position = tuple(new_position)
        camera.focal_point = tuple(new_focal)
        camera.up = tuple(new_up)
        self.plotter.render()

    def _end_camera_orbit(self) -> None:
        self.camera_orbit_active = False
        self.camera_orbit_pivot = None
        self.camera_orbit_start_position = None
        self.camera_orbit_start_focal = None
        self.camera_orbit_start_up = None
        self.camera_orbit_start_mouse = None
