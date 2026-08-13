#!/usr/bin/env python3
"""Generate a deterministic indoor point-cloud demo scene."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


class PointCloudBuilder:
    def __init__(self, seed: int = 7) -> None:
        self.rng = np.random.default_rng(seed)
        self.chunks: list[np.ndarray] = []

    def add(self, points: np.ndarray) -> None:
        self.chunks.append(np.asarray(points, dtype=np.float32))

    def sample_box(
        self,
        center: tuple[float, float, float],
        size: tuple[float, float, float],
        spacing: float = 0.04,
        surfaces: tuple[str, ...] = ("top", "bottom", "front", "back", "left", "right"),
    ) -> None:
        center_array = np.asarray(center, dtype=np.float32)
        half = np.asarray(size, dtype=np.float32) / 2.0
        axes = {
            "x": np.arange(-half[0], half[0] + spacing * 0.5, spacing, dtype=np.float32),
            "y": np.arange(-half[1], half[1] + spacing * 0.5, spacing, dtype=np.float32),
            "z": np.arange(-half[2], half[2] + spacing * 0.5, spacing, dtype=np.float32),
        }
        for surface in surfaces:
            if surface in {"top", "bottom"}:
                z = half[2] if surface == "top" else -half[2]
                xx, yy = np.meshgrid(axes["x"], axes["y"], indexing="xy")
                points = np.column_stack((xx.ravel(), yy.ravel(), np.full(xx.size, z)))
            elif surface in {"front", "back"}:
                y = half[1] if surface == "front" else -half[1]
                xx, zz = np.meshgrid(axes["x"], axes["z"], indexing="xy")
                points = np.column_stack((xx.ravel(), np.full(xx.size, y), zz.ravel()))
            else:
                x = half[0] if surface == "right" else -half[0]
                yy, zz = np.meshgrid(axes["y"], axes["z"], indexing="xy")
                points = np.column_stack((np.full(yy.size, x), yy.ravel(), zz.ravel()))
            self.add(points + center_array)

    def sample_cylinder(
        self,
        center: tuple[float, float, float],
        radius: float,
        height: float,
        spacing: float = 0.04,
        angle_count: int | None = None,
    ) -> None:
        center_array = np.asarray(center, dtype=np.float32)
        angle_count = angle_count or max(24, int(2.0 * np.pi * radius / spacing))
        angles = np.linspace(0.0, 2.0 * np.pi, angle_count, endpoint=False, dtype=np.float32)
        z_values = np.arange(-height / 2.0, height / 2.0 + spacing * 0.5, spacing, dtype=np.float32)
        theta, z = np.meshgrid(angles, z_values, indexing="xy")
        side = np.column_stack((radius * np.cos(theta).ravel(), radius * np.sin(theta).ravel(), z.ravel()))
        self.add(side + center_array)

        radial_values = np.arange(0.0, radius + spacing * 0.5, spacing, dtype=np.float32)
        radial, theta = np.meshgrid(radial_values, angles, indexing="xy")
        for z_value in (-height / 2.0, height / 2.0):
            cap = np.column_stack(
                (
                    radial.ravel() * np.cos(theta.ravel()),
                    radial.ravel() * np.sin(theta.ravel()),
                    np.full(radial.size, z_value),
                )
            )
            self.add(cap + center_array)

    def sample_sphere(
        self,
        center: tuple[float, float, float],
        radius: float,
        spacing: float = 0.04,
    ) -> None:
        center_array = np.asarray(center, dtype=np.float32)
        latitude_count = max(12, int(np.pi * radius / spacing))
        longitude_count = max(24, int(2.0 * np.pi * radius / spacing))
        latitude = np.linspace(-np.pi / 2.0, np.pi / 2.0, latitude_count, dtype=np.float32)
        longitude = np.linspace(0.0, 2.0 * np.pi, longitude_count, endpoint=False, dtype=np.float32)
        lat, lon = np.meshgrid(latitude, longitude, indexing="ij")
        points = np.column_stack(
            (
                radius * np.cos(lat).ravel() * np.cos(lon).ravel(),
                radius * np.cos(lat).ravel() * np.sin(lon).ravel(),
                radius * np.sin(lat).ravel(),
            )
        )
        self.add(points + center_array)

    def sample_ground(self, extent: float = 5.0, spacing: float = 0.06) -> None:
        values = np.arange(-extent, extent + spacing * 0.5, spacing, dtype=np.float32)
        xx, yy = np.meshgrid(values, values, indexing="xy")
        ground = np.column_stack((xx.ravel(), yy.ravel(), np.zeros(xx.size, dtype=np.float32)))
        self.add(ground)

    def build(self) -> np.ndarray:
        points = np.concatenate(self.chunks, axis=0)
        points = np.unique(np.round(points, decimals=5), axis=0)
        order = self.rng.permutation(len(points))
        return points[order]


def build_scene() -> np.ndarray:
    scene = PointCloudBuilder()
    scene.sample_ground()

    # Dining table: 1.4 m x 0.8 m, with a 0.75 m tabletop height.
    scene.sample_box((2.35, 1.75, 0.71), (1.4, 0.8, 0.08), spacing=0.025)
    for x in (1.72, 2.98):
        for y in (1.43, 2.07):
            scene.sample_box((x, y, 0.34), (0.07, 0.07, 0.68), spacing=0.025)

    # Dining chair: 0.45 m seat height and 0.9 m total height.
    scene.sample_box((2.35, 0.75, 0.45), (0.48, 0.48, 0.07), spacing=0.022)
    scene.sample_box((2.35, 0.98, 0.69), (0.48, 0.06, 0.48), spacing=0.022)
    for x in (2.15, 2.55):
        for y in (0.57, 0.93):
            scene.sample_box((x, y, 0.22), (0.055, 0.055, 0.44), spacing=0.022)

    # Two-seat sofa along the rear-left side.
    scene.sample_box((-2.45, 2.55, 0.28), (1.9, 0.82, 0.32), spacing=0.035)
    scene.sample_box((-2.45, 2.82, 0.68), (1.9, 0.28, 0.78), spacing=0.035)
    scene.sample_box((-3.32, 2.52, 0.56), (0.18, 0.78, 0.58), spacing=0.03)
    scene.sample_box((-1.58, 2.52, 0.56), (0.18, 0.78, 0.58), spacing=0.03)
    scene.sample_box((-2.9, 2.38, 0.49), (0.78, 0.48, 0.14), spacing=0.03)
    scene.sample_box((-2.0, 2.38, 0.49), (0.78, 0.48, 0.14), spacing=0.03)

    # Low coffee table in front of the sofa.
    scene.sample_box((-2.4, 1.35, 0.42), (1.05, 0.58, 0.07), spacing=0.025)
    for x in (-2.82, -1.98):
        for y in (1.14, 1.56):
            scene.sample_box((x, y, 0.2), (0.06, 0.06, 0.4), spacing=0.025)

    # Low cabinet with three visible door panels.
    scene.sample_box((-3.75, -1.45, 0.48), (1.35, 0.42, 0.96), spacing=0.035)
    for x in (-4.18, -3.75, -3.32):
        scene.sample_box((x, -1.225, 0.49), (0.36, 0.03, 0.78), spacing=0.025, surfaces=("front",))

    # Cardboard boxes of different sizes near the right wall.
    scene.sample_box((3.35, -1.55, 0.28), (0.62, 0.52, 0.56), spacing=0.03)
    scene.sample_box((3.45, -1.48, 0.73), (0.42, 0.38, 0.34), spacing=0.028)
    scene.sample_box((2.75, -2.15, 0.2), (0.48, 0.42, 0.4), spacing=0.028)

    # Cylindrical trash bin and a small round stool.
    scene.sample_cylinder((1.65, -2.5, 0.32), radius=0.22, height=0.64, spacing=0.025)
    scene.sample_cylinder((-1.25, -2.55, 0.23), radius=0.28, height=0.46, spacing=0.025)

    # Potted plant: pot, trunk and several overlapping foliage clusters.
    scene.sample_cylinder((-3.75, 0.1, 0.22), radius=0.24, height=0.44, spacing=0.025)
    scene.sample_cylinder((-3.75, 0.1, 0.68), radius=0.045, height=0.7, spacing=0.02)
    for center, radius in (
        ((-3.75, 0.1, 1.05), 0.32),
        ((-3.98, 0.08, 0.94), 0.25),
        ((-3.55, 0.02, 0.94), 0.27),
        ((-3.72, 0.28, 0.9), 0.24),
    ):
        scene.sample_sphere(center, radius, spacing=0.035)

    # Floor lamp with a circular base, narrow pole and box-like shade.
    scene.sample_cylinder((0.95, 3.45, 0.035), radius=0.27, height=0.07, spacing=0.025)
    scene.sample_cylinder((0.95, 3.45, 0.85), radius=0.035, height=1.65, spacing=0.025)
    scene.sample_box((0.95, 3.45, 1.68), (0.42, 0.42, 0.34), spacing=0.025)

    # A small suitcase near the cabinet.
    scene.sample_box((-2.55, -2.45, 0.35), (0.62, 0.25, 0.7), spacing=0.028)
    scene.sample_box((-2.55, -2.45, 0.76), (0.22, 0.05, 0.14), spacing=0.022)
    return scene.build()


def write_pcd(path: Path, points: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii") as stream:
        stream.write(
            "VERSION .7\n"
            "FIELDS x y z\n"
            "SIZE 4 4 4\n"
            "TYPE F F F\n"
            "COUNT 1 1 1\n"
            f"WIDTH {len(points)}\n"
            "HEIGHT 1\n"
            "VIEWPOINT 0 0 0 1 0 0 0\n"
            f"POINTS {len(points)}\n"
            "DATA ascii\n"
        )
        np.savetxt(stream, points, fmt="%.5f")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", nargs="?", type=Path, default=Path("tests/data/test.pcd"))
    args = parser.parse_args()
    points = build_scene()
    write_pcd(args.output, points)
    print(f"Wrote {len(points):,} points to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
