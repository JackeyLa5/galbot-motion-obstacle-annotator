from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def rpy_to_matrix(roll: float, pitch: float, yaw: float) -> FloatArray:
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)

    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=float,
    )


def matrix_to_rpy(matrix: FloatArray) -> FloatArray:
    pitch = math.asin(float(np.clip(-matrix[2, 0], -1.0, 1.0)))
    if abs(math.cos(pitch)) > 1e-8:
        roll = math.atan2(matrix[2, 1], matrix[2, 2])
        yaw = math.atan2(matrix[1, 0], matrix[0, 0])
    else:
        roll = math.atan2(-matrix[1, 2], matrix[1, 1])
        yaw = 0.0
    return np.array([roll, pitch, yaw], dtype=float)


def matrix_to_quaternion(matrix: FloatArray) -> FloatArray:
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * scale
        qx = (matrix[2, 1] - matrix[1, 2]) / scale
        qy = (matrix[0, 2] - matrix[2, 0]) / scale
        qz = (matrix[1, 0] - matrix[0, 1]) / scale
    elif matrix[0, 0] > matrix[1, 1] and matrix[0, 0] > matrix[2, 2]:
        scale = math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
        qw = (matrix[2, 1] - matrix[1, 2]) / scale
        qx = 0.25 * scale
        qy = (matrix[0, 1] + matrix[1, 0]) / scale
        qz = (matrix[0, 2] + matrix[2, 0]) / scale
    elif matrix[1, 1] > matrix[2, 2]:
        scale = math.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
        qw = (matrix[0, 2] - matrix[2, 0]) / scale
        qx = (matrix[0, 1] + matrix[1, 0]) / scale
        qy = 0.25 * scale
        qz = (matrix[1, 2] + matrix[2, 1]) / scale
    else:
        scale = math.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
        qw = (matrix[1, 0] - matrix[0, 1]) / scale
        qx = (matrix[0, 2] + matrix[2, 0]) / scale
        qy = (matrix[1, 2] + matrix[2, 1]) / scale
        qz = 0.25 * scale

    quaternion = np.array([qx, qy, qz, qw], dtype=float)
    quaternion /= np.linalg.norm(quaternion)
    if quaternion[3] < 0.0:
        quaternion *= -1.0
    return quaternion


def quaternion_to_matrix(quaternion: FloatArray) -> FloatArray:
    quaternion = np.asarray(quaternion, dtype=float)
    norm = np.linalg.norm(quaternion)
    if norm < 1e-12:
        raise ValueError("Quaternion norm must be greater than zero")
    qx, qy, qz, qw = quaternion / norm
    return np.array(
        [
            [1.0 - 2.0 * (qy * qy + qz * qz), 2.0 * (qx * qy - qz * qw), 2.0 * (qx * qz + qy * qw)],
            [2.0 * (qx * qy + qz * qw), 1.0 - 2.0 * (qx * qx + qz * qz), 2.0 * (qy * qz - qx * qw)],
            [2.0 * (qx * qz - qy * qw), 2.0 * (qy * qz + qx * qw), 1.0 - 2.0 * (qx * qx + qy * qy)],
        ],
        dtype=float,
    )


def compose_pose(position: FloatArray, quaternion: FloatArray) -> FloatArray:
    matrix = np.eye(4, dtype=float)
    matrix[:3, :3] = quaternion_to_matrix(quaternion)
    matrix[:3, 3] = position
    return matrix


def compose_box_transform(center: FloatArray, rpy: FloatArray, size: FloatArray) -> FloatArray:
    matrix = np.eye(4, dtype=float)
    matrix[:3, :3] = rpy_to_matrix(*rpy) @ np.diag(size)
    matrix[:3, 3] = center
    return matrix


def decompose_box_transform(matrix: FloatArray) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    center = matrix[:3, 3].copy()
    linear = matrix[:3, :3]
    size = np.linalg.norm(linear, axis=0)
    if np.any(size < 1e-9):
        raise ValueError("Box dimensions must be greater than zero")

    rotation = linear / size
    u, _, vh = np.linalg.svd(rotation)
    rotation = u @ vh
    if np.linalg.det(rotation) < 0.0:
        rotation[:, -1] *= -1.0
    rpy = matrix_to_rpy(rotation)
    quaternion = matrix_to_quaternion(rotation)
    return center, rpy, size, quaternion
