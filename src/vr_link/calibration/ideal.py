"""Ideal (distortion-free) stereo calibration of simulated eye cameras.

Simulators render exact pinhole cameras, so the calibration follows from the
camera model instead of a checkerboard run. Two entry points:

* :func:`ideal_stereo_from_fovy` - from the rig description (per-eye render
  size, MuJoCo ``fovy``, toe-out angle, baseline). This is the
  ``make_mujoco_calibration.py`` model.
* :func:`ideal_stereo_from_intrinsics` - from live camera objects: per-eye
  intrinsic matrix and camera-to-world orientation in the OpenCV/ROS optical
  convention (+X right, +Y down, +Z forward). Isaac Sim's ``quat_w_ros`` is
  exactly that. :func:`ideal_stereo_from_metadata` adapts the MuJoCo camera
  convention (+X right, +Y up, -Z forward) published in the vision SHM header.

Both emit the tan-series fisheye coefficients (:data:`TAN_SERIES_K`) so the
receiver's fisheye shader reproduces a pinhole projection, and the OpenCV
``stereoRectify``-style R1/R2 that rotate each eye's optical frame into the
common mid frame (the slerp midpoint of the two eye orientations).
"""

from __future__ import annotations

import math
from typing import Dict, Sequence

import numpy as np

from .schema import R_KEYS, TAN_SERIES_K, SIDES

# MuJoCo camera axes (+X right, +Y up, looks along -Z) -> OpenCV optical axes
# (+X right, +Y down, +Z forward): flip Y and Z.
MUJOCO_CAM_TO_OPENCV = np.diag([1.0, -1.0, -1.0])


# --- rotation helpers ---------------------------------------------------------


def rotation_y(deg: float) -> np.ndarray:
    """Ry(deg) in OpenCV camera coordinates (row-major 3x3)."""

    c = math.cos(math.radians(deg))
    s = math.sin(math.radians(deg))
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)


def quat_wxyz_to_matrix(quat: Sequence[float]) -> np.ndarray:
    """Rotation matrix from a unit ``(w, x, y, z)`` quaternion."""

    w, x, y, z = (float(v) for v in quat)
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm <= 0.0 or not math.isfinite(norm):
        raise ValueError(f"degenerate quaternion {quat!r}")
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def matrix_to_quat_wxyz(rotation: np.ndarray) -> np.ndarray:
    """Unit ``(w, x, y, z)`` quaternion from a rotation matrix (Shepperd's method)."""

    m = np.asarray(rotation, dtype=np.float64)
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        q = (0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s)
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        q = ((m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s)
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        q = ((m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s)
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        q = ((m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s)
    quat = np.array(q, dtype=np.float64)
    return quat / np.linalg.norm(quat)


def quat_slerp(q0: Sequence[float], q1: Sequence[float], t: float) -> np.ndarray:
    """Spherical interpolation between two ``(w, x, y, z)`` quaternions."""

    a = np.asarray(q0, dtype=np.float64)
    b = np.asarray(q1, dtype=np.float64)
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    if float(np.dot(a, b)) < 0.0:  # take the shorter arc
        b = -b
    dot = float(np.clip(np.dot(a, b), -1.0, 1.0))
    if dot > 0.9995:  # nearly parallel: normalise the linear blend
        result = a + t * (b - a)
        return result / np.linalg.norm(result)
    theta = math.acos(dot)
    return (math.sin((1.0 - t) * theta) * a + math.sin(t * theta) * b) / math.sin(theta)


def rectifying_rotations(r_left_world: np.ndarray, r_right_world: np.ndarray) -> "tuple[np.ndarray, np.ndarray]":
    """R1/R2 (eye optical frame -> mid frame) from camera-to-world rotations."""

    q_mid = quat_slerp(matrix_to_quat_wxyz(r_left_world), matrix_to_quat_wxyz(r_right_world), 0.5)
    r_mid = quat_wxyz_to_matrix(q_mid)
    return r_mid.T @ np.asarray(r_left_world, dtype=np.float64), r_mid.T @ np.asarray(r_right_world, dtype=np.float64)


# --- intrinsics ------------------------------------------------------------------


def intrinsics_from_fovy(width: int, height: int, fovy_deg: float, *, principal: str = "index_center") -> np.ndarray:
    """Pinhole K for a square-pixel camera whose vertical FOV spans the image height.

    ``principal="index_center"`` places the principal point at ``((w-1)/2, (h-1)/2)``
    (pixel-index centre, the MuJoCo/OpenCV convention used by the generated
    YAML); ``"half"`` uses ``(w/2, h/2)``.
    """

    if width <= 0 or height <= 0:
        raise ValueError(f"image size must be positive, got {width}x{height}")
    if not 0.0 < fovy_deg < 180.0:
        raise ValueError(f"fovy must be in (0, 180) degrees, got {fovy_deg}")
    focal = (height / 2.0) / math.tan(math.radians(fovy_deg) / 2.0)
    if principal == "index_center":
        cx, cy = (width - 1) / 2.0, (height - 1) / 2.0
    elif principal == "half":
        cx, cy = width / 2.0, height / 2.0
    else:
        raise ValueError(f"unknown principal point convention {principal!r}")
    return np.array([[focal, 0.0, cx], [0.0, focal, cy], [0.0, 0.0, 1.0]], dtype=np.float64)


def _camera_entry(k: np.ndarray, rotation: np.ndarray) -> Dict[str, object]:
    k = np.asarray(k, dtype=np.float64)
    if k.shape != (3, 3):
        raise ValueError(f"intrinsic matrix must be 3x3, got {k.shape}")
    rotation = np.asarray(rotation, dtype=np.float64)
    if rotation.shape != (3, 3):
        raise ValueError(f"rotation must be 3x3, got {rotation.shape}")
    entry: Dict[str, object] = {
        "fx": float(k[0, 0]),
        "fy": float(k[1, 1]),
        "cx": float(k[0, 2]),
        "cy": float(k[1, 2]),
    }
    entry.update({f"k{i + 1}": float(v) for i, v in enumerate(TAN_SERIES_K)})
    entry["r"] = [float(v) for v in rotation.reshape(-1)]
    return entry


# --- public builders -------------------------------------------------------------


def ideal_stereo_from_intrinsics(
    width: int,
    height: int,
    left_k: np.ndarray,
    right_k: np.ndarray,
    left_r_world: np.ndarray,
    right_r_world: np.ndarray,
) -> Dict[str, object]:
    """Calibration dict from per-eye K and camera-to-world rotations (OpenCV optical axes)."""

    r_left, r_right = rectifying_rotations(left_r_world, right_r_world)
    return {
        "image_width": int(width),
        "image_height": int(height),
        "camera_model": "pinhole_as_fisheye_tan_series",
        "left": _camera_entry(left_k, r_left),
        "right": _camera_entry(right_k, r_right),
    }


def ideal_stereo_from_fovy(
    width: int,
    height: int,
    fovy_deg: float,
    toe_out_deg: float,
    baseline_mm: float,
) -> Dict[str, object]:
    """Calibration dict from the rig description (per-eye size, fovy, toe-out yaw, baseline).

    In OpenCV camera coordinates a camera yawed outward by ``toe`` degrees
    needs R = Ry(-toe) for the left eye and Ry(+toe) for the right eye; the
    signs match the measured hardware YAML (left r02 < 0, right r02 > 0).
    """

    k = intrinsics_from_fovy(width, height, fovy_deg)
    cal = {
        "image_width": int(width),
        "image_height": int(height),
        "camera_model": "pinhole_as_fisheye_tan_series",
        "fovy_deg": float(fovy_deg),
        "toe_out_deg": float(toe_out_deg),
        "baseline_mm": float(baseline_mm),
        "left": _camera_entry(k, rotation_y(-toe_out_deg)),
        "right": _camera_entry(k, rotation_y(+toe_out_deg)),
    }
    return cal


def ideal_stereo_from_metadata(meta) -> Dict[str, object]:
    """Calibration dict from :class:`vr_link.sources.StereoCameraMetadata` (MuJoCo axes)."""

    if meta.eye_width <= 0 or meta.eye_height <= 0:
        raise ValueError("camera metadata has no eye size")
    left_k = intrinsics_from_fovy(meta.eye_width, meta.eye_height, meta.left_fovy_deg)
    right_k = intrinsics_from_fovy(meta.eye_width, meta.eye_height, meta.right_fovy_deg)
    left_r = quat_wxyz_to_matrix(meta.left_quat_wxyz) @ MUJOCO_CAM_TO_OPENCV
    right_r = quat_wxyz_to_matrix(meta.right_quat_wxyz) @ MUJOCO_CAM_TO_OPENCV
    cal = ideal_stereo_from_intrinsics(meta.eye_width, meta.eye_height, left_k, right_k, left_r, right_r)
    cal["baseline_mm"] = float(meta.baseline_m) * 1000.0
    return cal


def flatten_sides(cal: Dict[str, object]) -> Dict[str, float]:
    """``{left_fx: ..., left_r00: ..., right_...}`` view used by the flat YAML format."""

    flat: Dict[str, float] = {}
    for side in SIDES:
        cam = cal[side]
        for key in ("fx", "fy", "cx", "cy", "k1", "k2", "k3", "k4"):
            flat[f"{side}_{key}"] = float(cam[key])
        for name, value in zip(R_KEYS, cam["r"]):
            flat[f"{side}_{name}"] = float(value)
    return flat


__all__ = [
    "MUJOCO_CAM_TO_OPENCV",
    "flatten_sides",
    "ideal_stereo_from_fovy",
    "ideal_stereo_from_intrinsics",
    "ideal_stereo_from_metadata",
    "intrinsics_from_fovy",
    "matrix_to_quat_wxyz",
    "quat_slerp",
    "quat_wxyz_to_matrix",
    "rectifying_rotations",
    "rotation_y",
]
