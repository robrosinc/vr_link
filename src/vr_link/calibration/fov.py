"""Field of view implied by a calibration (what the lens actually sees).

For the OpenCV fisheye model the image radius of a ray at angle ``theta``
from the optical axis is ``r = f * theta_d(theta)`` with
``theta_d = theta * (1 + k1 theta^2 + k2 theta^4 + k3 theta^6 + k4 theta^8)``.
Inverting it at the image borders gives the half-angles the sensor covers;
their sums are the vertical/horizontal field of view. The vertical value is
the simulator's ``fovy``. Treating the fisheye ``fy`` as a pinhole focal
(``2 atan(H / 2 fy)``) underestimates the FOV and must not be used for that.

Border values are only as good as the calibration near the borders: when
the checkerboard never reached the image edge the polynomial is
extrapolated there, and the two eyes disagree. The summary reports both eyes
so that shows up.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Sequence

from .schema import SIDES, validate_calibration


def fisheye_forward(theta: float, k: Sequence[float]) -> float:
    """``theta_d`` of the OpenCV fisheye model for a ray angle ``theta`` (rad)."""

    t2 = theta * theta
    return theta * (1.0 + k[0] * t2 + k[1] * t2 ** 2 + k[2] * t2 ** 3 + k[3] * t2 ** 4)


def fisheye_theta(radius_px: float, focal_px: float, k: Sequence[float], *, tolerance: float = 1e-12) -> float:
    """Ray angle (rad) whose fisheye image radius is ``radius_px``.

    Bisection on the forward model (a fixed-point iteration diverges for the
    tan-series coefficients near the image edge). Assumes the forward model
    is increasing up to the first angle where it exceeds ``theta_d``.
    """

    if focal_px <= 0.0:
        raise ValueError("focal length must be positive")
    theta_d = radius_px / focal_px
    if theta_d <= 0.0:
        return 0.0
    high = theta_d
    while fisheye_forward(high, k) < theta_d:
        high *= 1.5
        if high > math.pi:
            raise ValueError(f"fisheye model never reaches theta_d={theta_d:.3f}; coefficients {tuple(k)} are not usable there")
    low = 0.0
    for _ in range(200):
        mid = 0.5 * (low + high)
        if fisheye_forward(mid, k) < theta_d:
            low = mid
        else:
            high = mid
        if high - low < tolerance:
            break
    return 0.5 * (low + high)


def _border_angle(radius_px: float, focal_px: float, k: Sequence[float]) -> float:
    """``fisheye_theta`` or NaN when the polynomial never reaches that radius."""

    try:
        return fisheye_theta(radius_px, focal_px, k)
    except ValueError:
        return math.nan


def camera_fov(cam: Mapping[str, Any], width: int, height: int) -> Dict[str, float]:
    """Half-angles at the four image borders and the resulting FOV (degrees) for one eye.

    A border the calibrated polynomial cannot reach (the checkerboard never
    covered it, so the fit is meaningless there) yields NaN.
    """

    k = (cam["k1"], cam["k2"], cam["k3"], cam["k4"])
    up = _border_angle(cam["cy"], cam["fy"], k)
    down = _border_angle(height - cam["cy"], cam["fy"], k)
    left = _border_angle(cam["cx"], cam["fx"], k)
    right = _border_angle(width - cam["cx"], cam["fx"], k)
    return {
        "up_deg": math.degrees(up),
        "down_deg": math.degrees(down),
        "left_deg": math.degrees(left),
        "right_deg": math.degrees(right),
        "fovy_deg": math.degrees(up + down),
        "fovx_deg": math.degrees(left + right),
        "pinhole_fovy_deg": 2.0 * math.degrees(math.atan(height / (2.0 * cam["fy"]))),
    }


def stereo_fov_summary(cal: Mapping[str, Any]) -> Dict[str, Dict[str, float]]:
    """``{"left": camera_fov(...), "right": camera_fov(...)}`` of a calibration mapping."""

    valid = validate_calibration(cal)
    width, height = valid["image_width"], valid["image_height"]
    return {side: camera_fov(valid[side], width, height) for side in SIDES}


def format_fov_summary(cal: Mapping[str, Any]) -> str:
    """Human-readable summary (used by vr-link-calibrate-cams)."""

    summary = stereo_fov_summary(cal)
    width, height = cal["image_width"], cal["image_height"]

    def deg(value: float) -> str:
        return "  n/a" if math.isnan(value) else f"{value:5.1f}"

    lines = [f"Field of view from the fisheye model ({width}x{height} per eye):"]
    for side in SIDES:
        f = summary[side]
        lines.append(
            f"  {side:5s} fovy {deg(f['fovy_deg'])} deg (up {deg(f['up_deg'])}, down {deg(f['down_deg'])})   "
            f"fovx {deg(f['fovx_deg'])} deg (left {deg(f['left_deg'])}, right {deg(f['right_deg'])})   "
            f"[pinhole-from-fy {f['pinhole_fovy_deg']:.1f}]"
        )
    fovx = (summary["left"]["fovx_deg"], summary["right"]["fovx_deg"])
    if any(math.isnan(v) for v in fovx):
        lines.append(
            "  note: n/a = the calibrated polynomial never reaches that image border; the checkerboard did not "
            "cover it, so the model is invalid there. Capture more views at the image edges."
        )
    elif abs(fovx[0] - fovx[1]) > 10.0:
        lines.append(
            f"  note: horizontal FOV differs by {abs(fovx[0] - fovx[1]):.0f} deg between eyes; the edge values are "
            "extrapolated beyond the checkerboard coverage. Capture more views at the image edges."
        )
    lines.append("  Simulator fovy = the vertical value above (MuJoCo <camera fovy>, hit-bench VR_EYE_CAMERA_INTRINSICS).")
    return "\n".join(lines)


__all__ = ["camera_fov", "fisheye_forward", "fisheye_theta", "format_fov_summary", "stereo_fov_summary"]
