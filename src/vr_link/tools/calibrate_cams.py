"""``vr-link-calibrate-cams``: stereo (fisheye or pinhole) calibration from SBS checkerboard shots.

Input: a capture folder of ``*_sbs.png`` written by ``vr-link-capture-checkerboard``
(``--input-dir``, or ``--captures-root`` to take the newest ``yymmdd_HHMMSS`` folder).
Output (``--output-dir``): the full OpenCV result (``stereo_calibration.yaml``,
FileStorage), the flat Unreal YAML served on port 9101
(``stereo_calibration_unreal.yaml``), the rectification maps
(``calibration_maps.npz``) and debug images.

The fisheye path keeps only the rectifying rotations from
``cv2.fisheye.stereoRectify`` (its projection matrices collapse for large
inter-camera rotations) and rebuilds P1/P2/Q around a common focal. Samples
that make the fisheye intrinsic calibration ill-conditioned are dropped by a
leave-one-out retry.
"""

from __future__ import annotations

import argparse
import shutil
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from vr_link.calibration.fov import format_fov_summary
from vr_link.calibration.yaml_io import MEASURED_YAML_COMMENT, dump_unreal_yaml

from ._common import default_calibration_root, default_captures_root, detect_checkerboard, find_latest_capture_dir, load_tools_config, split_sbs


# --- geometry helpers ------------------------------------------------------------


def make_object_points(board_cols: int, board_rows: int, square_size: float) -> np.ndarray:
    object_points = np.zeros((board_rows * board_cols, 3), np.float32)
    grid = np.mgrid[0:board_cols, 0:board_rows].T.reshape(-1, 2)
    object_points[:, :2] = grid * float(square_size)
    return object_points


def roi_intersection(roi1, roi2):
    x1, y1, w1, h1 = [int(value) for value in roi1]
    x2, y2, w2, h2 = [int(value) for value in roi2]
    left = max(x1, x2)
    top = max(y1, y2)
    right = min(x1 + w1, x2 + w2)
    bottom = min(y1 + h1, y2 + h2)
    if right <= left or bottom <= top:
        return (0, 0, 0, 0)
    return (left, top, right - left, bottom - top)


def skew_matrix(vector):
    x, y, z = np.asarray(vector, dtype=np.float64).reshape(3)
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64)


def fundamental_from_rt(camera_matrix_left, camera_matrix_right, rotation, translation):
    essential = skew_matrix(translation) @ rotation
    try:
        fundamental = np.linalg.inv(camera_matrix_right).T @ essential @ np.linalg.inv(camera_matrix_left)
        if abs(fundamental[2, 2]) > 1e-12:
            fundamental = fundamental / fundamental[2, 2]
    except np.linalg.LinAlgError:
        fundamental = np.zeros((3, 3), dtype=np.float64)
    return essential, fundamental


def build_rectified_projection(camera_matrix_left, camera_matrix_right, translation, image_size, focal_scale):
    """P1/P2/Q for a rectified pinhole target around a common focal (see module docstring)."""

    width, height = image_size
    baseline = float(np.linalg.norm(translation))
    focal = 0.5 * (float(camera_matrix_left[0, 0]) + float(camera_matrix_right[0, 0])) * float(focal_scale)
    cx = width / 2.0
    cy = height / 2.0
    proj_left = np.array([[focal, 0.0, cx, 0.0], [0.0, focal, cy, 0.0], [0.0, 0.0, 1.0, 0.0]], dtype=np.float64)
    proj_right = np.array([[focal, 0.0, cx, -focal * baseline], [0.0, focal, cy, 0.0], [0.0, 0.0, 1.0, 0.0]], dtype=np.float64)
    inv_baseline = 1.0 / baseline if baseline > 1e-12 else 0.0
    disparity_to_depth = np.array(
        [[1.0, 0.0, 0.0, -cx], [0.0, 1.0, 0.0, -cy], [0.0, 0.0, 0.0, focal], [0.0, 0.0, inv_baseline, 0.0]],
        dtype=np.float64,
    )
    return proj_left, proj_right, disparity_to_depth


def valid_roi_from_maps(map_x, map_y, image_size):
    width, height = image_size
    valid = (map_x >= 0) & (map_x < width) & (map_y >= 0) & (map_y < height)
    ys, xs = np.where(valid)
    if len(xs) == 0 or len(ys) == 0:
        return (0, 0, 0, 0)
    left = int(xs.min())
    top = int(ys.min())
    return (left, top, int(xs.max()) + 1 - left, int(ys.max()) + 1 - top)


def parse_ill_conditioned_input_index(error) -> Optional[int]:
    match = re.search(r"input array\s+(\d+)", str(error))
    return int(match.group(1)) if match else None


def is_fisheye_init_extrinsics_error(error) -> bool:
    return "InitExtrinsics" in str(error)


class FisheyeCalibrationError(RuntimeError):
    def __init__(self, stage: str, original: Exception) -> None:
        self.stage = stage
        self.original = original
        super().__init__(f"{stage}: {original}")


# --- fisheye ---------------------------------------------------------------------


def make_fisheye_object_points(object_points):
    return np.ascontiguousarray(object_points.reshape(-1, 1, 3), dtype=np.float64)


def make_fisheye_image_points(image_points):
    return np.ascontiguousarray(image_points.reshape(-1, 1, 2), dtype=np.float64)


def fisheye_calibration_settings(check_cond: bool):
    import cv2

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 1e-7)
    flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC | cv2.fisheye.CALIB_FIX_SKEW
    if check_cond:
        flags |= cv2.fisheye.CALIB_CHECK_COND
    return flags, criteria


def run_fisheye_intrinsic_calibration(fisheye_object_points, fisheye_image_points, image_size, calibration_flags, criteria):
    import cv2

    camera_matrix = np.zeros((3, 3), dtype=np.float64)
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)
    rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for _ in fisheye_object_points]
    tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for _ in fisheye_object_points]
    rms, camera_matrix, dist_coeffs, _, _ = cv2.fisheye.calibrate(
        fisheye_object_points, fisheye_image_points, image_size, camera_matrix, dist_coeffs, rvecs, tvecs, calibration_flags, criteria,
    )
    return rms, camera_matrix, dist_coeffs


def can_calibrate_fisheye_intrinsics(object_points, image_points, image_size, check_cond) -> bool:
    calibration_flags, criteria = fisheye_calibration_settings(check_cond)
    run_fisheye_intrinsic_calibration(
        [make_fisheye_object_points(p) for p in object_points],
        [make_fisheye_image_points(p) for p in image_points],
        image_size, calibration_flags, criteria,
    )
    return True


def without_index(values, index):
    return values[:index] + values[index + 1 :]


def find_fisheye_intrinsic_drop_candidate(object_points, image_points, image_size, check_cond, min_valid_pairs) -> Optional[int]:
    import cv2

    if len(object_points) - 1 < min_valid_pairs:
        return None
    for index in range(len(object_points)):
        try:
            can_calibrate_fisheye_intrinsics(without_index(object_points, index), without_index(image_points, index), image_size, check_cond)
            return index
        except cv2.error:
            continue
    return None


def calibrate_pinhole(object_points, image_points_left, image_points_right, image_size, alpha) -> Dict[str, Any]:
    import cv2

    print("Calibrating left camera with pinhole model...")
    rms_left, k_left, d_left, _, _ = cv2.calibrateCamera(object_points, image_points_left, image_size, None, None)
    print("Calibrating right camera with pinhole model...")
    rms_right, k_right, d_right, _, _ = cv2.calibrateCamera(object_points, image_points_right, image_size, None, None)
    print("Running pinhole stereoCalibrate...")
    stereo_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)
    rms_stereo, k_left, d_left, k_right, d_right, rotation, translation, essential, fundamental = cv2.stereoCalibrate(
        object_points, image_points_left, image_points_right, k_left, d_left, k_right, d_right, image_size,
        criteria=stereo_criteria, flags=cv2.CALIB_FIX_INTRINSIC,
    )
    print("Running pinhole stereoRectify...")
    rect_left, rect_right, proj_left, proj_right, q, roi_left, roi_right = cv2.stereoRectify(
        k_left, d_left, k_right, d_right, image_size, rotation, translation, flags=cv2.CALIB_ZERO_DISPARITY, alpha=alpha,
    )
    print("Building pinhole remap tables...")
    map_left = cv2.initUndistortRectifyMap(k_left, d_left, rect_left, proj_left, image_size, cv2.CV_32FC1)
    map_right = cv2.initUndistortRectifyMap(k_right, d_right, rect_right, proj_right, image_size, cv2.CV_32FC1)
    return {
        "rms_left": rms_left, "rms_right": rms_right, "rms_stereo": rms_stereo,
        "K1": k_left, "D1": d_left, "K2": k_right, "D2": d_right, "R": rotation, "T": translation, "E": essential, "F": fundamental,
        "R1": rect_left, "R2": rect_right, "P1": proj_left, "P2": proj_right, "Q": q,
        "roi_left": roi_left, "roi_right": roi_right, "maps": (*map_left, *map_right),
    }


def calibrate_fisheye(object_points, image_points_left, image_points_right, image_size, balance, fov_scale, check_cond) -> Dict[str, Any]:
    import cv2

    if not hasattr(cv2, "fisheye"):
        raise RuntimeError("This OpenCV build does not include cv2.fisheye.")
    fisheye_object_points = [make_fisheye_object_points(p) for p in object_points]
    fisheye_left = [make_fisheye_image_points(p) for p in image_points_left]
    fisheye_right = [make_fisheye_image_points(p) for p in image_points_right]
    point_count = fisheye_object_points[0].shape[0]
    for index, (object_set, left_set, right_set) in enumerate(zip(fisheye_object_points, fisheye_left, fisheye_right), start=1):
        if object_set.shape != (point_count, 1, 3):
            raise ValueError(f"Invalid fisheye object point shape at pair {index}: {object_set.shape}")
        if left_set.shape != (point_count, 1, 2):
            raise ValueError(f"Invalid fisheye left point shape at pair {index}: {left_set.shape}")
        if right_set.shape != (point_count, 1, 2):
            raise ValueError(f"Invalid fisheye right point shape at pair {index}: {right_set.shape}")

    calibration_flags, criteria = fisheye_calibration_settings(check_cond)
    print("Calibrating left camera with fisheye model...")
    try:
        rms_left, k_left, d_left = run_fisheye_intrinsic_calibration(fisheye_object_points, fisheye_left, image_size, calibration_flags, criteria)
    except cv2.error as exc:
        raise FisheyeCalibrationError("left_intrinsics", exc) from exc
    print("Calibrating right camera with fisheye model...")
    try:
        rms_right, k_right, d_right = run_fisheye_intrinsic_calibration(fisheye_object_points, fisheye_right, image_size, calibration_flags, criteria)
    except cv2.error as exc:
        raise FisheyeCalibrationError("right_intrinsics", exc) from exc

    print("Running fisheye stereoCalibrate...")
    rotation = np.eye(3, dtype=np.float64)
    translation = np.zeros((3, 1), dtype=np.float64)
    try:
        # OpenCV >= 4.11 appends rvecs/tvecs; slice the first 7 values to stay compatible.
        stereo_result = cv2.fisheye.stereoCalibrate(
            fisheye_object_points, fisheye_left, fisheye_right, k_left, d_left, k_right, d_right, image_size,
            rotation, translation, flags=cv2.fisheye.CALIB_FIX_INTRINSIC, criteria=criteria,
        )
        rms_stereo, k_left, d_left, k_right, d_right, rotation, translation = stereo_result[:7]
    except cv2.error as exc:
        raise FisheyeCalibrationError("stereo", exc) from exc

    essential, fundamental = fundamental_from_rt(k_left, k_right, rotation, translation)
    print("Running fisheye stereoRectify...")
    rectify_result = cv2.fisheye.stereoRectify(
        k_left, d_left, k_right, d_right, image_size, rotation, translation,
        flags=cv2.CALIB_ZERO_DISPARITY, balance=balance, fov_scale=fov_scale,
    )
    rect_left, rect_right = rectify_result[0], rectify_result[1]
    proj_left, proj_right, q = build_rectified_projection(k_left, k_right, translation, image_size, fov_scale)
    print("Building fisheye remap tables...")
    map_left = cv2.fisheye.initUndistortRectifyMap(k_left, d_left, rect_left, proj_left, image_size, cv2.CV_32FC1)
    map_right = cv2.fisheye.initUndistortRectifyMap(k_right, d_right, rect_right, proj_right, image_size, cv2.CV_32FC1)
    return {
        "rms_left": rms_left, "rms_right": rms_right, "rms_stereo": rms_stereo,
        "K1": k_left, "D1": d_left, "K2": k_right, "D2": d_right, "R": rotation, "T": translation, "E": essential, "F": fundamental,
        "R1": rect_left, "R2": rect_right, "P1": proj_left, "P2": proj_right, "Q": q,
        "roi_left": valid_roi_from_maps(map_left[0], map_left[1], image_size),
        "roi_right": valid_roi_from_maps(map_right[0], map_right[1], image_size),
        "maps": (*map_left, *map_right),
    }


# --- outputs ---------------------------------------------------------------------


def fisheye_coeffs_4(dist_coeffs) -> np.ndarray:
    coeffs = np.asarray(dist_coeffs, dtype=np.float64).reshape(-1)
    if coeffs.size < 4:
        coeffs = np.pad(coeffs, (0, 4 - coeffs.size), mode="constant")
    return coeffs[:4]


def camera_shader_entry(camera_matrix, dist_coeffs, rectification) -> Dict[str, Any]:
    """One eye of the calibration JSON schema (fx..k4 + r) from OpenCV results."""

    coeffs = fisheye_coeffs_4(dist_coeffs)
    return {
        "fx": float(camera_matrix[0, 0]), "fy": float(camera_matrix[1, 1]),
        "cx": float(camera_matrix[0, 2]), "cy": float(camera_matrix[1, 2]),
        "k1": float(coeffs[0]), "k2": float(coeffs[1]), "k3": float(coeffs[2]), "k4": float(coeffs[3]),
        "r": [float(v) for v in np.asarray(rectification, dtype=np.float64).reshape(-1)],
    }


def measured_calibration(result: Dict[str, Any], image_size) -> Dict[str, Any]:
    return {
        "image_width": int(image_size[0]),
        "image_height": int(image_size[1]),
        "left": camera_shader_entry(result["K1"], result["D1"], result["R1"]),
        "right": camera_shader_entry(result["K2"], result["D2"], result["R2"]),
    }


def render_measured_yaml(result, image_size, config, input_dir, camera_model, valid_count) -> str:
    camera_config = config.get("camera", {})
    board_config = config.get("checkerboard", {})
    meta = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source_input_dir": str(input_dir),
        "camera_model": str(camera_model),
        "left_half": str(camera_config.get("left_half", "right")),
        "valid_pair_count": int(valid_count),
        "image_width": int(image_size[0]),
        "image_height": int(image_size[1]),
        "rms_left": float(result["rms_left"]),
        "rms_right": float(result["rms_right"]),
        "rms_stereo": float(result["rms_stereo"]),
        "baseline": float(np.linalg.norm(result["T"])),
        "baseline_unit": str(board_config.get("square_unit", "")),
    }
    return dump_unreal_yaml(measured_calibration(result, image_size), meta=meta, comments=MEASURED_YAML_COMMENT, float_digits=None)


def write_opencv_yaml(output_path: Path, image_size, config, input_dir, camera_model, valid_count, result, common_roi) -> None:
    import cv2

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fs = cv2.FileStorage(str(output_path), cv2.FILE_STORAGE_WRITE)
    if not fs.isOpened():
        raise RuntimeError(f"Failed to open calibration output: {output_path}")
    board_config = config.get("checkerboard", {})
    camera_config = config.get("camera", {})
    calibration_config = config.get("calibration", {})
    fs.write("generated_at", datetime.now().isoformat(timespec="seconds"))
    fs.write("input_dir", str(input_dir))
    fs.write("camera_model", str(camera_model))
    fs.write("valid_pair_count", int(valid_count))
    fs.write("image_width", int(image_size[0]))
    fs.write("image_height", int(image_size[1]))
    fs.write("left_half", str(camera_config.get("left_half", "right")))
    fs.write("board_cols", int(board_config.get("board_cols", 9)))
    fs.write("board_rows", int(board_config.get("board_rows", 6)))
    fs.write("square_size", float(board_config.get("square_size", 25.0)))
    fs.write("square_unit", str(board_config.get("square_unit", "mm")))
    fs.write("alpha", float(calibration_config.get("alpha", 0.0)))
    fs.write("fisheye_balance", float(calibration_config.get("fisheye_balance", 0.0)))
    fs.write("fisheye_fov_scale", float(calibration_config.get("fisheye_fov_scale", 1.0)))
    fs.write("fisheye_check_cond", int(bool(calibration_config.get("fisheye_check_cond", False))))
    fs.write("rms_left", float(result["rms_left"]))
    fs.write("rms_right", float(result["rms_right"]))
    fs.write("rms_stereo", float(result["rms_stereo"]))
    fs.write("baseline", float(np.linalg.norm(result["T"])))
    for key in ("K1", "D1", "K2", "D2", "R", "T", "E", "F", "R1", "R2", "P1", "P2", "Q"):
        fs.write(key, result[key])
    fs.write("validRoi1", np.array(result["roi_left"], dtype=np.int32))
    fs.write("validRoi2", np.array(result["roi_right"], dtype=np.int32))
    fs.write("commonRoi", np.array(common_roi, dtype=np.int32))
    fs.release()


def draw_horizontal_guides(image, step=40) -> None:
    import cv2

    for y in range(step, image.shape[0], step):
        cv2.line(image, (0, y), (image.shape[1], y), (0, 255, 0), 1)


def save_debug_images(debug_base: Path, valid_files, left_half, pattern_size, maps) -> None:
    import cv2

    corners_dir = debug_base / "corners"
    rectified_dir = debug_base / "rectified"
    corners_dir.mkdir(parents=True, exist_ok=True)
    rectified_dir.mkdir(parents=True, exist_ok=True)
    map_left_1, map_left_2, map_right_1, map_right_2 = maps
    for image_path in valid_files:
        frame = cv2.imread(str(image_path))
        if frame is None:
            continue
        left, right = split_sbs(frame, left_half)
        left_found, left_corners = detect_checkerboard(left, pattern_size)
        right_found, right_corners = detect_checkerboard(right, pattern_size)
        left_view = left.copy()
        right_view = right.copy()
        if left_found:
            cv2.drawChessboardCorners(left_view, pattern_size, left_corners, left_found)
        if right_found:
            cv2.drawChessboardCorners(right_view, pattern_size, right_corners, right_found)
        cv2.imwrite(str(corners_dir / f"{image_path.stem}_corners.png"), cv2.hconcat([left_view, right_view]))
        rectified = cv2.hconcat([
            cv2.remap(left, map_left_1, map_left_2, cv2.INTER_LINEAR),
            cv2.remap(right, map_right_1, map_right_2, cv2.INTER_LINEAR),
        ])
        draw_horizontal_guides(rectified)
        cv2.imwrite(str(rectified_dir / f"{image_path.stem}_rectified.png"), rectified)


# --- driver ------------------------------------------------------------------------


def collect_pairs(image_paths: List[Path], left_half: str, pattern_size, object_template):
    import cv2

    object_points, image_points_left, image_points_right, valid_files, failed_files = [], [], [], [], []
    image_size = None
    for image_path in image_paths:
        frame = cv2.imread(str(image_path))
        if frame is None:
            failed_files.append((image_path, "read failed"))
            continue
        left, right = split_sbs(frame, left_half)
        current_size = (left.shape[1], left.shape[0])
        if image_size is None:
            image_size = current_size
        elif current_size != image_size:
            failed_files.append((image_path, f"size mismatch: {current_size}"))
            continue
        left_found, left_corners = detect_checkerboard(left, pattern_size)
        right_found, right_corners = detect_checkerboard(right, pattern_size)
        if left_found and right_found:
            object_points.append(object_template.copy())
            image_points_left.append(left_corners)
            image_points_right.append(right_corners)
            valid_files.append(image_path)
            print(f"OK   {image_path.name}")
        else:
            reason = f"left={'OK' if left_found else 'NO'}, right={'OK' if right_found else 'NO'}"
            failed_files.append((image_path, reason))
            print(f"SKIP {image_path.name}: {reason}")
    return object_points, image_points_left, image_points_right, valid_files, failed_files, image_size


def calibrate_with_retry(camera_model, object_points, image_points_left, image_points_right, valid_files, failed_files, image_size, cal_cfg, min_valid_pairs):
    import cv2

    if camera_model != "fisheye":
        return calibrate_pinhole(object_points, image_points_left, image_points_right, image_size, float(cal_cfg.get("alpha", 0.0)))
    balance = float(cal_cfg.get("fisheye_balance", 0.0))
    fov_scale = float(cal_cfg.get("fisheye_fov_scale", 1.0))
    check_cond = bool(cal_cfg.get("fisheye_check_cond", False))
    while True:
        try:
            return calibrate_fisheye(object_points, image_points_left, image_points_right, image_size, balance, fov_scale, check_cond)
        except (cv2.error, FisheyeCalibrationError) as exc:
            bad_index = parse_ill_conditioned_input_index(exc)
            reason = None
            if bad_index is not None:
                reason = f"fisheye ill-conditioned input array {bad_index}"
            elif isinstance(exc, FisheyeCalibrationError) and exc.stage in ("left_intrinsics", "right_intrinsics") and is_fisheye_init_extrinsics_error(exc):
                print(f"Fisheye intrinsic calibration failed during {exc.stage}; searching for a removable bad sample...")
                source_points = image_points_left if exc.stage == "left_intrinsics" else image_points_right
                bad_index = find_fisheye_intrinsic_drop_candidate(object_points, source_points, image_size, check_cond, min_valid_pairs)
                if bad_index is not None:
                    reason = f"fisheye {exc.stage} InitExtrinsics failure; removed by leave-one-out retry"
            if bad_index is None or bad_index < 0 or bad_index >= len(valid_files):
                raise
            bad_file = valid_files.pop(bad_index)
            object_points.pop(bad_index)
            image_points_left.pop(bad_index)
            image_points_right.pop(bad_index)
            failed_files.append((bad_file, f"{reason}; removed and retried"))
            print(f"DROP {bad_file.name}: {reason}; retrying")
            if len(valid_files) < min_valid_pairs:
                raise RuntimeError(
                    f"Only {len(valid_files)} valid pairs remain after dropping ill-conditioned fisheye samples. Need at least {min_valid_pairs}."
                ) from exc


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stereo calibration from SBS checkerboard captures")
    p.add_argument("--config", default=None, metavar="PATH", help="calibration_tools.yaml (default: packaged)")
    group = p.add_mutually_exclusive_group()
    group.add_argument("--input-dir", type=Path, help="capture folder with *_sbs.png")
    group.add_argument("--captures-root", type=Path, help="use the newest yymmdd_HHMMSS folder under this root (default local:captures)")
    p.add_argument("--output-dir", type=Path, default=None, help="where the YAML/npz/debug outputs go (default local:calibration/<capture folder name>)")
    p.add_argument("--serve-as", default=None, metavar="NAME", help="also copy the Unreal YAML to local:calibration/NAME, the file the stream configs serve (e.g. igris_c_eyes_stereo_unreal.yaml)")
    p.add_argument("--camera-model", choices=("fisheye", "pinhole"), default=None, help="override calibration.camera_model")
    p.add_argument("--no-debug-images", action="store_true", help="skip the corner/rectified debug images")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    config, config_path = load_tools_config(args.config)
    camera_config = config.get("camera", {})
    board_config = config.get("checkerboard", {})
    cal_cfg = config.get("calibration", {})

    left_half = str(camera_config.get("left_half", "right")).lower()
    pattern_size = (int(board_config.get("board_cols", 9)), int(board_config.get("board_rows", 6)))
    square_size = float(board_config.get("square_size", 25.0))
    min_valid_pairs = int(cal_cfg.get("min_valid_pairs", 10))
    camera_model = (args.camera_model or str(cal_cfg.get("camera_model", "fisheye"))).lower()
    if camera_model not in ("pinhole", "fisheye"):
        raise ValueError("calibration.camera_model must be 'pinhole' or 'fisheye'.")

    input_dir = args.input_dir if args.input_dir is not None else find_latest_capture_dir(args.captures_root or default_captures_root())
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    image_pattern = str(cal_cfg.get("image_pattern", "*_sbs.png"))
    image_paths = sorted(input_dir.glob(image_pattern))
    if not image_paths:
        raise FileNotFoundError(f"No images matching {image_pattern} in {input_dir}")

    print(f"Config: {config_path}")
    print(f"Input: {input_dir}")
    print(f"Images: {len(image_paths)}")
    print(f"Camera model: {camera_model}")

    object_template = make_object_points(pattern_size[0], pattern_size[1], square_size)
    object_points, image_points_left, image_points_right, valid_files, failed_files, image_size = collect_pairs(image_paths, left_half, pattern_size, object_template)
    if len(valid_files) < min_valid_pairs:
        raise RuntimeError(f"Only {len(valid_files)} valid pairs found. Need at least {min_valid_pairs}.")
    if image_size is None:
        raise RuntimeError("No readable calibration images found.")

    result = calibrate_with_retry(camera_model, object_points, image_points_left, image_points_right, valid_files, failed_files, image_size, cal_cfg, min_valid_pairs)
    common_roi = roi_intersection(result["roi_left"], result["roi_right"])

    output_dir: Path = args.output_dir if args.output_dir is not None else default_calibration_root() / input_dir.name
    output_dir.mkdir(parents=True, exist_ok=True)
    output_yaml = output_dir / str(cal_cfg.get("output_yaml", "stereo_calibration.yaml"))
    output_unreal_yaml = output_dir / str(cal_cfg.get("output_unreal_yaml", "stereo_calibration_unreal.yaml"))
    output_maps = output_dir / str(cal_cfg.get("output_maps", "calibration_maps.npz"))
    debug_base = output_dir / str(cal_cfg.get("debug_dir", "debug")) / f"{input_dir.name}_{camera_model}"

    write_opencv_yaml(output_yaml, image_size, config, input_dir, camera_model, len(valid_files), result, common_roi)
    output_unreal_yaml.write_text(render_measured_yaml(result, image_size, config, input_dir, camera_model, len(valid_files)), encoding="utf-8")
    map_left_1, map_left_2, map_right_1, map_right_2 = result["maps"]
    np.savez_compressed(
        output_maps,
        map1_left=map_left_1, map2_left=map_left_2, map1_right=map_right_1, map2_right=map_right_2,
        image_size=np.array(image_size, dtype=np.int32),
        valid_roi_left=np.array(result["roi_left"], dtype=np.int32),
        valid_roi_right=np.array(result["roi_right"], dtype=np.int32),
        common_roi=np.array(common_roi, dtype=np.int32),
        camera_model=np.array(camera_model),
    )
    if not args.no_debug_images:
        save_debug_images(debug_base, valid_files, left_half, pattern_size, result["maps"])

    print("")
    print(f"Camera model: {camera_model}")
    print(f"Valid pairs: {len(valid_files)} / {len(image_paths)}")
    print(f"Left RMS:    {result['rms_left']:.6f}")
    print(f"Right RMS:   {result['rms_right']:.6f}")
    print(f"Stereo RMS:  {result['rms_stereo']:.6f}")
    print(f"Baseline:    {float(np.linalg.norm(result['T'])):.6f} {board_config.get('square_unit', '')}")
    if camera_model == "fisheye":
        print(format_fov_summary(measured_calibration(result, image_size)))
    print(f"YAML:        {output_yaml}")
    print(f"Unreal YAML: {output_unreal_yaml}")
    if args.serve_as:
        served = default_calibration_root() / args.serve_as
        served.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(output_unreal_yaml, served)
        print(f"Served as:   {served}")
    print(f"Maps:        {output_maps}")
    if not args.no_debug_images:
        print(f"Debug:       {debug_base}")
    if failed_files:
        print("")
        print("Skipped images:")
        for image_path, reason in failed_files:
            print(f"  {image_path.name}: {reason}")


if __name__ == "__main__":
    main()
