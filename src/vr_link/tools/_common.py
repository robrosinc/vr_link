"""Shared helpers of the checkerboard capture and calibration tools (OpenCV)."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

from vr_link.pipeline.config import PathLike, data_path, packaged_path

DEFAULT_TOOLS_CONFIG = "configs/calibration_tools.yaml"


def default_captures_root() -> Path:
    """``data:captures``: where vr-link-capture-checkerboard writes its timestamped folders."""

    return data_path("captures")


def default_calibration_root() -> Path:
    """``data:calibration``: calibrate-cams outputs and the served calibration files."""

    return data_path("calibration")


def load_tools_config(path: Optional[PathLike]) -> Tuple[Dict[str, Any], Path]:
    """``calibration_tools.yaml`` (packaged default when ``path`` is None) -> (mapping, path)."""

    import yaml

    target = Path(path) if path else packaged_path(DEFAULT_TOOLS_CONFIG)
    if not target.exists():
        raise FileNotFoundError(f"Missing config file: {target}")
    data = yaml.safe_load(target.read_text(encoding="utf-8"))
    if not isinstance(data, Mapping):
        raise ValueError(f"{target} must contain a mapping at the top level.")
    return dict(data), target


def camera_api_from_config(api_name) -> int:
    import cv2

    if api_name is None:
        return cv2.CAP_ANY
    name = str(api_name).strip()
    if not name or name.lower() in ("default", "cap_any", "any"):
        return cv2.CAP_ANY
    if name.startswith("cv2."):
        name = name[4:]
    if not hasattr(cv2, name):
        raise ValueError(f"Unknown OpenCV camera API: {api_name}")
    return getattr(cv2, name)


def split_sbs(frame, left_half: str):
    """(left_eye, right_eye) views of an SBS frame; ``left_half`` names the eye in the left half."""

    height, width = frame.shape[:2]
    midpoint = width // 2
    first = frame[:, :midpoint]
    second = frame[:, midpoint:]
    if left_half == "left":
        return first, second
    if left_half == "right":
        return second, first
    raise ValueError("camera.left_half must be 'left' or 'right'.")


def detect_checkerboard(image, pattern_size):
    import cv2

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    found, corners = cv2.findChessboardCorners(gray, pattern_size, flags)
    if found:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    return found, corners


def put_label(image, text, origin, color=(255, 255, 255), scale=0.65):
    import cv2

    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 2
    (text_width, text_height), baseline = cv2.getTextSize(text, font, scale, thickness)
    x, y = origin
    padding = 5
    cv2.rectangle(image, (x - padding, y - text_height - padding), (x + text_width + padding, y + baseline + padding), (0, 0, 0), -1)
    cv2.putText(image, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)


def save_frame(output_dir: Path, frame, count: int, image_format: str) -> Path:
    import cv2

    path = output_dir / f"{count:04d}_sbs.{image_format}"
    if not cv2.imwrite(str(path), frame):
        raise RuntimeError(f"Failed to write image: {path}")
    return path


def write_metadata(output_dir: Path, config: Mapping[str, Any], config_path: Path, saved_count: int, source: str) -> None:
    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "config_path": str(config_path),
        "source": source,
        "saved_count": saved_count,
        "camera": config.get("camera", {}),
        "checkerboard": config.get("checkerboard", {}),
        "capture": config.get("capture", {}),
    }
    (output_dir / "capture_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def find_latest_capture_dir(resources_root: Path) -> Path:
    """Newest ``yymmdd_HHMMSS`` folder under ``resources_root``."""

    if not resources_root.exists():
        raise FileNotFoundError(f"Capture directory does not exist: {resources_root}")
    candidates = []
    for child in resources_root.iterdir():
        if not child.is_dir():
            continue
        try:
            timestamp = datetime.strptime(child.name, "%y%m%d_%H%M%S")
        except ValueError:
            continue
        candidates.append((timestamp, child))
    if not candidates:
        raise FileNotFoundError(f"No yymmdd_HHMMSS capture folders found in {resources_root}")
    candidates.sort(key=lambda item: item[0])
    return candidates[-1][1]


__all__ = [
    "DEFAULT_TOOLS_CONFIG",
    "camera_api_from_config",
    "detect_checkerboard",
    "find_latest_capture_dir",
    "load_tools_config",
    "put_label",
    "save_frame",
    "split_sbs",
    "write_metadata",
]
