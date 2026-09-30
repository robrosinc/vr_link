"""Stereo calibration JSON contract served on port 9101 and in VSTR codec=2.

The receiver (vr_system ``ASettingPawn`` / ``ASimTeleopPawn``) parses::

    {"image_width": 640, "image_height": 480,
     "left":  {"fx","fy","cx","cy","k1","k2","k3","k4", "r": [9 floats, row-major]},
     "right": {...}}

* ``fx fy cx cy``: per-eye pinhole intrinsics in pixels of the per-eye image.
* ``k1..k4``: OpenCV fisheye coefficients (theta_d = theta * (1 + k1 theta^2 +
  k2 theta^4 + k3 theta^6 + k4 theta^8)). The receiver's screen shader always
  applies this model, so an ideal pinhole camera must be sent as the Taylor
  series of tan(theta) (:data:`TAN_SERIES_K`), NOT as zeros (zeros mean an
  equidistant projection there).
* ``r``: OpenCV ``stereoRectify`` R1/R2, rotating the eye's optical frame into
  the common mid frame.
"""

from __future__ import annotations

import json
import math
from typing import Any, Dict, Mapping

# Taylor series of tan(theta) = theta * (1 + t^2/3 + 2 t^4/15 + 17 t^6/315 + 62 t^8/2835 + ...):
# feeding these as fisheye k makes the receiver reproduce a pinhole projection.
TAN_SERIES_K = (1.0 / 3.0, 2.0 / 15.0, 17.0 / 315.0, 62.0 / 2835.0)

CAMERA_FIELDS = ("fx", "fy", "cx", "cy", "k1", "k2", "k3", "k4")
# Row-major rotation matrix element order expected in the "r" array.
R_KEYS = ("r00", "r01", "r02", "r10", "r11", "r12", "r20", "r21", "r22")
SIDES = ("left", "right")

# Calibration request protocol (port 9101): the client sends this token
# followed by a newline and reads the JSON reply until EOF.
CALIBRATION_REQUEST_TOKEN = b"GET_CALIBRATION"
CALIBRATION_REQUEST_MAX_BYTES = 256

# Metadata keys accepted on input and emitted only with include_meta=True.
META_KEYS = ("camera_model", "baseline_mm", "left_half", "fovy_deg", "toe_out_deg")


class CalibrationError(ValueError):
    """The calibration mapping does not satisfy the receiver contract."""


def _finite(value: Any, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise CalibrationError(f"{name} must be a number, got {value!r}") from error
    if not math.isfinite(number):
        raise CalibrationError(f"{name} must be finite, got {number}")
    return number


def _camera(side: str, raw: Any) -> Dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise CalibrationError(f"{side} must be a mapping, got {type(raw).__name__}")
    cam: Dict[str, Any] = {}
    for field in CAMERA_FIELDS:
        if field not in raw:
            raise CalibrationError(f"{side} is missing '{field}'")
        cam[field] = _finite(raw[field], f"{side}.{field}")
    rotation = raw.get("r")
    if not isinstance(rotation, (list, tuple)) or len(rotation) != len(R_KEYS):
        raise CalibrationError(f"{side}.r must be a 9-element array")
    cam["r"] = [_finite(v, f"{side}.r[{i}]") for i, v in enumerate(rotation)]
    return cam


def validate_calibration(cal: Mapping[str, Any]) -> Dict[str, Any]:
    """Return a normalised copy of ``cal`` or raise :class:`CalibrationError`."""

    if not isinstance(cal, Mapping):
        raise CalibrationError(f"calibration must be a mapping, got {type(cal).__name__}")
    out: Dict[str, Any] = {}
    for key in ("image_width", "image_height"):
        if key not in cal:
            raise CalibrationError(f"missing '{key}'")
        try:
            value = int(cal[key])
        except (TypeError, ValueError) as error:
            raise CalibrationError(f"{key} must be an integer, got {cal[key]!r}") from error
        if value <= 0:
            raise CalibrationError(f"{key} must be positive, got {value}")
        out[key] = value
    for side in SIDES:
        if side not in cal:
            raise CalibrationError(f"missing '{side}'")
        out[side] = _camera(side, cal[side])
    for key in META_KEYS:
        if key in cal:
            out[key] = cal[key]
    return out


def strip_meta(cal: Mapping[str, Any]) -> Dict[str, Any]:
    """The wire subset of a calibration mapping (no metadata keys)."""

    return {key: cal[key] for key in ("image_width", "image_height", *SIDES)}


def encode_payload(cal: Mapping[str, Any], *, include_meta: bool = False) -> bytes:
    """Validate and serialise to the JSON bytes served on 9101 / codec=2."""

    valid = validate_calibration(cal)
    body = valid if include_meta else strip_meta(valid)
    return json.dumps(body, sort_keys=True).encode("utf-8")


def decode_payload(data: bytes) -> Dict[str, Any]:
    """Parse and validate a JSON payload."""

    try:
        raw = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CalibrationError(f"invalid calibration JSON: {error}") from error
    return validate_calibration(raw)


__all__ = [
    "CALIBRATION_REQUEST_MAX_BYTES",
    "CALIBRATION_REQUEST_TOKEN",
    "CAMERA_FIELDS",
    "META_KEYS",
    "R_KEYS",
    "SIDES",
    "TAN_SERIES_K",
    "CalibrationError",
    "decode_payload",
    "encode_payload",
    "strip_meta",
    "validate_calibration",
]
