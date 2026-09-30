"""``Calibration`` and ``CalibrationRequest`` messages.

The mapping shape is the one :mod:`vr_link.calibration.schema` validates::

    {"image_width": int, "image_height": int,
     "left":  {"fx","fy","cx","cy","k1","k2","k3","k4", "r": [9 floats]},
     "right": {...}}

so file loaders, ideal-model generators and this wire form share one type.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping

import flatbuffers
import numpy as np
import vr_link_generated as _vr_link

from ..calibration.schema import CAMERA_FIELDS, SIDES, validate_calibration
from .envelope import DecodedEnvelope, MessageKind, finish_envelope, read_table

_ADDERS = {
    "fx": _vr_link.CameraModelAddFx,
    "fy": _vr_link.CameraModelAddFy,
    "cx": _vr_link.CameraModelAddCx,
    "cy": _vr_link.CameraModelAddCy,
    "k1": _vr_link.CameraModelAddK1,
    "k2": _vr_link.CameraModelAddK2,
    "k3": _vr_link.CameraModelAddK3,
    "k4": _vr_link.CameraModelAddK4,
}


def _build_camera(builder: flatbuffers.Builder, cam: Mapping[str, Any]) -> int:
    r = builder.CreateNumpyVector(np.asarray(cam["r"], dtype=np.float32))
    _vr_link.CameraModelStart(builder)
    for key in CAMERA_FIELDS:
        _ADDERS[key](builder, float(cam[key]))
    _vr_link.CameraModelAddR(builder, r)
    return _vr_link.CameraModelEnd(builder)


def build_calibration_payload(cal: Mapping[str, Any], *, seq: int = 0, wall_ns: int = 0, sim_ns: int = 0) -> bytes:
    cal = validate_calibration(cal)
    builder = flatbuffers.Builder(512)
    left = _build_camera(builder, cal["left"])
    right = _build_camera(builder, cal["right"])
    _vr_link.CalibrationStart(builder)
    _vr_link.CalibrationAddImageWidth(builder, cal["image_width"])
    _vr_link.CalibrationAddImageHeight(builder, cal["image_height"])
    _vr_link.CalibrationAddLeft(builder, left)
    _vr_link.CalibrationAddRight(builder, right)
    body = _vr_link.CalibrationEnd(builder)
    return finish_envelope(builder, body, _vr_link.Message.Calibration, seq=seq, wall_ns=wall_ns, sim_ns=sim_ns)


def build_calibration_request_payload(*, seq: int = 0, wall_ns: int = 0) -> bytes:
    builder = flatbuffers.Builder(32)
    _vr_link.CalibrationRequestStart(builder)
    body = _vr_link.CalibrationRequestEnd(builder)
    return finish_envelope(builder, body, _vr_link.Message.CalibrationRequest, seq=seq, wall_ns=wall_ns)


def _camera_to_dict(cam) -> Dict[str, Any]:
    if cam is None:
        raise ValueError("Calibration is missing a camera")
    out: Dict[str, Any] = {
        "fx": cam.Fx(), "fy": cam.Fy(), "cx": cam.Cx(), "cy": cam.Cy(),
        "k1": cam.K1(), "k2": cam.K2(), "k3": cam.K3(), "k4": cam.K4(),
    }
    out["r"] = [float(v) for v in cam.RAsNumpy()] if cam.RLength() else []
    return out


def decode_calibration(env: DecodedEnvelope) -> Dict[str, Any]:
    """Decode to the validated mapping shape (raises CalibrationError on bad content)."""

    cal = read_table(env, MessageKind.CALIBRATION, _vr_link.Calibration)
    raw = {"image_width": cal.ImageWidth(), "image_height": cal.ImageHeight()}
    for side, getter in zip(SIDES, (cal.Left, cal.Right)):
        raw[side] = _camera_to_dict(getter())
    return validate_calibration(raw)
