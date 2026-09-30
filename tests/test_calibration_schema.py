"""The calibration JSON contract shared with the Unreal receiver."""
import json

import pytest

from vr_link.calibration import schema


def _cam(**overrides):
    cam = {"fx": 240.0, "fy": 240.0, "cx": 319.5, "cy": 239.5, "k1": 0.1, "k2": 0.2, "k3": 0.3, "k4": 0.4,
           "r": [1, 0, 0, 0, 1, 0, 0, 0, 1]}
    cam.update(overrides)
    return cam


def _cal(**overrides):
    cal = {"image_width": 640, "image_height": 480, "left": _cam(), "right": _cam()}
    cal.update(overrides)
    return cal


def test_encode_payload_has_the_legacy_shape_and_no_meta_by_default():
    payload = schema.encode_payload(_cal(camera_model="fisheye"))
    body = json.loads(payload)
    assert set(body) == {"image_width", "image_height", "left", "right"}
    assert set(body["left"]) == set(schema.CAMERA_FIELDS) | {"r"}
    assert len(body["left"]["r"]) == 9
    with_meta = json.loads(schema.encode_payload(_cal(camera_model="fisheye"), include_meta=True))
    assert with_meta["camera_model"] == "fisheye"


def test_decode_roundtrip_normalises_types():
    cal = schema.decode_payload(schema.encode_payload(_cal()))
    assert cal["image_width"] == 640
    assert isinstance(cal["left"]["fx"], float)
    assert cal["right"]["r"] == [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]


@pytest.mark.parametrize(
    "broken",
    [
        _cal(image_width=0),
        {k: v for k, v in _cal().items() if k != "right"},
        _cal(left=_cam(r=[1, 0, 0])),
        _cal(left={k: v for k, v in _cam().items() if k != "k4"}),
        _cal(left=_cam(fx=float("nan"))),
        _cal(left=_cam(cx="wide")),
    ],
)
def test_invalid_calibrations_are_rejected(broken):
    with pytest.raises(schema.CalibrationError):
        schema.validate_calibration(broken)


def test_decode_rejects_garbage():
    with pytest.raises(schema.CalibrationError):
        schema.decode_payload(b"not json")
