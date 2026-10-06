"""Ideal equidistant-fisheye calibration (Isaac Sim eye cameras) and its config plumbing."""
import math

import numpy as np
import pytest

from vr_link.calibration import ideal
from vr_link.calibration.schema import validate_calibration
from vr_link.pipeline import config

pytest.importorskip("yaml")

# hit_embodiments igris_c/sensors.py eye mounts: LSM22100 EFL 1.30 mm / 3.0 um -> 433.3 px/rad
# at 1280x720, scaled by 480/720 for the 640x480 render.
FOCAL_PX = 1.30 / 0.003 * 480 / 720


def test_fisheye_focal_matches_the_eye_module_derivation():
    assert FOCAL_PX == pytest.approx(288.89, abs=0.01)


def test_fisheye_entry_has_zero_distortion_and_authored_principal_point():
    cal = ideal.ideal_stereo_fisheye_from_focal(640, 480, 288.89, 14.0, 52.152, cx=320.0, cy=240.0)
    assert cal["camera_model"] == "fisheye_equidistant"
    assert (cal["image_width"], cal["image_height"]) == (640, 480)
    for side in ("left", "right"):
        cam = cal[side]
        assert (cam["fx"], cam["fy"], cam["cx"], cam["cy"]) == pytest.approx((288.89, 288.89, 320.0, 240.0))
        assert (cam["k1"], cam["k2"], cam["k3"], cam["k4"]) == (0.0, 0.0, 0.0, 0.0)
    validate_calibration(cal)


def test_fisheye_principal_point_defaults_to_half_size():
    cal = ideal.ideal_stereo_fisheye_from_focal(640, 480, 288.89, 14.0, 52.152)
    assert (cal["left"]["cx"], cal["left"]["cy"]) == (320.0, 240.0)


def test_fisheye_rectification_matches_the_pinhole_rig():
    fisheye = ideal.ideal_stereo_fisheye_from_focal(640, 480, 288.89, 14.0, 52.152)
    pinhole = ideal.ideal_stereo_from_fovy(640, 480, 90.0, 14.0, 52.152)
    for side in ("left", "right"):
        assert fisheye[side]["r"] == pytest.approx(pinhole[side]["r"], abs=1e-12)
    assert np.asarray(fisheye["left"]["r"])[2] == pytest.approx(-math.sin(math.radians(14.0)))


def test_fisheye_equidistant_field_of_view():
    # r = f * theta: the 640 px width spans 2 * (320 / 288.89) rad.
    half_h = 320.0 / 288.89
    half_v = 240.0 / 288.89
    assert math.degrees(2 * half_h) == pytest.approx(126.9, abs=0.1)
    assert math.degrees(2 * half_v) == pytest.approx(95.2, abs=0.1)


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
def test_fisheye_rejects_bad_focal(bad):
    with pytest.raises(ValueError):
        ideal.ideal_stereo_fisheye_from_focal(640, 480, bad, 14.0, 52.152)


def test_packaged_isaac_dds_config_loads():
    path = config.packaged_path("configs/stream_dds_isaac.yaml")
    cfg = config.load_pump_config(path, "dds")
    rig = cfg.calibration.ideal
    assert cfg.calibration.provider == "ideal_from_config"
    assert rig.model == "fisheye_equidistant" and rig.focal_px == pytest.approx(288.89)
    assert (rig.cx, rig.cy, rig.toe_out_deg, rig.baseline_mm) == (320.0, 240.0, 14.0, 52.152)
    assert cfg.signals.source is config.SourceKind.ISAAC and cfg.signals.recording == "none"
    assert cfg.dds.camera_topic == "sensor/eyes_stereo/compressed" and cfg.dds.namespace == ""
    assert cfg.robot_state.dds_fields() == {"actual", "reference", "hand_actual", "hand_command"}


def test_ideal_model_defaults_to_pinhole_and_rejects_unknown(tmp_path):
    path = tmp_path / "s.yaml"
    path.write_text("calibration: {provider: ideal_from_config}\n", encoding="utf-8")
    cfg = config.load_pump_config(path, "dds")
    assert cfg.calibration.ideal.model == "pinhole" and cfg.calibration.ideal.focal_px is None
    path.write_text("calibration: {provider: ideal_from_config, ideal: {model: orthographic}}\n", encoding="utf-8")
    with pytest.raises(config.ConfigError):
        config.load_pump_config(path, "dds")


def test_fisheye_model_requires_focal_px(tmp_path):
    path = tmp_path / "s.yaml"
    path.write_text("calibration: {provider: ideal_from_config, ideal: {model: fisheye_equidistant}}\n", encoding="utf-8")
    with pytest.raises(config.ConfigError):
        config.load_pump_config(path, "dds")
