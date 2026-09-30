"""Field of view derived from a calibration: the number the simulator's fovy must match."""
import math
from importlib import resources

import pytest

from vr_link.calibration import fov, ideal

pytest.importorskip("yaml")
from vr_link.calibration.yaml_io import load_unreal_yaml  # noqa: E402


def test_ideal_pinhole_model_recovers_its_own_fovy():
    cal = ideal.ideal_stereo_from_fovy(640, 480, 90.0, 14.0, 52.152)
    summary = fov.stereo_fov_summary(cal)
    for side in ("left", "right"):
        # tan-series k: theta_d ~= tan(theta) (9th-order Taylor), so the inversion returns the
        # pinhole angle; the truncation error grows with the angle (0.24 deg at 53 deg).
        assert summary[side]["fovy_deg"] == pytest.approx(90.0, abs=0.2)
        assert summary[side]["fovx_deg"] == pytest.approx(2 * math.degrees(math.atan(320 / 240)), abs=0.5)


def test_local_measured_calibration_is_a_93_to_95_degree_fisheye():
    path = resources.files("vr_link.data") / "calibration" / "igris_c_eyes_stereo_unreal.yaml"
    if not path.is_file():
        pytest.skip("local measured calibration not present in this checkout")
    cal = load_unreal_yaml(path)
    summary = fov.stereo_fov_summary(cal)
    # Both measurements of the eyes_stereo module so far (16:9 and 4:3 modes) put the
    # vertical FOV between 92.5 and 95 degrees with the two eyes within 0.5 deg.
    for side in ("left", "right"):
        assert 92.5 <= summary[side]["fovy_deg"] <= 95.0
    assert abs(summary["left"]["fovy_deg"] - summary["right"]["fovy_deg"]) < 0.5
    # The naive pinhole reading of fy is far off and must not be used as fovy.
    assert summary["left"]["pinhole_fovy_deg"] < 82.0
    text = fov.format_fov_summary(cal)
    assert "fovy" in text and "Simulator fovy" in text


def test_summary_flags_borders_the_polynomial_cannot_reach():
    # k of the first (16:9) measurement's left eye: the fit never reaches the horizontal border.
    cam = {"fx": 220.1, "fy": 220.3, "cx": 321.7, "cy": 183.4,
           "k1": 0.04727, "k2": -0.08029, "k3": 0.05202, "k4": -0.01304, "r": [1, 0, 0, 0, 1, 0, 0, 0, 1]}
    cal = {"image_width": 640, "image_height": 360, "left": cam, "right": dict(cam)}
    summary = fov.stereo_fov_summary(cal)
    assert summary["left"]["fovy_deg"] == pytest.approx(92.8, abs=0.3)
    assert math.isnan(summary["left"]["fovx_deg"])
    text = fov.format_fov_summary(cal)
    assert "n/a" in text and "note:" in text


def test_fisheye_theta_inverts_the_forward_model():
    k = (0.05, -0.08, 0.05, -0.01)
    for theta in (0.1, 0.5, 0.9, 1.2):
        t2 = theta * theta
        radius = 220.0 * theta * (1 + k[0] * t2 + k[1] * t2**2 + k[2] * t2**3 + k[3] * t2**4)
        assert fov.fisheye_theta(radius, 220.0, k) == pytest.approx(theta, abs=1e-9)
