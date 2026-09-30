"""Ideal calibration math against the legacy make_mujoco_calibration golden values."""
import math

import numpy as np
import pytest

from vr_link.calibration import ideal
from vr_link.calibration.schema import TAN_SERIES_K
from vr_link.sources import StereoCameraMetadata

# Values of igris_c_camera/application/stream_tcp/mujoco_stereo_calibration.yaml
# (fovy 90, 640x480, 14 deg toe-out, 52.152 mm baseline).
GOLDEN = {
    "fx": 240.0, "fy": 240.0, "cx": 319.5, "cy": 239.5,
    "k": (0.333333333333, 0.133333333333, 0.053968253968, 0.021869488536),
    "left_r": (0.970295726276, 0.0, -0.2419218956, 0.0, 1.0, 0.0, 0.2419218956, 0.0, 0.970295726276),
    "right_r": (0.970295726276, 0.0, 0.2419218956, 0.0, 1.0, 0.0, -0.2419218956, 0.0, 0.970295726276),
}

# igris_c_v2_vision.xml.xacro camera quaternions: q_yaw(+/-14 deg about body Z) * base(0.5, 0.5, -0.5, -0.5).
MJCF_LEFT_QUAT = (0.55720775, 0.55720775, -0.43533840, -0.43533840)
MJCF_RIGHT_QUAT = (0.43533840, 0.43533840, -0.55720775, -0.55720775)


def test_from_fovy_matches_golden():
    cal = ideal.ideal_stereo_from_fovy(640, 480, 90.0, 14.0, 52.152)
    for side, r_key in (("left", "left_r"), ("right", "right_r")):
        cam = cal[side]
        assert (cam["fx"], cam["fy"], cam["cx"], cam["cy"]) == pytest.approx((GOLDEN["fx"], GOLDEN["fy"], GOLDEN["cx"], GOLDEN["cy"]))
        assert (cam["k1"], cam["k2"], cam["k3"], cam["k4"]) == pytest.approx(GOLDEN["k"], abs=1e-12)
        assert cam["r"] == pytest.approx(GOLDEN[r_key], abs=1e-12)
    assert cal["image_width"] == 640 and cal["image_height"] == 480
    assert cal["baseline_mm"] == pytest.approx(52.152)


def test_from_intrinsics_is_relative_to_the_midpoint_frame():
    # Ported from hit-bench: identity vs 180 deg about Z -> each eye is +/-90 deg from the mid frame.
    k_left = np.array([[500.0, 0.0, 320.0], [0.0, 510.0, 240.0], [0.0, 0.0, 1.0]])
    r_identity = np.eye(3)
    r_z180 = ideal.quat_wxyz_to_matrix((0.0, 0.0, 0.0, 1.0))
    cal = ideal.ideal_stereo_from_intrinsics(640, 480, k_left, k_left, r_identity, r_z180)
    assert cal["left"]["fx"] == 500.0 and cal["right"]["fy"] == 510.0
    assert cal["left"]["k1"] == pytest.approx(TAN_SERIES_K[0])
    assert np.asarray(cal["left"]["r"]).reshape(3, 3) == pytest.approx(np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]]), abs=1e-9)
    assert np.asarray(cal["right"]["r"]).reshape(3, 3) == pytest.approx(np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]), abs=1e-9)


def test_mujoco_metadata_reproduces_the_fovy_model():
    meta = StereoCameraMetadata(
        eye_width=640, eye_height=480, left_fovy_deg=90.0, right_fovy_deg=90.0,
        left_pos=(0.095516, 0.026076, 0.121), left_quat_wxyz=MJCF_LEFT_QUAT,
        right_pos=(0.095516, -0.026076, 0.121), right_quat_wxyz=MJCF_RIGHT_QUAT,
        baseline_m=2 * 0.026076,
    )
    from_meta = ideal.ideal_stereo_from_metadata(meta)
    from_fovy = ideal.ideal_stereo_from_fovy(640, 480, 90.0, 14.0, 52.152)
    for side in ("left", "right"):
        for key in ("fx", "fy", "cx", "cy", "k1", "k2", "k3", "k4"):
            assert from_meta[side][key] == pytest.approx(from_fovy[side][key], abs=1e-9)
        assert from_meta[side]["r"] == pytest.approx(from_fovy[side]["r"], abs=1e-7)
    assert from_meta["baseline_mm"] == pytest.approx(52.152)


def test_quaternion_matrix_roundtrip():
    rng = np.random.default_rng(1)
    for _ in range(20):
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        back = ideal.matrix_to_quat_wxyz(ideal.quat_wxyz_to_matrix(q))
        assert min(np.linalg.norm(back - q), np.linalg.norm(back + q)) < 1e-9


def test_intrinsics_from_fovy_conventions():
    k = ideal.intrinsics_from_fovy(640, 480, 90.0)
    assert k[0, 0] == pytest.approx(240.0) and k[0, 2] == 319.5
    assert ideal.intrinsics_from_fovy(640, 480, 90.0, principal="half")[0, 2] == 320.0
    with pytest.raises(ValueError):
        ideal.intrinsics_from_fovy(640, 480, 180.0)


def test_rotation_y_sign_convention():
    assert ideal.rotation_y(-14.0)[0, 2] == pytest.approx(-math.sin(math.radians(14.0)))
