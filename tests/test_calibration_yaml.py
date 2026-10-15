"""Flat calibration YAML: packaged data files load, and dump/load round-trips."""
import pytest

from vr_link.calibration import ideal, schema, yaml_io
from vr_link.cli.make_calibration import render_ideal_yaml
from vr_link.pipeline.config import data_path

pytest.importorskip("yaml")


# Measured/generated calibration files are machine-local data (data/, git-ignored);
# the tests that read them skip when this machine does not have them.
def _local_data(name: str):
    path = data_path(f"calibration/{name}")
    if not path.is_file():
        pytest.skip(f"local calibration data {name} not present in this checkout")
    return path


def test_local_measured_calibration_loads():
    cal = yaml_io.load_unreal_yaml(_local_data("igris_c_eyes_stereo_unreal.yaml"))
    assert (cal["image_width"], cal["image_height"]) in ((640, 360), (640, 480))
    assert cal["camera_model"] == "fisheye"
    assert cal["left"]["r"][2] < 0 < cal["right"]["r"][2]  # toe-out signs


def test_local_mujoco_calibration_is_consistent_with_its_own_rig_parameters():
    """The local ideal YAML must equal what the generator makes from its header (fovy/toe-out/baseline)."""

    path = _local_data("mujoco_stereo_calibration.yaml")
    loaded = yaml_io.load_unreal_yaml(path)
    regenerated = ideal.ideal_stereo_from_fovy(
        loaded["image_width"], loaded["image_height"], float(loaded["fovy_deg"]), float(loaded["toe_out_deg"]), loaded["baseline_mm"],
    )
    assert ideal.flatten_sides(loaded) == pytest.approx(ideal.flatten_sides(regenerated), abs=1e-9), (
        "mujoco_stereo_calibration.yaml does not match its fovy/toe-out header; regenerate it with vr-link-make-calibration"
    )


def test_generator_output_round_trips(tmp_path):
    cal = ideal.ideal_stereo_from_fovy(640, 480, 94.35, 14.13, 64.22)
    path = tmp_path / "ideal.yaml"
    path.write_text(render_ideal_yaml(cal, generated_at="2026-09-29T00:00:00"), encoding="utf-8")
    loaded = yaml_io.load_unreal_yaml(path)
    assert loaded["fovy_deg"] == pytest.approx(94.35) and loaded["baseline_mm"] == pytest.approx(64.22)
    assert ideal.flatten_sides(loaded) == pytest.approx(ideal.flatten_sides(cal), abs=1e-9)


def test_dump_load_roundtrip(tmp_path):
    cal = ideal.ideal_stereo_from_fovy(1280, 720, 80.0, 10.0, 60.0)
    path = yaml_io.write_unreal_yaml(tmp_path / "cal.yaml", cal)
    loaded = yaml_io.load_unreal_yaml(path)
    assert (loaded["image_width"], loaded["image_height"]) == (1280, 720)
    assert ideal.flatten_sides(loaded) == pytest.approx(ideal.flatten_sides(schema.validate_calibration(cal)), abs=1e-11)
    assert loaded["baseline_mm"] == pytest.approx(60.0)


def test_size_override_and_missing_keys(tmp_path):
    cal = ideal.ideal_stereo_from_fovy(640, 480, 90.0, 14.0, 52.152)
    path = yaml_io.write_unreal_yaml(tmp_path / "cal.yaml", cal)
    assert yaml_io.load_unreal_yaml(path, image_width=1280, image_height=960)["image_width"] == 1280
    (tmp_path / "broken.yaml").write_text("image_width: 640\nimage_height: 480\nleft_fx: 1.0\n", encoding="utf-8")
    with pytest.raises(schema.CalibrationError):
        yaml_io.load_unreal_yaml(tmp_path / "broken.yaml")


def test_json_file_roundtrip(tmp_path):
    cal = ideal.ideal_stereo_from_fovy(640, 480, 90.0, 14.0, 52.152)
    path = tmp_path / "cal.json"
    path.write_bytes(schema.encode_payload(cal))
    assert yaml_io.calibration_from_json(path, image_width=320)["image_width"] == 320
