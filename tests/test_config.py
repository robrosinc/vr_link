"""Config loading: packaged defaults, path resolution, 2.0 keys and validation."""
import pytest

from vr_link.messages import SourceKind
from vr_link.pipeline import config

pytest.importorskip("yaml")


@pytest.mark.parametrize("kind", config.SOURCE_KINDS)
def test_packaged_defaults_load(kind):
    path = config.default_config_path(kind)
    assert path.exists(), path
    cfg = config.load_pump_config(path, kind)
    assert cfg.stream.port == 9100 and cfg.calibration.port == 9101
    if kind == "shm":
        assert cfg.calibration.provider == "ideal_from_source" and cfg.calibration.ideal.fovy_deg == 90.0
        assert cfg.robot_state.actual == "shm" and cfg.robot_state.base_pose == "shm" and cfg.signals.recording == "shm"
        assert cfg.signals.source is SourceKind.MUJOCO and cfg.needs_shm()
        assert cfg.robot_state.dds_fields() == {"reference", "hand_command"} and cfg.needs_dds()
    else:
        assert cfg.calibration.provider == "yaml" and cfg.encoder.codec == "libx264"
        assert cfg.calibration.yaml_path == config.local_path("calibration/igris_c_eyes_stereo_unreal.yaml")
        assert cfg.signals.source is SourceKind.ROBOT and cfg.needs_dds()
        assert cfg.tolerances.actual_ms == 5.0 and cfg.tolerances.reference_ms == 15.0
    assert cfg.dds.domain_id == 0


def test_dds_fields_and_topics(tmp_path):
    # Packaged configs leave the namespace empty on purpose; --dds-namespace fills it and
    # DdsChannel.create refuses an empty one.
    path = tmp_path / "s.yaml"
    path.write_text("robot_state: {actual: dds}\ndds: {topics: {lowstate: rt_low}}\n", encoding="utf-8")
    cfg = config.load_pump_config(path, "usb")
    assert cfg.needs_dds() and cfg.dds.namespace == ""
    assert cfg.dds.topics.lowstate == "rt_low" and cfg.dds.topics.refstate == "gmr/refstate"


def test_unknown_source_kind_is_rejected():
    with pytest.raises(config.ConfigError):
        config.normalize_source_kind("mujoco")  # the 1.x alias for shm is gone


def test_shm_section_and_legacy_camera_topic_key(tmp_path):
    path = tmp_path / "s.yaml"
    path.write_text("shm: {shm_name: /x}\ndds: {topic: cam/raw, namespace: n}\n", encoding="utf-8")
    cfg = config.load_pump_config(path, "shm")
    assert cfg.source_kind == "shm" and cfg.shm.shm_name == "/x" and cfg.dds.camera_topic == "cam/raw"


def test_packaged_mujoco_dds_config_loads():
    # MuJoCo over DDS: the camera topic carries no camera metadata, so the ideal rig comes from
    # the file; the HUD label is MuJoCo. Everything else matches stream_dds.yaml.
    path = config.packaged_path("configs/stream_dds_mujoco.yaml")
    cfg = config.load_pump_config(path, "dds")
    assert cfg.calibration.provider == "ideal_from_config" and cfg.calibration.ideal.fovy_deg == 90.0
    assert cfg.signals.source is SourceKind.MUJOCO and cfg.signals.recording == "none"
    assert cfg.dds.camera_topic == "sensor/eyes_stereo/compressed" and cfg.dds.namespace == ""
    assert cfg.robot_state.dds_fields() == {"actual", "reference", "hand_actual", "hand_command"}
    assert cfg.robot_state.base_pose == "none"


def test_relative_paths_resolve_against_the_config_file(tmp_path):
    (tmp_path / "cal.yaml").write_text("x: 1\n", encoding="utf-8")
    path = tmp_path / "stream.yaml"
    path.write_text("calibration: {provider: yaml, yaml: cal.yaml}\ntcp: {encoder: nvenc, bitrate: '4M'}\n", encoding="utf-8")
    cfg = config.load_pump_config(path, "usb")
    assert cfg.calibration.yaml_path == tmp_path / "cal.yaml"
    assert cfg.encoder.codec == "h264_nvenc" and cfg.encoder.bitrate == 4_000_000


@pytest.mark.parametrize("text", [
    "calibration: {provider: magic}\n",
    "robot_state: {reference: shm}\n",
    "robot_state: {base_pose: dds}\n",
    "signals: {source: hologram}\n",
    "signals: {recording: dds}\n",
    "dds: {namespace: n, topic_naming: ros}\n",
])
def test_invalid_values_are_rejected(tmp_path, text):
    path = tmp_path / "stream.yaml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(config.ConfigError):
        config.load_pump_config(path, "usb")


def test_shm_fields_need_a_shm_capable_source(tmp_path):
    path = tmp_path / "stream.yaml"
    path.write_text("robot_state: {actual: shm}\n", encoding="utf-8")
    with pytest.raises(config.ConfigError):
        config.load_pump_config(path, "usb")
    assert config.load_pump_config(path, "shm").needs_shm()
    with pytest.raises(config.ConfigError):
        config.default_config_path("none")  # the 2.0 state-only mode is gone


def test_config_path_precedence(monkeypatch, tmp_path):
    monkeypatch.delenv(config.CONFIG_PATH_ENV, raising=False)
    assert config.resolve_config_path(None, "usb") == config.default_config_path("usb")
    monkeypatch.setenv(config.CONFIG_PATH_ENV, str(tmp_path / "env.yaml"))
    assert config.resolve_config_path(None, "usb") == tmp_path / "env.yaml"
    assert config.resolve_config_path(tmp_path / "cli.yaml", "usb") == tmp_path / "cli.yaml"


def test_path_prefixes_and_local_root(monkeypatch, tmp_path):
    monkeypatch.delenv(config.LOCAL_DIR_ENV, raising=False)
    assert config.local_root().name == "local" and (config.local_root().parent / "pyproject.toml").is_file()
    assert config.resolve_data_path("pkg:configs/stream_usb.yaml") == config.default_config_path("usb")
    assert config.resolve_data_path("data:configs/stream_usb.yaml") == config.default_config_path("usb")  # 1.x spelling
    assert config.resolve_data_path("local:calibration/x.yaml") == config.local_root() / "calibration/x.yaml"
    monkeypatch.setenv(config.LOCAL_DIR_ENV, str(tmp_path / "vrl"))
    assert config.resolve_data_path("local:captures") == tmp_path / "vrl" / "captures"
