"""Stream config loading: packaged defaults, path resolution, provider validation."""
import pytest

from vr_link import config

pytest.importorskip("yaml")


@pytest.mark.parametrize("kind", config.SOURCE_KINDS)
def test_packaged_defaults_load(kind):
    path = config.default_config_path(kind)
    assert path.exists(), path
    cfg = config.load_pump_config(path, kind)
    assert cfg.stream.port == 9100 and cfg.calibration.port == 9101
    assert cfg.encoder.codec == "libx264"
    if kind == "mujoco":
        assert cfg.calibration.provider == "ideal_from_source"
        assert cfg.calibration.ideal.fovy_deg == 90.0
    else:
        assert cfg.calibration.provider == "yaml"
        assert cfg.calibration.yaml_path is not None and cfg.calibration.yaml_path.exists()


def test_relative_paths_resolve_against_the_config_file(tmp_path):
    (tmp_path / "cal.yaml").write_text("x: 1\n", encoding="utf-8")
    path = tmp_path / "stream.yaml"
    path.write_text("calibration: {provider: yaml, yaml: cal.yaml}\ntcp: {encoder: nvenc, bitrate: '4M'}\n", encoding="utf-8")
    cfg = config.load_pump_config(path, "usb")
    assert cfg.calibration.yaml_path == tmp_path / "cal.yaml"
    assert cfg.encoder.codec == "h264_nvenc" and cfg.encoder.bitrate == 4_000_000


def test_unknown_provider_and_source_are_rejected(tmp_path):
    path = tmp_path / "stream.yaml"
    path.write_text("calibration: {provider: magic}\n", encoding="utf-8")
    with pytest.raises(config.ConfigError):
        config.load_pump_config(path, "usb")
    with pytest.raises(config.ConfigError):
        config.default_config_path("isaac")


def test_config_path_precedence(monkeypatch, tmp_path):
    monkeypatch.delenv(config.CONFIG_PATH_ENV, raising=False)
    assert config.resolve_config_path(None, "usb") == config.default_config_path("usb")
    monkeypatch.setenv(config.CONFIG_PATH_ENV, str(tmp_path / "env.yaml"))
    assert config.resolve_config_path(None, "usb") == tmp_path / "env.yaml"
    assert config.resolve_config_path(tmp_path / "cli.yaml", "usb") == tmp_path / "cli.yaml"
