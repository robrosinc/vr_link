"""``vr-link-stream`` argument parsing and config overrides."""
import pytest

from vr_link.cli.stream import apply_overrides, parse_args
from vr_link.pipeline import config

pytest.importorskip("yaml")


def test_head_rgb_flag_overrides_the_config_and_defaults_to_the_file():
    base = config.load_pump_config(config.packaged_path("configs/stream_dds_isaac.yaml"), "dds")
    assert not base.head_rgb.enable
    assert not apply_overrides(base, parse_args(["--source", "dds"])).head_rgb.enable
    assert apply_overrides(base, parse_args(["--source", "dds", "--head-rgb", "on"])).head_rgb.enable
    assert not apply_overrides(base, parse_args(["--source", "dds", "--head-rgb", "off"])).head_rgb.enable
    with pytest.raises(SystemExit):
        parse_args(["--source", "dds", "--head-rgb", "yes"])


def test_other_overrides_still_apply():
    cfg = config.load_pump_config(config.default_config_path("usb"), "usb")
    args = parse_args(["--source", "usb", "--port", "9200", "--dds-namespace", "igris_c_test", "--encoder", "nvenc"])
    cfg = apply_overrides(cfg, args)
    assert cfg.stream.port == 9200 and cfg.dds.namespace == "igris_c_test" and cfg.encoder.codec == "nvenc"
