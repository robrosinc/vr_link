"""Stream pump configuration (``stream_*.yaml``) for the standalone senders.

The YAML keeps the layout of the legacy ``igris_c_camera`` ``stream_config*.yaml``::

    eyes_stereo: {cam_index, resolution: [w, h], fps, enable_stereo}
    tcp:         {host, port, bitrate, gop, fourcc, encoder}
    calibration: {enable, host, port, provider, yaml, json, image_width, image_height,
                  request_timeout, ideal: {fovy_deg, toe_out_deg, baseline_mm}}
    dds:         {domain_id, namespace, topic, config_uri, frame_id}
    mujoco:      {shm_name, read_timeout_sec}

Paths inside a file resolve against the file's own directory; the ``data:``
prefix addresses this package's bundled data (``data:calibration/x.yaml``).
The location of the file itself: ``--config`` > ``$VR_LINK_STREAM_CONFIG`` >
the packaged default for the source kind.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Union

from .encoder import normalize_encoder_name, parse_bitrate

PathLike = Union[str, Path]
CONFIG_PATH_ENV = "VR_LINK_STREAM_CONFIG"
DATA_PREFIX = "data:"
SOURCE_KINDS = ("usb", "dds", "mujoco")
CALIBRATION_PROVIDERS = ("yaml", "json", "ideal_from_source", "ideal_from_config")


class ConfigError(ValueError):
    """The stream configuration is invalid."""


# --- path helpers ----------------------------------------------------------------


def packaged_data_path(relative: str) -> Path:
    """Filesystem path of a file under ``vr_link/data`` (editable and wheel installs)."""

    return Path(str(resources.files("vr_link.data") / relative))


def resolve_data_path(value: PathLike, *, base_dir: Optional[Path] = None) -> Path:
    """``data:`` prefix -> packaged data; absolute as-is; relative -> ``base_dir``."""

    text = str(value)
    if text.startswith(DATA_PREFIX):
        return packaged_data_path(text[len(DATA_PREFIX):])
    path = Path(text)
    if path.is_absolute() or base_dir is None:
        return path
    return Path(base_dir) / path


def default_config_path(source_kind: str) -> Path:
    if source_kind not in SOURCE_KINDS:
        raise ConfigError(f"unknown source kind {source_kind!r} (expected one of {SOURCE_KINDS})")
    return packaged_data_path(f"configs/stream_{source_kind}.yaml")


def resolve_config_path(cli_arg: Optional[PathLike], source_kind: str, *, env: str = CONFIG_PATH_ENV) -> Path:
    """``--config`` > ``$VR_LINK_STREAM_CONFIG`` > packaged default."""

    if cli_arg:
        return Path(cli_arg)
    env_path = os.environ.get(env)
    if env_path:
        return Path(env_path)
    return default_config_path(source_kind)


# --- dataclasses ------------------------------------------------------------------


@dataclass
class StreamConfig:
    host: str = "0.0.0.0"
    port: int = 9100
    width: int = 1280
    height: int = 480
    fps: float = 30.0


@dataclass
class EncoderConfig:
    codec: str = "libx264"
    bitrate: int = 8_000_000
    gop: int = 30
    fps: float = 30.0


@dataclass
class IdealRigConfig:
    fovy_deg: float = 90.0
    toe_out_deg: float = 14.0
    baseline_mm: float = 52.152


@dataclass
class CalibrationServerConfig:
    enable: bool = True
    host: str = "0.0.0.0"
    port: int = 9101
    provider: str = "yaml"
    yaml_path: Optional[Path] = None
    json_path: Optional[Path] = None
    image_width: int = 0
    image_height: int = 0
    request_timeout: float = 2.0
    ideal: IdealRigConfig = field(default_factory=IdealRigConfig)


@dataclass
class UsbSourceConfig:
    device: str = "/dev/eyes_stereo"
    fourcc: str = "MJPG"
    apply_tuning: bool = True


@dataclass
class DdsSourceConfig:
    domain_id: int = 1
    namespace: str = ""
    topic: str = "igris_c/sensor/eyes_stereo"
    config_uri: str = ""
    frame_id: Optional[str] = None


@dataclass
class MujocoSourceConfig:
    shm_name: str = "/igris_c_mujoco_vision_shm"
    read_timeout_sec: float = 1.0


@dataclass
class PumpConfig:
    source_kind: str
    config_path: Optional[Path]
    stream: StreamConfig
    encoder: EncoderConfig
    calibration: CalibrationServerConfig
    usb: UsbSourceConfig
    dds: DdsSourceConfig
    mujoco: MujocoSourceConfig
    enable_stereo: bool = True


# --- loading -----------------------------------------------------------------------


def _yaml():
    try:
        import yaml  # noqa: WPS433
    except ImportError as error:  # pragma: no cover - environment dependent
        raise ImportError("pyyaml is required to read stream configs: install the 'yaml' extra") from error
    return yaml


def load_config_file(path: PathLike) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        data = _yaml().safe_load(handle)
    if data is None:
        return {}
    if not isinstance(data, Mapping):
        raise ConfigError(f"{path}: top level must be a mapping")
    return dict(data)


def build_pump_config(data: Mapping[str, Any], source_kind: str, *, config_path: Optional[Path] = None) -> PumpConfig:
    """Resolve a parsed YAML mapping into :class:`PumpConfig`."""

    if source_kind not in SOURCE_KINDS:
        raise ConfigError(f"unknown source kind {source_kind!r} (expected one of {SOURCE_KINDS})")
    base_dir = Path(config_path).resolve().parent if config_path else None
    eyes = dict(data.get("eyes_stereo") or {})
    tcp = dict(data.get("tcp") or {})
    cal = dict(data.get("calibration") or {})
    dds = dict(data.get("dds") or {})
    mujoco = dict(data.get("mujoco") or {})
    resolution = eyes.get("resolution", [1280, 480])
    if not isinstance(resolution, (list, tuple)) or len(resolution) != 2:
        raise ConfigError("eyes_stereo.resolution must be [width, height]")
    fps = float(eyes.get("fps", 30))

    stream = StreamConfig(
        host=str(tcp.get("host", "0.0.0.0")),
        port=int(tcp.get("port", 9100)),
        width=int(resolution[0]),
        height=int(resolution[1]),
        fps=fps,
    )
    encoder = EncoderConfig(
        codec=normalize_encoder_name(str(tcp.get("encoder", "sw"))),
        bitrate=parse_bitrate(tcp.get("bitrate", 8_000_000)),
        gop=int(tcp.get("gop", 30)),
        fps=fps,
    )
    provider = str(cal.get("provider", "yaml"))
    if provider not in CALIBRATION_PROVIDERS:
        raise ConfigError(f"calibration.provider must be one of {CALIBRATION_PROVIDERS}, got {provider!r}")
    ideal_raw = dict(cal.get("ideal") or {})
    calibration = CalibrationServerConfig(
        enable=bool(cal.get("enable", True)),
        host=str(cal.get("host", "0.0.0.0")),
        port=int(cal.get("port", 9101)),
        provider=provider,
        yaml_path=resolve_data_path(cal["yaml"], base_dir=base_dir) if cal.get("yaml") else None,
        json_path=resolve_data_path(cal["json"], base_dir=base_dir) if cal.get("json") else None,
        image_width=int(cal.get("image_width", 0)),
        image_height=int(cal.get("image_height", 0)),
        request_timeout=float(cal.get("request_timeout", 2.0)),
        ideal=IdealRigConfig(
            fovy_deg=float(ideal_raw.get("fovy_deg", 90.0)),
            toe_out_deg=float(ideal_raw.get("toe_out_deg", 14.0)),
            baseline_mm=float(ideal_raw.get("baseline_mm", 52.152)),
        ),
    )
    usb = UsbSourceConfig(
        device=str(eyes.get("cam_index", "/dev/eyes_stereo")),
        fourcc=str(tcp.get("fourcc", "MJPG")),
        apply_tuning=bool(eyes.get("apply_tuning", True)),
    )
    config_uri = dds.get("config_uri") or dds.get("config") or ""
    if config_uri and not str(config_uri).startswith("file://"):
        config_uri = f"file://{resolve_data_path(config_uri, base_dir=base_dir).resolve()}"
    dds_cfg = DdsSourceConfig(
        domain_id=int(dds.get("domain_id", 1)),
        namespace=str(dds.get("namespace", "")),
        topic=str(dds.get("topic", "igris_c/sensor/eyes_stereo")),
        config_uri=str(config_uri),
        frame_id=(dds.get("frame_id") or None),
    )
    mujoco_cfg = MujocoSourceConfig(
        shm_name=str(mujoco.get("shm_name", "/igris_c_mujoco_vision_shm")),
        read_timeout_sec=float(mujoco.get("read_timeout_sec", 1.0)),
    )
    return PumpConfig(
        source_kind=source_kind,
        config_path=Path(config_path) if config_path else None,
        stream=stream,
        encoder=encoder,
        calibration=calibration,
        usb=usb,
        dds=dds_cfg,
        mujoco=mujoco_cfg,
        enable_stereo=bool(eyes.get("enable_stereo", True)),
    )


def load_pump_config(path: PathLike, source_kind: str) -> PumpConfig:
    return build_pump_config(load_config_file(path), source_kind, config_path=Path(path))


__all__ = [
    "CALIBRATION_PROVIDERS",
    "CONFIG_PATH_ENV",
    "CalibrationServerConfig",
    "ConfigError",
    "DdsSourceConfig",
    "EncoderConfig",
    "IdealRigConfig",
    "MujocoSourceConfig",
    "PumpConfig",
    "SOURCE_KINDS",
    "StreamConfig",
    "UsbSourceConfig",
    "build_pump_config",
    "default_config_path",
    "load_config_file",
    "load_pump_config",
    "packaged_data_path",
    "resolve_config_path",
    "resolve_data_path",
]
