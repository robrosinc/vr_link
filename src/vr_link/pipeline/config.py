"""Sender configuration (``stream_*.yaml``).

The YAML keeps the layout of the legacy ``igris_c_camera`` ``stream_config*.yaml``
and adds the robot-state sources::

    eyes_stereo: {cam_index, resolution: [w, h], fps, enable_stereo, apply_tuning, left_half}
    tcp:         {host, port, bitrate, gop, fourcc, encoder}
    head_rgb:    {enable, camera_topic, frame_id, resolution: [w, h], fps, bitrate, gop}
    calibration: {enable, host, port, provider, yaml, json, image_width, image_height,
                  request_timeout,
                  ideal: {model: pinhole|fisheye_equidistant, fovy_deg, focal_px, cx, cy,
                          toe_out_deg, baseline_mm}}
    dds:         {domain_id, namespace, topic_naming, config_uri, camera_topic, frame_id,
                  topics: {lowstate, handstate, handcmd, refstate, recording_state, motion_feedback},
                  buffer_capacity}
    shm:         {shm_name, read_timeout_sec}
    robot_state: {actual, reference, hand_actual, hand_command, base_pose,
                  tolerances_ms: {actual, reference, hand_actual, hand_command}}
    signals:     {source, recording: shm|dds|none, motion_feedback: dds|none}

Paths inside a file resolve against the file's own directory. Two prefixes
address well-known roots instead: ``pkg:`` is this package
(``pkg:configs/stream_usb.yaml``, shipped in the wheel) and ``data:`` is the
machine-local, git-ignored data root (``data:calibration/x.yaml``: measured
calibrations, checkerboard captures, calibration outputs). The data root is
``$VR_LINK_DATA_DIR``, else ``<checkout>/data`` when the package runs from a
source checkout (editable install, including a submodule checkout of a host
repository), else ``~/.vr_link``.
The location of the config file itself: ``--config`` > ``$VR_LINK_STREAM_CONFIG``
> the packaged default for the source kind.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Union

from ..messages.signals import SourceKind
from ..network.dds.subscribers import DdsStateTopics
from ..sources.usb import LEFT_HALF_VALUES
from ..video.encoder import normalize_encoder_name, parse_bitrate
from .robot_state import MatchTolerances, RobotStateSpec, SignalsSpec

PathLike = Union[str, Path]

CONFIG_PATH_ENV = "VR_LINK_STREAM_CONFIG"
DATA_DIR_ENV = "VR_LINK_DATA_DIR"
PKG_PREFIX = "pkg:"
# Machine-local data root (2.1: renamed from "local:"; the 1.x "data:" alias of
# "pkg:" is gone, no consumer used it).
DATA_PREFIX = "data:"
SOURCE_KINDS = ("usb", "dds", "shm")
CALIBRATION_PROVIDERS = ("yaml", "json", "ideal_from_source", "ideal_from_config")
# "pinhole": the simulator renders a pinhole; serve fx from fovy with the tan-series k.
# "fisheye_equidistant": the simulator renders the OpenCV fisheye model itself (Isaac Sim);
# serve focal_px with k = 0.
IDEAL_MODELS = ("pinhole", "fisheye_equidistant")
DEFAULT_CONFIG_FILES = {"usb": "stream_usb.yaml", "dds": "stream_dds.yaml", "shm": "stream_shm.yaml"}


class ConfigError(ValueError):
    """The stream configuration is invalid."""


# --- path helpers ----------------------------------------------------------------


def packaged_path(relative: str) -> Path:
    """Filesystem path of a file inside the ``vr_link`` package (editable and wheel installs)."""

    return Path(str(resources.files("vr_link") / relative))


def data_root() -> Path:
    """Root of machine-local data: ``$VR_LINK_DATA_DIR`` > ``<checkout>/data`` > ``~/.vr_link``.

    The checkout is detected by walking up from the package directory to a
    ``pyproject.toml`` (an editable install, which is also how host repositories
    install their vr_link submodule); a wheel install has none.
    """

    env = os.environ.get(DATA_DIR_ENV)
    if env:
        return Path(env).expanduser()
    package_dir = Path(__file__).resolve().parents[1]  # src/vr_link
    for parent in package_dir.parents:
        if (parent / "pyproject.toml").is_file():
            return parent / "data"
    return Path.home() / ".vr_link"


def data_path(relative: str = "") -> Path:
    return data_root() / relative if relative else data_root()


def resolve_data_path(value: PathLike, *, base_dir: Optional[Path] = None) -> Path:
    """``pkg:``/``data:`` prefixes -> their roots; absolute as-is; relative -> ``base_dir``."""

    text = str(value)
    if text.startswith(PKG_PREFIX):
        return packaged_path(text[len(PKG_PREFIX):])
    if text.startswith(DATA_PREFIX):
        return data_path(text[len(DATA_PREFIX):])
    path = Path(text).expanduser()
    if path.is_absolute() or base_dir is None:
        return path
    return Path(base_dir) / path


def normalize_source_kind(source_kind: str) -> str:
    kind = str(source_kind)
    if kind not in SOURCE_KINDS:
        raise ConfigError(f"unknown source kind {source_kind!r} (expected one of {SOURCE_KINDS})")
    return kind


def default_config_path(source_kind: str) -> Path:
    return packaged_path(f"configs/{DEFAULT_CONFIG_FILES[normalize_source_kind(source_kind)]}")


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
    model: str = "pinhole"
    fovy_deg: float = 90.0
    toe_out_deg: float = 14.0
    baseline_mm: float = 52.152
    # fisheye_equidistant only: px/rad focal and the authored principal point (None = w/2, h/2).
    focal_px: Optional[float] = None
    cx: Optional[float] = None
    cy: Optional[float] = None


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
    # The eye the camera puts in the LEFT half of its SBS frame. The stream carries the
    # left eye in the left half (what igris_c_camera publishes over DDS), so "right"
    # (the eyes_stereo hardware) makes the source swap the halves.
    left_half: str = "right"


@dataclass
class DdsConfig:
    """One CycloneDDS participant for the camera topic and the robot-state topics."""

    domain_id: int = 0
    namespace: str = ""
    topic_naming: str = "native"
    config_uri: str = ""
    camera_topic: str = "sensor/eyes_stereo/compressed"
    frame_id: Optional[str] = None
    topics: DdsStateTopics = field(default_factory=DdsStateTopics)
    buffer_capacity: int = 512


@dataclass
class HeadRgbConfig:
    """Second video stream (``VideoStream.HEAD_RGB``) from the head colour camera's DDS topic.

    Off by default: a headset build without stream demultiplexing would feed
    these frames to its stereo decoder. The encoder codec is ``tcp.encoder``.
    """

    enable: bool = False
    camera_topic: str = "sensor/d435_color/compressed"
    frame_id: Optional[str] = None
    width: int = 640
    height: int = 480
    fps: float = 15.0
    bitrate: int = 2_000_000
    gop: int = 15


@dataclass
class ShmSourceConfig:
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
    dds: DdsConfig
    shm: ShmSourceConfig
    robot_state: RobotStateSpec
    tolerances: MatchTolerances
    signals: SignalsSpec
    enable_stereo: bool = True
    head_rgb: HeadRgbConfig = field(default_factory=HeadRgbConfig)

    def needs_dds(self) -> bool:
        return (
            self.source_kind == "dds"
            or self.head_rgb.enable
            or bool(self.robot_state.dds_fields())
            or self.signals.recording == "dds"
            or self.signals.motion_feedback == "dds"
        )

    def needs_shm(self) -> bool:
        return self.source_kind == "shm" or self.robot_state.uses_shm() or self.signals.recording == "shm"


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


def _section(data: Mapping[str, Any], *names: str) -> Dict[str, Any]:
    for name in names:
        if data.get(name):
            return dict(data[name])
    return {}


def build_pump_config(data: Mapping[str, Any], source_kind: str, *, config_path: Optional[Path] = None) -> PumpConfig:
    """Resolve a parsed YAML mapping into :class:`PumpConfig` (raises ConfigError)."""

    source_kind = normalize_source_kind(source_kind)
    base_dir = Path(config_path).resolve().parent if config_path else None
    eyes = _section(data, "eyes_stereo")
    tcp = _section(data, "tcp")
    cal = _section(data, "calibration")
    dds = _section(data, "dds")
    shm = _section(data, "shm")
    head = _section(data, "head_rgb")
    robot = _section(data, "robot_state")
    signals_raw = _section(data, "signals")

    resolution = eyes.get("resolution", [1280, 480])
    if not isinstance(resolution, (list, tuple)) or len(resolution) != 2:
        raise ConfigError("eyes_stereo.resolution must be [width, height]")
    fps = float(eyes.get("fps", 30))
    stream = StreamConfig(
        host=str(tcp.get("host", "0.0.0.0")), port=int(tcp.get("port", 9100)),
        width=int(resolution[0]), height=int(resolution[1]), fps=fps,
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
    ideal_model = str(ideal_raw.get("model", "pinhole"))
    if ideal_model not in IDEAL_MODELS:
        raise ConfigError(f"calibration.ideal.model must be one of {IDEAL_MODELS}, got {ideal_model!r}")
    focal_px = ideal_raw.get("focal_px")
    if ideal_model == "fisheye_equidistant" and focal_px is None:
        raise ConfigError("calibration.ideal.model fisheye_equidistant requires focal_px")
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
            model=ideal_model,
            fovy_deg=float(ideal_raw.get("fovy_deg", 90.0)),
            toe_out_deg=float(ideal_raw.get("toe_out_deg", 14.0)),
            baseline_mm=float(ideal_raw.get("baseline_mm", 52.152)),
            focal_px=None if focal_px is None else float(focal_px),
            cx=None if ideal_raw.get("cx") is None else float(ideal_raw["cx"]),
            cy=None if ideal_raw.get("cy") is None else float(ideal_raw["cy"]),
        ),
    )

    left_half = str(eyes.get("left_half", "right")).lower()
    if left_half not in LEFT_HALF_VALUES:
        raise ConfigError(f"eyes_stereo.left_half must be left or right, got {left_half!r}")
    usb = UsbSourceConfig(
        device=str(eyes.get("cam_index", "/dev/eyes_stereo")),
        fourcc=str(tcp.get("fourcc", "MJPG")),
        apply_tuning=bool(eyes.get("apply_tuning", True)),
        left_half=left_half,
    )

    config_uri = dds.get("config_uri") or dds.get("config") or ""
    if config_uri and not str(config_uri).startswith("file://"):
        config_uri = f"file://{resolve_data_path(config_uri, base_dir=base_dir).resolve()}"
    topics_raw = dict(dds.get("topics") or {})
    topic_naming = str(dds.get("topic_naming", "native"))
    if topic_naming not in ("native", "ros_compatible"):
        raise ConfigError(f"dds.topic_naming must be native or ros_compatible, got {topic_naming!r}")
    dds_cfg = DdsConfig(
        domain_id=int(dds.get("domain_id", 0)),
        namespace=str(dds.get("namespace") or ""),
        topic_naming=topic_naming,
        config_uri=str(config_uri),
        camera_topic=str(dds.get("camera_topic") or dds.get("topic") or DdsConfig.camera_topic),
        frame_id=(dds.get("frame_id") or None),
        topics=DdsStateTopics(
            lowstate=str(topics_raw.get("lowstate", "lowstate")),
            handstate=str(topics_raw.get("handstate", "handstate")),
            handcmd=str(topics_raw.get("handcmd", "handcmd")),
            refstate=str(topics_raw.get("refstate", "gmr/refstate")),
            recording_state=str(topics_raw.get("recording_state", DdsStateTopics.recording_state)),
            motion_feedback=str(topics_raw.get("motion_feedback", DdsStateTopics.motion_feedback)),
        ),
        buffer_capacity=int(dds.get("buffer_capacity", 512)),
    )

    shm_cfg = ShmSourceConfig(
        shm_name=str(shm.get("shm_name", "/igris_c_mujoco_vision_shm")),
        read_timeout_sec=float(shm.get("read_timeout_sec", 1.0)),
    )

    head_resolution = head.get("resolution", [HeadRgbConfig.width, HeadRgbConfig.height])
    if not isinstance(head_resolution, (list, tuple)) or len(head_resolution) != 2:
        raise ConfigError("head_rgb.resolution must be [width, height]")
    head_cfg = HeadRgbConfig(
        enable=bool(head.get("enable", False)),
        camera_topic=str(head.get("camera_topic") or HeadRgbConfig.camera_topic),
        frame_id=(head.get("frame_id") or None),
        width=int(head_resolution[0]), height=int(head_resolution[1]),
        fps=float(head.get("fps", HeadRgbConfig.fps)),
        bitrate=parse_bitrate(head.get("bitrate", HeadRgbConfig.bitrate)),
        gop=int(head.get("gop", HeadRgbConfig.gop)),
    )

    tol_raw = dict(robot.get("tolerances_ms") or {})
    try:
        robot_spec = RobotStateSpec(
            actual=str(robot.get("actual", "none")),
            reference=str(robot.get("reference", "none")),
            hand_actual=str(robot.get("hand_actual", "none")),
            hand_command=str(robot.get("hand_command", "none")),
            base_pose=str(robot.get("base_pose", "none")),
        ).validate()
        signals_spec = SignalsSpec(
            source=SourceKind.parse(str(signals_raw.get("source", "unknown"))),
            recording=str(signals_raw.get("recording", "none")),
            motion_feedback=str(signals_raw.get("motion_feedback", "none")),
        ).validate()
    except ValueError as error:
        raise ConfigError(str(error)) from error
    tolerances = MatchTolerances(
        actual_ms=float(tol_raw.get("actual", 5.0)),
        reference_ms=float(tol_raw.get("reference", 15.0)),
        hand_actual_ms=float(tol_raw.get("hand_actual", 40.0)),
        hand_command_ms=float(tol_raw.get("hand_command", 40.0)),
    )

    # dds.namespace may stay empty in the file (the CLI's --dds-namespace fills it); the
    # DDS channel itself refuses an empty namespace at start-up.
    if (robot_spec.uses_shm() or signals_spec.recording == "shm") and source_kind != "shm":
        raise ConfigError("robot_state/signals fields from shm need --source shm")

    return PumpConfig(
        source_kind=source_kind,
        config_path=Path(config_path) if config_path else None,
        stream=stream,
        encoder=encoder,
        calibration=calibration,
        usb=usb,
        dds=dds_cfg,
        shm=shm_cfg,
        robot_state=robot_spec,
        tolerances=tolerances,
        signals=signals_spec,
        enable_stereo=bool(eyes.get("enable_stereo", True)),
        head_rgb=head_cfg,
    )


def load_pump_config(path: PathLike, source_kind: str) -> PumpConfig:
    return build_pump_config(load_config_file(path), source_kind, config_path=Path(path))


__all__ = [
    "CALIBRATION_PROVIDERS",
    "CONFIG_PATH_ENV",
    "DATA_DIR_ENV",
    "PKG_PREFIX",
    "CalibrationServerConfig",
    "ConfigError",
    "DdsConfig",
    "EncoderConfig",
    "HeadRgbConfig",
    "IdealRigConfig",
    "PumpConfig",
    "SOURCE_KINDS",
    "ShmSourceConfig",
    "StreamConfig",
    "UsbSourceConfig",
    "build_pump_config",
    "data_path",
    "data_root",
    "default_config_path",
    "load_config_file",
    "load_pump_config",
    "normalize_source_kind",
    "packaged_path",
    "resolve_config_path",
    "resolve_data_path",
]
