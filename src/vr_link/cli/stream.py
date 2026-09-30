"""``vr-link-stream``: the VRL2 downlink sender (video + calibration + robot state).

Port 9100 carries framed FlatBuffers ``Envelope`` records (Calibration once,
then per frame Signals -> RobotState -> VideoFrame); port 9101 answers a
``CalibrationRequest`` with a ``Calibration``. Only the frame source and the
robot-state sources differ between modes::

    vr-link-stream --source usb   [--config stream_usb.yaml]   NUC USB camera + robot DDS state
    vr-link-stream --source dds   [--config stream_dds.yaml]   Jetson camera topic + robot DDS state
    vr-link-stream --source shm   [--config stream_shm.yaml]   simulator SHM (video + state) [+ DDS reference]
    vr-link-stream --source none  [--config state_only.yaml]   no camera: RobotState/Signals only

Calibration providers (``calibration.provider``): ``yaml`` / ``json`` serve a
file (measured hardware calibration); ``ideal_from_source`` computes the ideal
model from the camera metadata a simulator publishes (SHM layout >= 2), falling
back to ``ideal_from_config`` (``calibration.ideal`` + the source frame size).
"""

from __future__ import annotations

import argparse
import logging
import signal
import threading
from typing import List, Optional, Sequence

from vr_link.calibration.ideal import ideal_stereo_from_fovy, ideal_stereo_from_metadata
from vr_link.calibration.schema import CalibrationError
from vr_link.network.calibration_server import CalibrationProvider, CalibrationServer, json_file_provider, yaml_file_provider
from vr_link.network.tcp_server import EnvelopeTcpServer
from vr_link.pipeline.config import PumpConfig, load_pump_config, normalize_source_kind, resolve_config_path
from vr_link.pipeline.pump import StreamPump
from vr_link.pipeline.robot_state import DdsStateBuffers, RobotStateAssembler, SignalsProvider
from vr_link.pipeline.state_only import StateOnlyPump
from vr_link.sources import FrameSource
from vr_link.video.encoder import resolve_h264_encoder

logger = logging.getLogger("vr_link.stream")


class DdsRuntime:
    """The process-wide DDS participant plus the robot-state subscriptions (all optional)."""

    def __init__(self, cfg: PumpConfig) -> None:
        from vr_link.network.dds import DdsChannel, start_state_subscriptions
        from vr_link.utils.time_sync import StampedRingBuffer

        self.channel = DdsChannel.create(
            domain_id=cfg.dds.domain_id, namespace=cfg.dds.namespace,
            topic_naming=cfg.dds.topic_naming, config_uri=cfg.dds.config_uri,
        )
        cap = cfg.dds.buffer_capacity
        self.buffers = DdsStateBuffers(
            actual=StampedRingBuffer(cap, name="actual"), reference=StampedRingBuffer(cap, name="reference"),
            hand_actual=StampedRingBuffer(cap, name="hand_actual"), hand_command=StampedRingBuffer(cap, name="hand_command"),
        )
        self.subscriptions = start_state_subscriptions(self.channel, cfg.dds.topics, self.buffers, needed=cfg.robot_state.dds_fields())

    def close(self) -> None:
        for sub in self.subscriptions:
            sub.stop()
        self.channel.release()


def build_source(cfg: PumpConfig, dds: Optional[DdsRuntime]) -> Optional[FrameSource]:
    if cfg.source_kind == "usb":
        from vr_link.sources.usb import UsbStereoSource

        return UsbStereoSource(
            cfg.usb.device, cfg.stream.width, cfg.stream.height, cfg.stream.fps,
            fourcc=cfg.usb.fourcc, apply_tuning=cfg.usb.apply_tuning,
        )
    if cfg.source_kind == "dds":
        from vr_link.sources.dds_camera import DdsCameraSource

        assert dds is not None
        return DdsCameraSource(dds.channel, cfg.dds.camera_topic, frame_id=cfg.dds.frame_id, width=cfg.stream.width, height=cfg.stream.height)
    if cfg.source_kind == "shm":
        from vr_link.sources.shm import ShmSource

        return ShmSource(cfg.shm.shm_name)
    if cfg.source_kind == "none":
        return None
    raise ValueError(f"unknown source kind {cfg.source_kind!r}")


def build_calibration_provider(cfg: PumpConfig, source: Optional[FrameSource]) -> Optional[CalibrationProvider]:
    cal = cfg.calibration
    if not cal.enable:
        return None
    if cal.provider == "yaml":
        if cal.yaml_path is None:
            logger.info("Calibration enabled but no yaml source configured; skipping")
            return None
        return yaml_file_provider(cal.yaml_path, image_width=cal.image_width, image_height=cal.image_height)
    if cal.provider == "json":
        if cal.json_path is None:
            logger.info("Calibration enabled but no json source configured; skipping")
            return None
        return json_file_provider(cal.json_path, image_width=cal.image_width, image_height=cal.image_height)
    if source is None:
        logger.info("Calibration provider %s needs a frame source; skipping (state-only)", cal.provider)
        return None

    def from_config():
        size = source.frame_size
        if size is None:
            raise CalibrationError("source frame size unknown yet")
        width, height = size
        return ideal_stereo_from_fovy(width // 2, height, cal.ideal.fovy_deg, cal.ideal.toe_out_deg, cal.ideal.baseline_mm)

    if cal.provider == "ideal_from_config":
        return from_config

    def from_source():
        meta = source.camera_metadata()
        if meta is None:
            return from_config()
        return ideal_stereo_from_metadata(meta)

    return from_source


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="VRL2 downlink sender: stereo H.264 + calibration + robot state (USB camera, DDS camera, simulator SHM, or state only)")
    p.add_argument("--source", required=True, choices=("usb", "dds", "shm", "none", "mujoco"), help="frame source ('mujoco' = shm)")
    p.add_argument("--config", default=None, metavar="PATH", help="config YAML (default: $VR_LINK_STREAM_CONFIG or the packaged default for --source)")
    p.add_argument("--port", type=int, default=None, help="override tcp.port")
    p.add_argument("--calibration-port", type=int, default=None, help="override calibration.port")
    p.add_argument("--encoder", default=None, help="override tcp.encoder (libx264, h264_nvenc, h264_vaapi, ...)")
    p.add_argument("--shm-name", default=None, help="shm: override shm.shm_name")
    p.add_argument("--dds-namespace", default=None, help="override dds.namespace")
    p.add_argument("--dds-domain", type=int, default=None, help="override dds.domain_id")
    p.add_argument("--max-fps", type=float, default=None, help="pace the stream to this rate (default: source rate)")
    p.add_argument("-v", "--verbose", action="store_true", help="log every packet")
    args = p.parse_args(argv)
    args.source = normalize_source_kind(args.source)
    return args


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    config_path = resolve_config_path(args.config, args.source)
    logger.info("Loading config from %s", config_path)
    cfg = load_pump_config(config_path, args.source)
    if args.source != "none" and not cfg.enable_stereo:
        logger.info("eyes_stereo disabled by config; nothing to stream")
        return
    if args.port is not None:
        cfg.stream.port = args.port
    if args.calibration_port is not None:
        cfg.calibration.port = args.calibration_port
    if args.encoder is not None:
        cfg.encoder.codec = args.encoder
    if args.shm_name is not None:
        cfg.shm.shm_name = args.shm_name
    if args.dds_namespace is not None:
        cfg.dds.namespace = args.dds_namespace
    if args.dds_domain is not None:
        cfg.dds.domain_id = args.dds_domain
    if cfg.needs_dds() and not cfg.dds.namespace:
        raise SystemExit("dds.namespace is required (config or --dds-namespace)")

    closers: List = []
    stop = threading.Event()

    def on_signal(_signum, _frame):
        logger.info("Shutting down.")
        stop.set()

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)

    dds = None
    try:
        if cfg.needs_dds():
            try:
                dds = DdsRuntime(cfg)
            except ImportError as error:
                raise SystemExit(
                    f"{error}\nThis config needs DDS for robot_state {sorted(cfg.robot_state.dds_fields())}"
                    f"{' and the camera' if cfg.source_kind == 'dds' else ''}; install the wheel or set those fields to 'none'."
                ) from error
            closers.append(dds.close)
        source = build_source(cfg, dds)
        provider = build_calibration_provider(cfg, source)
        server = EnvelopeTcpServer(host=cfg.stream.host, port=cfg.stream.port, send_calibration=provider is not None)
        calibration_server = None
        if provider is not None:
            calibration_server = CalibrationServer(provider, host=cfg.calibration.host, port=cfg.calibration.port, request_timeout=cfg.calibration.request_timeout)
        assembler = RobotStateAssembler(cfg.robot_state, cfg.tolerances, dds.buffers if dds is not None else None)
        signals = SignalsProvider(cfg.signals)

        server.start()
        closers.append(server.stop)
        if calibration_server is not None:
            calibration_server.start()
            closers.append(calibration_server.stop)

        if source is None:
            shm = None
            if cfg.needs_shm():
                from vr_link.sources.shm import ShmSource

                shm = ShmSource(cfg.shm.shm_name)
            logger.info("Listening on %s:%d (state only, %.0f Hz, robot_state %s). Waiting for headset...", cfg.stream.host, server.port, cfg.state_only.rate_hz, cfg.robot_state)
            StateOnlyPump(server, assembler, signals, rate_hz=cfg.state_only.rate_hz, max_age_ms=cfg.state_only.max_age_ms, shm=shm, calibration=provider).run(stop)
        else:
            cfg.encoder.codec = resolve_h264_encoder(cfg.encoder.codec)
            pump = StreamPump(
                source, server, encoder_cfg=cfg.encoder, max_fps=args.max_fps, calibration=provider,
                read_timeout=cfg.shm.read_timeout_sec if args.source == "shm" else 1.0, verbose=args.verbose,
                robot_state=(lambda frame: assembler.assemble(frame.capture_ns, frame.robot_block)) if not cfg.robot_state.is_empty() else None,
                signals=lambda frame: signals.sample(frame.robot_block),
                stats_text=assembler.format_stats if cfg.robot_state.dds_fields() else None,
            )
            logger.info("Listening on %s:%d (%s source, encoder %s, robot_state %s). Waiting for headset...", cfg.stream.host, server.port, args.source, cfg.encoder.codec, cfg.robot_state)
            pump.run(stop)
    finally:
        for close in reversed(closers):
            try:
                close()
            except Exception:  # noqa: BLE001
                logger.exception("shutdown step failed")


if __name__ == "__main__":
    main()
