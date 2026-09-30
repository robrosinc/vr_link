"""``vr-link-stream``: stereo H.264 VSTR sender for a USB camera, the robot DDS topic or MuJoCo.

Port 9100 carries the video (48-byte header + Annex-B access units) plus the
in-band camera-parameter record; port 9101 answers ``GET_CALIBRATION`` with
the stereo calibration JSON. Only the frame source differs between the three
modes; the wire/encoder path is shared with the Isaac runner in hit-bench.

    vr-link-stream --source usb     [--config stream_usb.yaml]
    vr-link-stream --source dds     [--config stream_dds.yaml]
    vr-link-stream --source mujoco  [--config stream_mujoco.yaml]

Calibration providers (``calibration.provider`` in the config):

* ``yaml`` / ``json``: serve a file (the real robot's measured calibration).
* ``ideal_from_source``: compute the ideal model from what the source
  publishes (MuJoCo SHM layout v2). Falls back to ``ideal_from_config`` when
  the source has no metadata (layout v1).
* ``ideal_from_config``: compute it from ``calibration.ideal`` and the source
  frame size.
"""

from __future__ import annotations

import argparse
import logging
import signal
import threading
from typing import Optional, Sequence

from vr_link.calibration.ideal import ideal_stereo_from_fovy, ideal_stereo_from_metadata
from vr_link.calibration.schema import CalibrationError
from vr_link.calibration.server import CalibrationProvider, CalibrationServer, json_file_provider, yaml_file_provider
from vr_link.config import PumpConfig, load_pump_config, resolve_config_path
from vr_link.encoder import resolve_h264_encoder
from vr_link.pump import StreamPump
from vr_link.sources import FrameSource
from vr_link.stream_server import VstrTcpStreamServer

logger = logging.getLogger("vr_link.stream")


def build_source(cfg: PumpConfig) -> FrameSource:
    if cfg.source_kind == "usb":
        from vr_link.sources.usb import UsbStereoSource

        return UsbStereoSource(
            cfg.usb.device, cfg.stream.width, cfg.stream.height, cfg.stream.fps,
            fourcc=cfg.usb.fourcc, apply_tuning=cfg.usb.apply_tuning,
        )
    if cfg.source_kind == "dds":
        from vr_link.sources.dds import DdsStereoSource

        return DdsStereoSource(
            cfg.dds.domain_id, cfg.dds.namespace, cfg.dds.topic,
            config_uri=cfg.dds.config_uri, frame_id=cfg.dds.frame_id,
            width=cfg.stream.width, height=cfg.stream.height,
        )
    if cfg.source_kind == "mujoco":
        from vr_link.sources.mujoco_shm import MujocoShmSource

        return MujocoShmSource(cfg.mujoco.shm_name)
    raise ValueError(f"unknown source kind {cfg.source_kind!r}")


def build_calibration_provider(cfg: PumpConfig, source: FrameSource) -> Optional[CalibrationProvider]:
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
    p = argparse.ArgumentParser(description="Stereo H.264 VSTR sender (USB camera, robot DDS topic or MuJoCo SHM)")
    p.add_argument("--source", required=True, choices=("usb", "dds", "mujoco"), help="frame source")
    p.add_argument("--config", default=None, metavar="PATH", help="stream config YAML (default: $VR_LINK_STREAM_CONFIG or the packaged default for --source)")
    p.add_argument("--port", type=int, default=None, help="override tcp.port")
    p.add_argument("--calibration-port", type=int, default=None, help="override calibration.port")
    p.add_argument("--encoder", default=None, help="override tcp.encoder (libx264, h264_nvenc, h264_vaapi, ...)")
    p.add_argument("--shm-name", default=None, help="mujoco: override mujoco.shm_name")
    p.add_argument("--max-fps", type=float, default=None, help="pace the stream to this rate (default: source rate)")
    p.add_argument("-v", "--verbose", action="store_true", help="log every packet")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    config_path = resolve_config_path(args.config, args.source)
    logger.info("Loading config from %s", config_path)
    cfg = load_pump_config(config_path, args.source)
    if not cfg.enable_stereo:
        logger.info("eyes_stereo disabled by config; nothing to stream")
        return
    if args.port is not None:
        cfg.stream.port = args.port
    if args.calibration_port is not None:
        cfg.calibration.port = args.calibration_port
    if args.encoder is not None:
        cfg.encoder.codec = args.encoder
    if args.shm_name is not None:
        cfg.mujoco.shm_name = args.shm_name
    cfg.encoder.codec = resolve_h264_encoder(cfg.encoder.codec)

    source = build_source(cfg)
    provider = build_calibration_provider(cfg, source)
    server = VstrTcpStreamServer(host=cfg.stream.host, port=cfg.stream.port, send_camera_params=provider is not None)
    calibration_server = None
    if provider is not None:
        calibration_server = CalibrationServer(
            provider, host=cfg.calibration.host, port=cfg.calibration.port, request_timeout=cfg.calibration.request_timeout,
        )
    pump = StreamPump(
        source, server, encoder_cfg=cfg.encoder, max_fps=args.max_fps, calibration=provider,
        read_timeout=cfg.mujoco.read_timeout_sec if args.source == "mujoco" else 1.0, verbose=args.verbose,
    )

    stop = threading.Event()

    def on_signal(_signum, _frame):
        logger.info("Shutting down.")
        stop.set()

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)

    server.start()
    if calibration_server is not None:
        calibration_server.start()
    logger.info("Listening on %s:%d (%s source, encoder %s). Waiting for headset...", cfg.stream.host, server.port, args.source, cfg.encoder.codec)
    try:
        pump.run(stop)
    finally:
        if calibration_server is not None:
            calibration_server.stop()
        server.stop()


if __name__ == "__main__":
    main()
