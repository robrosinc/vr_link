"""Frame source -> H.264 encoder -> VSTR server loop for the standalone senders.

The pump owns the blocking loop the legacy ``camera_tcp_stream.py`` scripts
ran per connection; the connection handling itself is the non-blocking
:class:`vr_link.stream_server.VstrTcpStreamServer` shared with the Isaac
runner, so all senders queue records the same way, drop delta frames until a
new headset's first IDR, and disconnect a receiver that falls too far behind.

Per frame: skip while no headset is connected (the source keeps draining),
honour a keyframe request from a fresh connection, optionally pace to
``max_fps``, encode, and queue the access unit with the frame's acquisition
time and simulation time (0 for real cameras).

The encoder is created from the first frame's size (producers such as the
MuJoCo segment fix the size at runtime) and recreated if the size changes.
When a calibration provider is given, its payload is also handed to the
server as the in-band codec=2 record so receivers without the 9101 request
flow still get camera parameters.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Callable, Optional

from .calibration.schema import CalibrationError
from .calibration.server import CalibrationProvider
from .config import EncoderConfig
from .encoder import H264Encoder
from .sources import Frame, FrameSource
from .stream_server import VstrTcpStreamServer

logger = logging.getLogger(__name__)

IDLE_WARN_INTERVAL_SEC = 5.0
FPS_REPORT_INTERVAL_SEC = 2.0

EncoderFactory = Callable[[int, int], H264Encoder]


class StreamPump:
    """Drives one :class:`FrameSource` into one :class:`VstrTcpStreamServer`."""

    def __init__(
        self,
        source: FrameSource,
        server: VstrTcpStreamServer,
        *,
        encoder_cfg: EncoderConfig,
        max_fps: Optional[float] = None,
        calibration: Optional[CalibrationProvider] = None,
        encoder_factory: Optional[EncoderFactory] = None,
        read_timeout: float = 1.0,
        verbose: bool = False,
    ) -> None:
        self.source = source
        self.server = server
        self.encoder_cfg = encoder_cfg
        self.max_fps = max_fps
        self.calibration = calibration
        self.read_timeout = float(read_timeout)
        self.verbose = verbose
        self._encoder_factory = encoder_factory or self._default_encoder
        self.encoder: Optional[H264Encoder] = None
        self.frames_sent = 0
        self.frames_dropped_no_client = 0
        self._params_published = False
        self._last_sent_ns: Optional[int] = None

    # -- helpers -------------------------------------------------------------------

    def _default_encoder(self, width: int, height: int) -> H264Encoder:
        cfg = self.encoder_cfg
        return H264Encoder(width, height, cfg.fps, codec=cfg.codec, bitrate=cfg.bitrate, gop=cfg.gop)

    def _ensure_encoder(self, frame: Frame) -> H264Encoder:
        if self.encoder is not None and (self.encoder.width, self.encoder.height) != (frame.width, frame.height):
            logger.warning("Frame size changed to %dx%d; recreating the encoder", frame.width, frame.height)
            self.encoder.close()
            self.encoder = None
        if self.encoder is None:
            self.encoder = self._encoder_factory(frame.width, frame.height)
            logger.info("Encoding %dx%d with %s", frame.width, frame.height, self.encoder.codec_name)
            self._params_published = False
        return self.encoder

    def _publish_params(self, frame: Frame) -> None:
        if self.calibration is None or self._params_published:
            return
        try:
            cal = self.calibration()
        except CalibrationError as error:
            logger.debug("camera params not ready for the in-band record: %s", error)
            return
        except Exception:
            logger.exception("calibration provider failed; no in-band camera params")
            self._params_published = True  # do not retry a broken provider every frame
            return
        self.server.set_camera_params(dict(cal), width=frame.width // 2, height=frame.height)
        self._params_published = True

    # -- loop ------------------------------------------------------------------------

    def run(self, stop: threading.Event) -> None:
        """Block until ``stop`` is set; the source is opened here and closed on exit."""

        self.source.open()
        last_idle_warn = time.monotonic()
        fps_count = 0
        fps_t0 = time.monotonic()
        try:
            while not stop.is_set():
                frame = self.source.read(self.read_timeout)
                if frame is None:
                    now = time.monotonic()
                    if now - last_idle_warn >= IDLE_WARN_INTERVAL_SEC:
                        logger.warning("No frames from the source; still waiting...")
                        last_idle_warn = now
                    continue
                last_idle_warn = time.monotonic()

                if not self.server.client_connected:
                    self.frames_dropped_no_client += 1
                    continue

                encoder = self._ensure_encoder(frame)
                self._publish_params(frame)
                if self.server.consume_keyframe_request():
                    encoder.request_keyframe()

                if self.max_fps and self._last_sent_ns is not None:
                    if frame.capture_ns - self._last_sent_ns < int(1e9 / self.max_fps):
                        continue

                for unit in encoder.encode(frame.pixels, frame.pixel_format):
                    self.server.send_video(
                        unit.data,
                        is_key=unit.is_key,
                        width=frame.width,
                        height=frame.height,
                        wall_ns=frame.capture_ns,
                        sim_ns=frame.sim_ns or 0,
                    )
                    if self.verbose:
                        logger.info("frame=%d %s bytes=%7d", frame.index, "KEY" if unit.is_key else "   ", len(unit.data))
                self._last_sent_ns = frame.capture_ns
                self.frames_sent += 1

                fps_count += 1
                now = time.monotonic()
                if now - fps_t0 >= FPS_REPORT_INTERVAL_SEC:
                    logger.info("capture rate: %.1f fps", fps_count / (now - fps_t0))
                    fps_count = 0
                    fps_t0 = now
        finally:
            if self.encoder is not None:
                self.encoder.close()
                self.encoder = None
            self.source.close()


__all__ = ["StreamPump"]
