"""Frame source -> H.264 encoder -> VSTR server loop for the standalone senders.

The pump owns the blocking loop the legacy ``camera_tcp_stream.py`` scripts
ran per connection; the connection handling itself is the non-blocking
:class:`vr_link.network.tcp_server.EnvelopeTcpServer`, so every
runner, so all senders queue records the same way, drop delta frames until a
new headset's first IDR, and disconnect a receiver that falls too far behind.

Per frame: skip while no headset is connected (the source keeps draining),
honour a keyframe request from a fresh connection, optionally pace to
``max_fps``, encode, and queue the access unit with the frame's acquisition
time and simulation time (0 for real cameras).

While the source delivers no frames (NO CAMERA targets, camera down) and a
headset is connected, a state-only heartbeat (``Signals`` and, when a
``robot_state_now`` provider is given, ``RobotState``) goes out at
``state_heartbeat_hz`` so the HUD keeps showing the producer feedback and the
robot state. The source read timeout is capped to the heartbeat period.

The encoder is created from the first frame's size (producers such as the
MuJoCo segment fix the size at runtime) and recreated if the size changes.
``stream_id`` tags every ``VideoFrame`` (``VideoStream``); a secondary stream
(head RGB) runs its own pump on its own thread against the same server with
no calibration or state providers, so only its video records go out.
When a calibration provider is given, its value is handed to the server as
the once-per-connection in-band ``Calibration`` record. ``robot_state`` and
``signals`` are per-frame providers (see :mod:`vr_link.pipeline.robot_state`)
whose results ride along with each video frame.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Callable, Optional

from ..calibration.schema import CalibrationError
from ..network.calibration_server import CalibrationProvider
from .config import EncoderConfig
from ..video.encoder import H264Encoder
from ..sources import Frame, FrameSource
from ..messages.robot_state import RobotStateSample
from ..messages.video import VideoStream
from ..messages.signals import SignalsSample
from ..network.tcp_server import EnvelopeTcpServer

logger = logging.getLogger(__name__)

IDLE_WARN_INTERVAL_SEC = 5.0
FPS_REPORT_INTERVAL_SEC = 2.0

EncoderFactory = Callable[[int, int], H264Encoder]
RobotStateProvider = Callable[[Frame], Optional[RobotStateSample]]
SignalsProvider = Callable[[Frame], Optional[SignalsSample]]
# Frame-less providers for the state heartbeat: robot state at a wall time, signals now.
RobotStateNowProvider = Callable[[int], Optional[RobotStateSample]]
SignalsNowProvider = Callable[[], Optional[SignalsSample]]
DEFAULT_STATE_HEARTBEAT_HZ = 10.0


class StreamPump:
    """Drives one :class:`FrameSource` into one :class:`EnvelopeTcpServer`."""

    def __init__(
        self,
        source: FrameSource,
        server: EnvelopeTcpServer,
        *,
        encoder_cfg: EncoderConfig,
        max_fps: Optional[float] = None,
        calibration: Optional[CalibrationProvider] = None,
        encoder_factory: Optional[EncoderFactory] = None,
        read_timeout: float = 1.0,
        verbose: bool = False,
        robot_state: Optional[RobotStateProvider] = None,
        signals: Optional[SignalsProvider] = None,
        stats_text: Optional[Callable[[], str]] = None,
        robot_state_now: Optional[RobotStateNowProvider] = None,
        signals_now: Optional[SignalsNowProvider] = None,
        state_heartbeat_hz: float = DEFAULT_STATE_HEARTBEAT_HZ,
        stream_id: int = VideoStream.EYES_STEREO,
    ) -> None:
        self.source = source
        self.server = server
        self.stream_id = VideoStream(int(stream_id))
        self.label = self.stream_id.name.lower()
        self.robot_state = robot_state
        self.signals = signals
        self.stats_text = stats_text  # e.g. RobotStateAssembler.format_stats, appended to the rate log
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
        # State heartbeat (no frames): providers and pacing.
        self.robot_state_now = robot_state_now
        self.signals_now = signals_now
        self.state_heartbeat_hz = float(state_heartbeat_hz)
        self.heartbeats_sent = 0
        self._last_heartbeat_monotonic: Optional[float] = None
        if self._heartbeat_enabled():
            self.read_timeout = min(self.read_timeout, 1.0 / self.state_heartbeat_hz)

    # -- helpers -------------------------------------------------------------------

    def _heartbeat_enabled(self) -> bool:
        return self.state_heartbeat_hz > 0.0 and (self.signals_now is not None or self.robot_state_now is not None)

    def _maybe_send_state_heartbeat(self) -> None:
        """Send Signals/RobotState without a frame, at most once per heartbeat period."""

        if not self._heartbeat_enabled() or not self.server.client_connected:
            return
        now = time.monotonic()
        if self._last_heartbeat_monotonic is not None and now - self._last_heartbeat_monotonic < 1.0 / self.state_heartbeat_hz:
            return
        self._last_heartbeat_monotonic = now
        wall_ns = time.time_ns()
        state = self.robot_state_now(wall_ns) if self.robot_state_now is not None else None
        sig = self.signals_now() if self.signals_now is not None else None
        if state is None and sig is None:
            return
        self.server.send_state(wall_ns=wall_ns, sim_ns=0, robot_state=state, signals=sig)
        self.heartbeats_sent += 1

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
            logger.info("%s: encoding %dx%d with %s", self.label, frame.width, frame.height, self.encoder.codec_name)
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
        self.server.set_calibration(cal)
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
                        logger.warning("%s: no frames from the source; still waiting...", self.label)
                        last_idle_warn = now
                    self._maybe_send_state_heartbeat()
                    continue
                last_idle_warn = time.monotonic()

                if not self.server.client_connected:
                    self.frames_dropped_no_client += 1
                    continue

                encoder = self._ensure_encoder(frame)
                self._publish_params(frame)
                if self.server.consume_keyframe_request(self.stream_id):
                    encoder.request_keyframe()

                if self.max_fps and self._last_sent_ns is not None:
                    if frame.capture_ns - self._last_sent_ns < int(1e9 / self.max_fps):
                        continue

                units = encoder.encode(frame.pixels, frame.pixel_format)
                # Assemble after encoding: gives DDS samples stamped near capture_ns a few ms to arrive.
                state = self.robot_state(frame) if (self.robot_state is not None and units) else None
                sig = self.signals(frame) if (self.signals is not None and units) else None
                for unit in units:
                    self.server.send_video(
                        unit.data,
                        is_key=unit.is_key,
                        width=frame.width,
                        height=frame.height,
                        wall_ns=frame.capture_ns,
                        sim_ns=frame.sim_ns or 0,
                        robot_state=state,
                        signals=sig,
                        stream_id=self.stream_id,
                    )
                    state, sig = None, None  # one state record per frame even if the encoder returns several units
                    if self.verbose:
                        logger.info("%s frame=%d %s bytes=%7d", self.label, frame.index, "KEY" if unit.is_key else "   ", len(unit.data))
                self._last_sent_ns = frame.capture_ns
                self.frames_sent += 1

                fps_count += 1
                now = time.monotonic()
                if now - fps_t0 >= FPS_REPORT_INTERVAL_SEC:
                    extra = f"; robot_state match: {self.stats_text()}" if self.stats_text is not None else ""
                    logger.info("%s capture rate: %.1f fps%s", self.label, fps_count / (now - fps_t0), extra)
                    fps_count = 0
                    fps_t0 = now
        finally:
            if self.encoder is not None:
                self.encoder.close()
                self.encoder = None
            self.source.close()


__all__ = ["StreamPump"]
