"""VRL2 downlink server: one headset at a time, framed Envelopes in order.

The PC is the TCP server and the headset connects (port 9100). Everything the
headset needs arrives on this one socket as ``frame_payload(Envelope)``
records:

* ``Calibration`` once per connection, before the first other record;
* per video frame ``Signals`` -> ``RobotState`` -> ``VideoFrame``, all with
  the same ``wall_ns``/``sim_ns`` and consecutive ``seq``;
* in state-only mode (no camera) ``Signals`` -> ``RobotState`` at a fixed rate.

Delivery rules (unchanged from the 1.x VSTR server): records queued while no
client is attached are dropped; video records are dropped until the first key
frame of a connection; nothing is dropped mid-stream; a client that falls
behind ``max_queue_bytes`` is disconnected. ``send_*`` never block on the
network - a writer thread drains the per-connection queue.
"""

from __future__ import annotations

import logging
import queue
import select
import socket
import threading
from typing import Any, Mapping, Optional

from ..messages.calibration import build_calibration_payload
from ..messages.envelope import VRL2_MAX_PAYLOAD_BYTES, frame_payload
from ..messages.robot_state import RobotStateSample, build_robot_state_payload
from ..messages.signals import SignalsSample, build_signals_payload
from ..messages.video import build_video_frame_payload
from ..utils.clocks import wall_ns as now_wall_ns

logger = logging.getLogger(__name__)

_ACCEPT_POLL_SEC = 0.5
_QUEUE_POLL_SEC = 0.5


class EnvelopeTcpServer:
    """Accepts one headset at a time and pushes VRL2 records to it."""

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 9100,
        *,
        send_calibration: bool = True,
        max_queue_bytes: int = 64 * 1024 * 1024,
    ) -> None:
        self._host = host
        self._port = port
        self._send_calibration = send_calibration
        self._max_queue_bytes = max_queue_bytes

        self._lock = threading.Lock()
        self._calibration: Optional[Mapping[str, Any]] = None
        self._queue: Optional["queue.Queue[bytes]"] = None
        self._queued_bytes = 0
        self._awaiting_keyframe = True
        self._sequence = 0
        self._connected = False
        self._calibration_sent = False
        self.connections = 0

        self._keyframe_requested = threading.Event()
        self._stop_requested = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._server_socket: Optional[socket.socket] = None

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((self._host, self._port))
        server.listen(1)
        server.settimeout(_ACCEPT_POLL_SEC)
        self._port = server.getsockname()[1]  # resolves port 0 (tests)
        self._server_socket = server
        self._thread = threading.Thread(target=self._run, name="vrl2-tcp-server", daemon=True)
        self._thread.start()
        logger.info("VRL2 downlink listening on %s:%d", self._host, self._port)

    def stop(self) -> None:
        self._stop_requested.set()
        if self._thread is not None:
            self._thread.join(timeout=10.0)
        if self._server_socket is not None:
            self._server_socket.close()

    def __enter__(self) -> "EnvelopeTcpServer":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()

    # -- sender API ----------------------------------------------------------

    @property
    def port(self) -> int:
        return self._port

    @property
    def client_connected(self) -> bool:
        with self._lock:
            return self._connected

    def consume_keyframe_request(self) -> bool:
        """True once per new connection: the encoder should emit an IDR next."""

        if self._keyframe_requested.is_set():
            self._keyframe_requested.clear()
            return True
        return False

    def set_calibration(self, cal: Mapping[str, Any]) -> None:
        """Store the calibration sent once per connection before any other record."""

        # Build once to validate early; the per-connection record is rebuilt with its seq.
        build_calibration_payload(cal)
        with self._lock:
            self._calibration = dict(cal)

    def send_video(
        self,
        access_unit: bytes,
        *,
        is_key: bool,
        width: int,
        height: int,
        wall_ns: int,
        sim_ns: int = 0,
        robot_state: Optional[RobotStateSample] = None,
        signals: Optional[SignalsSample] = None,
    ) -> None:
        """Queue [Calibration once] -> Signals -> RobotState -> VideoFrame for one frame."""

        if len(access_unit) > VRL2_MAX_PAYLOAD_BYTES:
            logger.warning("access unit %d bytes exceeds the %d-byte record limit; dropped", len(access_unit), VRL2_MAX_PAYLOAD_BYTES)
            return
        with self._lock:
            if self._queue is None:
                return
            if self._awaiting_keyframe:
                if not is_key:
                    return
                self._awaiting_keyframe = False
            self._enqueue_calibration_locked()
            self._enqueue_state_locked(wall_ns, sim_ns, robot_state, signals)
            self._enqueue_locked(
                build_video_frame_payload(
                    access_unit, is_key=is_key, width=width, height=height,
                    seq=self._next_sequence_locked(), wall_ns=wall_ns, sim_ns=sim_ns,
                )
            )
            drop_client = self._queued_bytes > self._max_queue_bytes
        if drop_client:
            logger.warning("Client fell behind (>%d queued bytes); disconnecting it.", self._max_queue_bytes)
            self._detach_client()

    def send_state(
        self,
        *,
        wall_ns: int,
        sim_ns: int = 0,
        robot_state: Optional[RobotStateSample] = None,
        signals: Optional[SignalsSample] = None,
    ) -> None:
        """Queue Signals/RobotState without video (state-only mode). Not keyframe-gated."""

        with self._lock:
            if self._queue is None:
                return
            self._enqueue_calibration_locked()
            self._enqueue_state_locked(wall_ns, sim_ns, robot_state, signals)
            drop_client = self._queued_bytes > self._max_queue_bytes
        if drop_client:
            logger.warning("Client fell behind (>%d queued bytes); disconnecting it.", self._max_queue_bytes)
            self._detach_client()

    # -- internals -----------------------------------------------------------

    def _enqueue_calibration_locked(self) -> None:
        if not self._send_calibration or self._calibration is None or self._calibration_sent:
            return
        self._calibration_sent = True
        self._enqueue_locked(build_calibration_payload(self._calibration, seq=self._next_sequence_locked(), wall_ns=now_wall_ns()))

    def _enqueue_state_locked(self, wall_ns: int, sim_ns: int, robot_state: Optional[RobotStateSample], signals: Optional[SignalsSample]) -> None:
        if signals is not None:
            self._enqueue_locked(build_signals_payload(signals, seq=self._next_sequence_locked(), wall_ns=wall_ns, sim_ns=sim_ns))
        if robot_state is not None:
            self._enqueue_locked(build_robot_state_payload(robot_state, seq=self._next_sequence_locked(), wall_ns=wall_ns, sim_ns=sim_ns))

    def _next_sequence_locked(self) -> int:
        sequence = self._sequence
        self._sequence += 1
        return sequence

    def _enqueue_locked(self, payload: bytes) -> None:
        if self._queue is None:
            return
        record = frame_payload(payload)
        self._queue.put(record)
        self._queued_bytes += len(record)

    def _detach_client(self) -> None:
        with self._lock:
            if self._queue is not None:
                self._queue.put(b"")  # sentinel: writer closes the connection

    def _run(self) -> None:
        assert self._server_socket is not None
        while not self._stop_requested.is_set():
            try:
                conn, addr = self._server_socket.accept()
            except TimeoutError:
                continue
            except OSError:
                break

            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            logger.info("Headset connected from %s:%d", addr[0], addr[1])
            with self._lock:
                self._queue = queue.Queue()
                self._queued_bytes = 0
                self._sequence = 0
                self._awaiting_keyframe = True
                self._calibration_sent = False
                self._connected = True
                self.connections += 1
            self._keyframe_requested.set()
            try:
                self._writer_loop(conn)
            finally:
                with self._lock:
                    self._queue = None
                    self._queued_bytes = 0
                    self._connected = False
                conn.close()
                logger.info("Connection closed. Waiting for next headset...")

    def _writer_loop(self, conn: socket.socket) -> None:
        while not self._stop_requested.is_set():
            # One-way protocol: inbound readability means EOF (or stray bytes to discard).
            readable, _, _ = select.select([conn], [], [], 0)
            if readable:
                try:
                    inbound = conn.recv(4096)
                except OSError:
                    logger.info("Client connection lost.")
                    return
                if not inbound:
                    logger.info("Client closed the connection.")
                    return
            with self._lock:
                record_queue = self._queue
            if record_queue is None:
                return
            try:
                record = record_queue.get(timeout=_QUEUE_POLL_SEC)
            except queue.Empty:
                continue
            if record == b"":
                return
            try:
                conn.sendall(record)
            except (BrokenPipeError, ConnectionResetError, OSError) as exc:
                logger.info("Client disconnected: %s", exc)
                return
            with self._lock:
                self._queued_bytes -= len(record)
