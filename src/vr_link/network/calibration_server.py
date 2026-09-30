"""Calibration request server (port 9101) and the providers that feed it.

Protocol (Unreal ``ASettingPawn::RequestCalibration``): the headset connects,
sends one framed ``Envelope(CalibrationRequest)`` and reads one framed
``Envelope(Calibration)`` until the server closes the connection. One
request per connection. Any other framed record, or none at all within
``request_timeout``, is answered anyway (logged), so a plain connect also
works as a request.

The payload comes from a *provider*, a zero-argument callable returning the
calibration mapping, evaluated on every request so simulators can answer
with their live camera model. A provider that raises (for example
:class:`LatestValueProvider` before its first ``set``) makes the server close
the connection without a reply; the headset then reports a failed request
and keeps NEXT disabled, which is the intended "not ready yet" behaviour.
"""

from __future__ import annotations

import logging
import os
import socket
import threading
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Union

from ..calibration.schema import CalibrationError, validate_calibration
from ..messages.calibration import build_calibration_payload, build_calibration_request_payload, decode_calibration
from ..messages.envelope import VRL2_HEADER_SIZE, FramingError, MessageKind, RecordReader, frame_payload, parse_envelope
from ..utils.clocks import wall_ns

logger = logging.getLogger(__name__)

CalibrationProvider = Callable[[], Mapping[str, Any]]
_ACCEPT_POLL_SEC = 0.5
_RECV_CHUNK = 256
_REQUEST_MAX_BYTES = 4096


# --- providers ------------------------------------------------------------------


def static_provider(cal: Mapping[str, Any]) -> CalibrationProvider:
    """Serve one validated calibration forever."""

    valid = validate_calibration(cal)
    return lambda: valid


class LatestValueProvider:
    """Thread-safe holder filled by the producer thread, read by the server thread.

    Raises :class:`CalibrationError` until :meth:`set` has been called once.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._value: Optional[Dict[str, Any]] = None

    def set(self, cal: Mapping[str, Any]) -> None:
        valid = validate_calibration(cal)
        with self._lock:
            self._value = valid

    def clear(self) -> None:
        with self._lock:
            self._value = None

    @property
    def ready(self) -> bool:
        with self._lock:
            return self._value is not None

    def __call__(self) -> Dict[str, Any]:
        with self._lock:
            if self._value is None:
                raise CalibrationError("calibration not available yet")
            return dict(self._value)


def yaml_file_provider(path: Union[str, Path], *, image_width: Optional[int] = None, image_height: Optional[int] = None) -> CalibrationProvider:
    """Serve a flat calibration YAML, re-reading it when its mtime changes."""

    from ..calibration.yaml_io import load_unreal_yaml

    target = Path(path)
    cache: Dict[str, Any] = {}

    def provider() -> Mapping[str, Any]:
        mtime = os.stat(target).st_mtime_ns
        if cache.get("mtime") != mtime:
            cache["value"] = load_unreal_yaml(target, image_width=image_width, image_height=image_height)
            cache["mtime"] = mtime
        return cache["value"]

    return provider


def json_file_provider(path: Union[str, Path], *, image_width: Optional[int] = None, image_height: Optional[int] = None) -> CalibrationProvider:
    """Serve a ready JSON calibration file, re-reading it when its mtime changes."""

    from ..calibration.yaml_io import calibration_from_json

    target = Path(path)
    cache: Dict[str, Any] = {}

    def provider() -> Mapping[str, Any]:
        mtime = os.stat(target).st_mtime_ns
        if cache.get("mtime") != mtime:
            cache["value"] = calibration_from_json(target, image_width, image_height)
            cache["mtime"] = mtime
        return cache["value"]

    return provider


# --- server ---------------------------------------------------------------------


class CalibrationServer:
    """Answers ``Envelope(CalibrationRequest)`` on ``host:port`` from a daemon thread."""

    def __init__(
        self,
        provider: CalibrationProvider,
        *,
        host: str = "0.0.0.0",
        port: int = 9101,
        request_timeout: float = 2.0,
    ) -> None:
        self._provider = provider
        self._host = host
        self._port = int(port)
        self._request_timeout = float(request_timeout)
        self._server: Optional[socket.socket] = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self.served_count = 0
        self.failed_count = 0

    @property
    def port(self) -> int:
        """Bound port (differs from the requested one only when it was 0)."""

        return self._port

    def start(self) -> None:
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((self._host, self._port))
        server.listen(4)
        server.settimeout(_ACCEPT_POLL_SEC)
        self._port = server.getsockname()[1]
        self._server = server
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="calibration-server", daemon=True)
        self._thread.start()
        logger.info("Calibration server listening on %s:%d", self._host, self._port)

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        if self._server is not None:
            self._server.close()
            self._server = None

    def __enter__(self) -> "CalibrationServer":
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()

    # -- internals -------------------------------------------------------------

    def _run(self) -> None:
        assert self._server is not None
        while not self._stop.is_set():
            try:
                conn, addr = self._server.accept()
            except socket.timeout:
                continue
            except OSError:
                return
            try:
                self._serve_one(conn, addr)
            finally:
                conn.close()

    def _serve_one(self, conn: socket.socket, addr) -> None:
        kind = self._read_request(conn)
        if kind is not MessageKind.CALIBRATION_REQUEST:
            logger.warning("Calibration: no CalibrationRequest from %s (got %s); responding anyway", addr[0], kind.name if kind else "nothing")
        try:
            payload = build_calibration_payload(self._provider(), seq=0, wall_ns=wall_ns())
        except CalibrationError as error:
            self.failed_count += 1
            logger.warning("Calibration: no payload for %s: %s", addr[0], error)
            return
        except Exception:
            self.failed_count += 1
            logger.exception("Calibration: provider failed for %s", addr[0])
            return
        record = frame_payload(payload)
        try:
            conn.sendall(record)
        except OSError as error:
            self.failed_count += 1
            logger.error("Calibration: send to %s failed: %s", addr[0], error)
            return
        self.served_count += 1
        logger.info("Sent calibration (%d bytes) to %s:%d", len(record), addr[0], addr[1])

    def _read_request(self, conn: socket.socket) -> Optional[MessageKind]:
        """Read one framed record (lenient); None when nothing parsable arrived in time."""

        conn.settimeout(self._request_timeout)
        reader = RecordReader(max_payload_bytes=_REQUEST_MAX_BYTES)
        received = 0
        try:
            while received < _REQUEST_MAX_BYTES:
                chunk = conn.recv(_RECV_CHUNK)
                if not chunk:
                    break
                received += len(chunk)
                records = reader.feed(chunk)
                if records:
                    return parse_envelope(records[0]).kind
                if reader.framing_errors and reader.pending_bytes < VRL2_HEADER_SIZE:
                    return None  # garbage (e.g. a 1.x text request); do not wait for more
        except socket.timeout:
            pass
        except FramingError as error:
            logger.warning("Calibration: unparsable request: %s", error)
        except OSError as error:
            logger.warning("Calibration: request read failed: %s", error)
        return None


def request_calibration(host: str, port: int, *, timeout: float = 5.0) -> Dict[str, Any]:
    """Client side of the protocol (tests, diagnostics): returns the decoded calibration mapping."""

    with socket.create_connection((host, port), timeout=timeout) as conn:
        conn.sendall(frame_payload(build_calibration_request_payload(seq=0, wall_ns=wall_ns())))
        reader = RecordReader()
        while True:
            chunk = conn.recv(65536)
            if not chunk:
                break
            records = reader.feed(chunk)
            if records:
                return decode_calibration(parse_envelope(records[0]))
    raise CalibrationError("calibration server closed the connection without a reply")


__all__ = [
    "CalibrationProvider",
    "CalibrationServer",
    "LatestValueProvider",
    "json_file_provider",
    "request_calibration",
    "static_provider",
    "yaml_file_provider",
]
