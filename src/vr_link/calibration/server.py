"""Calibration request server (port 9101) and the providers that feed it.

Protocol (Unreal ``ASettingPawn::RequestCalibration``): the headset connects,
sends ``GET_CALIBRATION\\n`` and reads the JSON reply until the server closes
the connection. One request per connection.

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

from .schema import CALIBRATION_REQUEST_MAX_BYTES, CALIBRATION_REQUEST_TOKEN, CalibrationError, encode_payload, validate_calibration

logger = logging.getLogger(__name__)

CalibrationProvider = Callable[[], Mapping[str, Any]]
_ACCEPT_POLL_SEC = 0.5
_RECV_CHUNK = 64


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

    from .yaml_io import load_unreal_yaml

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

    from .yaml_io import calibration_from_json

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
    """Answers ``GET_CALIBRATION`` on ``host:port`` from a daemon thread."""

    def __init__(
        self,
        provider: CalibrationProvider,
        *,
        host: str = "0.0.0.0",
        port: int = 9101,
        request_timeout: float = 2.0,
        include_meta: bool = False,
    ) -> None:
        self._provider = provider
        self._host = host
        self._port = int(port)
        self._request_timeout = float(request_timeout)
        self._include_meta = include_meta
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
        request = self._read_request(conn)
        if request and CALIBRATION_REQUEST_TOKEN not in request:
            logger.warning("Calibration: unexpected request %r from %s; responding anyway", request, addr[0])
        try:
            payload = encode_payload(self._provider(), include_meta=self._include_meta)
        except CalibrationError as error:
            self.failed_count += 1
            logger.warning("Calibration: no payload for %s: %s", addr[0], error)
            return
        except Exception:
            self.failed_count += 1
            logger.exception("Calibration: provider failed for %s", addr[0])
            return
        try:
            conn.sendall(payload)
        except OSError as error:
            self.failed_count += 1
            logger.error("Calibration: send to %s failed: %s", addr[0], error)
            return
        self.served_count += 1
        logger.info("Sent calibration (%d bytes) to %s:%d", len(payload), addr[0], addr[1])

    def _read_request(self, conn: socket.socket) -> bytes:
        """Read up to a newline (lenient); returns whatever arrived before the timeout."""

        conn.settimeout(self._request_timeout)
        buf = bytearray()
        try:
            while b"\n" not in buf and len(buf) < CALIBRATION_REQUEST_MAX_BYTES:
                chunk = conn.recv(_RECV_CHUNK)
                if not chunk:
                    break
                buf.extend(chunk)
        except socket.timeout:
            pass
        except OSError as error:
            logger.warning("Calibration: request read failed: %s", error)
        return bytes(buf)


def request_calibration(host: str, port: int, *, timeout: float = 5.0) -> bytes:
    """Client side of the protocol (tests, diagnostics): returns the raw JSON reply."""

    with socket.create_connection((host, port), timeout=timeout) as conn:
        conn.sendall(CALIBRATION_REQUEST_TOKEN + b"\n")
        chunks = []
        while True:
            chunk = conn.recv(4096)
            if not chunk:
                break
            chunks.append(chunk)
    return b"".join(chunks)


__all__ = [
    "CalibrationProvider",
    "CalibrationServer",
    "LatestValueProvider",
    "json_file_provider",
    "request_calibration",
    "static_provider",
    "yaml_file_provider",
]
