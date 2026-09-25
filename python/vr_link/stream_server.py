"""Raw-TCP H.264 stream server speaking the Unreal ``TCPActor`` wire protocol.

Replaces the earlier WebXR/WebSocket bridge (see git history) with the legacy
Unreal headset transport: the headset is the TCP client (it connects out to
this host:port), so this process listens, accepts one headset at a time, and
pushes frames until the connection drops, then waits for the next connection.

Wire record = 48-byte big-endian header + payload. The header layout mirrors
``igris_c_camera/camera_tcp_stream.py`` (the reference implementation of the
receiver contract): magic "VSTR", version, codec, flags, sequence, width,
height, payload size, ``sec``/``nanosec`` wall-clock acquisition time, and a
u64 ``sim_ns`` slot that this server fills with *simulation* time so the
receiver can align frames with the 50 Hz control sequence even when the
simulator does not run in real time.

Protocol extension over the legacy sender: when camera parameters are set, one
record with ``codec = WIRE_CODEC_PARAMS_JSON`` carrying a JSON payload is sent
once per connection before any video. The receiver must parse or skip that
record; disable ``send_camera_params`` when talking to a receiver without this
handling.

Second extension: when ``send_video`` is given a head pose, one record with
``codec = WIRE_CODEC_HEAD_POSE`` is sent on the same stream immediately before
its video record, sharing the video record's sequence and timestamps (the
receiver matches on ``sim_ns``). The 28-byte payload is 7 big-endian float32:
position xyz (m) then quaternion wxyz, in the frame the pose-uplink head command is
given in. The receiver uses it for latency ("Correction Delay") reprojection
and tolerates its absence, so the pose may be enabled/disabled freely.

The simulation loop never blocks on the network: records are handed to a
per-connection writer queue, and a client that falls too far behind is
disconnected instead of accumulating unbounded latency.
"""

from __future__ import annotations

import json
import logging
import queue
import select
import socket
import struct
import threading
import time

import flatbuffers
import vr_link_generated as _vr_link

from .transport import StreamTransport

logger = logging.getLogger(__name__)

# --- Wire protocol constants (must match the Unreal TCPActor receiver) ------
WIRE_MAGIC = 0x56535452  # "VSTR"
WIRE_VERSION = 1
WIRE_CODEC_H264 = 1
WIRE_CODEC_PARAMS_JSON = 2  # extension: camera-parameter JSON record
WIRE_CODEC_HEAD_POSE = 3  # extension: per-frame rendered head-pose record
WIRE_CODEC_BASE_POSE = 4  # extension: per-frame robot base-pose record (same layout as codec=3)
WIRE_CODEC_SIM_STATUS = 5  # extension: FlatBuffers robros.vr.SimStatus payload (vr_link_schema)
WIRE_FLAG_KEYFRAME = 1 << 0  # codec=1 records
WIRE_FLAG_POSE_VALID = 1 << 0  # codec=3/4 records
HEAD_POSE_FORMAT = ">7f"  # px py pz (m), quat wxyz — big-endian float32
HEAD_POSE_PAYLOAD_SIZE = struct.calcsize(HEAD_POSE_FORMAT)  # 28
HEADER_FORMAT = ">IIIIIIIIIIQ"  # 10x u32 + 1x u64, big-endian
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)  # 48
MAX_PACKET_BYTES = 2 * 1024 * 1024  # receiver's MaxPacketBytes default

_ACCEPT_POLL_SEC = 0.5
_QUEUE_POLL_SEC = 0.5


def build_header(
    codec: int,
    flags: int,
    sequence: int,
    width: int,
    height: int,
    payload_size: int,
    wall_ns: int,
    sim_ns: int,
) -> bytes:
    """Pack the 48-byte big-endian wire header."""

    return struct.pack(
        HEADER_FORMAT,
        WIRE_MAGIC,
        WIRE_VERSION,
        codec,
        flags,
        sequence & 0xFFFFFFFF,
        width,
        height,
        payload_size,
        (wall_ns // 1_000_000_000) & 0xFFFFFFFF,
        wall_ns % 1_000_000_000,
        sim_ns & 0xFFFFFFFFFFFFFFFF,
    )


def build_sim_status_payload(*, recording: bool, timestamp_ns: int) -> bytes:
    """FlatBuffers ``robros.vr.SimStatus`` payload for a ``codec=5`` record.

    State-only (never events), newest-wins on the receiver; the VSTR header
    supplies framing, sequence and timestamps. Contract lives in
    vr_link_schema (README, "Downlink status").
    """

    builder = flatbuffers.Builder(64)
    _vr_link.SimStatusStart(builder)
    _vr_link.SimStatusAddTimestampNs(builder, timestamp_ns & 0xFFFFFFFFFFFFFFFF)
    _vr_link.SimStatusAddRecording(builder, bool(recording))
    builder.Finish(_vr_link.SimStatusEnd(builder))
    return bytes(builder.Output())


class VstrTcpStreamServer(StreamTransport):
    """Accepts one headset at a time and pushes VSTR-framed records to it."""

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 9100,
        send_camera_params: bool = True,
        max_queue_bytes: int = 64 * 1024 * 1024,
    ) -> None:
        self._host = host
        self._port = port
        self._send_camera_params = send_camera_params
        self._max_queue_bytes = max_queue_bytes

        self._lock = threading.Lock()
        self._camera_params_json: bytes | None = None
        self._camera_params_size: tuple[int, int] = (0, 0)
        self._queue: queue.Queue[bytes] | None = None
        self._queued_bytes = 0
        self._awaiting_keyframe = True
        self._sequence = 0
        self._connected = False
        self._params_sent = False

        self._keyframe_requested = threading.Event()
        self._stop_requested = threading.Event()
        self._thread: threading.Thread | None = None
        self._server_socket: socket.socket | None = None

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        """Bind the listening socket and start the accept/send thread."""

        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((self._host, self._port))
        server.listen(1)
        server.settimeout(_ACCEPT_POLL_SEC)
        self._server_socket = server
        self._thread = threading.Thread(target=self._run, name="vstr-tcp-server", daemon=True)
        self._thread.start()
        logger.info("VSTR TCP server listening on %s:%d", self._host, self._port)

    def stop(self) -> None:
        """Stop the server thread and close the listening socket."""

        self._stop_requested.set()
        if self._thread is not None:
            self._thread.join(timeout=10.0)
        if self._server_socket is not None:
            self._server_socket.close()

    # -- simulation-side API -------------------------------------------------

    @property
    def client_connected(self) -> bool:
        with self._lock:
            return self._connected

    def consume_keyframe_request(self) -> bool:
        """Return True once if a client connected since the last check, else False."""

        if self._keyframe_requested.is_set():
            self._keyframe_requested.clear()
            return True
        return False

    def set_camera_params(self, params: dict, width: int, height: int) -> None:
        """Store the camera-parameter JSON sent once per connection before video.

        ``width``/``height`` fill the header dimension fields of the params
        record (per-eye image size). Call before or after clients connect; a
        connection made earlier receives the params before its next video record.
        """

        with self._lock:
            self._camera_params_json = json.dumps(params).encode("utf-8")
            self._camera_params_size = (int(width), int(height))

    def send_video(
        self,
        access_unit: bytes,
        *,
        is_key: bool,
        width: int,
        height: int,
        wall_ns: int,
        sim_ns: int,
        head_pose: tuple[float, ...] | None = None,
        base_pose: tuple[float, ...] | None = None,
        sim_status: bytes | None = None,
    ) -> None:
        """Frame one H.264 access unit and queue it for in-order delivery.

        Nothing is dropped mid-stream (H.264 delta frames depend on their
        predecessors); however, records queued while no client is connected are
        discarded, and non-key records are discarded until the first keyframe of
        each connection so the receiver can leave its WaitingForKeyframe state.

        ``head_pose`` is the rendered head pose of this frame as 7 floats
        (position xyz in meters, quaternion wxyz) in the pose-uplink command frame;
        when given, a ``codec=3`` record sharing this frame's sequence and
        timestamps is queued immediately before the video record. ``None``
        sends no pose record (the receiver keeps its last correction).

        ``base_pose`` is the robot base (base_link) pose at the same instant,
        same layout and frame; when given, a ``codec=4`` record is queued the
        same way. The VR client pairs it with the operator's chest to re-align
        its view without absorbing neck tracking error.

        ``sim_status`` is a pre-built FlatBuffers SimStatus payload (see
        ``build_sim_status_payload``); when given, a ``codec=5`` record sharing
        this frame's sequence and timestamps is queued before the video record.
        """

        if len(access_unit) > MAX_PACKET_BYTES:
            logger.warning(
                "payload %d > MaxPacketBytes %d; receiver will reject. Lower bitrate/resolution.",
                len(access_unit),
                MAX_PACKET_BYTES,
            )

        with self._lock:
            if self._queue is None:
                return
            if self._awaiting_keyframe:
                if not is_key:
                    return
                self._awaiting_keyframe = False
            params_record = self._take_params_record_locked()
            sequence = self._next_sequence_locked()
            def pose_record_or_none(codec: int, pose: tuple[float, ...] | None) -> bytes | None:
                if pose is None:
                    return None
                pose_header = build_header(
                    codec,
                    WIRE_FLAG_POSE_VALID,
                    sequence,
                    0,
                    0,
                    HEAD_POSE_PAYLOAD_SIZE,
                    wall_ns,
                    sim_ns,
                )
                return pose_header + struct.pack(HEAD_POSE_FORMAT, *pose)

            pose_record = pose_record_or_none(WIRE_CODEC_HEAD_POSE, head_pose)
            base_record = pose_record_or_none(WIRE_CODEC_BASE_POSE, base_pose)
            status_record = None
            if sim_status is not None:
                status_record = (
                    build_header(
                        WIRE_CODEC_SIM_STATUS,
                        0,
                        sequence,
                        0,
                        0,
                        len(sim_status),
                        wall_ns,
                        sim_ns,
                    )
                    + sim_status
                )
            header = build_header(
                WIRE_CODEC_H264,
                WIRE_FLAG_KEYFRAME if is_key else 0,
                sequence,
                width,
                height,
                len(access_unit),
                wall_ns,
                sim_ns,
            )
            self._enqueue_locked(params_record)
            self._enqueue_locked(pose_record)
            self._enqueue_locked(base_record)
            self._enqueue_locked(status_record)
            self._enqueue_locked(header + access_unit)
            drop_client = self._queued_bytes > self._max_queue_bytes

        if drop_client:
            logger.warning(
                "Client fell behind (>%d queued bytes); disconnecting it.", self._max_queue_bytes
            )
            self._detach_client()

    # -- server internals ----------------------------------------------------

    def _take_params_record_locked(self) -> bytes | None:
        """Build the once-per-connection params record, or None if sent/absent."""

        if not self._send_camera_params or self._camera_params_json is None:
            return None
        if self._params_sent:
            return None
        self._params_sent = True
        payload = self._camera_params_json
        width, height = self._camera_params_size
        header = build_header(
            WIRE_CODEC_PARAMS_JSON,
            0,
            self._next_sequence_locked(),
            width,
            height,
            len(payload),
            time.clock_gettime_ns(time.CLOCK_REALTIME),
            0,
        )
        return header + payload

    def _next_sequence_locked(self) -> int:
        sequence = self._sequence
        self._sequence += 1
        return sequence

    def _enqueue_locked(self, record: bytes | None) -> None:
        if record is None or self._queue is None:
            return
        self._queue.put(record)
        self._queued_bytes += len(record)

    def _detach_client(self) -> None:
        """Ask the writer loop to abandon the current connection."""

        with self._lock:
            if self._queue is not None:
                self._queue.put(b"")  # sentinel: writer closes the connection

    def _run(self) -> None:
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
                self._params_sent = False
                self._connected = True
            # A fresh encoder keyframe lets the receiver start decoding at once.
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
            # The protocol is one-way, so inbound readability signals EOF (or
            # stray data to discard). Without this check a silently departed
            # client would only be noticed on the next send, which may never
            # come while frames are gated off.
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
            try:
                record = record_queue.get(timeout=_QUEUE_POLL_SEC)
            except queue.Empty:
                continue
            if record == b"":  # detach sentinel
                return
            try:
                conn.sendall(record)
            except (BrokenPipeError, ConnectionResetError, OSError) as exc:
                logger.info("Client disconnected: %s", exc)
                return
            with self._lock:
                self._queued_bytes -= len(record)
