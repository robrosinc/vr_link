"""VR device backed by tracking data received over a raw TCP socket.

Pose counterpart of the VSTR video channel: the PC is the TCP server and the
headset (Unreal) is the client, mirroring the video channel so the headset
initiates both connections. The pose channel is a *separate* socket (default
port 9102; 9100 is video, 9101 the real-robot calibration channel) rather than
a lane inside the video socket, so small latency-critical pose packets never
queue behind large video records.

Wire contract, "VRS1" record (big-endian prefix + FlatBuffers payload):

    magic        u32  = 0x56525331 ("VRS1")
    length       u32  = payload size in bytes (records above 64 KiB dropped)
    payload           = ``robros.vr.VRState`` FlatBuffers message

The VRState schema (``vr_link.fbs``) and this module live in the
``vr_link_schema`` repository; ``vr_link_generated`` is the flatc binding
committed beside it (regenerated via ``scripts/generate.sh``). The record
carries the 24-joint skeleton, the raw HMD pose, both controllers' input state
(triggers, grips, buttons, thumbsticks) that fills
``VRInputState.left_input``/``right_input``, and the optional per-hand OpenXR
hand-tracking tables (``left_hand``/``right_hand``: tracked flag, hand-tracking
wrist, 26 wrist-relative keypoints) that fill
``VRInputState.left_hand``/``right_hand`` - absent on sender builds without
hand tracking.

Body joints pass through the optional per-joint ``body_joint_adapter`` the
consumer supplies (``None`` = raw wire quaternions, as documented in the README
"Data conventions"). This is where a retargeting consumer maps the sender's
bone-frame convention onto its own (hit-bench binds GMR's
``general_motion_retargeting.vr_uplink_convention.to_xrobot_convention``).
The head and hand poses are never adapted: the head feeds the separate
head-teleoperation logic and the hands are wrist-relative already.

Coordinate convention: the sender (Unreal) converts poses into the simulation
world frame BEFORE transmission - Z-up right-handed, meters, wxyz quaternions.
If the convention ever changes, the decode functions in this module are the
single seam to adapt on the PC side.

The decode/build helpers are module-level and transport-agnostic: a future
UDP device can reuse them unchanged (one record per datagram, no
resynchronization needed).
"""

from __future__ import annotations

import logging
import select
import socket
import struct
import threading
import time
from collections.abc import Callable

import flatbuffers
import vr_link_generated as _vr_link

from .types import (
    BODY_JOINT_NAMES,
    HAND_KEYPOINT_COUNT,
    ControllerInput,
    HandTrackingState,
    TrackedPose,
    VRDevice,
    VRInputState,
)

logger = logging.getLogger(__name__)

_ACCEPT_POLL_SEC = 0.5
_RECV_CHUNK_BYTES = 4096

BODY_JOINT_COUNT = len(BODY_JOINT_NAMES)  # 24 (body joints only)
HEAD_VALID_BIT = BODY_JOINT_COUNT  # valid_mask bit 24 in build_vr_state_record

VRS1_MAGIC = 0x56525331  # "VRS1"
VRS1_HEADER_FORMAT = ">II"  # magic, payload length
VRS1_HEADER_SIZE = struct.calcsize(VRS1_HEADER_FORMAT)  # 8
# Sanity cap on the length field: a genuine VRState is well under 4 KiB even
# with both hand-tracking tables (2 x 27 x 32 B of poses on top of the ~2 KiB
# base record), so anything larger means a corrupt or hostile length and the
# magic is skipped.
VRS1_MAX_PAYLOAD_BYTES = 64 * 1024
_VRS1_MAGIC_BYTES = struct.pack(">I", VRS1_MAGIC)
# Longest magic prefix worth keeping when no full magic is found in the buffer.
_MAX_MAGIC_PREFIX = len(_VRS1_MAGIC_BYTES) - 1

Pose = tuple[tuple[float, float, float], tuple[float, float, float, float]]
Quat = tuple[float, float, float, float]
# Per-body-joint orientation hook: (schema joint name, wire quaternion wxyz)
# -> quaternion wxyz. Called once per body joint in BODY_JOINT_NAMES order.
BodyJointAdapter = Callable[[str, Quat], Quat]


def _append_controller_input(builder: flatbuffers.Builder, controller: ControllerInput) -> int:
    """Append one ControllerInput table to the builder and return its offset."""

    _vr_link.ControllerInputStart(builder)
    _vr_link.ControllerInputAddTrigger(builder, controller.trigger)
    _vr_link.ControllerInputAddGrip(builder, controller.grip)
    _vr_link.ControllerInputAddButtonFirst(builder, controller.button_first)
    _vr_link.ControllerInputAddButtonSecond(builder, controller.button_second)
    _vr_link.ControllerInputAddButtonThumbstick(builder, controller.button_thumbstick)
    _vr_link.ControllerInputAddThumbstickX(builder, controller.thumbstick[0])
    _vr_link.ControllerInputAddThumbstickY(builder, controller.thumbstick[1])
    return _vr_link.ControllerInputEnd(builder)


def _append_hand_state(builder: flatbuffers.Builder, hand: HandTrackingState) -> int:
    """Append one HandState table to the builder and return its offset.

    The keypoint count is taken as given (not validated) so tests can build
    malformed records and exercise the decode-side degradation.
    """

    _vr_link.HandStateStartKeypointsVector(builder, len(hand.keypoints))
    for keypoint in reversed(hand.keypoints):
        _vr_link.CreatePose(builder, *keypoint.position_m, *keypoint.quat_wxyz, keypoint.valid)
    keypoints_offset = builder.EndVector()

    _vr_link.HandStateStart(builder)
    _vr_link.HandStateAddTracked(builder, hand.tracked)
    # Struct fields must be created inline while their table is being built.
    _vr_link.HandStateAddWrist(
        builder,
        _vr_link.CreatePose(builder, *hand.wrist.position_m, *hand.wrist.quat_wxyz, hand.wrist.valid),
    )
    _vr_link.HandStateAddKeypoints(builder, keypoints_offset)
    return _vr_link.HandStateEnd(builder)


def build_vr_state_record(
    joints: list[Pose],
    valid_mask: int,
    timestamp_ns: int,
    head: Pose = ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)),
    left_input: ControllerInput = ControllerInput(),
    right_input: ControllerInput = ControllerInput(),
    left_hand: HandTrackingState | None = None,
    right_hand: HandTrackingState | None = None,
    record_toggle_count: int = 0,
) -> bytes:
    """Pack one framed VRS1 record (used by tests and any Python-side sender).

    ``valid_mask`` bit i marks body joint i valid; bit 24 marks the head
    slot valid. When ``left_hand``/``right_hand`` are None the hand tables are
    absent from the record (byte-identical to a pre-hand-tracking record).
    """

    if len(joints) != BODY_JOINT_COUNT:
        raise ValueError(f"Expected {BODY_JOINT_COUNT} body joints, got {len(joints)}.")

    builder = flatbuffers.Builder(2048)
    left_offset = _append_controller_input(builder, left_input)
    right_offset = _append_controller_input(builder, right_input)
    left_hand_offset = None if left_hand is None else _append_hand_state(builder, left_hand)
    right_hand_offset = None if right_hand is None else _append_hand_state(builder, right_hand)

    # Vectors of structs are written inline, last element first.
    _vr_link.VRStateStartBodyJointsVector(builder, BODY_JOINT_COUNT)
    for index in reversed(range(BODY_JOINT_COUNT)):
        position, quat = joints[index]
        _vr_link.CreatePose(builder, *position, *quat, bool(valid_mask & (1 << index)))
    joints_offset = builder.EndVector()

    _vr_link.VRStateStart(builder)
    _vr_link.VRStateAddTimestampNs(builder, timestamp_ns & 0xFFFFFFFFFFFFFFFF)
    _vr_link.VRStateAddRecordToggleCount(builder, record_toggle_count & 0xFFFFFFFF)
    _vr_link.VRStateAddBodyJoints(builder, joints_offset)
    # Struct fields must be created inline while their table is being built.
    head_position, head_quat = head
    _vr_link.VRStateAddHead(
        builder,
        _vr_link.CreatePose(
            builder,
            *head_position,
            *head_quat,
            bool(valid_mask & (1 << HEAD_VALID_BIT)),
        ),
    )
    _vr_link.VRStateAddLeftInput(builder, left_offset)
    _vr_link.VRStateAddRightInput(builder, right_offset)
    if left_hand_offset is not None:
        _vr_link.VRStateAddLeftHand(builder, left_hand_offset)
    if right_hand_offset is not None:
        _vr_link.VRStateAddRightHand(builder, right_hand_offset)
    builder.Finish(_vr_link.VRStateEnd(builder))

    payload = bytes(builder.Output())
    return struct.pack(VRS1_HEADER_FORMAT, VRS1_MAGIC, len(payload)) + payload


def _decoded_tracked_pose(pose, *, quat_wxyz: Quat) -> TrackedPose:
    return TrackedPose(
        position_m=(pose.Px(), pose.Py(), pose.Pz()),
        quat_wxyz=quat_wxyz,
        valid=bool(pose.Valid()),
    )


def _decoded_controller_input(controller) -> ControllerInput:
    """ControllerInput dataclass from a decoded table (defaults when absent)."""

    if controller is None:
        return ControllerInput()
    return ControllerInput(
        trigger=controller.Trigger(),
        grip=controller.Grip(),
        button_first=controller.ButtonFirst(),
        button_second=controller.ButtonSecond(),
        button_thumbstick=controller.ButtonThumbstick(),
        thumbstick=(controller.ThumbstickX(), controller.ThumbstickY()),
    )


def _decoded_hand_state(hand) -> HandTrackingState | None:
    """HandTrackingState from a decoded HandState table (None when absent).

    A present-but-malformed hand (keypoint count != 26) degrades to
    tracked=False instead of dropping the whole record: the body, head, and
    controller slots in the same record are independently valid, and an
    untracked hand already means "hold the previous finger targets" downstream.

    Hand poses are delivered raw (sim-world wrist, wrist-frame keypoints); the
    body-joint adapter applies to body joints only.
    """

    if hand is None:
        return None
    if hand.KeypointsLength() != HAND_KEYPOINT_COUNT:
        logger.warning(
            "HandState with %d keypoints (expected %d); treating as untracked.",
            hand.KeypointsLength(),
            HAND_KEYPOINT_COUNT,
        )
        return HandTrackingState(tracked=False)

    wrist_pose = hand.Wrist()
    if wrist_pose is None:
        wrist = TrackedPose()
    else:
        wrist = _decoded_tracked_pose(
            wrist_pose,
            quat_wxyz=(wrist_pose.Qw(), wrist_pose.Qx(), wrist_pose.Qy(), wrist_pose.Qz()),
        )

    keypoints = []
    for index in range(HAND_KEYPOINT_COUNT):
        pose = hand.Keypoints(index)
        keypoints.append(
            _decoded_tracked_pose(pose, quat_wxyz=(pose.Qw(), pose.Qx(), pose.Qy(), pose.Qz()))
        )

    return HandTrackingState(tracked=bool(hand.Tracked()), wrist=wrist, keypoints=tuple(keypoints))


def decode_vr_state_payload(
    payload: bytes,
    *,
    timestamp: float,
    body_joint_adapter: BodyJointAdapter | None = None,
) -> VRInputState | None:
    """Decode one VRS1 FlatBuffers payload into a :class:`VRInputState`.

    Body joints pass through ``body_joint_adapter`` when given (called with the
    schema joint name and the wire wxyz quaternion, in BODY_JOINT_NAMES order);
    with ``None`` they are returned raw. The head slot is always raw. Returns
    ``None`` on any malformed payload: the Python flatbuffers runtime has no
    verifier API (unlike C++), so a corrupt buffer surfaces as an arbitrary
    decoding exception and the record is dropped while the stream
    resynchronizes on the next magic. An adapter exception drops the record
    the same way.
    """

    try:
        state = _vr_link.VRState.GetRootAs(payload, 0)
        if state.BodyJointsLength() != BODY_JOINT_COUNT:
            logger.warning(
                "Dropping VRS1 record with %d body joints (expected %d).",
                state.BodyJointsLength(),
                BODY_JOINT_COUNT,
            )
            return None

        body_joints = []
        for index in range(BODY_JOINT_COUNT):
            pose = state.BodyJoints(index)
            received_quat = (pose.Qw(), pose.Qx(), pose.Qy(), pose.Qz())
            quat = (
                received_quat
                if body_joint_adapter is None
                else body_joint_adapter(BODY_JOINT_NAMES[index], received_quat)
            )
            body_joints.append(_decoded_tracked_pose(pose, quat_wxyz=quat))

        head_pose = state.Head()
        if head_pose is None:
            head = TrackedPose()
        else:
            head = _decoded_tracked_pose(
                head_pose,
                quat_wxyz=(head_pose.Qw(), head_pose.Qx(), head_pose.Qy(), head_pose.Qz()),
            )

        return VRInputState(
            timestamp=timestamp,
            head=head,
            body_joints=tuple(body_joints),
            left_input=_decoded_controller_input(state.LeftInput()),
            right_input=_decoded_controller_input(state.RightInput()),
            left_hand=_decoded_hand_state(state.LeftHand()),
            right_hand=_decoded_hand_state(state.RightHand()),
            connected=True,
            record_toggle_count=state.RecordToggleCount(),
        )
    except Exception:  # noqa: BLE001 - unverifiable buffer, see docstring
        logger.warning("Dropping undecodable VRS1 record.", exc_info=True)
        return None


class TcpVRDevice(VRDevice):
    """Receives the device-fused 24-joint skeleton over a dedicated TCP socket.

    A daemon thread accepts one client at a time, reassembles VRS1 records
    from the byte stream (resynchronizing on the magic after partial or
    corrupt data), and keeps only the newest payload. ``advance()`` never
    blocks: it decodes the newest payload into a :class:`VRInputState`
    (through ``body_joint_adapter`` when given), reporting ``connected=False``
    when no client is attached or no record arrived within ``stale_timeout``
    seconds.
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 9102,
        stale_timeout: float = 0.5,
        body_joint_adapter: BodyJointAdapter | None = None,
    ) -> None:
        self._host = host
        self._port = port
        self._stale_timeout = stale_timeout
        self._body_joint_adapter = body_joint_adapter

        self._lock = threading.Lock()
        self._latest_payload: bytes | None = None  # newest VRS1 payload
        self._latest_recv_time: float = 0.0
        self._client_attached = False

        self._stop_requested = threading.Event()
        self._thread: threading.Thread | None = None
        self._server_socket: socket.socket | None = None

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        """Bind the listening socket and start the receive thread."""

        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((self._host, self._port))
        server.listen(1)
        server.settimeout(_ACCEPT_POLL_SEC)
        self._server_socket = server
        self._thread = threading.Thread(target=self._run, name="tcp-vr-device", daemon=True)
        self._thread.start()
        logger.info("TCP VR device listening on %s:%d", self._host, self._port)

    def close(self) -> None:
        """Stop the receive thread and close the listening socket."""

        self._stop_requested.set()
        if self._thread is not None:
            self._thread.join(timeout=10.0)
        if self._server_socket is not None:
            self._server_socket.close()

    # -- VRDevice interface --------------------------------------------------

    @property
    def receiving(self) -> bool:
        """Whether a client is attached and a record arrived recently."""

        now = time.monotonic()
        with self._lock:
            return (
                self._client_attached
                and self._latest_payload is not None
                and (now - self._latest_recv_time) <= self._stale_timeout
            )

    def advance(self, dt: float) -> VRInputState:
        del dt  # the device is purely event-driven; dt is unused
        now = time.monotonic()
        with self._lock:
            payload = self._latest_payload
            fresh = self._client_attached and (now - self._latest_recv_time) <= self._stale_timeout

        if payload is None or not fresh:
            return VRInputState(timestamp=now, connected=False)
        state = decode_vr_state_payload(payload, timestamp=now, body_joint_adapter=self._body_joint_adapter)
        if state is None:
            return VRInputState(timestamp=now, connected=False)
        return state

    def reset(self) -> None:
        with self._lock:
            self._latest_payload = None
            self._latest_recv_time = 0.0

    # -- receive internals ---------------------------------------------------

    def _run(self) -> None:
        while not self._stop_requested.is_set():
            try:
                conn, addr = self._server_socket.accept()
            except TimeoutError:
                continue
            except OSError:
                break

            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            logger.info("VR pose client connected from %s:%d", addr[0], addr[1])
            with self._lock:
                self._client_attached = True

            try:
                self._receive_loop(conn)
            finally:
                with self._lock:
                    self._client_attached = False
                    self._latest_payload = None
                conn.close()
                logger.info("VR pose connection closed. Waiting for next client...")

    def _receive_loop(self, conn: socket.socket) -> None:
        buffer = bytearray()
        while not self._stop_requested.is_set():
            readable, _, _ = select.select([conn], [], [], _ACCEPT_POLL_SEC)
            if not readable:
                continue
            try:
                chunk = conn.recv(_RECV_CHUNK_BYTES)
            except OSError:
                logger.info("VR pose connection lost.")
                return
            if not chunk:
                logger.info("VR pose client closed the connection.")
                return

            buffer.extend(chunk)
            newest = self._extract_newest_payload(buffer)
            if newest is not None:
                with self._lock:
                    self._latest_payload = newest
                    self._latest_recv_time = time.monotonic()

    @staticmethod
    def _extract_newest_payload(buffer: bytearray) -> bytes | None:
        """Consume complete VRS1 records from ``buffer``, return the newest payload.

        Resynchronizes on the magic after garbage or a mid-record join, and
        leaves any trailing partial record in the buffer. The returned bytes
        are the FlatBuffers payload with the 8-byte prefix stripped.
        """

        newest: bytes | None = None
        while True:
            start = buffer.find(_VRS1_MAGIC_BYTES)
            if start < 0:
                # No magic anywhere: keep only a potential magic prefix.
                del buffer[: max(0, len(buffer) - _MAX_MAGIC_PREFIX)]
                return newest
            if start > 0:
                del buffer[:start]
            if len(buffer) < VRS1_HEADER_SIZE:
                return newest

            _magic, payload_size = struct.unpack(VRS1_HEADER_FORMAT, bytes(buffer[:VRS1_HEADER_SIZE]))
            if payload_size == 0 or payload_size > VRS1_MAX_PAYLOAD_BYTES:
                logger.warning(
                    "Dropping VRS1 record with implausible payload size %d.",
                    payload_size,
                )
                del buffer[: len(_VRS1_MAGIC_BYTES)]  # resync on the next magic
                continue
            record_size = VRS1_HEADER_SIZE + payload_size
            if len(buffer) < record_size:
                return newest

            newest = bytes(buffer[VRS1_HEADER_SIZE:record_size])
            del buffer[:record_size]
