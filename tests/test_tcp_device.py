"""VRS1 (FlatBuffers) wire-protocol regression for vr_link.tcp_device.

Runs with only ``flatbuffers`` and ``pytest`` installed. Several tests slice
framed records at fixed byte offsets ([500:], [:100], [:200]); they rely on
``build_vr_state_record`` producing the same layout, which
``test_record_without_hands_is_byte_identical`` guards.
"""

from __future__ import annotations

import inspect
import socket
import struct
import time

import pytest

import vr_link.tcp_device as tcp
from vr_link import types

JOINT_COUNT = len(types.BODY_JOINT_NAMES)
ALL_VALID_MASK = (1 << (JOINT_COUNT + 1)) - 1  # 24 body joints + head bit


def _joints(offset: float = 0.0):
    return [((offset + 0.01 * index, 0.2, 0.3), (1.0, 0.0, 0.0, 0.0)) for index in range(JOINT_COUNT)]


def _vrs1_record(offset: float = 0.0, grip: float = 0.0) -> bytes:
    return tcp.build_vr_state_record(
        _joints(offset),
        valid_mask=ALL_VALID_MASK,
        timestamp_ns=42,
        head=((0.0, 0.0, 1.7), (1.0, 0.0, 0.0, 0.0)),
        left_input=types.ControllerInput(trigger=0.25, thumbstick=(0.1, -0.2)),
        right_input=types.ControllerInput(grip=grip, button_first=1.0),
    )


def _extract(data: bytes):
    buffer = bytearray(data)
    return tcp.TcpVRDevice._extract_newest_payload(buffer), buffer


def _swap_wx(joint_name: str, quat):
    """Trivial adapter for tests: swaps w and x so the effect is visible."""

    w, x, y, z = quat
    return (x, w, y, z)


def test_round_trip_fills_poses_and_controller_inputs():
    payload, _ = _extract(_vrs1_record(grip=0.87))
    assert payload is not None

    state = tcp.decode_vr_state_payload(payload, timestamp=1.0)
    assert state is not None and state.connected
    assert len(state.body_joints) == JOINT_COUNT
    assert all(pose.valid for pose in state.body_joints)
    assert state.body_joints[3].position_m[0] == pytest.approx(0.03, abs=1e-6)
    assert state.right_input.grip == pytest.approx(0.87, abs=1e-6)
    assert state.right_input.button_first == pytest.approx(1.0)
    assert state.left_input.trigger == pytest.approx(0.25, abs=1e-6)
    assert state.left_input.thumbstick == pytest.approx((0.1, -0.2), abs=1e-6)
    assert state.left_input.tracked is False and state.right_input.tracked is False
    assert state.head.valid and state.head.position_m[2] == pytest.approx(1.7, abs=1e-6)


def test_controller_tracked_round_trips_and_is_omitted_when_false():
    # A record with both controllers untracked carries no `tracked` field, so a
    # 2.0 reader sees the same bytes and a 2.1 reader gets the default.
    untracked = tcp.build_vr_state_record(_joints(), valid_mask=ALL_VALID_MASK, timestamp_ns=1)
    held = tcp.build_vr_state_record(
        _joints(), valid_mask=ALL_VALID_MASK, timestamp_ns=1, left_input=types.ControllerInput(tracked=True)
    )
    assert len(held) > len(untracked)
    state = tcp.decode_vr_state_payload(_extract(untracked)[0], timestamp=1.0)
    assert state.left_input.tracked is False and state.right_input.tracked is False
    state = tcp.decode_vr_state_payload(_extract(held)[0], timestamp=1.0)
    assert state.left_input.tracked is True and state.right_input.tracked is False


def test_record_toggle_count_round_trips_and_defaults_to_zero():
    # Explicit counter round-trips.
    record = tcp.build_vr_state_record(
        _joints(),
        valid_mask=ALL_VALID_MASK,
        timestamp_ns=42,
        record_toggle_count=7,
    )
    payload, _ = _extract(record)
    state = tcp.decode_vr_state_payload(payload, timestamp=1.0)
    assert state is not None and state.record_toggle_count == 7

    # A pre-field record (builder default) decodes to the schema default 0.
    payload, _ = _extract(_vrs1_record())
    state = tcp.decode_vr_state_payload(payload, timestamp=1.0)
    assert state is not None and state.record_toggle_count == 0


def test_default_decode_keeps_wire_quaternions_raw():
    """Without an adapter the receiver returns exactly what the sender put on the wire."""

    joints = _joints()
    joints[0] = (joints[0][0], (0.0, 0.0, 1.0, 0.0))
    record = tcp.build_vr_state_record(joints, valid_mask=ALL_VALID_MASK, timestamp_ns=42)
    payload, _ = _extract(record)
    state = tcp.decode_vr_state_payload(payload, timestamp=1.0)
    assert state.body_joints[0].quat_wxyz == pytest.approx((0.0, 0.0, 1.0, 0.0), abs=1e-6)
    assert state.body_joints[1].quat_wxyz == pytest.approx((1.0, 0.0, 0.0, 0.0), abs=1e-6)
    assert state.head.quat_wxyz == pytest.approx((1.0, 0.0, 0.0, 0.0), abs=1e-6)


def test_body_joint_adapter_receives_joint_names_and_maps_body_only():
    calls = []

    def adapter(joint_name, quat):
        calls.append((joint_name, quat))
        return _swap_wx(joint_name, quat)

    payload, _ = _extract(_vrs1_record())
    state = tcp.decode_vr_state_payload(payload, timestamp=1.0, body_joint_adapter=adapter)
    assert [name for name, _ in calls] == list(types.BODY_JOINT_NAMES)
    assert all(quat == pytest.approx((1.0, 0.0, 0.0, 0.0), abs=1e-6) for _, quat in calls)
    assert all(pose.quat_wxyz == pytest.approx((0.0, 1.0, 0.0, 0.0), abs=1e-6) for pose in state.body_joints)
    # Head is never adapted.
    assert state.head.quat_wxyz == pytest.approx((1.0, 0.0, 0.0, 0.0), abs=1e-6)


def test_newest_record_wins():
    payload, buffer = _extract(_vrs1_record(offset=0.1) + _vrs1_record(offset=0.2))
    assert not buffer
    state = tcp.decode_vr_state_payload(payload, timestamp=1.0)
    assert state.body_joints[0].position_m[0] == pytest.approx(0.2, abs=1e-6)


def test_resynchronizes_after_mid_stream_join():
    """A receiver attaching mid-record sees a record tail before the first magic."""

    tail = _vrs1_record(offset=0.9)[500:]
    payload, _ = _extract(tail + _vrs1_record(offset=0.3))
    assert payload is not None
    state = tcp.decode_vr_state_payload(payload, timestamp=1.0)
    assert state.body_joints[0].position_m[0] == pytest.approx(0.3, abs=1e-6)


def test_stream_recovers_after_boundary_spanning_corruption():
    """A stale magic makes one boundary-spanning corrupt payload, then recovery.

    Length-prefixed framing cannot detect a truncated record whose header
    survived: the extractor consumes bytes spanning into the next record and
    the decode of that payload fails (handled as connected=False upstream).
    The records that follow must decode normally again.
    """

    buffer = bytearray(_vrs1_record(offset=0.9)[:100] + _vrs1_record(offset=0.3))
    corrupt = tcp.TcpVRDevice._extract_newest_payload(buffer)
    assert corrupt is not None
    assert tcp.decode_vr_state_payload(corrupt, timestamp=1.0) is None

    buffer.extend(_vrs1_record(offset=0.4))
    recovered = tcp.TcpVRDevice._extract_newest_payload(buffer)
    assert recovered is not None
    state = tcp.decode_vr_state_payload(recovered, timestamp=1.0)
    assert state is not None
    assert state.body_joints[0].position_m[0] == pytest.approx(0.4, abs=1e-6)


def test_partial_record_completes_across_chunks():
    record = _vrs1_record(offset=0.6)
    buffer = bytearray(record[:200])
    assert tcp.TcpVRDevice._extract_newest_payload(buffer) is None
    buffer.extend(record[200:])
    payload = tcp.TcpVRDevice._extract_newest_payload(buffer)
    assert payload is not None
    state = tcp.decode_vr_state_payload(payload, timestamp=1.0)
    assert state.body_joints[0].position_m[0] == pytest.approx(0.6, abs=1e-6)


def test_oversized_length_field_is_skipped():
    bogus = struct.pack(tcp.VRS1_HEADER_FORMAT, tcp.VRS1_MAGIC, 2**24) + b"x" * 64
    payload, _ = _extract(bogus + _vrs1_record(offset=0.7))
    assert payload is not None
    state = tcp.decode_vr_state_payload(payload, timestamp=1.0)
    assert state.body_joints[0].position_m[0] == pytest.approx(0.7, abs=1e-6)


def test_undecodable_payload_returns_none():
    assert tcp.decode_vr_state_payload(b"\x01\x02\x03", timestamp=1.0) is None


def _hand(tracked: bool = True, keypoint_count: int | None = None, x_step: float = 0.01):
    count = types.HAND_KEYPOINT_COUNT if keypoint_count is None else keypoint_count
    keypoints = tuple(
        types.TrackedPose(
            position_m=(x_step * index, 0.0, 0.0),
            # Wrist-relative: entry 1 (the wrist itself) is identity.
            quat_wxyz=(1.0, 0.0, 0.0, 0.0),
            valid=True,
        )
        for index in range(count)
    )
    return types.HandTrackingState(
        tracked=tracked,
        wrist=types.TrackedPose(position_m=(0.1, 0.2, 1.0), quat_wxyz=(0.0, 0.0, 1.0, 0.0), valid=True),
        keypoints=keypoints,
    )


def test_hands_absent_decode_to_none():
    payload, _ = _extract(_vrs1_record())
    state = tcp.decode_vr_state_payload(payload, timestamp=1.0)
    assert state.left_hand is None
    assert state.right_hand is None


def test_record_without_hands_is_byte_identical():
    """Guards the byte-offset-sensitive resync/corruption tests above."""

    with_none = tcp.build_vr_state_record(
        _joints(), valid_mask=ALL_VALID_MASK, timestamp_ns=42, left_hand=None, right_hand=None
    )
    without = tcp.build_vr_state_record(_joints(), valid_mask=ALL_VALID_MASK, timestamp_ns=42)
    assert with_none == without


def test_hand_round_trip():
    record = tcp.build_vr_state_record(
        _joints(),
        valid_mask=ALL_VALID_MASK,
        timestamp_ns=42,
        left_hand=_hand(x_step=0.01),
        right_hand=_hand(x_step=0.02),
    )
    payload, _ = _extract(record)
    # Even with an adapter, hand poses stay raw.
    state = tcp.decode_vr_state_payload(payload, timestamp=1.0, body_joint_adapter=_swap_wx)

    for hand, x_step in ((state.left_hand, 0.01), (state.right_hand, 0.02)):
        assert hand is not None and hand.tracked
        assert hand.wrist.valid
        assert hand.wrist.position_m == pytest.approx((0.1, 0.2, 1.0), abs=1e-6)
        assert hand.wrist.quat_wxyz == pytest.approx((0.0, 0.0, 1.0, 0.0), abs=1e-6)
        assert len(hand.keypoints) == types.HAND_KEYPOINT_COUNT
        assert hand.keypoints[1].position_m == pytest.approx((x_step, 0.0, 0.0), abs=1e-6)
        assert hand.keypoints[1].quat_wxyz == pytest.approx((1.0, 0.0, 0.0, 0.0), abs=1e-6)
        assert hand.keypoints[25].position_m[0] == pytest.approx(25 * x_step, abs=1e-6)
        assert all(keypoint.valid for keypoint in hand.keypoints)


def test_untracked_hand_round_trip():
    record = tcp.build_vr_state_record(
        _joints(), valid_mask=ALL_VALID_MASK, timestamp_ns=42, left_hand=_hand(tracked=False)
    )
    payload, _ = _extract(record)
    state = tcp.decode_vr_state_payload(payload, timestamp=1.0)
    assert state.left_hand is not None and not state.left_hand.tracked
    assert state.right_hand is None


def test_malformed_keypoint_count_degrades_to_untracked():
    """A bad hand table must not drop the record: body/head stay usable."""

    record = tcp.build_vr_state_record(
        _joints(offset=0.5), valid_mask=ALL_VALID_MASK, timestamp_ns=42, left_hand=_hand(keypoint_count=5)
    )
    payload, _ = _extract(record)
    state = tcp.decode_vr_state_payload(payload, timestamp=1.0)
    assert state is not None and state.connected
    assert state.body_joints[0].position_m[0] == pytest.approx(0.5, abs=1e-6)
    assert state.left_hand is not None and not state.left_hand.tracked
    assert state.left_hand.keypoints == ()


def test_default_port_is_9102():
    assert inspect.signature(tcp.TcpVRDevice).parameters["port"].default == 9102


def _receive_one(vr_device: tcp.TcpVRDevice, record: bytes):
    vr_device.start()
    try:
        port = vr_device._server_socket.getsockname()[1]
        with socket.create_connection(("127.0.0.1", port), timeout=5.0) as conn:
            conn.sendall(record)
            deadline = time.monotonic() + 5.0
            state = None
            while time.monotonic() < deadline:
                state = vr_device.advance(0.0)
                if state.connected:
                    break
                time.sleep(0.01)
            return state
    finally:
        vr_device.close()


def test_device_receives_vrs1_over_socket():
    """End-to-end: real socket, receive thread, advance() sees the input."""

    state = _receive_one(tcp.TcpVRDevice(host="127.0.0.1", port=0), _vrs1_record(grip=0.6))
    assert state is not None and state.connected
    assert state.right_input.grip == pytest.approx(0.6, abs=1e-6)
    assert state.body_joints[0].quat_wxyz == pytest.approx((1.0, 0.0, 0.0, 0.0), abs=1e-6)


def test_device_passes_adapter_through_advance():
    state = _receive_one(
        tcp.TcpVRDevice(host="127.0.0.1", port=0, body_joint_adapter=_swap_wx), _vrs1_record()
    )
    assert state is not None and state.connected
    assert state.body_joints[0].quat_wxyz == pytest.approx((0.0, 1.0, 0.0, 0.0), abs=1e-6)
    assert state.head.quat_wxyz == pytest.approx((1.0, 0.0, 0.0, 0.0), abs=1e-6)


def test_teleop_state_round_trips_and_defaults_to_unknown():
    assert tcp.decode_vr_state_payload(_vrs1_record()[8:], timestamp=0.0).teleop_state is types.TeleopState.UNKNOWN
    record = tcp.build_vr_state_record(
        _joints(), valid_mask=ALL_VALID_MASK, timestamp_ns=1, teleop_state=types.TeleopState.RESUMING
    )
    assert tcp.decode_vr_state_payload(record[8:], timestamp=0.0).teleop_state is types.TeleopState.RESUMING
    # UNKNOWN leaves the field at its default: byte-identical to a pre-2.1 record.
    assert tcp.build_vr_state_record(_joints(), valid_mask=ALL_VALID_MASK, timestamp_ns=1, teleop_state=types.TeleopState.UNKNOWN) == \
        tcp.build_vr_state_record(_joints(), valid_mask=ALL_VALID_MASK, timestamp_ns=1)
    assert types.TeleopState.from_wire(99) is types.TeleopState.UNKNOWN
    assert [s.value for s in types.TeleopState] == [0, 1, 2, 3, 4, 5]
