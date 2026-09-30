"""VRL2 framing: golden prefix, stream splitting/resync, Envelope round-trips."""
import struct

import pytest

from vr_link.messages import (
    SCHEMA_VERSION,
    VRL2_HEADER_SIZE,
    VRL2_MAGIC,
    FramingError,
    MessageKind,
    RecordReader,
    SignalsSample,
    SourceKind,
    build_calibration_request_payload,
    build_signals_payload,
    decode_signals,
    frame_payload,
    parse_envelope,
)


def test_frame_prefix_is_magic_then_length_big_endian():
    framed = frame_payload(b"abc")
    assert framed[:4] == b"VRL2"
    assert struct.unpack(">I", framed[4:8]) == (3,)
    assert framed[8:] == b"abc"
    assert VRL2_MAGIC == 0x56524C32 and VRL2_HEADER_SIZE == 8


def test_frame_rejects_empty_and_oversize_payloads():
    with pytest.raises(FramingError):
        frame_payload(b"")
    with pytest.raises(FramingError):
        frame_payload(b"x" * (8 * 1024 * 1024 + 1))


def test_envelope_round_trip_carries_timestamps_and_kind():
    payload = build_signals_payload(SignalsSample(True, SourceKind.MUJOCO), seq=7, wall_ns=123, sim_ns=456)
    env = parse_envelope(payload)
    assert (env.schema_version, env.seq, env.wall_ns, env.sim_ns, env.kind) == (SCHEMA_VERSION, 7, 123, 456, MessageKind.SIGNALS)
    assert decode_signals(env) == SignalsSample(True, SourceKind.MUJOCO)


def test_calibration_request_is_an_empty_body():
    env = parse_envelope(build_calibration_request_payload(seq=1, wall_ns=5))
    assert env.kind is MessageKind.CALIBRATION_REQUEST and env.sim_ns == 0
    with pytest.raises(FramingError):
        decode_signals(env)


def test_parse_rejects_garbage():
    with pytest.raises(FramingError):
        parse_envelope(b"\x00" * 4)
    with pytest.raises(FramingError):
        parse_envelope(b"not an envelope at all, definitely")


def test_record_reader_splits_and_reassembles_partial_chunks():
    payloads = [build_signals_payload(SignalsSample(i % 2 == 0), seq=i, wall_ns=i) for i in range(3)]
    stream = b"".join(frame_payload(p) for p in payloads)
    reader = RecordReader()
    out = []
    for i in range(0, len(stream), 5):
        out.extend(reader.feed(stream[i:i + 5]))
    assert out == payloads and reader.pending_bytes == 0 and reader.framing_errors == 0


def test_record_reader_resyncs_after_garbage_and_bad_length():
    good = frame_payload(build_signals_payload(SignalsSample(), seq=9, wall_ns=9))
    bad_length = struct.pack(">II", VRL2_MAGIC, 0) + b"zz"
    reader = RecordReader()
    out = reader.feed(b"junkVR" + bad_length + b"L2broken" + good + b"VR")
    assert out == [good[8:]]
    assert reader.framing_errors >= 2
    assert reader.pending_bytes == 2  # a possible magic prefix is kept
    out = reader.feed(good[2:])
    assert out == [good[8:]]
