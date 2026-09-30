"""The 48-byte VSTR header is a fixed contract with the Unreal receiver."""
import struct

import pytest

from vr_link import wire


GOLDEN_HEADER = (
    b"VSTR"  # magic
    + b"\x00\x00\x00\x01"  # version
    + b"\x00\x00\x00\x01"  # codec H264
    + b"\x00\x00\x00\x01"  # flags keyframe
    + b"\x00\x00\x00\x07"  # sequence
    + b"\x00\x00\x05\x00"  # width 1280
    + b"\x00\x00\x01\xe0"  # height 480
    + b"\x00\x00\x00\x64"  # payload 100
    + struct.pack(">I", 1_700_000_000)  # sec
    + struct.pack(">I", 123_456_789)  # nanosec
    + struct.pack(">Q", 42)  # sim_ns
)


def test_build_header_matches_golden_bytes():
    header = wire.build_header(
        wire.WIRE_CODEC_H264, wire.WIRE_FLAG_KEYFRAME, 7, 1280, 480, 100,
        1_700_000_000_123_456_789, 42,
    )
    assert len(header) == wire.HEADER_SIZE == 48
    assert header == GOLDEN_HEADER


def test_parse_header_roundtrip():
    parsed = wire.parse_header(GOLDEN_HEADER + b"payload-bytes")
    assert parsed.codec == wire.WIRE_CODEC_H264
    assert parsed.flags & wire.WIRE_FLAG_KEYFRAME
    assert parsed.sequence == 7
    assert (parsed.width, parsed.height, parsed.payload_size) == (1280, 480, 100)
    assert parsed.wall_ns == 1_700_000_000_123_456_789
    assert parsed.sim_ns == 42


def test_sequence_and_sim_ns_wrap_instead_of_overflowing():
    header = wire.parse_header(wire.build_header(1, 0, 2**32 + 5, 0, 0, 0, 0, 2**64 + 9))
    assert header.sequence == 5
    assert header.sim_ns == 9


@pytest.mark.parametrize(
    "buf",
    [b"", GOLDEN_HEADER[:47], b"XXXX" + GOLDEN_HEADER[4:], GOLDEN_HEADER[:4] + b"\x00\x00\x00\x02" + GOLDEN_HEADER[8:]],
)
def test_parse_header_rejects_bad_input(buf):
    with pytest.raises(wire.WireError):
        wire.parse_header(buf)
