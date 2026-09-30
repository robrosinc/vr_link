"""VRL2 framing and the ``Envelope`` root shared by every downlink message.

Framing (identical in shape to the VRS1 uplink)::

    u32 magic   0x56524C32 ("VRL2"), big-endian
    u32 length  FlatBuffers payload size in bytes, big-endian
    payload     robros.vr.Envelope { schema_version, seq, wall_ns, sim_ns, body }

Builders in the sibling modules produce the *payload*; the TCP server frames
it with :func:`frame_payload`. Receivers split the byte stream with
:class:`RecordReader` and decode each payload with :func:`parse_envelope`.
"""

from __future__ import annotations

import enum
import struct
from dataclasses import dataclass
from typing import List, Optional

import flatbuffers
import flatbuffers.table
import vr_link_generated as _vr_link

VRL2_MAGIC = 0x56524C32  # "VRL2"
VRL2_MAGIC_BYTES = VRL2_MAGIC.to_bytes(4, "big")
VRL2_HEADER_FORMAT = ">II"
VRL2_HEADER_SIZE = struct.calcsize(VRL2_HEADER_FORMAT)  # 8
# Must equal the headset's MaxPacketBytes; a 1280x480 IDR is far below this.
VRL2_MAX_PAYLOAD_BYTES = 8 * 1024 * 1024
SCHEMA_VERSION = 2

_U32 = 0xFFFFFFFF
_U64 = 0xFFFFFFFFFFFFFFFF


class FramingError(ValueError):
    """Bytes are not a valid VRL2 record or Envelope."""


class MessageKind(enum.IntEnum):
    """``Envelope.body`` union discriminator (values are the generated ``Message`` enum)."""

    NONE = _vr_link.Message.NONE
    VIDEO_FRAME = _vr_link.Message.VideoFrame
    CALIBRATION = _vr_link.Message.Calibration
    CALIBRATION_REQUEST = _vr_link.Message.CalibrationRequest
    ROBOT_STATE = _vr_link.Message.RobotState
    SIGNALS = _vr_link.Message.Signals


@dataclass(frozen=True)
class DecodedEnvelope:
    schema_version: int
    seq: int
    wall_ns: int
    sim_ns: int
    kind: MessageKind
    # Union body as a raw table; decode with the matching messages.<kind> module.
    table: Optional[flatbuffers.table.Table]

    def expect(self, kind: MessageKind) -> flatbuffers.table.Table:
        if self.kind is not kind or self.table is None:
            raise FramingError(f"expected {kind.name}, got {self.kind.name}")
        return self.table


def frame_payload(payload: bytes) -> bytes:
    """Prefix ``payload`` with the 8-byte VRL2 header."""

    size = len(payload)
    if size == 0 or size > VRL2_MAX_PAYLOAD_BYTES:
        raise FramingError(f"payload size {size} outside 1..{VRL2_MAX_PAYLOAD_BYTES}")
    return struct.pack(VRL2_HEADER_FORMAT, VRL2_MAGIC, size) + payload


def finish_envelope(
    builder: flatbuffers.Builder, body_offset: int, body_type: int, *, seq: int, wall_ns: int, sim_ns: int = 0
) -> bytes:
    """Wrap an already-built union body in an Envelope and return the payload bytes."""

    _vr_link.EnvelopeStart(builder)
    _vr_link.EnvelopeAddSchemaVersion(builder, SCHEMA_VERSION)
    _vr_link.EnvelopeAddSeq(builder, int(seq) & _U32)
    _vr_link.EnvelopeAddWallNs(builder, int(wall_ns) & _U64)
    _vr_link.EnvelopeAddSimNs(builder, int(sim_ns) & _U64)
    _vr_link.EnvelopeAddBodyType(builder, int(body_type))
    _vr_link.EnvelopeAddBody(builder, body_offset)
    builder.Finish(_vr_link.EnvelopeEnd(builder))
    return bytes(builder.Output())


def parse_envelope(payload: bytes) -> DecodedEnvelope:
    """Decode one Envelope payload (without the 8-byte prefix)."""

    if len(payload) < 8:
        raise FramingError(f"payload too short for an Envelope ({len(payload)} bytes)")
    try:
        env = _vr_link.Envelope.GetRootAs(payload, 0)
        version = env.SchemaVersion()
        body_type = env.BodyType()
        table = env.Body()
        decoded = DecodedEnvelope(version, env.Seq(), env.WallNs(), env.SimNs(), MessageKind(body_type), table)
    except FramingError:
        raise
    except Exception as error:  # struct/index errors from a corrupt buffer, unknown union value
        raise FramingError(f"malformed Envelope: {error}") from error
    if decoded.schema_version != SCHEMA_VERSION:
        raise FramingError(f"schema_version {decoded.schema_version}, expected {SCHEMA_VERSION}")
    return decoded


class RecordReader:
    """Incremental splitter of a VRL2 byte stream into payloads.

    Resynchronises on the magic after garbage or an oversize length and
    counts those events in :attr:`framing_errors`.
    """

    def __init__(self, max_payload_bytes: int = VRL2_MAX_PAYLOAD_BYTES) -> None:
        self._buffer = bytearray()
        self._max_payload = int(max_payload_bytes)
        self.framing_errors = 0

    def reset(self) -> None:
        self._buffer.clear()

    @property
    def pending_bytes(self) -> int:
        return len(self._buffer)

    def feed(self, chunk: bytes) -> List[bytes]:
        self._buffer.extend(chunk)
        records: List[bytes] = []
        while len(self._buffer) >= VRL2_HEADER_SIZE:
            magic, length = struct.unpack_from(VRL2_HEADER_FORMAT, self._buffer, 0)
            if magic != VRL2_MAGIC or length == 0 or length > self._max_payload:
                self.framing_errors += 1
                self._resync()
                continue
            end = VRL2_HEADER_SIZE + length
            if len(self._buffer) < end:
                break
            records.append(bytes(self._buffer[VRL2_HEADER_SIZE:end]))
            del self._buffer[:end]
        return records

    def _resync(self) -> None:
        # Drop bytes up to the next magic candidate; keep a partial tail that
        # might be the start of one.
        index = self._buffer.find(VRL2_MAGIC_BYTES, 1)
        if index < 0:
            keep = 0
            for tail in (3, 2, 1):
                if self._buffer.endswith(VRL2_MAGIC_BYTES[:tail]):
                    keep = tail
                    break
            del self._buffer[: len(self._buffer) - keep]
        else:
            del self._buffer[:index]


def read_table(env: DecodedEnvelope, kind: MessageKind, cls):
    """Instantiate generated table ``cls`` on the union body of ``env``."""

    table = env.expect(kind)
    obj = cls()
    obj.Init(table.Bytes, table.Pos)
    return obj
