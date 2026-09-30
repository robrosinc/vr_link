"""VSTR wire protocol: the 48-byte record header shared by every video sender.

One record on the 9100 downlink = this header + payload. The layout is the
contract of the Unreal ``TCPActor`` / ``FTcpVideoStreamClient`` receiver and
of the legacy ``igris_c_camera`` senders; every sender (Isaac, MuJoCo, USB,
DDS) must produce byte-identical headers, so the packing lives here only.

Header fields, all big-endian::

    u32 magic        "VSTR" (0x56535452)
    u32 version      1
    u32 codec        WIRE_CODEC_* (1 = H.264 access unit, 2 = camera params JSON,
                     3 = head pose, 4 = base pose, 5 = FlatBuffers SimStatus)
    u32 flags        WIRE_FLAG_* (keyframe for codec 1, pose valid for 3/4)
    u32 sequence     per-connection record counter
    u32 width        encoded frame width (0 for non-video records)
    u32 height       encoded frame height (0 for non-video records)
    u32 payload_size bytes following the header
    u32 sec          CLOCK_REALTIME acquisition time, seconds
    u32 nanosec      CLOCK_REALTIME acquisition time, nanoseconds
    u64 sim_ns       simulation time of the frame (0 when the sender has none)

``sim_ns`` is the slot the receiver matches pose/status records against
their video record on. The legacy hardware sender wrote the capture time
into it; here it is reserved for simulation time and is 0 for real cameras.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

import flatbuffers
import vr_link_generated as _vr_link

WIRE_MAGIC = 0x56535452  # "VSTR"
WIRE_VERSION = 1
WIRE_CODEC_H264 = 1
WIRE_CODEC_PARAMS_JSON = 2  # extension: camera-parameter JSON record
WIRE_CODEC_HEAD_POSE = 3  # extension: per-frame rendered head-pose record
WIRE_CODEC_BASE_POSE = 4  # extension: per-frame robot base-pose record (same layout as codec=3)
WIRE_CODEC_SIM_STATUS = 5  # extension: FlatBuffers robros.vr.SimStatus payload
WIRE_FLAG_KEYFRAME = 1 << 0  # codec=1 records
WIRE_FLAG_POSE_VALID = 1 << 0  # codec=3/4 records
HEAD_POSE_FORMAT = ">7f"  # px py pz (m), quat wxyz; big-endian float32
HEAD_POSE_PAYLOAD_SIZE = struct.calcsize(HEAD_POSE_FORMAT)  # 28
HEADER_FORMAT = ">IIIIIIIIIIQ"  # 10x u32 + 1x u64, big-endian
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)  # 48
MAX_PACKET_BYTES = 2 * 1024 * 1024  # receiver's MaxPacketBytes default

_U32 = 0xFFFFFFFF
_U64 = 0xFFFFFFFFFFFFFFFF


class WireError(ValueError):
    """A byte string is not a valid VSTR header."""


@dataclass(frozen=True)
class WireHeader:
    """Decoded 48-byte VSTR header (field order matches :data:`HEADER_FORMAT`)."""

    magic: int
    version: int
    codec: int
    flags: int
    sequence: int
    width: int
    height: int
    payload_size: int
    sec: int
    nanosec: int
    sim_ns: int

    @property
    def wall_ns(self) -> int:
        return self.sec * 1_000_000_000 + self.nanosec


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
        sequence & _U32,
        width,
        height,
        payload_size,
        (wall_ns // 1_000_000_000) & _U32,
        wall_ns % 1_000_000_000,
        sim_ns & _U64,
    )


def parse_header(buf: bytes) -> WireHeader:
    """Decode the first :data:`HEADER_SIZE` bytes of ``buf``.

    Raises :class:`WireError` on a short buffer, a wrong magic or an
    unsupported version.
    """

    if len(buf) < HEADER_SIZE:
        raise WireError(f"need {HEADER_SIZE} header bytes, got {len(buf)}")
    header = WireHeader(*struct.unpack(HEADER_FORMAT, bytes(buf[:HEADER_SIZE])))
    if header.magic != WIRE_MAGIC:
        raise WireError(f"bad magic 0x{header.magic:08X} (expected 0x{WIRE_MAGIC:08X})")
    if header.version != WIRE_VERSION:
        raise WireError(f"unsupported wire version {header.version} (expected {WIRE_VERSION})")
    return header


def build_sim_status_payload(*, recording: bool, timestamp_ns: int) -> bytes:
    """FlatBuffers ``robros.vr.SimStatus`` payload for a ``codec=5`` record.

    State-only (never events), newest-wins on the receiver; the VSTR header
    supplies framing, sequence and timestamps. Contract lives in this
    repository's README ("Downlink status").
    """

    builder = flatbuffers.Builder(64)
    _vr_link.SimStatusStart(builder)
    _vr_link.SimStatusAddTimestampNs(builder, timestamp_ns & _U64)
    _vr_link.SimStatusAddRecording(builder, bool(recording))
    builder.Finish(_vr_link.SimStatusEnd(builder))
    return bytes(builder.Output())
