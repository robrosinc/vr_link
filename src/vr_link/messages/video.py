"""``VideoFrame``: one H.264 Annex-B access unit of the SBS stereo frame."""

from __future__ import annotations

from dataclasses import dataclass

import flatbuffers
import vr_link_generated as _vr_link

from .envelope import DecodedEnvelope, MessageKind, finish_envelope, read_table

CODEC_H264 = 1


@dataclass(frozen=True)
class VideoFrameMsg:
    codec: int
    width: int
    height: int
    is_key: bool
    data: bytes


def build_video_frame_payload(
    access_unit: bytes, *, is_key: bool, width: int, height: int, seq: int, wall_ns: int, sim_ns: int = 0
) -> bytes:
    if width <= 0 or height <= 0:
        raise ValueError("video frame size must be positive")
    if not access_unit:
        raise ValueError("empty access unit")
    builder = flatbuffers.Builder(len(access_unit) + 128)
    data = builder.CreateByteVector(access_unit)
    _vr_link.VideoFrameStart(builder)
    _vr_link.VideoFrameAddCodec(builder, CODEC_H264)
    _vr_link.VideoFrameAddWidth(builder, int(width))
    _vr_link.VideoFrameAddHeight(builder, int(height))
    _vr_link.VideoFrameAddIsKey(builder, bool(is_key))
    _vr_link.VideoFrameAddData(builder, data)
    body = _vr_link.VideoFrameEnd(builder)
    return finish_envelope(builder, body, _vr_link.Message.VideoFrame, seq=seq, wall_ns=wall_ns, sim_ns=sim_ns)


def decode_video_frame(env: DecodedEnvelope) -> VideoFrameMsg:
    frame = read_table(env, MessageKind.VIDEO_FRAME, _vr_link.VideoFrame)
    data = frame.DataAsNumpy().tobytes() if frame.DataLength() else b""
    return VideoFrameMsg(frame.Codec(), frame.Width(), frame.Height(), bool(frame.IsKey()), data)
