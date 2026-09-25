"""Downlink framing and keyframe gating stay independent of the simulator."""
import queue
import struct

import vr_link_generated as wire
from vr_link.stream_server import (
    HEADER_FORMAT, HEADER_SIZE, VstrTcpStreamServer, build_sim_status_payload,
)
from vr_link.h264_encoder import _AnnexBSplitter


def test_status_payload_roundtrip():
    payload = build_sim_status_payload(recording=True, timestamp_ns=123456789)
    status = wire.SimStatus.GetRootAs(payload, 0)
    assert status.Recording()
    assert status.TimestampNs() == 123456789


def test_pose_and_video_share_sequence_and_timestamps_after_keyframe():
    server = VstrTcpStreamServer(send_camera_params=False)
    server._queue = queue.Queue()
    kwargs = dict(width=640, height=480, wall_ns=2_000_000_003, sim_ns=42)
    server.send_video(b'delta', is_key=False, **kwargs)
    assert server._queue.empty()
    pose = (1., 2., 3., 1., 0., 0., 0.)
    server.send_video(b'key', is_key=True, head_pose=pose, **kwargs)
    records = [server._queue.get_nowait(), server._queue.get_nowait()]
    headers = [struct.unpack(HEADER_FORMAT, r[:HEADER_SIZE]) for r in records]
    assert headers[0][2] == 3
    assert headers[1][2] == 1
    assert headers[0][4] == headers[1][4]
    assert headers[0][-3:] == headers[1][-3:] == (2, 3, 42)
    assert struct.unpack('>7f', records[0][HEADER_SIZE:]) == pose
    assert records[1][HEADER_SIZE:] == b'key'


def test_annex_b_splitter_handles_fragmented_start_codes():
    splitter = _AnnexBSplitter()
    stream = b'\x00\x00\x00\x01\x67abc\x00\x00\x01\x65def\x00\x00\x01\x41ghi\x00\x00\x01\x09x'
    frames = []
    for byte in stream:
        frames.extend(splitter.feed(bytes([byte])))
    assert len(frames) == 2
    assert [is_key for _, is_key in frames] == [True, False]
    assert b'abc' in frames[0][0] and b'def' in frames[0][0]
