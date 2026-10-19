"""EnvelopeTcpServer: record order, calibration-once, keyframe gate."""
import queue

import numpy as np

from vr_link.calibration import ideal
from vr_link.messages import (
    MessageKind,
    RecordReader,
    RobotStateSample,
    SignalsSample,
    SourceKind,
    VideoStream,
    decode_calibration,
    decode_video_frame,
    parse_envelope,
)
from vr_link.messages.robot_state import vector
from vr_link.network.tcp_server import EnvelopeTcpServer

CAL = ideal.ideal_stereo_from_fovy(320, 240, 90.0, 14.0, 52.152)


def _attached_server(**kwargs) -> EnvelopeTcpServer:
    server = EnvelopeTcpServer(**kwargs)
    server._queue = queue.Queue()  # simulate an attached client without sockets
    return server


def _drain(server):
    reader = RecordReader()
    out = []
    while not server._queue.empty():
        out.extend(parse_envelope(p) for p in reader.feed(server._queue.get_nowait()))
    return out


def test_non_key_video_is_dropped_until_the_first_keyframe():
    server = _attached_server(send_calibration=False)
    server.send_video(b"delta", is_key=False, width=64, height=48, wall_ns=1, sim_ns=0)
    assert server._queue.empty()
    server.send_video(b"key", is_key=True, width=64, height=48, wall_ns=2, sim_ns=0)
    server.send_video(b"delta2", is_key=False, width=64, height=48, wall_ns=3, sim_ns=0)
    envs = _drain(server)
    assert [e.kind for e in envs] == [MessageKind.VIDEO_FRAME] * 2
    assert [decode_video_frame(e).data for e in envs] == [b"key", b"delta2"]
    assert [e.seq for e in envs] == [0, 1]


def test_calibration_once_then_signals_robot_state_video_share_stamps():
    server = _attached_server()
    server.set_calibration(CAL)
    state = RobotStateSample(body_actual=vector(np.zeros(31), 77))
    sig = SignalsSample(recording=True, source=SourceKind.ISAAC)
    kwargs = dict(width=64, height=48, robot_state=state, signals=sig)
    server.send_video(b"key", is_key=True, wall_ns=2_000_000_003, sim_ns=42, **kwargs)
    server.send_video(b"delta", is_key=False, wall_ns=2_000_000_004, sim_ns=43, **kwargs)
    envs = _drain(server)
    kinds = [e.kind for e in envs]
    assert kinds == [
        MessageKind.CALIBRATION,
        MessageKind.SIGNALS, MessageKind.ROBOT_STATE, MessageKind.VIDEO_FRAME,
        MessageKind.SIGNALS, MessageKind.ROBOT_STATE, MessageKind.VIDEO_FRAME,
    ]
    assert [e.seq for e in envs] == list(range(7))
    assert decode_calibration(envs[0])["image_width"] == 320
    first = envs[1:4]
    assert {(e.wall_ns, e.sim_ns) for e in first} == {(2_000_000_003, 42)}
    assert {(e.wall_ns, e.sim_ns) for e in envs[4:]} == {(2_000_000_004, 43)}


def test_nothing_is_queued_without_a_client():
    server = EnvelopeTcpServer(send_calibration=False)
    server.send_video(b"key", is_key=True, width=64, height=48, wall_ns=1, sim_ns=0, signals=SignalsSample())
    assert server._queue is None and not server.client_connected


def test_backlog_over_limit_detaches_the_client():
    server = _attached_server(send_calibration=False, max_queue_bytes=64)
    server.send_video(b"k" * 200, is_key=True, width=64, height=48, wall_ns=1, sim_ns=0)
    records = []
    while not server._queue.empty():
        records.append(server._queue.get_nowait())
    assert records[-1] == b""  # detach sentinel follows the oversized record


def test_each_stream_waits_for_its_own_keyframe_and_head_frames_carry_no_state():
    server = _attached_server(send_calibration=False)
    head = dict(width=640, height=480, stream_id=VideoStream.HEAD_RGB)
    stereo = dict(width=64, height=48, signals=SignalsSample())
    server.send_video(b"h-delta", is_key=False, wall_ns=1, **head)  # dropped: head has no IDR yet
    server.send_video(b"s-key", is_key=True, wall_ns=2, **stereo)
    server.send_video(b"h-delta2", is_key=False, wall_ns=3, **head)  # still dropped despite the stereo IDR
    server.send_video(b"h-key", is_key=True, wall_ns=4, **head)
    server.send_video(b"s-delta", is_key=False, wall_ns=5, **stereo)
    server.send_video(b"h-delta3", is_key=False, wall_ns=6, **head)
    envs = _drain(server)
    kinds = [e.kind for e in envs]
    assert kinds == [
        MessageKind.SIGNALS, MessageKind.VIDEO_FRAME,  # stereo frame with its state
        MessageKind.VIDEO_FRAME,                       # head key frame, no state
        MessageKind.SIGNALS, MessageKind.VIDEO_FRAME,
        MessageKind.VIDEO_FRAME,
    ]
    assert [e.seq for e in envs] == list(range(6))
    frames = [decode_video_frame(e) for e in envs if e.kind is MessageKind.VIDEO_FRAME]
    assert [(f.stream_id, f.data) for f in frames] == [
        (VideoStream.EYES_STEREO, b"s-key"), (VideoStream.HEAD_RGB, b"h-key"),
        (VideoStream.EYES_STEREO, b"s-delta"), (VideoStream.HEAD_RGB, b"h-delta3"),
    ]
    assert [(f.width, f.height) for f in frames if f.stream_id is VideoStream.HEAD_RGB] == [(640, 480)] * 2


def test_keyframe_requests_are_per_stream_and_renewed_per_connection():
    server = EnvelopeTcpServer(send_calibration=False)
    assert not server.consume_keyframe_request()  # no client yet
    server._attach_client()
    assert server.consume_keyframe_request() and not server.consume_keyframe_request()
    # A stream that first appears during a connection is owed an IDR once as well.
    assert server.consume_keyframe_request(VideoStream.HEAD_RGB) and not server.consume_keyframe_request(VideoStream.HEAD_RGB)
    server._attach_client()  # the next headset: both known streams owe an IDR again
    assert server.consume_keyframe_request(VideoStream.HEAD_RGB) and server.consume_keyframe_request(VideoStream.EYES_STEREO)
    assert not server.consume_keyframe_request(VideoStream.HEAD_RGB)
    assert server.connections == 2
