"""EnvelopeTcpServer: record order, calibration-once, keyframe gate, state-only sends."""
import queue

import numpy as np

from vr_link.calibration import ideal
from vr_link.messages import (
    MessageKind,
    RecordReader,
    RobotStateSample,
    SignalsSample,
    SourceKind,
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


def test_send_state_is_not_keyframe_gated_and_sends_calibration_once():
    server = _attached_server()
    server.set_calibration(CAL)
    server.send_state(wall_ns=5, sim_ns=0, signals=SignalsSample())
    server.send_state(wall_ns=6, sim_ns=0, robot_state=RobotStateSample(hand_actual=vector(np.ones(12))))
    kinds = [e.kind for e in _drain(server)]
    assert kinds == [MessageKind.CALIBRATION, MessageKind.SIGNALS, MessageKind.ROBOT_STATE]


def test_nothing_is_queued_without_a_client():
    server = EnvelopeTcpServer(send_calibration=False)
    server.send_video(b"key", is_key=True, width=64, height=48, wall_ns=1, sim_ns=0)
    server.send_state(wall_ns=1, signals=SignalsSample())
    assert server._queue is None and not server.client_connected


def test_backlog_over_limit_detaches_the_client():
    server = _attached_server(send_calibration=False, max_queue_bytes=64)
    server.send_video(b"k" * 200, is_key=True, width=64, height=48, wall_ns=1, sim_ns=0)
    records = []
    while not server._queue.empty():
        records.append(server._queue.get_nowait())
    assert records[-1] == b""  # detach sentinel follows the oversized record
