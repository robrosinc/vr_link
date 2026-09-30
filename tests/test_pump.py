"""Source -> encoder -> Envelope server loop over loopback."""
import socket
import threading
import time

import numpy as np
import pytest

from vr_link.calibration import ideal
from vr_link.messages import (
    MessageKind,
    RecordReader,
    RobotStateSample,
    SignalsSample,
    SourceKind,
    decode_calibration,
    decode_robot_state,
    decode_signals,
    decode_video_frame,
    parse_envelope,
)
from vr_link.messages.robot_state import vector
from vr_link.network.calibration_server import static_provider
from vr_link.network.tcp_server import EnvelopeTcpServer
from vr_link.pipeline.config import EncoderConfig
from vr_link.pipeline.pump import StreamPump
from vr_link.sources import Frame
from vr_link.video.encoder import probe_h264_encoder

pytest.importorskip("av")

W, H = 64, 48
T0 = 1_700_000_000_000_000_000


class FakeSource:
    def __init__(self, count: int) -> None:
        self.count = count
        self.opened = False
        self.closed = False
        self._index = 0

    def open(self):
        self.opened = True

    def close(self):
        self.closed = True

    @property
    def frame_size(self):
        return (W, H)

    @property
    def fps_hint(self):
        return 30.0

    def camera_metadata(self):
        return None

    def read(self, timeout):
        if self._index >= self.count:
            time.sleep(min(timeout, 0.01))
            return None
        self._index += 1
        pixels = np.full((H, W, 3), self._index * 10, dtype=np.uint8)
        return Frame(pixels=pixels, pixel_format="rgb24", capture_ns=T0 + self._index, sim_ns=1000 * self._index, index=self._index)


class EnvelopeClient:
    def __init__(self, conn):
        self.conn = conn
        self.reader = RecordReader()
        self.pending = []

    def next(self):
        while not self.pending:
            chunk = self.conn.recv(65536)
            if not chunk:
                raise ConnectionError("server closed")
            self.pending.extend(self.reader.feed(chunk))
        return parse_envelope(self.pending.pop(0))


@pytest.fixture(scope="module")
def sw_available():
    if not probe_h264_encoder("libx264", W, H):
        pytest.skip("libx264 not available in this PyAV build")


def _encoder_cfg():
    return EncoderConfig(codec="libx264", bitrate=200_000, gop=30, fps=30.0)


def test_pump_sends_calibration_then_state_and_keyframe_video(sw_available):
    source = FakeSource(count=6)
    server = EnvelopeTcpServer(host="127.0.0.1", port=0)
    server.start()
    stop = threading.Event()
    cal = ideal.ideal_stereo_from_fovy(W // 2, H, 90.0, 14.0, 52.152)

    def robot_state(frame):
        # Pretend a DDS sample matched this frame: stamp it with the frame time.
        return RobotStateSample(body_actual=vector(np.full(31, frame.index, dtype=np.float32), frame.capture_ns))

    def signals(frame):
        return SignalsSample(recording=frame.index % 2 == 0, source=SourceKind.MUJOCO)

    pump = StreamPump(source, server, encoder_cfg=_encoder_cfg(), calibration=static_provider(cal), read_timeout=0.05,
                      robot_state=robot_state, signals=signals)
    thread = threading.Thread(target=pump.run, args=(stop,), daemon=True)
    try:
        with socket.create_connection(("127.0.0.1", server.port), timeout=5.0) as conn:
            conn.settimeout(5.0)
            client = EnvelopeClient(conn)
            thread.start()
            env = client.next()
            assert env.kind is MessageKind.CALIBRATION
            assert decode_calibration(env)["left"]["fx"] == pytest.approx(cal["left"]["fx"])

            sig_env, state_env, video_env = client.next(), client.next(), client.next()
            assert (sig_env.kind, state_env.kind, video_env.kind) == (MessageKind.SIGNALS, MessageKind.ROBOT_STATE, MessageKind.VIDEO_FRAME)
            assert sig_env.seq + 1 == state_env.seq and state_env.seq + 1 == video_env.seq
            assert (sig_env.wall_ns, sig_env.sim_ns) == (state_env.wall_ns, state_env.sim_ns) == (video_env.wall_ns, video_env.sim_ns)
            assert video_env.sim_ns > 0 and video_env.sim_ns % 1000 == 0 and video_env.wall_ns > T0
            frame = decode_video_frame(video_env)
            assert frame.is_key and (frame.width, frame.height) == (W, H)
            assert frame.data[:4] in (b"\x00\x00\x00\x01", b"\x00\x00\x01\x00")
            state = decode_robot_state(state_env)
            assert state.body_actual is not None and state.body_actual.stamp_ns == state_env.wall_ns
            assert decode_signals(sig_env).source is SourceKind.MUJOCO

            second = [client.next() for _ in range(3)]
            assert [e.kind for e in second] == [MessageKind.SIGNALS, MessageKind.ROBOT_STATE, MessageKind.VIDEO_FRAME]
            assert second[2].seq == video_env.seq + 3
    finally:
        stop.set()
        thread.join(timeout=5.0)
        server.stop()
    assert source.opened and source.closed
    assert pump.frames_sent >= 2


def test_pump_drops_frames_while_no_client(sw_available):
    source = FakeSource(count=3)
    server = EnvelopeTcpServer(host="127.0.0.1", port=0, send_calibration=False)
    server.start()
    stop = threading.Event()
    pump = StreamPump(source, server, encoder_cfg=_encoder_cfg(), read_timeout=0.01)
    thread = threading.Thread(target=pump.run, args=(stop,), daemon=True)
    thread.start()
    deadline = time.monotonic() + 5.0
    while pump.frames_dropped_no_client < 3 and time.monotonic() < deadline:
        time.sleep(0.01)
    stop.set()
    thread.join(timeout=5.0)
    server.stop()
    assert pump.frames_dropped_no_client == 3
    assert pump.frames_sent == 0
    assert pump.encoder is None
