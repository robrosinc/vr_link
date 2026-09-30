"""Source -> encoder -> VSTR server loop over loopback."""
import json
import socket
import threading
import time

import numpy as np
import pytest

from vr_link import wire
from vr_link.calibration import ideal
from vr_link.calibration.server import static_provider
from vr_link.config import EncoderConfig
from vr_link.encoder import probe_h264_encoder
from vr_link.pump import StreamPump
from vr_link.sources import Frame
from vr_link.stream_server import VstrTcpStreamServer

pytest.importorskip("av")

W, H = 64, 48


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
        return Frame(pixels=pixels, pixel_format="rgb24", capture_ns=1_700_000_000_000_000_000 + self._index, sim_ns=1000 * self._index, index=self._index)


def _recv_exact(conn, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("server closed")
        buf.extend(chunk)
    return bytes(buf)


def _recv_record(conn):
    header = wire.parse_header(_recv_exact(conn, wire.HEADER_SIZE))
    return header, _recv_exact(conn, header.payload_size)


@pytest.fixture(scope="module")
def sw_available():
    if not probe_h264_encoder("libx264", W, H):
        pytest.skip("libx264 not available in this PyAV build")


def test_pump_sends_params_then_keyframe_video(sw_available):
    source = FakeSource(count=6)
    server = VstrTcpStreamServer(host="127.0.0.1", port=0, send_camera_params=True)
    server.start()
    stop = threading.Event()
    cal = ideal.ideal_stereo_from_fovy(W // 2, H, 90.0, 14.0, 52.152)
    pump = StreamPump(source, server, encoder_cfg=EncoderConfig(codec="libx264", bitrate=200_000, gop=30, fps=30.0), calibration=static_provider(cal), read_timeout=0.05)
    thread = threading.Thread(target=pump.run, args=(stop,), daemon=True)
    try:
        with socket.create_connection(("127.0.0.1", server.port), timeout=5.0) as conn:
            conn.settimeout(5.0)
            thread.start()
            params_header, params = _recv_record(conn)
            assert params_header.codec == wire.WIRE_CODEC_PARAMS_JSON
            assert (params_header.width, params_header.height) == (W // 2, H)
            assert json.loads(params)["left"]["fx"] == pytest.approx(cal["left"]["fx"])
            video_header, access_unit = _recv_record(conn)
            assert video_header.codec == wire.WIRE_CODEC_H264
            assert video_header.flags & wire.WIRE_FLAG_KEYFRAME
            assert (video_header.width, video_header.height) == (W, H)
            assert video_header.sim_ns > 0 and video_header.sim_ns % 1000 == 0
            assert video_header.wall_ns > 1_700_000_000_000_000_000
            assert access_unit[:4] in (b"\x00\x00\x00\x01", b"\x00\x00\x01\x00")
            second_header, _ = _recv_record(conn)
            assert second_header.codec == wire.WIRE_CODEC_H264
            assert second_header.sequence == video_header.sequence + 1
    finally:
        stop.set()
        thread.join(timeout=5.0)
        server.stop()
    assert source.opened and source.closed
    assert pump.frames_sent >= 2


def test_pump_drops_frames_while_no_client(sw_available):
    source = FakeSource(count=3)
    server = VstrTcpStreamServer(host="127.0.0.1", port=0, send_camera_params=False)
    server.start()
    stop = threading.Event()
    pump = StreamPump(source, server, encoder_cfg=EncoderConfig(codec="libx264", bitrate=200_000, gop=30, fps=30.0), read_timeout=0.01)
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
