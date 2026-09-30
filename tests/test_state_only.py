"""StateOnlyPump: fixed-rate RobotState/Signals without video."""
import threading
import time

import numpy as np

from vr_link.messages import SourceKind
from vr_link.pipeline.robot_state import DdsStateBuffers, RobotStateAssembler, RobotStateSpec, SignalsProvider, SignalsSpec
from vr_link.pipeline.state_only import StateOnlyPump
from vr_link.utils.clocks import wall_ns


class FakeServer:
    def __init__(self):
        self.client_connected = True
        self.sent = []
        self.calibration = None

    def set_calibration(self, cal):
        self.calibration = cal

    def send_state(self, *, wall_ns, sim_ns=0, robot_state=None, signals=None):
        self.sent.append((wall_ns, sim_ns, robot_state, signals))


def test_state_only_sends_latest_dds_state_at_rate():
    buffers = DdsStateBuffers()
    buffers.actual.push(wall_ns(), np.zeros(31, np.float32))
    assembler = RobotStateAssembler(RobotStateSpec(actual="dds", hand_actual="dds"), dds=buffers)
    server = FakeServer()
    pump = StateOnlyPump(server, assembler, SignalsProvider(SignalsSpec(source=SourceKind.ROBOT)), rate_hz=200.0, max_age_ms=100.0,
                         calibration=lambda: {"image_width": 1, "image_height": 1, "left": {}, "right": {}})
    stop = threading.Event()
    thread = threading.Thread(target=pump.run, args=(stop,))
    thread.start()
    time.sleep(0.15)
    stop.set()
    thread.join(timeout=2.0)
    assert 10 <= len(server.sent) <= 60  # ~30 at 200 Hz over 150 ms, tolerant to scheduling
    _, sim_ns, state, signals = server.sent[0]
    assert sim_ns == 0 and state.body_actual is not None and state.hand_actual is None
    assert signals.source is SourceKind.ROBOT and signals.recording is False
    assert server.calibration is not None  # provider handed to the server once


def test_state_only_idles_without_a_client():
    server = FakeServer()
    server.client_connected = False
    pump = StateOnlyPump(server, RobotStateAssembler(RobotStateSpec()), SignalsProvider(SignalsSpec()), rate_hz=100.0)
    stop = threading.Event()
    thread = threading.Thread(target=pump.run, args=(stop,))
    thread.start()
    time.sleep(0.05)
    stop.set()
    thread.join(timeout=2.0)
    assert server.sent == []
