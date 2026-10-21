"""DDS layer against a fake igris_c_sdk injected into sys.modules (no CycloneDDS needed)."""
import sys
import types

import numpy as np
import pytest

from vr_link.network import dds as dds_pkg
from vr_link.network.dds import (
    REFERENCE_LOWCMD_KEY,
    DdsChannel,
    DdsStateTopics,
    StampedSubscription,
    body_cmd_q_by_id,
    header_stamp_ns,
    joint_state_q,
    motor_cmd_q_by_id,
    motor_state_q,
    start_state_subscriptions,
)
from vr_link.pipeline.robot_state import DdsStateBuffers
from vr_link.utils.time_sync import StampedRingBuffer

# --- a minimal stand-in for the pybind module (current 78bec2c spelling) ------------


class _Time:
    def __init__(self, sec, nanosec):
        self._sec, self._nanosec = sec, nanosec

    def sec(self):
        return self._sec

    def nanosec(self):
        return self._nanosec


class _Header:
    def __init__(self, sec=0, nanosec=0, frame_id=""):
        self._stamp = _Time(sec, nanosec)
        self._frame_id = frame_id

    def stamp(self):
        return self._stamp

    def frame_id(self):
        return self._frame_id


class _Joint:
    def __init__(self, q):
        self._q = q

    def q(self):
        return self._q


class _MotorCmd(_Joint):
    def __init__(self, motor_id, q):
        super().__init__(q)
        self._id = motor_id

    def id(self):
        return self._id


class _LowState:
    def __init__(self, sec, nanosec, values):
        self._header = _Header(sec, nanosec)
        self._joints = [_Joint(v) for v in values]

    def header(self):
        return self._header

    def joint_state(self):
        return self._joints


class _HandState:
    def __init__(self, sec, nanosec, values):
        self._header = _Header(sec, nanosec)
        self._motors = [_Joint(v) for v in values]

    def header(self):
        return self._header

    def motor_state(self):
        return self._motors


class _HandCmd:
    def __init__(self, cmds):
        self._header = _Header(0, 0)
        self._cmds = [_MotorCmd(i, q) for i, q in cmds]

    def header(self):
        return self._header

    def motor_cmd(self):
        return self._cmds


class _LowCmd:
    """igris_c_sdk LowCmd_: header + 31 motors tagged by id (the inference node writes id = index)."""

    def __init__(self, sec, nanosec, cmds):
        self._header = _Header(sec, nanosec, "igris_c_inference_public")
        self._motors = [_MotorCmd(i, q) for i, q in cmds]

    def header(self):
        return self._header

    def motors(self):
        return self._motors


class _Qos:
    @staticmethod
    def SensorData():
        return "sensor-data"


class _Factory:
    instance = None
    calls = []

    @classmethod
    def Instance(cls):
        if cls.instance is None:
            cls.instance = cls()
        return cls.instance

    def Init(self, domain_id, ns, config, naming):
        _Factory.calls.append(("Init", domain_id, ns, config, naming))

    def IsInitialized(self):
        return True

    def resolve(self, topic):
        return f"ns/{topic}"

    def Release(self):
        _Factory.calls.append(("Release",))


def _make_subscriber_class(registry, name):
    class _Sub:
        def __init__(self, topic, qos=None):
            self.topic, self.qos, self.callback, self.stopped = topic, qos, None, False
            registry.setdefault(name, []).append(self)

        def init(self, callback):
            self.callback = callback
            return True

        def stop(self):
            self.stopped = True

    _Sub.__name__ = name
    return _Sub


@pytest.fixture
def fake_sdk(monkeypatch):
    registry = {}
    sdk = types.ModuleType("igris_c_sdk")
    sdk.__version__ = "fake"
    sdk.ChannelFactory = _Factory
    sdk.QosProfile = _Qos
    sdk.parse_topic_naming = lambda value: f"naming:{value}"
    for name in ("LowState", "LowCmd", "HandState", "HandCmd", "MasterArmState", "CompressedImage"):
        setattr(sdk, f"{name}_", object())
        setattr(sdk, f"{name}Subscriber", _make_subscriber_class(registry, f"{name}Subscriber"))
    monkeypatch.setitem(sys.modules, "igris_c_sdk", sdk)
    _Factory.instance = None
    _Factory.calls = []
    sdk.registry = registry
    return sdk


# --- tests ---------------------------------------------------------------------------


def test_import_sdk_and_sdk_type(fake_sdk):
    sdk = dds_pkg.import_sdk()
    assert sdk is fake_sdk
    assert dds_pkg.sdk_type(sdk, "LowState") is fake_sdk.LowState_


def test_import_sdk_rejects_rosidl_shadow(monkeypatch):
    monkeypatch.setitem(sys.modules, "igris_c_sdk", types.ModuleType("igris_c_sdk"))
    with pytest.raises(ImportError):
        dds_pkg.import_sdk()


def test_channel_init_resolve_release(fake_sdk):
    with pytest.raises(ValueError):
        DdsChannel.create(domain_id=1, namespace="", sdk=fake_sdk)
    with pytest.raises(ValueError):
        DdsChannel.create(domain_id=1, namespace="igris_c_x", topic_naming="ros", sdk=fake_sdk)
    channel = DdsChannel.create(domain_id=3, namespace="igris_c_x", topic_naming="ros_compatible", config_uri="file:///x.xml", sdk=fake_sdk)
    assert _Factory.calls == [("Init", 3, "igris_c_x", "file:///x.xml", "naming:ros_compatible")]
    assert channel.resolve("lowstate") == "ns/lowstate"
    channel.release()
    channel.release()
    assert _Factory.calls[-1] == ("Release",) and _Factory.calls.count(("Release",)) == 1


def test_extractors_read_stamps_and_orders(fake_sdk):
    low = _LowState(5, 250, np.arange(31) * 0.5)
    assert header_stamp_ns(low) == 5_000_000_250
    stamp, values = joint_state_q()(low)
    assert stamp == 5_000_000_250 and values.dtype == np.float32 and values[30] == pytest.approx(15.0)
    stamp, values = motor_state_q()(_HandState(1, 0, np.linspace(0, 1, 12)))
    assert stamp == 1_000_000_000 and values.shape == (12,)
    # HandCmd arrives id-tagged and possibly reordered; missing ids stay NaN.
    cmd = _HandCmd([(21, 0.1), (11, 0.9), (26, 0.5)])
    stamp, values = motor_cmd_q_by_id()(cmd)
    assert values[0] == pytest.approx(0.9) and values[6] == pytest.approx(0.1) and values[11] == pytest.approx(0.5)
    assert np.isnan(values[1])
    assert stamp > 1_600_000_000_000_000_000  # no header stamp -> receive time


def test_body_cmd_extractor_places_motors_by_id(fake_sdk):
    # LowCmd motors in a shuffled order land by id; the header stamp is the publish time.
    cmd = _LowCmd(7, 500, [(i, 0.1 * i) for i in reversed(range(31))])
    stamp, values = body_cmd_q_by_id()(cmd)
    assert stamp == 7_000_000_500 and values.dtype == np.float32 and values.shape == (31,)
    assert values[0] == pytest.approx(0.0) and values[30] == pytest.approx(3.0) and values[12] == pytest.approx(1.2)
    # A missing id leaves NaN (the assembler then drops the sample); ids outside 0..30 are ignored.
    _, partial = body_cmd_q_by_id()(_LowCmd(7, 0, [(i, 1.0) for i in range(30)] + [(31, 9.0), (-1, 9.0)]))
    assert np.isnan(partial[30]) and np.count_nonzero(np.isnan(partial)) == 1


def test_subscription_pushes_to_buffer_and_swallows_errors(fake_sdk, caplog):
    channel = DdsChannel.create(domain_id=0, namespace="igris_c_x", sdk=fake_sdk)
    buffer = StampedRingBuffer(8)
    sub = StampedSubscription(channel, "LowState", "lowstate", joint_state_q(), buffer)
    sub.start()
    fake = fake_sdk.registry["LowStateSubscriber"][0]
    assert fake.topic == "lowstate" and fake.qos == "sensor-data"
    fake.callback(_LowState(2, 0, np.zeros(31)))
    fake.callback(_LowState(2, 0, np.zeros(3)))  # too short -> np.fromiter raises inside the extractor
    assert sub.received == 1 and sub.errors == 1 and len(buffer) == 1
    sub.stop()
    assert fake.stopped
    channel.release()


def test_start_state_subscriptions_only_for_needed_fields(fake_sdk):
    channel = DdsChannel.create(domain_id=0, namespace="igris_c_x", sdk=fake_sdk)
    buffers = DdsStateBuffers()
    subs = start_state_subscriptions(channel, DdsStateTopics(), buffers, needed={"actual", "hand_command"})
    assert sorted(s.type_name for s in subs) == ["HandCmd", "LowState"]
    assert set(fake_sdk.registry) == {"LowStateSubscriber", "HandCmdSubscriber"}
    fake_sdk.registry["HandCmdSubscriber"][0].callback(_HandCmd([(i, 0.5) for i in (11, 12, 13, 14, 15, 16, 21, 22, 23, 24, 25, 26)]))
    assert len(buffers.hand_command) == 1 and len(buffers.actual) == 0
    for sub in subs:
        sub.stop()
    channel.release()


def test_reference_from_lowcmd_subscribes_lowcmd_into_the_reference_buffer(fake_sdk):
    channel = DdsChannel.create(domain_id=0, namespace="igris_c_x", sdk=fake_sdk)
    buffers = DdsStateBuffers()
    subs = start_state_subscriptions(channel, DdsStateTopics(lowcmd="rt/lowcmd"), buffers, needed={REFERENCE_LOWCMD_KEY})
    assert [s.type_name for s in subs] == ["LowCmd"] and subs[0].topic == "rt/lowcmd"
    assert set(fake_sdk.registry) == {"LowCmdSubscriber"}  # no MasterArmState subscription
    fake_sdk.registry["LowCmdSubscriber"][0].callback(_LowCmd(3, 0, [(i, 0.5) for i in range(31)]))
    assert len(buffers.reference) == 1
    for sub in subs:
        sub.stop()
    channel.release()


def test_dds_camera_source_keeps_newest_sample(fake_sdk):
    cv2 = pytest.importorskip("cv2")
    from vr_link.sources.dds_camera import DdsCameraSource

    channel = DdsChannel.create(domain_id=0, namespace="igris_c_x", sdk=fake_sdk)
    source = DdsCameraSource(channel, "sensor/eyes_stereo/compressed")
    source.open()
    fake = fake_sdk.registry["CompressedImageSubscriber"][0]
    ok, jpeg = cv2.imencode(".jpg", np.full((4, 8, 3), 200, np.uint8))
    assert ok

    class _Image:
        def __init__(self, sec, data):
            self._header = _Header(sec, 0)
            self._data = data

        def header(self):
            return self._header

        def data(self):
            return self._data

    fake.callback(_Image(1, jpeg.tobytes()))
    fake.callback(_Image(2, list(jpeg.tobytes())))  # sequence-of-int form
    frame = source.read(timeout=0.1)
    assert frame is not None and frame.pixel_format == "bgr24" and frame.capture_ns == 2_000_000_000
    assert frame.pixels.shape == (4, 8, 3)
    assert source.read(timeout=0.01) is None  # nothing newer
    source.close()
    assert fake.stopped
    channel.release()
