"""Teleop signal publishers/subscribers against a fake igris_c_sdk (no CycloneDDS needed)."""

import sys
import types

import pytest

from vr_link.messages.signals import SourceKind
from vr_link.messages.teleop_dds import (
    CONTROLLER_FRAME_ID,
    CONTROLLER_TOPIC,
    RECORDING_STATE_TOPIC,
    ControllerSample,
    RecordingStateSample,
)
from vr_link.network.dds import DdsStateTopics
from vr_link.network.dds.teleop import (
    ControllerStatePublisher,
    ControllerStateSubscriber,
    RecordingStatePublisher,
    RecordingStateSubscriber,
)
from vr_link.pipeline.robot_state import SignalsProvider, SignalsSpec

# --- a pybind-shaped stand-in: calling an accessor with a value sets it, without reads it ---


class _Field:
    def __init__(self, value):
        self.value = value

    def __call__(self, *args):
        if args:
            self.value = args[0]
            return None
        return self.value


class _Time:
    def __init__(self):
        self.sec, self.nanosec = _Field(0), _Field(0)


class _Header:
    def __init__(self):
        self.stamp, self.frame_id = _Field(_Time()), _Field("")


class _Joy:
    def __init__(self):
        self.header, self.axes, self.buttons = _Field(_Header()), _Field([]), _Field([])


class _Qos:
    @staticmethod
    def SensorData():
        return "sensor-data"


class _Bus:
    """Topic -> subscriber callbacks; a publisher write delivers synchronously."""

    def __init__(self):
        self.subscribers = {}
        self.publishers = []
        self.constructions = []


def _make_classes(bus):
    class _Publisher:
        def __init__(self, *args):
            bus.constructions.append(("pub", args))
            self.topic = args[1] if len(args) == 3 else args[0]
            self.initialized = False
            self.stopped = False
            self.fail_write = False
            bus.publishers.append(self)

        def init(self):
            self.initialized = True
            return True

        def write(self, message):
            if self.fail_write:
                return False
            for callback in bus.subscribers.get(self.topic, []):
                callback(message)
            return True

        def stop(self):
            self.stopped = True

    class _Subscriber:
        def __init__(self, *args):
            bus.constructions.append(("sub", args))
            self.topic = args[1] if len(args) == 3 else args[0]
            self.stopped = False

        def init(self, callback):
            bus.subscribers.setdefault(self.topic, []).append(callback)
            return True

        def stop(self):
            self.stopped = True

    return _Publisher, _Subscriber


@pytest.fixture
def fake_sdk(monkeypatch):
    bus = _Bus()
    publisher_cls, subscriber_cls = _make_classes(bus)
    sdk = types.ModuleType("igris_c_sdk")
    sdk.__version__ = "fake"
    sdk.ChannelFactory = type("ChannelFactory", (), {})
    sdk.QosProfile = _Qos
    sdk.Joy_, sdk.Header_, sdk.Time_ = _Joy, _Header, _Time
    sdk.JoyPublisher, sdk.JoySubscriber = publisher_cls, subscriber_cls
    sdk.bus = bus
    monkeypatch.setitem(sys.modules, "igris_c_sdk", sdk)
    return sdk


def _controller(count=3, session=1):
    return ControllerSample(record_toggle_count=count, session_id=session, connected=True, left=(0.1,) * 7, right=(0.2,) * 7, stamp_ns=5_000_000_250)


# --- tests --------------------------------------------------------------------------------


def test_controller_round_trip_through_sdk_channels(fake_sdk):
    subscriber = ControllerStateSubscriber(fake_sdk)
    subscriber.start()
    publisher = ControllerStatePublisher(fake_sdk)
    assert publisher.topic == subscriber.topic == CONTROLLER_TOPIC
    assert subscriber.latest() is None and subscriber.age_s() is None
    assert publisher.publish(_controller()) is True
    sample, received = subscriber.latest()
    assert sample == _controller() and received > 0 and subscriber.count == 1
    assert subscriber.age_s() >= 0.0
    assert publisher.published == 1 and publisher.rejected == 0
    # Both default to the SDK singleton overload: (topic, qos).
    assert all(len(args) == 2 and args[1] == "sensor-data" for _, args in fake_sdk.bus.constructions)
    publisher.stop()
    subscriber.stop()
    assert fake_sdk.bus.publishers[0].stopped
    publisher.stop()  # idempotent


def test_explicit_factory_uses_the_three_argument_overload(fake_sdk):
    factory = fake_sdk.ChannelFactory()
    publisher = RecordingStatePublisher(fake_sdk, factory=factory, qos="reliable")
    subscriber = RecordingStateSubscriber(fake_sdk, factory=factory)
    subscriber.start()
    assert fake_sdk.bus.constructions[0] == ("pub", (factory, RECORDING_STATE_TOPIC, "reliable"))
    assert fake_sdk.bus.constructions[1] == ("sub", (factory, RECORDING_STATE_TOPIC, "sensor-data"))
    publisher.publish(RecordingStateSample(recording=True, demo_index=2))
    sample, _ = subscriber.latest()
    assert sample.recording is True and sample.demo_index == 2 and sample.stamp_ns > 1_600_000_000_000_000_000  # stamped at write


def test_write_stamps_from_the_sample_and_reports_rejections(fake_sdk):
    subscriber = ControllerStateSubscriber(fake_sdk)
    subscriber.start()
    publisher = ControllerStatePublisher(fake_sdk)
    publisher.publish(_controller())
    assert subscriber.latest()[0].stamp_ns == 5_000_000_250
    fake_sdk.bus.publishers[0].fail_write = True
    assert publisher.publish(_controller(count=4)) is False
    assert publisher.rejected == 1 and subscriber.latest()[0].record_toggle_count == 3


def test_subscriber_ignores_foreign_messages_and_counts_decode_errors(fake_sdk, caplog):
    subscriber = ControllerStateSubscriber(fake_sdk)
    subscriber.start()
    deliver = fake_sdk.bus.subscribers[CONTROLLER_TOPIC][0]
    gamepad = _Joy()
    gamepad.header().frame_id("")
    gamepad.axes([0.0, 1.0])
    gamepad.buttons([1, 0, 0, 0])
    deliver(gamepad)
    assert subscriber.latest() is None and subscriber.ignored == 1 and subscriber.errors == 0
    short = _Joy()
    short.header().frame_id(CONTROLLER_FRAME_ID)
    short.axes([0.0] * 3)
    short.buttons([1, 1, 1])
    deliver(short)
    assert subscriber.latest() is None and subscriber.errors == 1
    deliver(object())  # not even a Joy: counted, never raised into the SDK thread
    assert subscriber.errors == 2


def test_recording_state_subscriber_reports_stale_as_not_recording(fake_sdk, monkeypatch):
    import vr_link.network.dds.teleop as teleop

    clock = [100.0]
    monkeypatch.setattr(teleop.time, "monotonic", lambda: clock[0])
    subscriber = RecordingStateSubscriber(fake_sdk)
    subscriber.start()
    publisher = RecordingStatePublisher(fake_sdk)
    assert subscriber.recording() is False
    publisher.publish(RecordingStateSample(recording=True, demo_index=0))
    assert subscriber.recording() is True
    clock[0] += 0.9
    assert subscriber.recording(max_age_s=1.0) is True
    clock[0] += 0.2
    assert subscriber.recording(max_age_s=1.0) is False  # the recorder went quiet
    publisher.publish(RecordingStateSample(recording=False, demo_index=-1))
    assert subscriber.recording() is False
    subscriber.clear()
    assert subscriber.latest() is None


def test_signals_provider_reads_the_dds_recording_flag(fake_sdk):
    subscriber = RecordingStateSubscriber(fake_sdk)
    subscriber.start()
    publisher = RecordingStatePublisher(fake_sdk)
    spec = SignalsSpec(source=SourceKind.ISAAC, recording="dds")
    with pytest.raises(ValueError):
        SignalsProvider(spec)  # dds recording needs the subscriber
    provider = SignalsProvider(spec, recording_state=subscriber)
    assert provider.sample(None).recording is False
    publisher.publish(RecordingStateSample(recording=True, demo_index=1))
    sample = provider.sample(None)
    assert sample.recording is True and sample.source is SourceKind.ISAAC
    with pytest.raises(ValueError):
        SignalsSpec(recording="hologram").validate()


def test_state_topics_carry_the_recording_state_default():
    assert DdsStateTopics().recording_state == RECORDING_STATE_TOPIC
