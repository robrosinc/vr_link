"""SDK publishers and latest-value subscribers for the teleop signal topics.

The wire layout is :mod:`vr_link.messages.teleop_dds`; this module only moves
it through igris_c_sdk ``Joy_`` channels. Any process that holds an SDK
participant can host these endpoints: pass ``factory`` when the process owns
a ``ChannelFactory`` instance (the hit-tasks Isaac bridge), leave it ``None``
to use the SDK's process-wide singleton (``igris_c_gmr``, ``vr-link-stream``
through :class:`~vr_link.network.dds.channel.DdsChannel`).

Subscriber callbacks run on the SDK delivery thread: they decode, keep the
newest sample and never raise. Decoding failures are counted; messages with
a foreign ``frame_id`` are counted separately and otherwise ignored.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Callable, List, Optional, Sequence, Tuple

from ...messages.teleop_dds import (
    CONTROLLER_FRAME_ID,
    CONTROLLER_TOPIC,
    RECORDING_STATE_FRAME_ID,
    RECORDING_STATE_TOPIC,
    ControllerSample,
    RecordingStateSample,
    decode_controller,
    decode_recording_state,
    encode_controller,
    encode_recording_state,
)
from ...utils.clocks import split_ns, wall_ns
from . import sdk_type
from .subscribers import header_stamp_ns

logger = logging.getLogger(__name__)

Decoder = Callable[..., Optional[Any]]


def _sdk_endpoint(sdk, class_name: str, topic: str, factory, qos):
    """Instantiate ``sdk.<class_name>`` with the singleton or the explicit-factory overload."""

    endpoint_cls = getattr(sdk, class_name, None)
    if endpoint_cls is None:
        raise RuntimeError(f"igris_c_sdk has no {class_name}")
    if qos is None:
        qos = sdk.QosProfile.SensorData()
    if factory is None:
        return endpoint_cls(topic, qos)
    return endpoint_cls(factory, topic, qos)


class _JoyPublisher:
    def __init__(self, sdk, topic: str, frame_id: str, *, factory=None, qos=None) -> None:
        self.sdk = sdk
        self.topic = str(topic)
        self.frame_id = str(frame_id)
        self._joy_type = sdk_type(sdk, "Joy")
        self._header_type = sdk_type(sdk, "Header")
        self._time_type = sdk_type(sdk, "Time")
        self._publisher = _sdk_endpoint(sdk, "JoyPublisher", self.topic, factory, qos)
        if not self._publisher.init():
            raise RuntimeError(f"failed to initialize DDS publisher on {self.topic!r}")
        self.published = 0
        self.rejected = 0

    def _write(self, axes: Sequence[float], buttons: Sequence[int], stamp_ns: int) -> bool:
        sec, nanosec = split_ns(int(stamp_ns) or wall_ns())
        when = self._time_type()
        when.sec(int(sec))
        when.nanosec(int(nanosec))
        header = self._header_type()
        header.stamp(when)
        header.frame_id(self.frame_id)
        message = self._joy_type()
        message.header(header)
        message.axes([float(v) for v in axes])
        message.buttons([int(v) for v in buttons])
        if self._publisher.write(message) is False:
            self.rejected += 1
            return False
        self.published += 1
        if self.published == 1:
            logger.info("First %s written on %s", self.frame_id, self.topic)
        return True

    def stop(self) -> None:
        publisher, self._publisher = self._publisher, None
        if publisher is None:
            return
        try:
            stop = getattr(publisher, "stop", None)
            if stop is not None:
                stop()
        except Exception as error:  # noqa: BLE001
            logger.warning("stopping %s publisher failed: %s", self.frame_id, error)

    close = stop


class ControllerStatePublisher(_JoyPublisher):
    """``vr/controller``: hosted by the VRS1 receiver (igris_c_gmr)."""

    def __init__(self, sdk, topic: str = CONTROLLER_TOPIC, *, factory=None, qos=None) -> None:
        super().__init__(sdk, topic, CONTROLLER_FRAME_ID, factory=factory, qos=qos)

    def publish(self, sample: ControllerSample) -> bool:
        _, axes, buttons = encode_controller(sample)
        return self._write(axes, buttons, sample.stamp_ns)


class RecordingStatePublisher(_JoyPublisher):
    """``recording/state``: hosted by the process that owns the recorder."""

    def __init__(self, sdk, topic: str = RECORDING_STATE_TOPIC, *, factory=None, qos=None) -> None:
        super().__init__(sdk, topic, RECORDING_STATE_FRAME_ID, factory=factory, qos=qos)

    def publish(self, sample: RecordingStateSample) -> bool:
        _, axes, buttons = encode_recording_state(sample)
        return self._write(axes, buttons, sample.stamp_ns)


class _LatestJoySubscriber:
    """One ``JoySubscriber`` keeping only the newest decoded sample."""

    def __init__(self, sdk, topic: str, decoder: Decoder, *, factory=None, qos=None) -> None:
        self.sdk = sdk
        self.topic = str(topic)
        self._decoder = decoder
        self._factory = factory
        self._qos = qos
        self._subscriber = None
        self._lock = threading.Lock()
        self._latest: Optional[Any] = None
        self._received_monotonic = 0.0
        self.count = 0
        self.ignored = 0
        self.errors = 0

    def start(self) -> None:
        subscriber = _sdk_endpoint(self.sdk, "JoySubscriber", self.topic, self._factory, self._qos)
        if not subscriber.init(self._on_message):
            raise RuntimeError(f"failed to initialize DDS subscriber on {self.topic!r}")
        self._subscriber = subscriber
        logger.info("Subscribed Joy on %s", self.topic)

    def stop(self) -> None:
        subscriber, self._subscriber = self._subscriber, None
        if subscriber is not None:
            try:
                subscriber.stop()
            except Exception as error:  # noqa: BLE001
                logger.warning("stopping Joy subscriber on %s failed: %s", self.topic, error)

    close = stop

    def _on_message(self, message) -> None:
        # SDK thread: decode a copy, store, never raise.
        try:
            header = message.header()
            sample = self._decoder(
                str(header.frame_id()),
                [float(v) for v in message.axes()],
                [int(v) for v in message.buttons()],
                stamp_ns=header_stamp_ns(message),
            )
        except Exception as error:  # noqa: BLE001
            self.errors += 1
            if self.errors in (1, 10, 100) or self.errors % 1000 == 0:
                logger.warning("Joy on %s: decode failed (%d so far): %s", self.topic, self.errors, error)
            return
        if sample is None:
            self.ignored += 1
            if self.ignored == 1:
                logger.warning("Joy on %s carries a foreign frame_id; such messages are ignored", self.topic)
            return
        with self._lock:
            self._latest = sample
            self._received_monotonic = time.monotonic()
            self.count += 1

    def latest(self) -> Optional[Tuple[Any, float]]:
        """``(sample, receive time.monotonic())`` of the newest accepted message, or ``None``."""

        with self._lock:
            if self._latest is None:
                return None
            return self._latest, self._received_monotonic

    def age_s(self) -> Optional[float]:
        """Seconds since the newest accepted message, or ``None`` before the first one."""

        with self._lock:
            if self._latest is None:
                return None
            return time.monotonic() - self._received_monotonic

    def clear(self) -> None:
        with self._lock:
            self._latest = None
            self._received_monotonic = 0.0


class ControllerStateSubscriber(_LatestJoySubscriber):
    """``vr/controller`` consumer side (a recorder host)."""

    def __init__(self, sdk, topic: str = CONTROLLER_TOPIC, *, factory=None, qos=None) -> None:
        super().__init__(sdk, topic, decode_controller, factory=factory, qos=qos)

    def latest(self) -> Optional[Tuple[ControllerSample, float]]:  # narrow the type for callers
        return super().latest()


class RecordingStateSubscriber(_LatestJoySubscriber):
    """``recording/state`` consumer side (``vr-link-stream`` for the HUD)."""

    def __init__(self, sdk, topic: str = RECORDING_STATE_TOPIC, *, factory=None, qos=None) -> None:
        super().__init__(sdk, topic, decode_recording_state, factory=factory, qos=qos)

    def latest(self) -> Optional[Tuple[RecordingStateSample, float]]:
        return super().latest()

    def recording(self, max_age_s: float = 1.0) -> bool:
        """The recorder's flag, or ``False`` when nothing fresh arrived within ``max_age_s``."""

        entry = self.latest()
        if entry is None:
            return False
        sample, received = entry
        if time.monotonic() - received > float(max_age_s):
            return False
        return bool(sample.recording)


__all__: List[str] = [
    "ControllerStatePublisher",
    "ControllerStateSubscriber",
    "RecordingStatePublisher",
    "RecordingStateSubscriber",
]
