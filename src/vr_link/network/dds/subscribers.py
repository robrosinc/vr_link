"""Stamped DDS subscriptions feeding :class:`~vr_link.utils.time_sync.StampedRingBuffer`.

The SDK invokes callbacks on the CycloneDDS delivery thread with a loaned
sample. :class:`StampedSubscription` copies what it needs (stamp + a float32
vector) inside the callback and pushes it to the ring buffer; nothing else
runs on that thread and exceptions are counted, never propagated.

Extractors turn one message into ``(stamp_ns, np.ndarray)``:

* :func:`joint_state_q` for ``LowState_`` / ``MasterArmState_`` (31 joints);
* :func:`motor_state_q` for ``HandState_`` (12 motors, positional order);
* :func:`motor_cmd_q_by_id` for ``HandCmd_`` (12 motors reordered by ``id``);
* :func:`body_cmd_q_by_id` for ``LowCmd_`` (31 ``motors`` placed by ``id``, the
  policy's final joint command; ``robot_state.reference_topic: lowcmd``).
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from ...messages.robot_state import BODY_JOINT_COUNT, HAND_MOTOR_COUNT, HAND_MOTOR_IDS
from ...utils.clocks import stamp_to_ns, wall_ns
from ...utils.time_sync import StampedRingBuffer
from .channel import DdsChannel

logger = logging.getLogger(__name__)

Extractor = Callable[[Any], Tuple[int, np.ndarray]]

_ZERO_STAMP_WARNED: Set[int] = set()


def header_stamp_ns(msg) -> int:
    """``header.stamp`` as nanoseconds; 0 when the message has no usable stamp."""

    header = msg.header()
    stamp = getattr(header, "stamp", None)
    if stamp is not None:
        stamp = stamp()
        return stamp_to_ns(stamp.sec(), stamp.nanosec())
    # Older binding: Header.sec()/nanosec() directly.
    return stamp_to_ns(header.sec(), header.nanosec())


def _stamp_or_now(msg, tag: str) -> int:
    stamp = header_stamp_ns(msg)
    if stamp == 0:
        if id(tag) not in _ZERO_STAMP_WARNED:
            _ZERO_STAMP_WARNED.add(id(tag))
            logger.warning("%s samples carry no header stamp; using the receive time (matching quality degrades)", tag)
        stamp = wall_ns()
    return stamp


def joint_state_q(count: int = BODY_JOINT_COUNT) -> Extractor:
    def extract(msg):
        joints = msg.joint_state()
        values = np.fromiter((joints[i].q() for i in range(count)), dtype=np.float32, count=count)
        return _stamp_or_now(msg, "joint_state"), values

    return extract


def motor_state_q(count: int = HAND_MOTOR_COUNT) -> Extractor:
    def extract(msg):
        motors = msg.motor_state()
        values = np.fromiter((motors[i].q() for i in range(count)), dtype=np.float32, count=count)
        return _stamp_or_now(msg, "motor_state"), values

    return extract


def motor_cmd_q_by_id(ids: Sequence[int] = HAND_MOTOR_IDS) -> Extractor:
    index = {int(motor_id): slot for slot, motor_id in enumerate(ids)}

    def extract(msg):
        values = np.full(len(ids), np.nan, dtype=np.float32)
        for cmd in msg.motor_cmd():
            slot = index.get(int(cmd.id()))
            if slot is not None:
                values[slot] = cmd.q()
        return _stamp_or_now(msg, "motor_cmd"), values

    return extract


def body_cmd_q_by_id(count: int = BODY_JOINT_COUNT) -> Extractor:
    """``LowCmd_.motors[*].q`` placed by motor ``id`` (0..count-1, the LowState joint order).

    The inference node writes ``id = i`` for every joint; an id outside the
    range is ignored and a missing id stays NaN, so the assembler drops the
    sample instead of sending a partial command.
    """

    def extract(msg):
        values = np.full(count, np.nan, dtype=np.float32)
        for cmd in msg.motors():
            slot = int(cmd.id())
            if 0 <= slot < count:
                values[slot] = cmd.q()
        return _stamp_or_now(msg, "low_cmd"), values

    return extract


class StampedSubscription:
    """One SDK subscriber whose samples land in a ring buffer as ``(stamp_ns, vector)``."""

    def __init__(self, channel: DdsChannel, type_name: str, topic: str, extractor: Extractor, buffer: StampedRingBuffer, *, qos=None) -> None:
        self.channel = channel
        self.type_name = type_name
        self.topic = topic
        self.extractor = extractor
        self.buffer = buffer
        self.qos = qos
        self.received = 0
        self.errors = 0
        self._subscriber = None
        self._lock = threading.Lock()

    def start(self) -> None:
        sdk = self.channel.sdk
        subscriber_cls = getattr(sdk, f"{self.type_name}Subscriber", None)
        if subscriber_cls is None:
            raise RuntimeError(f"igris_c_sdk has no {self.type_name}Subscriber")
        qos = self.qos if self.qos is not None else sdk.QosProfile.SensorData()
        subscriber = subscriber_cls(self.topic, qos)
        if not subscriber.init(self._on_message):
            raise RuntimeError(f"failed to initialize DDS subscriber on {self.channel.resolve(self.topic)}")
        self._subscriber = subscriber
        logger.info("Subscribed %s on %s", self.type_name, self.channel.resolve(self.topic))

    def stop(self) -> None:
        subscriber, self._subscriber = self._subscriber, None
        if subscriber is not None:
            try:
                subscriber.stop()
            except Exception as error:
                logger.warning("stopping %s subscriber failed: %s", self.type_name, error)

    def _on_message(self, msg) -> None:
        # DDS thread: copy out, push, never raise.
        try:
            stamp_ns, values = self.extractor(msg)
            self.buffer.push(stamp_ns, values)
            self.received += 1
        except Exception as error:  # noqa: BLE001
            self.errors += 1
            if self.errors in (1, 10, 100) or self.errors % 1000 == 0:
                logger.warning("%s on %s: extractor failed (%d so far): %s", self.type_name, self.topic, self.errors, error)


@dataclass
class DdsStateTopics:
    """Bare topic names (the SDK adds namespace/prefix)."""

    lowstate: str = "lowstate"
    handstate: str = "handstate"
    handcmd: str = "handcmd"
    refstate: str = "gmr/refstate"
    lowcmd: str = "lowcmd"  # the inference node's LowCmd_, read for body_reference with robot_state.reference_topic: lowcmd
    recording_state: str = "recording/state"  # teleop signal (messages.teleop_dds), read for Signals.recording
    motion_feedback: str = "teleop/motion_feedback"  # teleop producer feedback (messages.teleop_dds), read for Signals.converging / arm_hold_*


# subscription key -> (message type, topic attribute, extractor factory, DdsStateBuffers field).
# Keys are the RobotStateSpec field names, plus "reference_lowcmd" for the reference
# group fed by LowCmd_ instead of MasterArmState_ (RobotStateSpec.dds_subscriptions()).
REFERENCE_LOWCMD_KEY = "reference_lowcmd"
_FIELD_SUBSCRIPTIONS: Dict[str, Tuple[str, str, Callable[[], Extractor], str]] = {
    "actual": ("LowState", "lowstate", joint_state_q, "actual"),
    "reference": ("MasterArmState", "refstate", joint_state_q, "reference"),
    REFERENCE_LOWCMD_KEY: ("LowCmd", "lowcmd", body_cmd_q_by_id, "reference"),
    "hand_actual": ("HandState", "handstate", motor_state_q, "hand_actual"),
    "hand_command": ("HandCmd", "handcmd", motor_cmd_q_by_id, "hand_command"),
}


def start_state_subscriptions(channel: DdsChannel, topics: DdsStateTopics, buffers, *, needed: Set[str]) -> List[StampedSubscription]:
    """Subscribe only the keys in ``needed`` (``RobotStateSpec.dds_subscriptions()``); returns the started subscriptions."""

    started: List[StampedSubscription] = []
    try:
        for name in sorted(needed):
            type_name, topic_attr, make_extractor, buffer_name = _FIELD_SUBSCRIPTIONS[name]
            sub = StampedSubscription(channel, type_name, getattr(topics, topic_attr), make_extractor(), buffers.get(buffer_name))
            sub.start()
            started.append(sub)
    except Exception:
        for sub in started:
            sub.stop()
        raise
    return started


__all__ = [
    "DdsStateTopics",
    "Extractor",
    "REFERENCE_LOWCMD_KEY",
    "StampedSubscription",
    "body_cmd_q_by_id",
    "header_stamp_ns",
    "joint_state_q",
    "motor_cmd_q_by_id",
    "motor_state_q",
    "start_state_subscriptions",
]
