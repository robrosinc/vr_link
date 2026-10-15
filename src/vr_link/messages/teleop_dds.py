"""DDS teleoperation signals carried in igris_c_sdk ``Joy_`` messages.

Three bare topics (the SDK ``ChannelFactory`` prefixes the namespace):

* ``vr/controller`` (producer -> recorder): the VRS1 receiver (``igris_c_teleop``)
  relays the headset controller state and the headset-UI recording toggle
  counter so a recorder in another process (the hit-tasks Isaac bridge, a
  robot-side recorder) can act on it.
* ``recording/state`` (recorder -> HUD): whether an episode recording is
  active, read by ``vr-link-stream`` for ``Signals.recording`` on the 9100
  downlink. Only the recorder knows this; the HUD must show the recorder's
  truth, not a count of button presses.
* ``teleop/motion_feedback`` (producer -> HUD, 2.1): the teleop producer's reference
  shaping state (convergence ramp active + progress, per-arm hold while the
  operator's bare hand is untracked), read by ``vr-link-stream`` for the
  ``Signals.converging`` / ``arm_hold_*`` fields. Published every producer
  tick while the skeleton is connected; the HUD reads a stale sample as idle.

``Joy_`` (vendored ``sensor_msgs/Joy``: ``header``, ``axes[]`` float32,
``buttons[]`` int32) is reused so no SDK/IDL change is needed; the index layout
below is this module's contract. ``header.frame_id`` carries a layout tag and
the decoders return ``None`` for any other tag, so a gamepad ``Joy_`` on the
same topic name is ignored rather than misread. Layout changes append fields
and bump the tag; decoders accept longer vectors and reject shorter ones.

This module has no SDK dependency: it works on plain lists so producers,
consumers and tests share one encoding. :mod:`vr_link.network.dds.teleop`
wraps it in SDK publishers and subscribers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

CONTROLLER_TOPIC = "vr/controller"
RECORDING_STATE_TOPIC = "recording/state"
CONTROLLER_FRAME_ID = "vr_link.controller.v1"
RECORDING_STATE_FRAME_ID = "vr_link.recording.v1"

# vr/controller buttons (int32).
CONTROLLER_BUTTON_RECORD_TOGGLE_COUNT = 0  # VRState.record_toggle_count, relayed unchanged
CONTROLLER_BUTTON_SESSION_ID = 1  # relay-side counter: +1 each time the headset (re)connects
CONTROLLER_BUTTON_CONNECTED = 2  # 1 while the relay has fresh VRS1 records
CONTROLLER_BUTTON_COUNT = 3

# vr/controller axes (float32): one ControllerInput per hand, left then right,
# in the VRS1 ControllerInput field order (thumbstick split into x, y).
CONTROLLER_AXIS_FIELDS = (
    "trigger", "grip", "button_first", "button_second", "button_thumbstick", "thumbstick_x", "thumbstick_y",
)
CONTROLLER_AXES_PER_HAND = len(CONTROLLER_AXIS_FIELDS)
CONTROLLER_LEFT_AXIS_OFFSET = 0
CONTROLLER_RIGHT_AXIS_OFFSET = CONTROLLER_AXES_PER_HAND
CONTROLLER_AXIS_COUNT = 2 * CONTROLLER_AXES_PER_HAND

# recording/state buttons (int32).
RECORDING_BUTTON_RECORDING = 0  # 1 while an episode recording is active
RECORDING_BUTTON_DEMO_INDEX = 1  # the recorder's current demo id; -1 when none is open
RECORDING_BUTTON_COUNT = 2

# teleop/motion_feedback buttons (int32) and axes (float32).
MOTION_FEEDBACK_TOPIC = "teleop/motion_feedback"
MOTION_FEEDBACK_FRAME_ID = "vr_link.motion_feedback.v1"
MOTION_FEEDBACK_BUTTON_CONVERGING = 0  # 1 while the convergence ramp shapes the reference
MOTION_FEEDBACK_BUTTON_ARM_HOLD_LEFT = 1  # 1 while the left arm reference is held (hand untracked)
MOTION_FEEDBACK_BUTTON_ARM_HOLD_RIGHT = 2
MOTION_FEEDBACK_BUTTON_COUNT = 3
MOTION_FEEDBACK_AXIS_CONVERGE_PROGRESS = 0  # 0..1; 1 when idle
MOTION_FEEDBACK_AXIS_COUNT = 1

INT32_MIN = -(2**31)
INT32_MAX = 2**31 - 1

_ZERO_HAND: Tuple[float, ...] = (0.0,) * CONTROLLER_AXES_PER_HAND


@dataclass(frozen=True)
class ControllerSample:
    """One ``vr/controller`` message; ``left``/``right`` follow ``CONTROLLER_AXIS_FIELDS``."""

    record_toggle_count: int
    session_id: int
    connected: bool
    left: Tuple[float, ...] = _ZERO_HAND
    right: Tuple[float, ...] = _ZERO_HAND
    stamp_ns: int = 0  # producer CLOCK_REALTIME; 0 lets the publisher stamp at write time


@dataclass(frozen=True)
class RecordingStateSample:
    """One ``recording/state`` message."""

    recording: bool
    demo_index: int = -1
    stamp_ns: int = 0


@dataclass(frozen=True)
class MotionFeedbackSample:
    """One ``teleop/motion_feedback`` message (teleop producer reference-shaping state)."""

    converging: bool = False
    converge_progress: float = 1.0
    arm_hold_left: bool = False
    arm_hold_right: bool = False
    stamp_ns: int = 0


def controller_axes(controller_input) -> Tuple[float, ...]:
    """``vr_link.types.ControllerInput`` (or any object with its attributes) -> 7 axes."""

    thumbstick = tuple(controller_input.thumbstick)
    if len(thumbstick) != 2:
        raise ValueError(f"thumbstick must be (x, y), got {thumbstick!r}")
    return (
        float(controller_input.trigger),
        float(controller_input.grip),
        float(controller_input.button_first),
        float(controller_input.button_second),
        float(controller_input.button_thumbstick),
        float(thumbstick[0]),
        float(thumbstick[1]),
    )


def controller_sample_from_vr_state(state, *, session_id: int, connected: bool, stamp_ns: int = 0) -> ControllerSample:
    """Build the relay message from a decoded VRS1 ``VRInputState``."""

    return ControllerSample(
        record_toggle_count=int(state.record_toggle_count),
        session_id=int(session_id),
        connected=bool(connected),
        left=controller_axes(state.left_input),
        right=controller_axes(state.right_input),
        stamp_ns=int(stamp_ns),
    )


def _int32(name: str, value) -> int:
    number = int(value)
    if not INT32_MIN <= number <= INT32_MAX:
        raise ValueError(f"{name} {number} does not fit Joy_.buttons (int32)")
    return number


def _hand(name: str, values: Sequence[float]) -> List[float]:
    axes = [float(v) for v in values]
    if len(axes) != CONTROLLER_AXES_PER_HAND:
        raise ValueError(f"{name} needs {CONTROLLER_AXES_PER_HAND} axes, got {len(axes)}")
    return axes


def encode_controller(sample: ControllerSample) -> Tuple[str, List[float], List[int]]:
    """``ControllerSample`` -> ``(frame_id, axes, buttons)``."""

    if sample.record_toggle_count < 0:
        raise ValueError(f"record_toggle_count must be non-negative, got {sample.record_toggle_count}")
    axes = _hand("left", sample.left) + _hand("right", sample.right)
    buttons = [
        _int32("record_toggle_count", sample.record_toggle_count),
        _int32("session_id", sample.session_id),
        1 if sample.connected else 0,
    ]
    return CONTROLLER_FRAME_ID, axes, buttons


def decode_controller(frame_id: str, axes: Sequence[float], buttons: Sequence[int], *, stamp_ns: int = 0) -> Optional[ControllerSample]:
    """``(frame_id, axes, buttons)`` -> ``ControllerSample``; ``None`` when the tag is not ours."""

    if str(frame_id) != CONTROLLER_FRAME_ID:
        return None
    axes_list = [float(v) for v in axes]
    buttons_list = [int(v) for v in buttons]
    if len(axes_list) < CONTROLLER_AXIS_COUNT or len(buttons_list) < CONTROLLER_BUTTON_COUNT:
        raise ValueError(
            f"{CONTROLLER_FRAME_ID} needs >= {CONTROLLER_AXIS_COUNT} axes and >= {CONTROLLER_BUTTON_COUNT} buttons, "
            f"got {len(axes_list)} and {len(buttons_list)}"
        )
    left = tuple(axes_list[CONTROLLER_LEFT_AXIS_OFFSET:CONTROLLER_LEFT_AXIS_OFFSET + CONTROLLER_AXES_PER_HAND])
    right = tuple(axes_list[CONTROLLER_RIGHT_AXIS_OFFSET:CONTROLLER_RIGHT_AXIS_OFFSET + CONTROLLER_AXES_PER_HAND])
    return ControllerSample(
        record_toggle_count=buttons_list[CONTROLLER_BUTTON_RECORD_TOGGLE_COUNT],
        session_id=buttons_list[CONTROLLER_BUTTON_SESSION_ID],
        connected=buttons_list[CONTROLLER_BUTTON_CONNECTED] != 0,
        left=left,
        right=right,
        stamp_ns=int(stamp_ns),
    )


def encode_recording_state(sample: RecordingStateSample) -> Tuple[str, List[float], List[int]]:
    """``RecordingStateSample`` -> ``(frame_id, axes, buttons)``; axes are empty."""

    buttons = [1 if sample.recording else 0, _int32("demo_index", sample.demo_index)]
    return RECORDING_STATE_FRAME_ID, [], buttons


def decode_recording_state(frame_id: str, axes: Sequence[float], buttons: Sequence[int], *, stamp_ns: int = 0) -> Optional[RecordingStateSample]:
    """``(frame_id, axes, buttons)`` -> ``RecordingStateSample``; ``None`` when the tag is not ours."""

    if str(frame_id) != RECORDING_STATE_FRAME_ID:
        return None
    buttons_list = [int(v) for v in buttons]
    if len(buttons_list) < RECORDING_BUTTON_COUNT:
        raise ValueError(f"{RECORDING_STATE_FRAME_ID} needs >= {RECORDING_BUTTON_COUNT} buttons, got {len(buttons_list)}")
    return RecordingStateSample(
        recording=buttons_list[RECORDING_BUTTON_RECORDING] != 0,
        demo_index=buttons_list[RECORDING_BUTTON_DEMO_INDEX],
        stamp_ns=int(stamp_ns),
    )


def encode_motion_feedback(sample: MotionFeedbackSample) -> Tuple[str, List[float], List[int]]:
    """``MotionFeedbackSample`` -> ``(frame_id, axes, buttons)``."""

    progress = float(sample.converge_progress)
    if not progress == progress:
        raise ValueError("converge_progress must not be NaN")
    buttons = [
        1 if sample.converging else 0,
        1 if sample.arm_hold_left else 0,
        1 if sample.arm_hold_right else 0,
    ]
    return MOTION_FEEDBACK_FRAME_ID, [min(1.0, max(0.0, progress))], buttons


def decode_motion_feedback(frame_id: str, axes: Sequence[float], buttons: Sequence[int], *, stamp_ns: int = 0) -> Optional[MotionFeedbackSample]:
    """``(frame_id, axes, buttons)`` -> ``MotionFeedbackSample``; ``None`` when the tag is not ours."""

    if str(frame_id) != MOTION_FEEDBACK_FRAME_ID:
        return None
    axes_list = [float(v) for v in axes]
    buttons_list = [int(v) for v in buttons]
    if len(axes_list) < MOTION_FEEDBACK_AXIS_COUNT or len(buttons_list) < MOTION_FEEDBACK_BUTTON_COUNT:
        raise ValueError(
            f"{MOTION_FEEDBACK_FRAME_ID} needs >= {MOTION_FEEDBACK_AXIS_COUNT} axes and >= {MOTION_FEEDBACK_BUTTON_COUNT} buttons, "
            f"got {len(axes_list)} and {len(buttons_list)}"
        )
    return MotionFeedbackSample(
        converging=buttons_list[MOTION_FEEDBACK_BUTTON_CONVERGING] != 0,
        converge_progress=axes_list[MOTION_FEEDBACK_AXIS_CONVERGE_PROGRESS],
        arm_hold_left=buttons_list[MOTION_FEEDBACK_BUTTON_ARM_HOLD_LEFT] != 0,
        arm_hold_right=buttons_list[MOTION_FEEDBACK_BUTTON_ARM_HOLD_RIGHT] != 0,
        stamp_ns=int(stamp_ns),
    )


class RecordToggleTracker:
    """Turn the relayed ``record_toggle_count`` into start/stop toggles.

    VRS1 rule (README, "Recording toggle"): the receiver seeds ``last_seen``
    from the first record of a connection and applies ``(new - last_seen) % 2``
    toggles on change, because newest-only sampling may skip records. Over DDS
    the TCP connection boundary is invisible to the consumer, so the relay
    stamps every message with a ``session_id`` that changes when the headset
    skeleton (re)connects; the first sample of a new session re-seeds without
    toggling. Parity is taken on the signed difference, so a counter that
    moves backwards within one session (a sender restart the relay did not see
    as a reconnect) still toggles by the parity of the jump.
    """

    def __init__(self) -> None:
        self._session_id: Optional[int] = None
        self._last_count: Optional[int] = None

    @property
    def seeded(self) -> bool:
        return self._last_count is not None

    def reset(self) -> None:
        self._session_id = None
        self._last_count = None

    def update(self, sample: ControllerSample) -> int:
        """Return the number of toggles (0 or 1) to apply for ``sample``."""

        count = int(sample.record_toggle_count)
        if self._session_id != int(sample.session_id) or self._last_count is None:
            self._session_id = int(sample.session_id)
            self._last_count = count
            return 0
        toggles = (count - self._last_count) % 2
        self._last_count = count
        return int(toggles)


__all__ = [
    "CONTROLLER_AXES_PER_HAND",
    "CONTROLLER_AXIS_COUNT",
    "CONTROLLER_AXIS_FIELDS",
    "CONTROLLER_BUTTON_CONNECTED",
    "CONTROLLER_BUTTON_COUNT",
    "CONTROLLER_BUTTON_RECORD_TOGGLE_COUNT",
    "CONTROLLER_BUTTON_SESSION_ID",
    "CONTROLLER_FRAME_ID",
    "CONTROLLER_LEFT_AXIS_OFFSET",
    "CONTROLLER_RIGHT_AXIS_OFFSET",
    "CONTROLLER_TOPIC",
    "ControllerSample",
    "MOTION_FEEDBACK_AXIS_CONVERGE_PROGRESS",
    "MOTION_FEEDBACK_AXIS_COUNT",
    "MOTION_FEEDBACK_BUTTON_ARM_HOLD_LEFT",
    "MOTION_FEEDBACK_BUTTON_ARM_HOLD_RIGHT",
    "MOTION_FEEDBACK_BUTTON_CONVERGING",
    "MOTION_FEEDBACK_BUTTON_COUNT",
    "MOTION_FEEDBACK_FRAME_ID",
    "MOTION_FEEDBACK_TOPIC",
    "MotionFeedbackSample",
    "RECORDING_BUTTON_COUNT",
    "RECORDING_BUTTON_DEMO_INDEX",
    "RECORDING_BUTTON_RECORDING",
    "RECORDING_STATE_FRAME_ID",
    "RECORDING_STATE_TOPIC",
    "RecordToggleTracker",
    "RecordingStateSample",
    "controller_axes",
    "controller_sample_from_vr_state",
    "decode_controller",
    "decode_motion_feedback",
    "decode_recording_state",
    "encode_controller",
    "encode_motion_feedback",
    "encode_recording_state",
]
