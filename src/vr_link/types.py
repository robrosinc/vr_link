"""VR input data model and device interface for the VRS1 uplink.

Shared by every Python receiver of the ``robros.vr.VRState`` record (the
simulator, the real-robot producer, offline tools). The interface is
device-agnostic so fake and real VR implementations can be exchanged without
changing the consumer.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

# Wire order of ``VRState.body_joints``: the PICO body-tracking 24-joint
# skeleton (SMPL-X body order), identical to GMR's "xrobot" source order
# (general_motion_retargeting/raw_xrobot_motion.py XROBOT_BODY_JOINT_NAMES).
BODY_JOINT_NAMES = (
    "Pelvis",
    "Left_Hip",
    "Right_Hip",
    "Spine1",
    "Left_Knee",
    "Right_Knee",
    "Spine2",
    "Left_Ankle",
    "Right_Ankle",
    "Spine3",
    "Left_Foot",
    "Right_Foot",
    "Neck",
    "Left_Collar",
    "Right_Collar",
    "Head",
    "Left_Shoulder",
    "Right_Shoulder",
    "Left_Elbow",
    "Right_Elbow",
    "Left_Wrist",
    "Right_Wrist",
    "Left_Hand",
    "Right_Hand",
)

# Compatibility alias for consumers written against the hit-bench module.
XROBOT_BODY_JOINT_NAMES = BODY_JOINT_NAMES


@dataclass(frozen=True)
class TrackedPose:
    position_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    quat_wxyz: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    valid: bool = False


# OpenXR XR_EXT_hand_tracking joint count and order (Palm, Wrist, thumb x4,
# then Metacarpal/Proximal/Intermediate/Distal/Tip for index..little).
HAND_KEYPOINT_COUNT = 26


@dataclass(frozen=True)
class HandTrackingState:
    """One hand-tracking sample for a single hand.

    ``keypoints`` are wrist-relative poses expressed in the wrist frame (26
    entries in OpenXR order, entry 1 is identity). When ``tracked`` is False
    the keypoints are stale and must not be acted on (hand-tracking runtimes
    keep serving last-known poses after tracking loss).
    """

    tracked: bool = False
    # Hand-tracking wrist in the sim world frame. Differs from the
    # body-tracking wrist in body_joints by a per-hand yaw convention offset.
    wrist: TrackedPose = TrackedPose()
    keypoints: tuple[TrackedPose, ...] = ()


@dataclass(frozen=True)
class ControllerInput:
    trigger: float = 0.0
    grip: float = 0.0
    button_first: float = 0.0  # right: A, left: X
    button_second: float = 0.0  # right: B, left: Y
    button_thumbstick: float = 0.0
    thumbstick: tuple[float, float] = (0.0, 0.0)  # (x, y), each in [-1, 1]


@dataclass(frozen=True)
class VRInputState:
    """Single snapshot of VR tracking input.

    Poses are expressed in the device's world frame using the same axis
    convention as the simulation world (x forward, y left, z up).
    """

    # Monotonic sample time in seconds.
    timestamp: float = 0.0

    head: TrackedPose = TrackedPose()

    left_input: ControllerInput = ControllerInput()
    right_input: ControllerInput = ControllerInput()

    # Device-fused full-body skeleton in BODY_JOINT_NAMES order (24 entries
    # when present, empty tuple when the device provides no skeleton).
    body_joints: tuple[TrackedPose, ...] = ()

    # Whether the device currently delivers valid tracking data.
    connected: bool = True

    episode_started: bool = False
    episode_ended: bool = False

    # Headset-UI recording toggle requests as a monotonic counter (VRS1
    # record_toggle_count). State, not an event: consumers apply
    # (new - last_seen) % 2 toggles and re-seed last_seen on reconnect.
    record_toggle_count: int = 0

    # Hand-tracking samples. None when the sender build/device provides no
    # hand tracking (schema table absent); present with tracked=False when the
    # hand is merely not tracked this frame.
    left_hand: HandTrackingState | None = None
    right_hand: HandTrackingState | None = None


class VRDevice(ABC):
    """Abstract VR input source polled once per control step."""

    @abstractmethod
    def advance(self, dt: float) -> VRInputState:
        """Advance the device clock by ``dt`` seconds and return the latest state."""
        raise NotImplementedError

    def reset(self) -> None:
        """Reset internal state; the default implementation does nothing."""

    def close(self) -> None:
        """Release device resources; the default implementation does nothing."""
