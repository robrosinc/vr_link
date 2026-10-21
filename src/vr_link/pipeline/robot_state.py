"""Assemble the per-frame ``RobotState`` and ``Signals`` from configured sources.

Each field of :class:`RobotStateSample` can come from a different place:

* ``dds``  a :class:`~vr_link.utils.time_sync.StampedRingBuffer` fed by the
  DDS subscribers (:mod:`vr_link.network.dds.subscribers`); the sample nearest
  to the frame's capture time within the field's tolerance is used, else the
  field is left invalid (never a stale value);
* ``shm``  the robot block the simulator wrote with the frame
  (:class:`~vr_link.sources.ShmRobotBlock`), already exact;
* ``none`` never sent.

The ``reference`` group has a second axis, ``reference_topic``: which DDS
topic/type feeds it when its source is ``dds``. ``refstate`` is the producer's
retargeted ``gmr/refstate`` (``MasterArmState_``), ``lowcmd`` the inference
node's final joint command (``LowCmd_``). Both name a key of ``dds.topics``.

The defaults reproduce what the headset matched itself in 1.x: LowState
within 5 ms, the gmr reference within 15 ms, hand state within 40 ms.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Literal, Optional, Set

import numpy as np

from ..messages.robot_state import BODY_JOINT_COUNT, HAND_MOTOR_COUNT, RobotStateSample, StampedVector
from ..messages.signals import SignalsSample, SourceKind
from ..network.dds.subscribers import REFERENCE_LOWCMD_KEY
from ..sources import ShmRobotBlock
from ..utils.clocks import ms_to_ns
from ..utils.time_sync import MatchStats, StampedRingBuffer

logger = logging.getLogger(__name__)

FieldSource = Literal["dds", "shm", "none"]
ReferenceTopic = Literal["refstate", "lowcmd"]
DDS_FIELDS = ("actual", "reference", "hand_actual", "hand_command")
SHM_FIELDS = ("actual", "hand_actual", "base_pose")
REFERENCE_TOPICS = ("refstate", "lowcmd")


@dataclass
class RobotStateSpec:
    """Where each RobotState group comes from."""

    actual: FieldSource = "none"
    reference: FieldSource = "none"       # dds | none
    hand_actual: FieldSource = "none"
    hand_command: FieldSource = "none"    # dds | none
    base_pose: FieldSource = "none"       # shm | none
    # Which dds.topics entry feeds the reference group when reference == "dds":
    # refstate (producer's retargeted MasterArmState_) or lowcmd (inference node's LowCmd_).
    reference_topic: ReferenceTopic = "refstate"

    def validate(self) -> "RobotStateSpec":
        for name in ("actual", "reference", "hand_actual", "hand_command", "base_pose"):
            value = getattr(self, name)
            if value not in ("dds", "shm", "none"):
                raise ValueError(f"robot_state.{name}: unknown source {value!r}")
            if value == "dds" and name not in DDS_FIELDS:
                raise ValueError(f"robot_state.{name} cannot come from dds")
            if value == "shm" and name not in SHM_FIELDS:
                raise ValueError(f"robot_state.{name} cannot come from shm")
        if self.reference_topic not in REFERENCE_TOPICS:
            raise ValueError(f"robot_state.reference_topic: unknown topic {self.reference_topic!r} (expected one of {REFERENCE_TOPICS})")
        return self

    def dds_fields(self) -> Set[str]:
        """Field names filled from DDS; keys of the match stats and buffers."""

        return {name for name in DDS_FIELDS if getattr(self, name) == "dds"}

    def dds_subscriptions(self) -> Set[str]:
        """Subscription keys for ``start_state_subscriptions``: the field names, with the reference
        group spelled ``reference_lowcmd`` when ``reference_topic`` is ``lowcmd``."""

        keys = self.dds_fields()
        if "reference" in keys and self.reference_topic == "lowcmd":
            keys.discard("reference")
            keys.add(REFERENCE_LOWCMD_KEY)
        return keys

    def uses_shm(self) -> bool:
        return any(getattr(self, name) == "shm" for name in SHM_FIELDS)

    def is_empty(self) -> bool:
        return not self.dds_fields() and not self.uses_shm()


@dataclass
class MatchTolerances:
    """Max |sample stamp - frame stamp| per DDS field, milliseconds."""

    actual_ms: float = 5.0
    reference_ms: float = 15.0
    hand_actual_ms: float = 40.0
    hand_command_ms: float = 40.0

    def ns(self, name: str) -> int:
        return ms_to_ns(getattr(self, f"{name}_ms"))


@dataclass
class DdsStateBuffers:
    """One stamped history per DDS-sourced field (filled by the subscriber layer)."""

    actual: StampedRingBuffer = field(default_factory=lambda: StampedRingBuffer(512, name="actual"))
    reference: StampedRingBuffer = field(default_factory=lambda: StampedRingBuffer(512, name="reference"))
    hand_actual: StampedRingBuffer = field(default_factory=lambda: StampedRingBuffer(512, name="hand_actual"))
    hand_command: StampedRingBuffer = field(default_factory=lambda: StampedRingBuffer(512, name="hand_command"))

    def get(self, name: str) -> StampedRingBuffer:
        return getattr(self, name)


_SAMPLE_FIELD = {"actual": "body_actual", "reference": "body_reference", "hand_actual": "hand_actual", "hand_command": "hand_command"}
_EXPECTED_LEN = {"actual": BODY_JOINT_COUNT, "reference": BODY_JOINT_COUNT, "hand_actual": HAND_MOTOR_COUNT, "hand_command": HAND_MOTOR_COUNT}


class RobotStateAssembler:
    """Builds the RobotStateSample matched to one frame."""

    def __init__(self, spec: RobotStateSpec, tolerances: Optional[MatchTolerances] = None, dds: Optional[DdsStateBuffers] = None) -> None:
        self.spec = spec.validate()
        self.tolerances = tolerances or MatchTolerances()
        self.dds = dds
        if self.spec.dds_fields() and self.dds is None:
            raise ValueError("robot_state uses dds fields but no DDS buffers were given")
        self.frames = 0
        self._warned: Set[str] = set()

    # -- helpers -----------------------------------------------------------------

    def _from_dds_nearest(self, name: str, target_ns: int) -> Optional[StampedVector]:
        assert self.dds is not None
        item = self.dds.get(name).nearest(target_ns, self.tolerances.ns(name))
        return None if item is None else self._checked(name, item.value, item.stamp_ns)

    def _checked(self, name: str, values, stamp_ns: int) -> Optional[StampedVector]:
        arr = np.asarray(values, dtype=np.float32).reshape(-1)
        if arr.shape[0] != _EXPECTED_LEN[name] or not np.all(np.isfinite(arr)):
            if name not in self._warned:
                self._warned.add(name)
                logger.warning("robot_state.%s: sample has %d values (expected %d) or non-finite entries; ignored", name, arr.shape[0], _EXPECTED_LEN[name])
            return None
        return StampedVector(arr, int(stamp_ns))

    @staticmethod
    def _from_shm(name: str, block: Optional[ShmRobotBlock]) -> Optional[StampedVector]:
        if block is None:
            return None
        if name == "actual" and block.has_body_actual:
            return StampedVector(np.asarray(block.body_actual, dtype=np.float32), block.state_stamp_realtime_ns)
        if name == "hand_actual" and block.has_hand_actual:
            return StampedVector(np.asarray(block.hand_actual, dtype=np.float32), block.state_stamp_realtime_ns)
        return None

    def _build(self, pick_dds, block: Optional[ShmRobotBlock]) -> RobotStateSample:
        fields: Dict[str, Optional[StampedVector]] = {}
        for name in DDS_FIELDS:
            source = getattr(self.spec, name)
            if source == "dds":
                fields[_SAMPLE_FIELD[name]] = pick_dds(name)
            elif source == "shm":
                fields[_SAMPLE_FIELD[name]] = self._from_shm(name, block)
            else:
                fields[_SAMPLE_FIELD[name]] = None
        base_pose = None
        if self.spec.base_pose == "shm" and block is not None and block.has_base_pose:
            base_pose = tuple(float(v) for v in block.base_pose)
        self.frames += 1
        return RobotStateSample(base_pose=base_pose, **fields)

    # -- API ---------------------------------------------------------------------

    def assemble(self, target_ns: int, shm_block: Optional[ShmRobotBlock] = None) -> RobotStateSample:
        """State matched to a frame captured at ``target_ns`` (CLOCK_REALTIME)."""

        return self._build(lambda name: self._from_dds_nearest(name, int(target_ns)), shm_block)

    def stats(self) -> Dict[str, MatchStats]:
        if self.dds is None:
            return {}
        return {name: self.dds.get(name).stats() for name in self.spec.dds_fields()}

    def format_stats(self) -> str:
        parts = []
        for name, stat in self.stats().items():
            parts.append(f"{name} {100.0 * stat.hit_rate:.0f}% (max {stat.max_error_ns / 1e6:.1f} ms)")
        return ", ".join(parts) if parts else "no dds fields"


RECORDING_SOURCES = ("shm", "dds", "none")
MOTION_FEEDBACK_SOURCES = ("dds", "none")
DDS_RECORDING_MAX_AGE_S = 1.0
DDS_MOTION_FEEDBACK_MAX_AGE_S = 1.0


@dataclass
class SignalsSpec:
    source: SourceKind = SourceKind.UNKNOWN
    # shm: the simulator's robot block flag; dds: the recorder's recording/state topic.
    recording: Literal["shm", "dds", "none"] = "none"
    # dds: the teleop producer's teleop/motion_feedback topic -> Signals.converging / arm_hold_*.
    motion_feedback: Literal["dds", "none"] = "none"

    def validate(self) -> "SignalsSpec":
        if self.recording not in RECORDING_SOURCES:
            raise ValueError(f"signals.recording: unknown source {self.recording!r}")
        if self.motion_feedback not in MOTION_FEEDBACK_SOURCES:
            raise ValueError(f"signals.motion_feedback: unknown source {self.motion_feedback!r}")
        return self


class SignalsProvider:
    """Signals for a frame: the configured source kind, the recording flag and the producer feedback.

    With ``recording: dds`` the flag is the newest ``recording/state`` sample
    (:class:`~vr_link.network.dds.teleop.RecordingStateSubscriber`); a sample
    older than ``max_age_s`` reads as not recording, so a recorder that went
    away does not leave the HUD showing a stale "recording". ``motion_feedback: dds``
    fills ``converging``/``arm_hold_*`` from the newest ``teleop/motion_feedback``
    sample the same way (stale reads as idle). The producer's ``converge_progress``
    is deprecated and not forwarded (``Signals.converge_progress`` is always 0).
    """

    def __init__(
        self,
        spec: SignalsSpec,
        recording_state=None,
        *,
        motion_feedback=None,
        max_age_s: float = DDS_RECORDING_MAX_AGE_S,
        motion_feedback_max_age_s: float = DDS_MOTION_FEEDBACK_MAX_AGE_S,
    ) -> None:
        self.spec = spec.validate()
        self.recording_state = recording_state
        self.motion_feedback = motion_feedback
        self.max_age_s = float(max_age_s)
        self.motion_feedback_max_age_s = float(motion_feedback_max_age_s)
        if self.spec.recording == "dds" and recording_state is None:
            raise ValueError("signals.recording is dds but no recording/state subscriber was given")
        if self.spec.motion_feedback == "dds" and motion_feedback is None:
            raise ValueError("signals.motion_feedback is dds but no teleop/motion_feedback subscriber was given")

    def sample(self, shm_block: Optional[ShmRobotBlock] = None) -> SignalsSample:
        recording = False
        if self.spec.recording == "shm" and shm_block is not None:
            recording = bool(shm_block.recording)
        elif self.spec.recording == "dds":
            recording = bool(self.recording_state.recording(self.max_age_s))
        converging, hold_left, hold_right = False, False, False
        if self.spec.motion_feedback == "dds":
            feedback = self.motion_feedback.state(self.motion_feedback_max_age_s)
            converging = bool(feedback.converging)
            hold_left, hold_right = bool(feedback.arm_hold_left), bool(feedback.arm_hold_right)
        return SignalsSample(
            recording=recording,
            source=self.spec.source,
            converging=converging,
            arm_hold_left=hold_left,
            arm_hold_right=hold_right,
        )


__all__ = [
    "DdsStateBuffers",
    "FieldSource",
    "MatchTolerances",
    "RobotStateAssembler",
    "RobotStateSpec",
    "SignalsProvider",
    "SignalsSpec",
]
