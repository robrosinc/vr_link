"""``RobotState``: robot joint state matched to the video frame it precedes.

Joint order contract (also documented in vr_link.fbs):

* ``body_*``: 31 radians in the igris_c_sdk joint-space order
  (:data:`BODY_SERIAL_JOINT_NAMES`), the order of ``LowState.joint_state``,
  ``LowCmd.motors`` (by ``id``), ``gmr/refstate`` and the MJCF serial actuators.
  ``body_reference`` is the inference node's final ``LowCmd`` by default
  (``robot_state.reference_topic: lowcmd``) or the producer's retargeted
  ``gmr/refstate`` (``reference_topic: refstate``); the receiver cannot tell which.
* ``hand_*``: 12 values in motor-id order :data:`HAND_MOTOR_IDS`
  (11..16 right hand, 21..26 left hand).

Groups the sender could not match are absent (``None``) and go on the wire
as ``valid=false`` with an empty vector; a receiver must ignore them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import flatbuffers
import numpy as np
import vr_link_generated as _vr_link

from .envelope import DecodedEnvelope, MessageKind, finish_envelope, read_table

BODY_JOINT_COUNT = 31
HAND_MOTOR_COUNT = 12
HAND_MOTOR_IDS: Tuple[int, ...] = (11, 12, 13, 14, 15, 16, 21, 22, 23, 24, 25, 26)

BODY_SERIAL_JOINT_NAMES: Tuple[str, ...] = (
    "waist_yaw", "waist_roll", "waist_pitch",
    "l_hip_pitch", "l_hip_roll", "l_hip_yaw", "l_knee_pitch", "l_ankle_pitch", "l_ankle_roll",
    "r_hip_pitch", "r_hip_roll", "r_hip_yaw", "r_knee_pitch", "r_ankle_pitch", "r_ankle_roll",
    "l_shoulder_pitch", "l_shoulder_roll", "l_shoulder_yaw", "l_elbow_pitch", "l_wrist_yaw", "l_wrist_roll", "l_wrist_pitch",
    "r_shoulder_pitch", "r_shoulder_roll", "r_shoulder_yaw", "r_elbow_pitch", "r_wrist_yaw", "r_wrist_roll", "r_wrist_pitch",
    "neck_yaw", "neck_pitch",
)
assert len(BODY_SERIAL_JOINT_NAMES) == BODY_JOINT_COUNT

_U64 = 0xFFFFFFFFFFFFFFFF


@dataclass(frozen=True)
class StampedVector:
    values: np.ndarray  # float32, fixed length
    stamp_ns: int = 0


@dataclass(frozen=True)
class RobotStateSample:
    body_actual: Optional[StampedVector] = None
    body_reference: Optional[StampedVector] = None
    hand_actual: Optional[StampedVector] = None
    hand_command: Optional[StampedVector] = None
    base_pose: Optional[Tuple[float, ...]] = None  # px py pz qw qx qy qz, sim world frame

    def is_empty(self) -> bool:
        return all(v is None for v in (self.body_actual, self.body_reference, self.hand_actual, self.hand_command, self.base_pose))


_GROUPS = (
    # (field, expected length, add vector, add valid, add stamp, read valid, read numpy, read length, read stamp)
    ("body_actual", BODY_JOINT_COUNT),
    ("body_reference", BODY_JOINT_COUNT),
    ("hand_actual", HAND_MOTOR_COUNT),
    ("hand_command", HAND_MOTOR_COUNT),
)


def _camel(name: str) -> str:
    return "".join(part.capitalize() for part in name.split("_"))


def _as_vector(name: str, expected: int, vec: StampedVector) -> np.ndarray:
    values = np.asarray(vec.values, dtype=np.float32).reshape(-1)
    if values.shape[0] != expected:
        raise ValueError(f"{name} must have {expected} values, got {values.shape[0]}")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} contains non-finite values")
    return values


def build_robot_state_payload(sample: RobotStateSample, *, seq: int, wall_ns: int, sim_ns: int = 0) -> bytes:
    builder = flatbuffers.Builder(512)
    offsets = {}
    for name, expected in _GROUPS:
        vec: Optional[StampedVector] = getattr(sample, name)
        if vec is not None:
            offsets[name] = builder.CreateNumpyVector(_as_vector(name, expected, vec))
    _vr_link.RobotStateStart(builder)
    for name, _ in _GROUPS:
        vec = getattr(sample, name)
        camel = _camel(name)
        if vec is None:
            getattr(_vr_link, f"RobotStateAdd{camel}Valid")(builder, False)
            continue
        getattr(_vr_link, f"RobotStateAdd{camel}")(builder, offsets[name])
        getattr(_vr_link, f"RobotStateAdd{camel}Valid")(builder, True)
        getattr(_vr_link, f"RobotStateAdd{camel}StampNs")(builder, int(vec.stamp_ns) & _U64)
    if sample.base_pose is not None:
        pose = tuple(float(v) for v in sample.base_pose)
        if len(pose) != 7:
            raise ValueError("base_pose must be (px, py, pz, qw, qx, qy, qz)")
        _vr_link.RobotStateAddBasePose(builder, _vr_link.CreatePose(builder, *pose, True))
    body = _vr_link.RobotStateEnd(builder)
    return finish_envelope(builder, body, _vr_link.Message.RobotState, seq=seq, wall_ns=wall_ns, sim_ns=sim_ns)


def decode_robot_state(env: DecodedEnvelope) -> RobotStateSample:
    state = read_table(env, MessageKind.ROBOT_STATE, _vr_link.RobotState)
    fields = {}
    for name, expected in _GROUPS:
        camel = _camel(name)
        valid = getattr(state, f"{camel}Valid")()
        length = getattr(state, f"{camel}Length")()
        if valid and length == expected:
            values = np.array(getattr(state, f"{camel}AsNumpy")(), dtype=np.float32, copy=True)
            fields[name] = StampedVector(values, getattr(state, f"{camel}StampNs")())
        else:
            fields[name] = None
    pose = state.BasePose()
    base_pose = None
    if pose is not None and pose.Valid():
        base_pose = (pose.Px(), pose.Py(), pose.Pz(), pose.Qw(), pose.Qx(), pose.Qy(), pose.Qz())
    return RobotStateSample(base_pose=base_pose, **fields)


def vector(values: Sequence[float], stamp_ns: int = 0) -> StampedVector:
    """Convenience constructor used by tests and simple providers."""

    return StampedVector(np.asarray(values, dtype=np.float32), int(stamp_ns))
