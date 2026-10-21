"""``Signals``: sender-side flags (replaces the 1.x SimStatus record).

2.1 adds the teleop producer feedback (``converging``,
``arm_hold_left``/``arm_hold_right``) that the headset HUD shows while the
robot-side reference converges to the operator or an arm is held because the
operator's bare hand is not tracked. The sender fills them from the producer's
``teleop/motion_feedback`` DDS topic (:mod:`vr_link.messages.teleop_dds`); they default
to idle when no producer reports.

HUD contract for ``converging`` (2026-10-21): an indicator while true, removed
when it drops to false (the HUD returns to its normal teleop-state display).
``converge_progress`` is deprecated: the wire field stays for compatibility but
senders write 0 and receivers must ignore it.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass

import flatbuffers
import vr_link_generated as _vr_link

from .envelope import DecodedEnvelope, MessageKind, finish_envelope, read_table


class SourceKind(enum.IntEnum):
    UNKNOWN = _vr_link.SourceKind.Unknown
    ROBOT = _vr_link.SourceKind.Robot
    MUJOCO = _vr_link.SourceKind.MuJoCo
    ISAAC = _vr_link.SourceKind.Isaac

    @classmethod
    def parse(cls, name: str) -> "SourceKind":
        try:
            return cls[str(name).upper()]
        except KeyError as error:
            raise ValueError(f"unknown source kind {name!r}; expected one of {[k.name.lower() for k in cls]}") from error


@dataclass(frozen=True)
class SignalsSample:
    recording: bool = False
    source: SourceKind = SourceKind.UNKNOWN
    # Teleop producer feedback (2.1); idle defaults when the producer does not report.
    converging: bool = False
    converge_progress: float = 0.0  # deprecated: decoded from old senders only, never encoded (always 0 on the wire)
    arm_hold_left: bool = False
    arm_hold_right: bool = False


def build_signals_payload(sample: SignalsSample, *, seq: int, wall_ns: int, sim_ns: int = 0) -> bytes:
    builder = flatbuffers.Builder(64)
    _vr_link.SignalsStart(builder)
    _vr_link.SignalsAddRecording(builder, bool(sample.recording))
    _vr_link.SignalsAddSource(builder, int(sample.source))
    _vr_link.SignalsAddConverging(builder, bool(sample.converging))
    _vr_link.SignalsAddConvergeProgress(builder, 0.0)  # deprecated field, fixed at 0
    _vr_link.SignalsAddArmHoldLeft(builder, bool(sample.arm_hold_left))
    _vr_link.SignalsAddArmHoldRight(builder, bool(sample.arm_hold_right))
    body = _vr_link.SignalsEnd(builder)
    return finish_envelope(builder, body, _vr_link.Message.Signals, seq=seq, wall_ns=wall_ns, sim_ns=sim_ns)


def decode_signals(env: DecodedEnvelope) -> SignalsSample:
    signals = read_table(env, MessageKind.SIGNALS, _vr_link.Signals)
    try:
        source = SourceKind(signals.Source())
    except ValueError:
        source = SourceKind.UNKNOWN
    return SignalsSample(
        recording=bool(signals.Recording()),
        source=source,
        converging=bool(signals.Converging()),
        converge_progress=float(signals.ConvergeProgress()),
        arm_hold_left=bool(signals.ArmHoldLeft()),
        arm_hold_right=bool(signals.ArmHoldRight()),
    )
