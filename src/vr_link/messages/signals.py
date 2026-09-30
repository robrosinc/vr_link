"""``Signals``: sender-side flags (replaces the 1.x SimStatus record)."""

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


def build_signals_payload(sample: SignalsSample, *, seq: int, wall_ns: int, sim_ns: int = 0) -> bytes:
    builder = flatbuffers.Builder(64)
    _vr_link.SignalsStart(builder)
    _vr_link.SignalsAddRecording(builder, bool(sample.recording))
    _vr_link.SignalsAddSource(builder, int(sample.source))
    body = _vr_link.SignalsEnd(builder)
    return finish_envelope(builder, body, _vr_link.Message.Signals, seq=seq, wall_ns=wall_ns, sim_ns=sim_ns)


def decode_signals(env: DecodedEnvelope) -> SignalsSample:
    signals = read_table(env, MessageKind.SIGNALS, _vr_link.Signals)
    try:
        source = SourceKind(signals.Source())
    except ValueError:
        source = SourceKind.UNKNOWN
    return SignalsSample(bool(signals.Recording()), source)
