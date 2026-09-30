"""RobotStateAssembler: per-field sources, tolerance matching, no stale values."""
import numpy as np
import pytest

from vr_link.messages import SourceKind
from vr_link.pipeline.robot_state import DdsStateBuffers, MatchTolerances, RobotStateAssembler, RobotStateSpec, SignalsProvider, SignalsSpec
from vr_link.sources import STATE_FLAG_RECORDING, STATE_VALID_BASE_POSE, STATE_VALID_BODY_ACTUAL, STATE_VALID_RECORDING, ShmRobotBlock

MS = 1_000_000
T = 1_700_000_000_000_000_000


def _block(valid, flags=0):
    return ShmRobotBlock(T + 5, 42, valid, flags, np.arange(31, dtype=np.float32), np.ones(12, np.float32), (1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0))


def test_spec_validation_rejects_impossible_sources():
    with pytest.raises(ValueError):
        RobotStateSpec(reference="shm").validate()
    with pytest.raises(ValueError):
        RobotStateSpec(base_pose="dds").validate()
    with pytest.raises(ValueError):
        RobotStateSpec(actual="redis").validate()
    with pytest.raises(ValueError):
        RobotStateAssembler(RobotStateSpec(actual="dds"))  # dds without buffers
    assert RobotStateSpec(actual="shm", reference="dds").dds_fields() == {"reference"}
    assert RobotStateSpec().is_empty()


def test_dds_fields_match_nearest_within_tolerance_else_invalid():
    buffers = DdsStateBuffers()
    for stamp in (T - 20 * MS, T - 3 * MS, T + 4 * MS):
        buffers.actual.push(stamp, np.full(31, stamp % 1000, np.float32))
    buffers.reference.push(T - 14 * MS, np.full(31, 7.0, np.float32))
    buffers.hand_actual.push(T - 60 * MS, np.zeros(12, np.float32))  # too old for 40 ms
    spec = RobotStateSpec(actual="dds", reference="dds", hand_actual="dds", hand_command="dds")
    assembler = RobotStateAssembler(spec, MatchTolerances(), buffers)
    sample = assembler.assemble(T)
    assert sample.body_actual.stamp_ns == T - 3 * MS  # nearest, not newest
    assert sample.body_reference.stamp_ns == T - 14 * MS  # inside 15 ms
    assert sample.hand_actual is None and sample.hand_command is None  # stale / never received
    assert sample.base_pose is None
    stats = assembler.stats()
    assert stats["actual"].hits == 1 and stats["hand_actual"].misses == 1
    assert "actual 100%" in assembler.format_stats()


def test_bad_length_dds_samples_are_ignored():
    buffers = DdsStateBuffers()
    buffers.actual.push(T, np.zeros(30, np.float32))
    assembler = RobotStateAssembler(RobotStateSpec(actual="dds"), dds=buffers)
    assert assembler.assemble(T).body_actual is None


def test_shm_fields_follow_the_valid_bits():
    assembler = RobotStateAssembler(RobotStateSpec(actual="shm", hand_actual="shm", base_pose="shm"))
    sample = assembler.assemble(T, _block(STATE_VALID_BODY_ACTUAL | STATE_VALID_BASE_POSE))
    assert sample.body_actual.stamp_ns == T + 5 and sample.body_actual.values[30] == 30.0
    assert sample.hand_actual is None  # bit clear
    assert sample.base_pose == pytest.approx((1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0))
    assert assembler.assemble(T, None).is_empty()


def test_mixed_shm_and_dds_sources():
    buffers = DdsStateBuffers()
    buffers.reference.push(T + 1 * MS, np.full(31, 2.0, np.float32))
    buffers.hand_command.push(T - 30 * MS, np.linspace(0, 1, 12).astype(np.float32))
    assembler = RobotStateAssembler(RobotStateSpec(actual="shm", reference="dds", hand_command="dds", base_pose="shm"), dds=buffers)
    sample = assembler.assemble(T, _block(STATE_VALID_BODY_ACTUAL))
    assert sample.body_actual is not None and sample.body_reference is not None and sample.hand_command is not None
    assert sample.base_pose is None  # base_pose bit clear in this block


def test_assemble_latest_uses_age_not_nearest():
    buffers = DdsStateBuffers()
    buffers.actual.push(T - 500 * MS, np.zeros(31, np.float32))
    assembler = RobotStateAssembler(RobotStateSpec(actual="dds"), dds=buffers)
    assert assembler.assemble_latest(T, 100 * MS).body_actual is None
    assert assembler.assemble_latest(T, 600 * MS).body_actual is not None


def test_signals_provider():
    provider = SignalsProvider(SignalsSpec(source=SourceKind.MUJOCO, recording="shm"))
    assert provider.sample(_block(STATE_VALID_RECORDING, STATE_FLAG_RECORDING)).recording is True
    assert provider.sample(_block(STATE_VALID_RECORDING, 0)).recording is False
    assert provider.sample(_block(0, STATE_FLAG_RECORDING)).recording is False  # flag not valid
    assert provider.sample(None) == provider.sample(None)
    assert SignalsProvider(SignalsSpec(source=SourceKind.ROBOT)).sample(None).source is SourceKind.ROBOT
    with pytest.raises(ValueError):
        SignalsSpec(recording="dds").validate()
