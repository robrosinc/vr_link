"""Vision segment reader: v1/v2/v3 headers, seqlock, metadata, robot-state block."""
import struct

import numpy as np
import pytest

from vr_link.sources import STATE_FLAG_RECORDING, STATE_VALID_BASE_POSE, STATE_VALID_BODY_ACTUAL, STATE_VALID_RECORDING, ShmRobotBlock, StereoCameraMetadata
from vr_link.sources import shm

W, H = 8, 4  # tiny SBS frame: 4x4 per eye
META = StereoCameraMetadata(
    eye_width=4, eye_height=4, left_fovy_deg=90.0, right_fovy_deg=90.0,
    left_pos=(0.095516, 0.026076, 0.121), left_quat_wxyz=(0.55720775, 0.55720775, -0.43533840, -0.43533840),
    right_pos=(0.095516, -0.026076, 0.121), right_quat_wxyz=(0.43533840, 0.43533840, -0.55720775, -0.55720775),
    baseline_m=0.052152,
)


ROBOT = ShmRobotBlock(
    state_stamp_realtime_ns=999, state_sim_time_ns=20_000,
    valid=STATE_VALID_BODY_ACTUAL | STATE_VALID_BASE_POSE | STATE_VALID_RECORDING, flags=STATE_FLAG_RECORDING,
    body_actual=np.arange(31, dtype=np.float32) * 0.1, hand_actual=np.zeros(12, np.float32),
    base_pose=(0.1, 0.2, 0.9, 1.0, 0.0, 0.0, 0.0),
)


def _segment(path, *, version, seq, index, stamp=123, sim=456, fill=7, robot=None, right_fill=None):
    header = shm.pack_shm_header(
        version=version, seq=seq, width=W, height=H, stamp_realtime_ns=stamp, sim_time_ns=sim,
        frame_index=index, metadata=META if version >= 2 else None, robot=robot,
    )
    pixels = np.full((H, W, 3), fill, dtype=np.uint8)
    if right_fill is not None:
        pixels[:, W // 2 :] = right_fill
    path.write_bytes(header + pixels.tobytes())
    return path


def test_halves_are_swapped_for_a_right_eye_left_producer(tmp_path):
    # The segment holds the right eye in the left half; the stream wants the left eye there.
    path = _segment(tmp_path / "seg", version=2, seq=2, index=1, fill=10, right_fill=20)
    source = shm.ShmSource(path=path, poll_sec=0.001, retry_sec=0.01)  # left_half defaults to "right"
    source.open()
    try:
        frame = source.read(timeout=0.5)
        assert (frame.pixels[:, : W // 2] == 20).all() and (frame.pixels[:, W // 2 :] == 10).all()
    finally:
        source.close()
    source = shm.ShmSource(path=path, poll_sec=0.001, retry_sec=0.01, left_half="left")
    source.open()
    try:
        frame = source.read(timeout=0.5)
        assert (frame.pixels[:, : W // 2] == 10).all() and (frame.pixels[:, W // 2 :] == 20).all()
    finally:
        source.close()
    with pytest.raises(ValueError):
        shm.ShmSource(path=path, left_half="top")


def test_header_formats_have_the_documented_sizes():
    assert shm.HEADER_V1_SIZE == 64
    assert shm.HEADER_V2_SIZE == 128
    assert shm.HEADER_V3_SIZE == 384 and struct.calcsize(shm.HEADER_V3_FORMAT) == 384
    # Offsets the C header static_asserts (include/vr_link/vision_shm.h).
    prefix = shm.HEADER_V2_FORMAT
    assert struct.calcsize(prefix + "QQIIII") == 160  # body_actual
    assert struct.calcsize(prefix + "QQIIII31f") == 284  # hand_actual
    assert struct.calcsize(prefix + "QQIIII31f12f") == 332  # base_pose


def test_v3_header_round_trips_the_robot_block():
    buf = shm.pack_shm_header(version=3, seq=6, width=W, height=H, stamp_realtime_ns=1, sim_time_ns=2, frame_index=7, metadata=META, robot=ROBOT)
    assert len(buf) == 384
    header = shm.parse_shm_header(buf)
    assert header.version == 3 and header.header_size == 384 and header.metadata.eye_width == 4
    robot = header.robot
    assert (robot.state_stamp_realtime_ns, robot.state_sim_time_ns) == (999, 20_000)
    assert robot.has_body_actual and robot.has_base_pose and not robot.has_hand_actual and robot.recording is True
    np.testing.assert_allclose(robot.body_actual, ROBOT.body_actual)
    assert robot.base_pose == pytest.approx(ROBOT.base_pose)
    # Default robot block: everything invalid, recording unknown.
    empty = shm.parse_shm_header(shm.pack_shm_header(version=3, seq=0, width=W, height=H, stamp_realtime_ns=0, sim_time_ns=0, frame_index=0, metadata=META)).robot
    assert empty.valid == 0 and empty.recording is None


def test_v3_with_foreign_joint_counts_is_rejected():
    buf = bytearray(shm.pack_shm_header(version=3, seq=0, width=W, height=H, stamp_realtime_ns=0, sim_time_ns=0, frame_index=0, metadata=META))
    struct.pack_into("<I", buf, 148, 30)
    with pytest.raises(shm.ShmLayoutError):
        shm.parse_shm_header(bytes(buf))


def test_parse_v1_and_v2_headers():
    v1 = shm.parse_shm_header(shm.pack_shm_header(version=1, seq=2, width=W, height=H, stamp_realtime_ns=1, sim_time_ns=2, frame_index=3))
    assert (v1.version, v1.seq, v1.width, v1.height, v1.frame_index, v1.metadata) == (1, 2, W, H, 3, None)
    v2 = shm.parse_shm_header(shm.pack_shm_header(version=2, seq=4, width=W, height=H, stamp_realtime_ns=1, sim_time_ns=2, frame_index=5, metadata=META))
    assert v2.version == 2 and v2.header_size == 128
    assert v2.metadata.eye_width == 4 and v2.metadata.left_fovy_deg == pytest.approx(90.0)
    assert v2.metadata.left_quat_wxyz == pytest.approx(META.left_quat_wxyz, abs=1e-7)
    assert v2.metadata.baseline_m == pytest.approx(0.052152, abs=1e-7)


@pytest.mark.parametrize("magic", [0, 0x4D4A5604, 0xDEADBEEF])
def test_unknown_layouts_are_rejected(magic):
    buf = struct.pack("<I", magic) + b"\x00" * 380
    with pytest.raises(shm.ShmLayoutError):
        shm.parse_shm_header(buf)


def test_reads_frames_with_sim_time_and_metadata(tmp_path):
    path = _segment(tmp_path / "seg", version=2, seq=2, index=1, stamp=1_000, sim=20_000)
    source = shm.ShmSource(path=path, poll_sec=0.001, retry_sec=0.01)
    source.open()
    try:
        frame = source.read(timeout=0.5)
        assert frame is not None
        assert frame.pixels.shape == (H, W, 3) and frame.pixel_format == "rgb24"
        assert (frame.capture_ns, frame.sim_ns, frame.index) == (1_000, 20_000, 1)
        assert source.frame_size == (W, H)
        assert source.camera_metadata().left_fovy_deg == pytest.approx(90.0)
        # Same frame index again: nothing new within the timeout.
        assert source.read(timeout=0.05) is None
        # A newer frame is picked up in place.
        _segment(path, version=2, seq=4, index=2, fill=9)
        frame2 = source.read(timeout=0.5)
        assert frame2.index == 2 and int(frame2.pixels[0, 0, 0]) == 9
        assert frame2.robot_block is None and source.read_state() is None  # v2 has no robot block
    finally:
        source.close()


def test_v3_frames_carry_the_robot_block_and_read_state_is_header_only(tmp_path):
    path = _segment(tmp_path / "seg", version=3, seq=2, index=1, robot=ROBOT)
    source = shm.ShmSource(path=path, poll_sec=0.001, retry_sec=0.01)
    source.open()
    try:
        frame = source.read(timeout=0.5)
        assert frame.robot_block is not None and frame.robot_block.has_body_actual
        np.testing.assert_allclose(frame.robot_block.body_actual, ROBOT.body_actual)
        state = source.read_state()
        assert state is not None and state.state_sim_time_ns == 20_000
        _segment(path, version=3, seq=3, index=2, robot=ROBOT)  # write in progress
        assert source.read_state(timeout=0.02) is None
    finally:
        source.close()


def test_odd_seq_is_a_write_in_progress(tmp_path):
    path = _segment(tmp_path / "seg", version=1, seq=3, index=1)
    source = shm.ShmSource(path=path, poll_sec=0.001, retry_sec=0.01)
    source.open()
    try:
        assert source.read(timeout=0.05) is None
        assert source.camera_metadata() is None
        assert source.fps_hint == 30.0
    finally:
        source.close()


def test_missing_segment_waits_without_raising(tmp_path):
    source = shm.ShmSource(path=tmp_path / "absent", poll_sec=0.001, retry_sec=0.01)
    source.open()
    try:
        assert source.read(timeout=0.05) is None
    finally:
        source.close()
