"""MuJoCo vision segment reader: v1/v2 headers, seqlock, metadata."""
import struct

import numpy as np
import pytest

from vr_link.sources import StereoCameraMetadata
from vr_link.sources import mujoco_shm as shm

W, H = 8, 4  # tiny SBS frame: 4x4 per eye
META = StereoCameraMetadata(
    eye_width=4, eye_height=4, left_fovy_deg=90.0, right_fovy_deg=90.0,
    left_pos=(0.095516, 0.026076, 0.121), left_quat_wxyz=(0.55720775, 0.55720775, -0.43533840, -0.43533840),
    right_pos=(0.095516, -0.026076, 0.121), right_quat_wxyz=(0.43533840, 0.43533840, -0.55720775, -0.55720775),
    baseline_m=0.052152,
)


def _segment(path, *, version, seq, index, stamp=123, sim=456, fill=7):
    header = shm.pack_shm_header(
        version=version, seq=seq, width=W, height=H, stamp_realtime_ns=stamp, sim_time_ns=sim,
        frame_index=index, metadata=META if version == 2 else None,
    )
    pixels = np.full((H, W, 3), fill, dtype=np.uint8).tobytes()
    path.write_bytes(header + pixels)
    return path


def test_header_formats_have_the_documented_sizes():
    assert shm.HEADER_V1_SIZE == 64
    assert shm.HEADER_V2_SIZE == 128


def test_parse_v1_and_v2_headers():
    v1 = shm.parse_shm_header(shm.pack_shm_header(version=1, seq=2, width=W, height=H, stamp_realtime_ns=1, sim_time_ns=2, frame_index=3))
    assert (v1.version, v1.seq, v1.width, v1.height, v1.frame_index, v1.metadata) == (1, 2, W, H, 3, None)
    v2 = shm.parse_shm_header(shm.pack_shm_header(version=2, seq=4, width=W, height=H, stamp_realtime_ns=1, sim_time_ns=2, frame_index=5, metadata=META))
    assert v2.version == 2 and v2.header_size == 128
    assert v2.metadata.eye_width == 4 and v2.metadata.left_fovy_deg == pytest.approx(90.0)
    assert v2.metadata.left_quat_wxyz == pytest.approx(META.left_quat_wxyz, abs=1e-7)
    assert v2.metadata.baseline_m == pytest.approx(0.052152, abs=1e-7)


@pytest.mark.parametrize("magic", [0, 0x4D4A5603, 0xDEADBEEF])
def test_unknown_layouts_are_rejected(magic):
    buf = struct.pack("<I", magic) + b"\x00" * 124
    with pytest.raises(shm.ShmLayoutError):
        shm.parse_shm_header(buf)


def test_reads_frames_with_sim_time_and_metadata(tmp_path):
    path = _segment(tmp_path / "seg", version=2, seq=2, index=1, stamp=1_000, sim=20_000)
    source = shm.MujocoShmSource(path=path, poll_sec=0.001, retry_sec=0.01)
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
    finally:
        source.close()


def test_odd_seq_is_a_write_in_progress(tmp_path):
    path = _segment(tmp_path / "seg", version=1, seq=3, index=1)
    source = shm.MujocoShmSource(path=path, poll_sec=0.001, retry_sec=0.01)
    source.open()
    try:
        assert source.read(timeout=0.05) is None
        assert source.camera_metadata() is None
        assert source.fps_hint == 30.0
    finally:
        source.close()


def test_missing_segment_waits_without_raising(tmp_path):
    source = shm.MujocoShmSource(path=tmp_path / "absent", poll_sec=0.001, retry_sec=0.01)
    source.open()
    try:
        assert source.read(timeout=0.05) is None
    finally:
        source.close()
