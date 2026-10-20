"""Simulator vision shared-memory segment as a frame (and robot-state) source.

The simulator renders ``Left_Eye``/``Right_Eye`` side by side (left eye in
the LEFT half, the layout igris_c_camera publishes for the real robot) into
a POSIX shared-memory segment (``/dev/shm/igris_c_mujoco_vision_shm``,
layout owned by this repository in ``include/vr_link/vision_shm.h``) under
a seqlock. This source mmaps the segment, accepts a frame only when ``seq``
is even and unchanged across the pixel copy, and skips frames superseded
before they are consumed (latest-only, low latency).

Three header layouts are understood; the layout version is the low byte of
the magic (``'MJV' | version``):

* v1, 64 bytes: ``<IIIIII QQQ 16x`` = magic, seq, width, height, channels,
  step, stamp_realtime_ns, sim_time_ns, frame_index, reserved.
* v2, 128 bytes: v1 fields followed by the static per-eye camera model
  ``ff 3f4f 3f4f f II 4x`` = left/right fovy (deg), left pos xyz + quat wxyz,
  right pos xyz + quat wxyz (parent-body frame, MuJoCo camera axes),
  baseline (m), eye width, eye height. It lets the pump serve the ideal
  calibration without a pre-generated YAML.
* v3, 384 bytes: v2 fields followed by the robot-state block written under
  the same seqlock as the pixels ``QQ IIII 31f 12f 7f 24x`` = state stamp
  (realtime ns), state sim time, valid bits, joint count (31), motor count
  (12), flags, body_actual[31], hand_actual[12], base_pose[7]. The reader
  hands it out as :class:`~vr_link.sources.ShmRobotBlock` on each frame.

A producer restart recreates the segment: the mapping is re-opened when the
inode changes or the magic is cleared.
"""

from __future__ import annotations

import logging
import mmap
import os
import struct
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from . import Frame, ShmRobotBlock, StereoCameraMetadata
from ..messages.robot_state import BODY_JOINT_COUNT, HAND_MOTOR_COUNT

logger = logging.getLogger(__name__)

SHM_DIR = Path("/dev/shm")
DEFAULT_SHM_NAME = "/igris_c_mujoco_vision_shm"
SHM_MAGIC_BASE = 0x4D4A5600  # 'MJV' << 8; low byte = layout version
SHM_CHANNELS = 3  # RGB8

HEADER_V1_FORMAT = "<IIIIIIQQQ16x"
HEADER_V1_SIZE = struct.calcsize(HEADER_V1_FORMAT)  # 64
HEADER_V2_FORMAT = "<IIIIIIQQQff3f4f3f4ffII4x"
HEADER_V2_SIZE = struct.calcsize(HEADER_V2_FORMAT)  # 128
HEADER_V3_FORMAT = HEADER_V2_FORMAT + "QQIIII31f12f7f24x"
HEADER_V3_SIZE = struct.calcsize(HEADER_V3_FORMAT)  # 384
HEADER_SIZES = {1: HEADER_V1_SIZE, 2: HEADER_V2_SIZE, 3: HEADER_V3_SIZE}
LAYOUT_VERSION = 3  # what pack_shm_header writes by default
MAX_HEADER_SIZE = max(HEADER_SIZES.values())

# Polling period while waiting for the producer to bump frame_index (frames arrive at 30 Hz).
DEFAULT_POLL_SEC = 0.001
# Retry period while the segment is missing or has no producer magic.
DEFAULT_RETRY_SEC = 1.0
IDLE_WARN_INTERVAL_SEC = 5.0


class ShmLayoutError(ValueError):
    """The segment header is not a layout this reader understands."""


@dataclass(frozen=True)
class ShmHeader:
    version: int
    seq: int
    width: int
    height: int
    channels: int
    step: int
    stamp_realtime_ns: int
    sim_time_ns: int
    frame_index: int
    metadata: Optional[StereoCameraMetadata]
    robot: Optional[ShmRobotBlock] = None

    @property
    def header_size(self) -> int:
        return HEADER_SIZES[self.version]

    @property
    def frame_bytes(self) -> int:
        return self.width * self.height * self.channels


def magic_version(magic: int) -> int:
    """Layout version encoded in the magic, or 0 when the magic is not ours."""

    if magic & 0xFFFFFF00 != SHM_MAGIC_BASE:
        return 0
    return magic & 0xFF


def parse_shm_header(buf: bytes) -> ShmHeader:
    """Decode a v1, v2 or v3 header from the first bytes of the segment."""

    if len(buf) < HEADER_V1_SIZE:
        raise ShmLayoutError(f"header needs {HEADER_V1_SIZE} bytes, got {len(buf)}")
    magic = struct.unpack_from("<I", buf, 0)[0]
    version = magic_version(magic)
    if version == 0:
        raise ShmLayoutError(f"magic 0x{magic:08X} is not an igris_c_mujoco vision segment")
    if version == 1:
        fields = struct.unpack_from(HEADER_V1_FORMAT, buf, 0)
        return ShmHeader(1, *fields[1:9], metadata=None)
    if version in (2, 3):
        size, fmt = HEADER_SIZES[version], (HEADER_V2_FORMAT if version == 2 else HEADER_V3_FORMAT)
        if len(buf) < size:
            raise ShmLayoutError(f"v{version} header needs {size} bytes, got {len(buf)}")
        fields = struct.unpack_from(fmt, buf, 0)
        (_magic, seq, width, height, channels, step, stamp, sim, index) = fields[:9]
        left_fovy, right_fovy = fields[9:11]
        left_pos = tuple(fields[11:14])
        left_quat = tuple(fields[14:18])
        right_pos = tuple(fields[18:21])
        right_quat = tuple(fields[21:25])
        baseline, eye_w, eye_h = fields[25:28]
        metadata = StereoCameraMetadata(
            eye_width=int(eye_w),
            eye_height=int(eye_h),
            left_fovy_deg=float(left_fovy),
            right_fovy_deg=float(right_fovy),
            left_pos=left_pos,
            left_quat_wxyz=left_quat,
            right_pos=right_pos,
            right_quat_wxyz=right_quat,
            baseline_m=float(baseline),
        )
        robot = None
        if version == 3:
            state_stamp, state_sim, valid, joint_count, motor_count, flags = fields[28:34]
            if joint_count != BODY_JOINT_COUNT or motor_count != HAND_MOTOR_COUNT:
                raise ShmLayoutError(
                    f"v3 robot block declares {joint_count} joints / {motor_count} motors; "
                    f"this reader expects {BODY_JOINT_COUNT} / {HAND_MOTOR_COUNT}"
                )
            body = np.array(fields[34:34 + BODY_JOINT_COUNT], dtype=np.float32)
            hand = np.array(fields[34 + BODY_JOINT_COUNT:34 + BODY_JOINT_COUNT + HAND_MOTOR_COUNT], dtype=np.float32)
            pose_start = 34 + BODY_JOINT_COUNT + HAND_MOTOR_COUNT
            robot = ShmRobotBlock(int(state_stamp), int(state_sim), int(valid), int(flags), body, hand, tuple(float(v) for v in fields[pose_start:pose_start + 7]))
        return ShmHeader(version, seq, width, height, channels, step, stamp, sim, index, metadata, robot)
    raise ShmLayoutError(f"unsupported vision SHM layout version {version} (this reader knows {sorted(HEADER_SIZES)})")


def pack_shm_header(
    *,
    version: int,
    seq: int,
    width: int,
    height: int,
    stamp_realtime_ns: int,
    sim_time_ns: int,
    frame_index: int,
    metadata: Optional[StereoCameraMetadata] = None,
    robot: Optional[ShmRobotBlock] = None,
) -> bytes:
    """Producer-side packer (tests and tools); mirrors ``parse_shm_header``.

    For v3 a missing ``robot`` block is written with all valid bits clear.
    """

    magic = SHM_MAGIC_BASE | version
    common = (magic, seq, width, height, SHM_CHANNELS, width * SHM_CHANNELS, stamp_realtime_ns, sim_time_ns, frame_index)
    if version == 1:
        return struct.pack(HEADER_V1_FORMAT, *common)
    if version in (2, 3):
        if metadata is None:
            raise ValueError(f"v{version} header needs camera metadata")
        m = metadata
        camera = (
            m.left_fovy_deg,
            m.right_fovy_deg,
            *m.left_pos,
            *m.left_quat_wxyz,
            *m.right_pos,
            *m.right_quat_wxyz,
            m.baseline_m,
            m.eye_width,
            m.eye_height,
        )
        if version == 2:
            return struct.pack(HEADER_V2_FORMAT, *common, *camera)
        if robot is None:
            robot = ShmRobotBlock(0, 0, 0, 0, np.zeros(BODY_JOINT_COUNT, np.float32), np.zeros(HAND_MOTOR_COUNT, np.float32), (0.0,) * 7)
        body = np.asarray(robot.body_actual, dtype=np.float32).reshape(-1)
        hand = np.asarray(robot.hand_actual, dtype=np.float32).reshape(-1)
        if body.shape[0] != BODY_JOINT_COUNT or hand.shape[0] != HAND_MOTOR_COUNT or len(robot.base_pose) != 7:
            raise ValueError("robot block needs 31 joints, 12 motors and a 7-float base pose")
        return struct.pack(
            HEADER_V3_FORMAT, *common, *camera,
            robot.state_stamp_realtime_ns, robot.state_sim_time_ns, robot.valid, BODY_JOINT_COUNT, HAND_MOTOR_COUNT, robot.flags,
            *body.tolist(), *hand.tolist(), *robot.base_pose,
        )
    raise ValueError(f"unknown layout version {version}")


class ShmSource:
    """Latest-frame reader of the vision segment (see module docstring)."""

    def __init__(
        self,
        shm_name: str = DEFAULT_SHM_NAME,
        *,
        shm_dir: Path = SHM_DIR,
        path: Optional[Path] = None,
        poll_sec: float = DEFAULT_POLL_SEC,
        retry_sec: float = DEFAULT_RETRY_SEC,
    ) -> None:
        self.path = Path(path) if path is not None else Path(shm_dir) / shm_name.lstrip("/")
        self.poll_sec = float(poll_sec)
        self.retry_sec = float(retry_sec)
        self._fd: Optional[int] = None
        self._mm: Optional[mmap.mmap] = None
        self._inode: Optional[int] = None
        self._header: Optional[ShmHeader] = None
        self._last_index = 0
        self._last_idle_warn = 0.0
        self._opened = False

    # -- mapping lifecycle -------------------------------------------------------

    def open(self) -> None:
        """Mark the source usable; the mapping itself is (re)attached lazily."""

        self._opened = True
        self.ensure_attached()

    def close(self) -> None:
        self._detach()
        self._opened = False

    def _detach(self) -> None:
        if self._mm is not None:
            self._mm.close()
            self._mm = None
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None
        self._inode = None
        self._header = None

    def _warn_idle(self, message: str) -> None:
        now = time.monotonic()
        if now - self._last_idle_warn >= IDLE_WARN_INTERVAL_SEC:
            logger.warning(message)
            self._last_idle_warn = now

    def _attach(self) -> bool:
        try:
            fd = os.open(self.path, os.O_RDONLY)
        except FileNotFoundError:
            self._warn_idle(f"{self.path} not found; is igris_c_mujoco running with the vision model?")
            return False
        try:
            st = os.fstat(fd)
            if st.st_size < HEADER_V1_SIZE:
                self._warn_idle(f"{self.path} is {st.st_size} B, smaller than the header; waiting")
                os.close(fd)
                return False
            mm = mmap.mmap(fd, st.st_size, access=mmap.ACCESS_READ)
        except OSError as error:
            logger.error("Cannot map %s: %s", self.path, error)
            os.close(fd)
            return False

        magic = struct.unpack_from("<I", mm, 0)[0]
        if magic == 0:
            self._warn_idle(f"{self.path}: no producer attached (magic 0); waiting for igris_c_mujoco")
            mm.close()
            os.close(fd)
            return False
        try:
            header = parse_shm_header(mm[:MAX_HEADER_SIZE])
        except ShmLayoutError as error:
            logger.error("%s: %s; refusing to attach", self.path, error)
            mm.close()
            os.close(fd)
            return False

        expected = header.header_size + header.frame_bytes
        if header.channels != SHM_CHANNELS or header.step != header.width * SHM_CHANNELS or st.st_size != expected:
            logger.error(
                "%s: inconsistent header (w=%d h=%d c=%d step=%d size=%d expected=%d); refusing to attach",
                self.path, header.width, header.height, header.channels, header.step, st.st_size, expected,
            )
            mm.close()
            os.close(fd)
            return False

        self._fd = fd
        self._mm = mm
        self._inode = st.st_ino
        self._header = header
        logger.info("Attached to %s: %dx%d RGB, layout v%d", self.path, header.width, header.height, header.version)
        return True

    def _still_valid(self) -> bool:
        try:
            st = os.stat(self.path)
        except FileNotFoundError:
            return False
        return st.st_ino == self._inode

    def ensure_attached(self) -> bool:
        if self._mm is not None:
            if self._still_valid():
                return True
            logger.warning("%s was recreated or removed; re-attaching", self.path)
            self._detach()
        return self._attach()

    # -- FrameSource -------------------------------------------------------------

    @property
    def frame_size(self) -> Optional[Tuple[int, int]]:
        if self._header is None:
            return None
        return (self._header.width, self._header.height)

    @property
    def fps_hint(self) -> Optional[float]:
        return 30.0  # VisionCapture::kFps

    def camera_metadata(self) -> Optional[StereoCameraMetadata]:
        if self._header is None:
            self.ensure_attached()
        return self._header.metadata if self._header is not None else None

    def current_header(self) -> Optional[ShmHeader]:
        if not self.ensure_attached() or self._mm is None:
            return None
        return parse_shm_header(self._mm[:MAX_HEADER_SIZE])

    def read_state(self, timeout: float = 0.0) -> Optional[ShmRobotBlock]:
        """Newest robot-state block (v3 only; None for v1/v2 or no producer). Header-only seqlock read."""

        deadline = time.monotonic() + timeout
        while True:
            header = self.current_header() if self.ensure_attached() else None
            if header is not None and header.seq % 2 == 0 and self._mm is not None:
                if struct.unpack_from("<I", self._mm, 4)[0] == header.seq:
                    return header.robot
            if time.monotonic() >= deadline:
                return None
            time.sleep(self.poll_sec)

    def skip_to_latest(self) -> None:
        """Ignore frames published before now (a freshly connected headset gets the live view)."""

        header = self.current_header()
        if header is not None:
            self._last_index = header.frame_index

    def read(self, timeout: float) -> Optional[Frame]:
        if not self._opened:
            raise RuntimeError("source not open")
        deadline = time.monotonic() + timeout
        while True:
            if not self.ensure_attached() or self._mm is None:
                if time.monotonic() >= deadline:
                    return None
                time.sleep(min(self.retry_sec, max(0.0, deadline - time.monotonic())))
                continue
            mm = self._mm
            magic = struct.unpack_from("<I", mm, 0)[0]
            if magic_version(magic) == 0:
                self._warn_idle(f"{self.path}: producer detached (magic cleared); waiting")
                self._detach()
                time.sleep(min(self.retry_sec, max(0.0, deadline - time.monotonic())))
                continue

            header = parse_shm_header(mm[:MAX_HEADER_SIZE])
            if header.frame_index > self._last_index and header.seq % 2 == 0:
                pixels = np.frombuffer(mm, dtype=np.uint8, count=header.frame_bytes, offset=header.header_size).copy()
                seq_after = struct.unpack_from("<I", mm, 4)[0]
                if seq_after == header.seq:
                    self._last_index = header.frame_index
                    self._header = header
                    stamp_ns = header.stamp_realtime_ns or time.clock_gettime_ns(time.CLOCK_REALTIME)
                    return Frame(
                        pixels=pixels.reshape(header.height, header.width, SHM_CHANNELS),
                        pixel_format="rgb24",
                        capture_ns=int(stamp_ns),
                        sim_ns=int(header.sim_time_ns),
                        index=int(header.frame_index),
                        robot_block=header.robot,
                    )
                continue  # torn read: the producer wrote during the copy, retry immediately

            if time.monotonic() >= deadline:
                return None
            time.sleep(self.poll_sec)


__all__ = [
    "DEFAULT_SHM_NAME",
    "HEADER_V1_FORMAT",
    "HEADER_V1_SIZE",
    "HEADER_V2_FORMAT",
    "HEADER_V2_SIZE",
    "HEADER_V3_FORMAT",
    "HEADER_V3_SIZE",
    "LAYOUT_VERSION",
    "ShmSource",
    "SHM_MAGIC_BASE",
    "ShmHeader",
    "ShmLayoutError",
    "magic_version",
    "pack_shm_header",
    "parse_shm_header",
]
