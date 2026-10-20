"""Stereo frame sources feeding the video pump.

A source delivers side-by-side (SBS) stereo frames, left eye in the LEFT
half (the layout igris_c_camera publishes over DDS and the headset renders
as-is), as one :class:`Frame` per call. Sources differ only in where pixels
come from: a USB camera (:mod:`vr_link.sources.usb`), the igris_c_mujoco
vision shared-memory segment (:mod:`vr_link.sources.shm`, layout owned by
``include/vr_link/vision_shm.h``; MuJoCo today, Isaac later) or the robot's
DDS image topic (:mod:`vr_link.sources.dds_camera`). Publishers (igris_c_camera,
the simulators) emit that layout directly; only the raw eyes_stereo USB
hardware has the right eye in the left half, and the USB source swaps the
halves itself (``eyes_stereo.left_half``).

Importing this package pulls in numpy only; each concrete source imports its
own backend (cv2, mmap, igris_c_sdk) lazily in ``open()``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Protocol, Tuple, runtime_checkable

import numpy as np

PixelFormat = Literal["rgb24", "bgr24"]


@dataclass
class Frame:
    """One SBS stereo frame.

    ``pixels`` is ``(height, width, 3)`` uint8 in ``pixel_format`` order.
    ``capture_ns`` is CLOCK_REALTIME at acquisition (the wire header
    sec/nanosec). ``sim_ns`` is the simulation time of the frame when the
    source has one (MuJoCo), else ``None``; ``index`` increases per frame.
    """

    pixels: np.ndarray
    pixel_format: PixelFormat
    capture_ns: int
    sim_ns: Optional[int]
    index: int
    # Robot state the producer wrote together with this frame (SHM layout v3), else None.
    robot_block: Optional["ShmRobotBlock"] = None

    @property
    def width(self) -> int:
        return int(self.pixels.shape[1])

    @property
    def height(self) -> int:
        return int(self.pixels.shape[0])


# state_valid bits of the SHM v3 robot-state block (include/vr_link/vision_shm.h).
STATE_VALID_BODY_ACTUAL = 1 << 0
STATE_VALID_HAND_ACTUAL = 1 << 1
STATE_VALID_BASE_POSE = 1 << 2
STATE_VALID_RECORDING = 1 << 3
STATE_FLAG_RECORDING = 1 << 0


@dataclass(frozen=True)
class ShmRobotBlock:
    """Robot state a simulator wrote alongside a frame (SHM layout v3).

    ``body_actual`` is 31 rad in the igris_c_sdk joint order, ``hand_actual``
    12 normalized motor values, ``base_pose`` 7 floats (px py pz qw qx qy qz)
    in the sim world frame. ``valid`` carries the ``STATE_VALID_*`` bits; a
    group whose bit is clear must be ignored.
    """

    state_stamp_realtime_ns: int
    state_sim_time_ns: int
    valid: int
    flags: int
    body_actual: np.ndarray
    hand_actual: np.ndarray
    base_pose: Tuple[float, ...]

    @property
    def has_body_actual(self) -> bool:
        return bool(self.valid & STATE_VALID_BODY_ACTUAL)

    @property
    def has_hand_actual(self) -> bool:
        return bool(self.valid & STATE_VALID_HAND_ACTUAL)

    @property
    def has_base_pose(self) -> bool:
        return bool(self.valid & STATE_VALID_BASE_POSE)

    @property
    def recording(self) -> Optional[bool]:
        """Recording flag, or None when the producer does not know it."""

        if not self.valid & STATE_VALID_RECORDING:
            return None
        return bool(self.flags & STATE_FLAG_RECORDING)


@dataclass(frozen=True)
class StereoCameraMetadata:
    """Per-eye camera model of a simulated stereo rig.

    Positions are meters and quaternions ``(w, x, y, z)`` in the frame of the
    body the cameras are attached to (MuJoCo ``cam_pos``/``cam_quat``
    semantics: camera looks down its -Z, +Y up). Enough to compute the ideal
    calibration on request (:func:`vr_link.calibration.ideal.ideal_stereo_from_metadata`).
    """

    eye_width: int
    eye_height: int
    left_fovy_deg: float
    right_fovy_deg: float
    left_pos: Tuple[float, float, float]
    left_quat_wxyz: Tuple[float, float, float, float]
    right_pos: Tuple[float, float, float]
    right_quat_wxyz: Tuple[float, float, float, float]
    baseline_m: float


@runtime_checkable
class FrameSource(Protocol):
    """Blocking-with-timeout stereo frame supplier driven by the pump thread."""

    def open(self) -> None:
        """Acquire the backend (device, mapping, subscription)."""
        ...

    def close(self) -> None:
        """Release the backend; idempotent."""
        ...

    @property
    def frame_size(self) -> Optional[Tuple[int, int]]:
        """``(width, height)`` of the SBS frame once known, else ``None``."""
        ...

    @property
    def fps_hint(self) -> Optional[float]:
        """Nominal frame rate, used for encoder timing; ``None`` if unknown."""
        ...

    def read(self, timeout: float) -> Optional[Frame]:
        """Return the next frame newer than the last one returned, or ``None`` on timeout."""
        ...

    def camera_metadata(self) -> Optional[StereoCameraMetadata]:
        """Camera model published by the source (simulators), else ``None``."""
        ...


__all__ = [
    "Frame", "FrameSource", "PixelFormat", "ShmRobotBlock", "StereoCameraMetadata",
    "STATE_FLAG_RECORDING", "STATE_VALID_BASE_POSE", "STATE_VALID_BODY_ACTUAL", "STATE_VALID_HAND_ACTUAL", "STATE_VALID_RECORDING",
]
