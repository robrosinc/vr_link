"""Stereo frame sources feeding the video pump.

A source delivers side-by-side (SBS) stereo frames, right eye in the LEFT
half (the hardware ``left_half: right`` convention every consumer assumes),
as one :class:`Frame` per call. Sources differ only in where pixels come
from: a USB camera (:mod:`vr_link.sources.usb`), the igris_c_mujoco vision
shared-memory segment (:mod:`vr_link.sources.mujoco_shm`) or the robot's DDS
image topic (:mod:`vr_link.sources.dds`). Isaac Sim is not a source here: the
simulator pushes frames from its own loop straight into the encoder.

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

    @property
    def width(self) -> int:
        return int(self.pixels.shape[1])

    @property
    def height(self) -> int:
        return int(self.pixels.shape[0])


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


__all__ = ["Frame", "FrameSource", "PixelFormat", "StereoCameraMetadata"]
