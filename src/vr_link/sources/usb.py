"""USB side-by-side stereo camera (the NUC ``/dev/eyes_stereo`` device) via OpenCV."""

from __future__ import annotations

import logging
import time
from typing import Optional, Tuple

import numpy as np

from . import Frame, StereoCameraMetadata, check_left_half, swap_sbs_halves
from .v4l2 import apply_stereo_tuning, init_manual

logger = logging.getLogger(__name__)


class UsbStereoSource:
    """Grabs SBS frames from a V4L2/OpenCV camera; ``read`` blocks on the driver.

    High-resolution USB cameras only reach full rate in MJPG; raw YUYV
    saturates the bus. Frames are resized to the configured size when the
    driver negotiates something else, so the encoded size is deterministic.

    ``left_half`` names the eye the device puts in the left half of its frame.
    The delivered frame always has the left eye in the left half (the layout
    igris_c_camera publishes over DDS), so ``"right"`` (the eyes_stereo
    hardware) swaps the two halves, exactly as igris_c_camera's stereo worker
    does before publishing.
    """

    def __init__(
        self,
        device: str,
        width: int,
        height: int,
        fps: float,
        *,
        fourcc: str = "MJPG",
        apply_tuning: bool = True,
        left_half: str = "right",
    ) -> None:
        self.left_half = check_left_half(left_half)
        self.device = str(device)
        self.width = int(width)
        self.height = int(height)
        self.fps = float(fps)
        self.fourcc = fourcc
        self.apply_tuning = apply_tuning
        self._cap = None
        self._index = 0

    def open(self) -> None:
        import cv2

        is_index = self.device.isdigit()
        cap = cv2.VideoCapture(int(self.device) if is_index else self.device)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open eyes_stereo camera: {self.device}")
        node = "" if is_index else self.device
        if node:
            init_manual(node)
        # Pixel format FIRST, then geometry; keep the driver queue at one frame.
        if self.fourcc:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.fourcc))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, self.fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.read()  # warm up so the device is streaming when controls apply
        if self.apply_tuning:
            apply_stereo_tuning(cap, node)
        actual_fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        fourcc_str = "".join(chr((actual_fourcc >> (8 * i)) & 0xFF) for i in range(4))
        logger.info(
            "eyes_stereo %s: %dx%d @ %.0ffps fourcc=%s",
            self.device,
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            cap.get(cv2.CAP_PROP_FPS),
            fourcc_str,
        )
        self._cap = cap

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    @property
    def frame_size(self) -> Optional[Tuple[int, int]]:
        return (self.width, self.height)

    @property
    def fps_hint(self) -> Optional[float]:
        return self.fps

    def read(self, timeout: float) -> Optional[Frame]:
        """One frame from the driver (``timeout`` is not honoured by OpenCV)."""

        if self._cap is None:
            raise RuntimeError("source not open")
        import cv2

        ok, pixels = self._cap.read()
        if not ok or pixels is None:
            return None
        capture_ns = time.clock_gettime_ns(time.CLOCK_REALTIME)
        if pixels.shape[1] != self.width or pixels.shape[0] != self.height:
            pixels = cv2.resize(pixels, (self.width, self.height))
        if self.left_half == "right":
            pixels = swap_sbs_halves(pixels)
        self._index += 1
        return Frame(pixels=np.ascontiguousarray(pixels), pixel_format="bgr24", capture_ns=capture_ns, sim_ns=None, index=self._index)

    def camera_metadata(self) -> Optional[StereoCameraMetadata]:
        return None


__all__ = ["UsbStereoSource"]
