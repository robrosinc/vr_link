"""V4L2 controls for the eyes_stereo USB camera.

``cv2.VideoCapture.set`` is unreliable for the camera-class controls
(exposure, focus), so those go through a raw ``VIDIOC_S_CTRL`` ioctl; the
image-class controls go through OpenCV. Tuning values are ported verbatim
from the robot's ``camera_image_stream.cpp::run_eyes_stereo`` so the TCP path
reproduces the DDS path's image tone.
"""

from __future__ import annotations

import logging
import struct

logger = logging.getLogger(__name__)

# --- V4L2 control ids (linux/videodev2.h) -----------------------------------
V4L2_CID_BASE = 0x00980900
V4L2_CID_AUTO_WHITE_BALANCE = V4L2_CID_BASE + 12
V4L2_CID_POWER_LINE_FREQUENCY = V4L2_CID_BASE + 24

V4L2_CID_CAMERA_CLASS_BASE = 0x009A0900
V4L2_CID_EXPOSURE_AUTO = V4L2_CID_CAMERA_CLASS_BASE + 1
V4L2_CID_EXPOSURE_ABSOLUTE = V4L2_CID_CAMERA_CLASS_BASE + 2
V4L2_CID_EXPOSURE_AUTO_PRIORITY = V4L2_CID_CAMERA_CLASS_BASE + 3
V4L2_CID_FOCUS_AUTO = V4L2_CID_CAMERA_CLASS_BASE + 12

V4L2_EXPOSURE_AUTO = 0
V4L2_EXPOSURE_MANUAL = 1

# VIDIOC_S_CTRL = _IOWR('V', 28, struct v4l2_control{u32 id; s32 value;}) -> 8 bytes.
VIDIOC_S_CTRL = 0xC008561C


def set_control(device: str, control_id: int, value: int) -> None:
    """``VIDIOC_S_CTRL`` on a ``/dev/...`` node; numeric indices are skipped."""

    if not device.startswith("/dev/"):
        return
    import fcntl
    import os

    try:
        fd = os.open(device, os.O_RDWR)
    except OSError as error:
        logger.warning("Cannot open %s for V4L2 control: %s", device, error)
        return
    try:
        fcntl.ioctl(fd, VIDIOC_S_CTRL, struct.pack("Ii", control_id, value))
    except OSError as error:
        logger.warning("V4L2 ioctl failed (id=0x%x value=%d): %s", control_id, value, error)
    finally:
        os.close(fd)


def init_manual(device: str) -> None:
    """Open-time defaults, mirroring camera.cpp::openCamera."""

    set_control(device, V4L2_CID_EXPOSURE_AUTO, V4L2_EXPOSURE_MANUAL)
    set_control(device, V4L2_CID_AUTO_WHITE_BALANCE, 0)
    set_control(device, V4L2_CID_FOCUS_AUTO, 0)


def apply_stereo_tuning(cap, device: str) -> None:
    """Image tone of the robot's DDS path (camera_image_stream.cpp::run_eyes_stereo)."""

    import cv2

    cap.set(cv2.CAP_PROP_BRIGHTNESS, 30.0)
    cap.set(cv2.CAP_PROP_CONTRAST, 34.0)
    cap.set(cv2.CAP_PROP_SATURATION, 67.0)
    cap.set(cv2.CAP_PROP_HUE, 0.0)
    cap.set(cv2.CAP_PROP_AUTO_WB, 1.0)
    cap.set(cv2.CAP_PROP_GAMMA, 161.0)
    cap.set(cv2.CAP_PROP_GAIN, 5.0)
    set_control(device, V4L2_CID_POWER_LINE_FREQUENCY, 2)  # 60Hz
    cap.set(cv2.CAP_PROP_WB_TEMPERATURE, 4935.0)
    cap.set(cv2.CAP_PROP_SHARPNESS, 4.0)
    cap.set(cv2.CAP_PROP_BACKLIGHT, 1.0)
    set_control(device, V4L2_CID_EXPOSURE_AUTO, V4L2_EXPOSURE_AUTO)
    set_control(device, V4L2_CID_EXPOSURE_ABSOLUTE, 128)
    set_control(device, V4L2_CID_EXPOSURE_AUTO_PRIORITY, 0)


__all__ = ["apply_stereo_tuning", "init_manual", "set_control"]
