"""UsbStereoSource: SBS half swap (hardware right-eye-left -> stream left-eye-left)."""
import numpy as np
import pytest

from vr_link.sources.usb import UsbStereoSource, swap_sbs_halves

W, H = 8, 4


class FakeCapture:
    """Stands in for cv2.VideoCapture: returns one preset frame per read."""

    def __init__(self, pixels):
        self.pixels = pixels

    def read(self):
        return True, self.pixels.copy()

    def release(self):
        pass


def sbs_frame():
    pixels = np.zeros((H, W, 3), dtype=np.uint8)
    pixels[:, : W // 2] = 10  # the device's left half
    pixels[:, W // 2 :] = 20  # the device's right half
    return pixels


def test_swap_sbs_halves_exchanges_the_halves():
    swapped = swap_sbs_halves(sbs_frame())
    assert (swapped[:, : W // 2] == 20).all() and (swapped[:, W // 2 :] == 10).all()
    assert swap_sbs_halves(swapped).tobytes() == sbs_frame().tobytes()


def test_read_swaps_when_the_device_has_the_right_eye_in_the_left_half():
    pytest.importorskip("cv2")
    source = UsbStereoSource("0", W, H, 30.0, left_half="right")
    source._cap = FakeCapture(sbs_frame())
    frame = source.read(1.0)
    assert frame is not None and frame.pixel_format == "bgr24"
    assert (frame.pixels[:, : W // 2] == 20).all() and (frame.pixels[:, W // 2 :] == 10).all()
    assert frame.pixels.flags["C_CONTIGUOUS"]


def test_read_passes_through_when_the_device_has_the_left_eye_in_the_left_half():
    pytest.importorskip("cv2")
    source = UsbStereoSource("0", W, H, 30.0, left_half="left")
    source._cap = FakeCapture(sbs_frame())
    frame = source.read(1.0)
    assert (frame.pixels[:, : W // 2] == 10).all() and (frame.pixels[:, W // 2 :] == 20).all()


def test_left_half_is_validated():
    with pytest.raises(ValueError):
        UsbStereoSource("0", W, H, 30.0, left_half="top")
