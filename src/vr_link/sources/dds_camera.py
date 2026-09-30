"""Robot eyes_stereo JPEG frames over CycloneDDS (``igris_c_sdk``) as a frame source.

igris_c_camera publishes the side-by-side frame as a ``sensor_msgs``
``CompressedImage_`` (JPEG); this source subscribes on the process-wide
:class:`~vr_link.network.dds.DdsChannel`, keeps only the newest sample
(cheap, on the DDS thread) and JPEG-decodes lazily in ``read`` so frames
superseded before they are consumed are never decoded.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Optional, Tuple

import numpy as np

from ..network.dds.channel import DdsChannel
from ..network.dds.subscribers import header_stamp_ns
from . import Frame, StereoCameraMetadata

logger = logging.getLogger(__name__)

# igris_c_camera's dds sink publishes CompressedImage_ on "<topic>/compressed".
DEFAULT_DDS_CAMERA_TOPIC = "igris_c/sensor/eyes_stereo/compressed"


def normalize_frame_id(value: Any) -> str:
    """``Header.frame_id()`` (a string, or a fixed char buffer in older bindings) -> str."""

    if isinstance(value, str):
        return value.split("\x00", 1)[0]
    if isinstance(value, (bytes, bytearray)):
        return bytes(value).split(b"\x00", 1)[0].decode(errors="ignore")
    try:
        chars = []
        for item in value:
            code = ord(item) if isinstance(item, str) else int(item)
            if code == 0:
                break
            chars.append(chr(code))
        return "".join(chars)
    except TypeError:
        return str(value)


def payload_bytes(data: Any) -> bytes:
    """``CompressedImage_.data()`` (bytes, buffer or a sequence of ints) -> bytes."""

    if isinstance(data, (bytes, bytearray, memoryview)):
        return bytes(data)
    try:
        return np.asarray(data, dtype=np.uint8).tobytes()
    except (TypeError, ValueError):
        return bytes(data)


class DdsCameraSource:
    """Newest-sample JPEG subscriber; ``read`` returns BGR frames."""

    def __init__(
        self,
        channel: DdsChannel,
        topic: str = DEFAULT_DDS_CAMERA_TOPIC,
        *,
        frame_id: Optional[str] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
    ) -> None:
        self.channel = channel
        self.topic = str(topic)
        self.frame_id = frame_id or None
        self.width = width
        self.height = height
        self._cond = threading.Condition()
        self._latest_payload: Optional[bytes] = None
        self._latest_capture_ns = 0
        self._seq = 0
        self._last_seq = 0
        self._subscriber = None
        self.received = 0
        self.errors = 0

    def open(self) -> None:
        sdk = self.channel.sdk
        subscriber_cls = getattr(sdk, "CompressedImageSubscriber", None) or getattr(sdk, "CompressedMessageSubscriber", None)
        if subscriber_cls is None:
            raise RuntimeError("igris_c_sdk has no CompressedImageSubscriber")
        subscriber = subscriber_cls(self.topic, sdk.QosProfile.SensorData())
        if not subscriber.init(self._on_message):
            raise RuntimeError(f"failed to initialize DDS camera subscriber on {self.channel.resolve(self.topic)}")
        self._subscriber = subscriber
        logger.info("DDS camera subscriber on %s; waiting for samples", self.channel.resolve(self.topic))

    def close(self) -> None:
        subscriber, self._subscriber = self._subscriber, None
        if subscriber is not None:
            try:
                subscriber.stop()
            except Exception as error:
                logger.warning("stopping the DDS camera subscriber failed: %s", error)

    def _on_message(self, message) -> None:
        # DDS thread: cheap and exception-safe (a raising callback kills the reader).
        try:
            if self.frame_id is not None:
                fid = normalize_frame_id(message.header().frame_id())
                if fid != self.frame_id:
                    return
            data_getter = getattr(message, "data", None) or getattr(message, "image_data")
            payload = payload_bytes(data_getter())
            capture_ns = header_stamp_ns(message) or time.time_ns()
            with self._cond:
                self._latest_payload = payload
                self._latest_capture_ns = capture_ns
                self._seq += 1
                self._cond.notify_all()
            self.received += 1
        except Exception as error:  # noqa: BLE001
            self.errors += 1
            if self.errors in (1, 10, 100) or self.errors % 1000 == 0:
                logger.warning("DDS camera callback failed (%d so far): %s", self.errors, error)

    # -- FrameSource ---------------------------------------------------------------

    @property
    def frame_size(self) -> Optional[Tuple[int, int]]:
        if self.width and self.height:
            return (int(self.width), int(self.height))
        return None

    @property
    def fps_hint(self) -> Optional[float]:
        return None

    def camera_metadata(self) -> Optional[StereoCameraMetadata]:
        return None

    def read(self, timeout: float) -> Optional[Frame]:
        import cv2  # lazy: the calibration extra

        with self._cond:
            if self._seq == self._last_seq:
                self._cond.wait(timeout)
            if self._seq == self._last_seq or self._latest_payload is None:
                return None
            payload, capture_ns, seq = self._latest_payload, self._latest_capture_ns, self._seq
            self._last_seq = seq
        image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            logger.warning("JPEG decode failed for a %d-byte sample", len(payload))
            return None
        if self.width and self.height and (image.shape[1], image.shape[0]) != (self.width, self.height):
            image = cv2.resize(image, (int(self.width), int(self.height)), interpolation=cv2.INTER_AREA)
        return Frame(pixels=image, pixel_format="bgr24", capture_ns=int(capture_ns), sim_ns=None, index=seq)


DdsStereoSource = DdsCameraSource  # 1.x name

__all__ = ["DEFAULT_DDS_CAMERA_TOPIC", "DdsCameraSource", "DdsStereoSource", "normalize_frame_id", "payload_bytes"]
