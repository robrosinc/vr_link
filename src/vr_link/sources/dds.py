"""Robot eyes_stereo JPEG frames over CycloneDDS (``igris_c_sdk``) as a frame source.

The robot's ``camera_image_stream`` publishes the side-by-side frame as a
JPEG ``CompressedMessage``; this source subscribes, keeps only the newest
sample (cheap, on the DDS thread) and JPEG-decodes lazily in ``read`` so
frames superseded before they are consumed are never decoded.

``igris_c_sdk`` is the workspace's Python binding (py3.12 build, not on
PyPI): it is imported in :meth:`DdsStereoSource.open`, from the environment
or from ``$IGRIS_C_SDK_ROOT``. Nothing in this module runs at import time.
"""

from __future__ import annotations

import logging
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np

from . import Frame, StereoCameraMetadata

logger = logging.getLogger(__name__)

DEFAULT_DDS_TOPIC = "igris_c/sensor/eyes_stereo"
DEFAULT_DDS_DOMAIN_ID = 1


def import_igris_c_sdk():
    """``igris_c_sdk`` from the environment, else from ``$IGRIS_C_SDK_ROOT``."""

    try:
        import igris_c_sdk  # noqa: WPS433
    except ImportError:
        pass
    else:
        return igris_c_sdk

    env_root = os.environ.get("IGRIS_C_SDK_ROOT")
    if env_root:
        for root in (Path(env_root), Path(env_root) / "python"):
            if (root / "igris_c_sdk" / "__init__.py").exists():
                sys.path.insert(0, str(root))
                try:
                    import igris_c_sdk  # noqa: WPS433
                except ImportError:
                    sys.path.remove(str(root))
                else:
                    return igris_c_sdk
    raise ImportError(
        "igris_c_sdk is not importable: install the workspace SDK Python binding (Python 3.12 build) "
        "in this environment or point IGRIS_C_SDK_ROOT at a checkout that contains igris_c_sdk/__init__.py"
    )


def resolve_config_uri(raw: Optional[str], *, base_dir: Optional[Path] = None) -> str:
    """``""``, a ``file://`` URI, or a path (relative to ``base_dir``) -> CycloneDDS URI."""

    if not raw:
        return ""
    raw = str(raw)
    if raw.startswith("file://"):
        return raw
    path = Path(raw)
    if not path.is_absolute() and base_dir is not None:
        path = Path(base_dir) / path
    return f"file://{os.path.abspath(path)}"


def normalize_frame_id(value: Any) -> str:
    """``Header.frame_id()`` is a fixed 256-char buffer; trim at the first NUL."""

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


class DdsStereoSource:
    """Newest-sample JPEG subscriber; ``read`` returns BGR frames."""

    def __init__(
        self,
        domain_id: int = DEFAULT_DDS_DOMAIN_ID,
        namespace: str = "",
        topic: str = DEFAULT_DDS_TOPIC,
        *,
        config_uri: str = "",
        frame_id: Optional[str] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
    ) -> None:
        self.domain_id = int(domain_id)
        self.namespace = str(namespace)
        self.topic = str(topic)
        self.config_uri = str(config_uri)
        self.frame_id = frame_id or None
        self.width = width
        self.height = height
        self._cond = threading.Condition()
        self._latest_payload: Optional[bytes] = None
        self._latest_capture_ns = 0
        self._seq = 0
        self._last_seq = 0
        self._factory = None
        self._subscriber = None

    def open(self) -> None:
        sdk = import_igris_c_sdk()
        logger.info(
            "Initializing DDS subscriber: domain=%d namespace=%r topic=%r config_uri=%r",
            self.domain_id, self.namespace, self.topic, self.config_uri or "(DDS default)",
        )
        factory = sdk.ChannelFactory()
        factory.Init(self.domain_id, self.namespace, self.config_uri)
        subscriber = sdk.CompressedMessageSubscriber(factory, self.topic, sdk.QosProfile.SensorData())
        if not subscriber.init(self._on_message):
            factory.Release()
            raise RuntimeError(f"Failed to initialize DDS subscriber on {self.topic}")
        self._factory = factory
        self._subscriber = subscriber
        logger.info("DDS subscriber initialized; waiting for samples on %s", self.topic)

    def close(self) -> None:
        subscriber, factory = self._subscriber, self._factory
        self._subscriber = self._factory = None
        try:
            if subscriber is not None:
                subscriber.stop()
        finally:
            if factory is not None:
                factory.Release()

    def _on_message(self, message) -> None:
        # DDS thread: cheap and exception-safe (a raising callback kills the reader).
        try:
            if self.frame_id is not None:
                fid = normalize_frame_id(message.header().frame_id())
                if fid != self.frame_id:
                    logger.warning("Dropping sample with frame_id=%r (expected %r)", fid, self.frame_id)
                    return
            payload = bytes(message.image_data())
            header = message.header()
            capture_ns = int(header.sec()) * 1_000_000_000 + int(header.nanosec())
            if capture_ns == 0:
                capture_ns = time.clock_gettime_ns(time.CLOCK_REALTIME)
            with self._cond:
                self._latest_payload = payload
                self._latest_capture_ns = capture_ns
                self._seq += 1
                self._cond.notify_all()
        except Exception:
            logger.exception("DDS message callback failed")

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
        if self._subscriber is None:
            raise RuntimeError("source not open")
        with self._cond:
            if self._seq == self._last_seq:
                self._cond.wait(timeout)
            if self._seq == self._last_seq:
                return None
            payload = self._latest_payload
            capture_ns = self._latest_capture_ns
            seq = self._seq
        self._last_seq = seq
        import cv2

        pixels = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
        if pixels is None or pixels.size == 0:
            logger.warning("JPEG decode failed for DDS sample seq=%d; skipping", seq)
            return None
        if self.width and self.height and (pixels.shape[1] != self.width or pixels.shape[0] != self.height):
            pixels = cv2.resize(pixels, (int(self.width), int(self.height)))
        return Frame(pixels=np.ascontiguousarray(pixels), pixel_format="bgr24", capture_ns=capture_ns, sim_ns=None, index=seq)


__all__ = ["DEFAULT_DDS_DOMAIN_ID", "DEFAULT_DDS_TOPIC", "DdsStereoSource", "import_igris_c_sdk", "normalize_frame_id", "resolve_config_uri"]
