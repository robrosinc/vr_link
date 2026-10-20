"""Low-latency H.264 encoding through PyAV (in-process libavcodec).

One :class:`H264Encoder` turns SBS RGB/BGR frames into Annex-B access units
the VSTR receiver can decode from any keyframe: no B-frames, no lookahead,
one access unit per submitted frame, SPS/PPS repeated in-band on every IDR
(the receiver leaves ``WaitingForKeyframe`` on the first IDR that carries
its parameter sets; a late joiner must never depend on out-of-band headers).

Encoder names are libavcodec names: ``libx264`` (CPU, always present in the
PyAV builds we ship), ``h264_nvenc`` (NVIDIA), ``h264_vaapi`` / ``h264_qsv``
(Intel). :func:`resolve_h264_encoder` probes the requested one and falls back
to ``libx264``. Hardware encoders need the host driver at runtime.

Keyframes on demand: a new headset connection needs an IDR immediately (the
GOP may be seconds long). :meth:`H264Encoder.request_keyframe` marks the next
frame as a forced IDR (``forced-idr`` option). The first access unit after a
request is checked for IDR + SPS/PPS; if the encoder ignored the request the
encoder switches to "recreate the codec context" mode for the rest of its
life, which is what the legacy per-connection encoder did and is always
correct.

The encoder itself is synchronous and not thread-safe. Callers that must not
pay encode time on their own thread (the 50 Hz simulation loop) wrap it in
:class:`EncoderWorker`, which drops *unencoded* input frames under
back-pressure; dropping inputs never breaks the H.264 reference chain,
dropping outputs would.
"""

from __future__ import annotations

import logging
import queue
import re
import threading
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# H.264 NAL unit types (H.264 spec Table 7-1).
NAL_IDR = 5
NAL_SPS = 7
NAL_PPS = 8

SW_ENCODER = "libx264"
_ENCODER_ALIASES = {"sw": SW_ENCODER, "x264": SW_ENCODER, "nvenc": "h264_nvenc", "vaapi": "h264_vaapi", "qsv": "h264_qsv"}

# Per-encoder private options. ``forced-idr`` makes a forced I picture an IDR
# with fresh parameter sets instead of a plain I slice.
_CODEC_OPTIONS: Dict[str, Dict[str, str]] = {
    "libx264": {
        "preset": "ultrafast",
        "tune": "zerolatency",
        "forced-idr": "1",
        # repeat-headers: SPS/PPS before every IDR. sliced-threads=1 keeps the
        # zerolatency slice threading so each frame's access unit leaves
        # encode() in the same call; frame threading (sliced-threads=0) holds
        # ~threads-1 frames (14 measured on a 16-core host) and desynchronises
        # the pixels from the capture_ns/robot_state the pump attaches. CBR
        # smooths keyframe bit spikes.
        "x264-params": "repeat-headers=1:bframes=0:sliced-threads=1:nal-hrd=cbr",
    },
    "h264_nvenc": {
        "preset": "p1",
        "tune": "ll",
        "zerolatency": "1",
        "delay": "0",
        "rc": "cbr",
        "forced-idr": "1",
        "bf": "0",
    },
}

_BITRATE_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([kKmM]?)\s*$")


def parse_bitrate(value: "int | str") -> int:
    """``8000000``, ``"8M"``, ``"800k"`` -> bits per second."""

    if isinstance(value, (int, np.integer)):
        return int(value)
    match = _BITRATE_RE.match(str(value))
    if not match:
        raise ValueError(f"invalid bitrate {value!r} (expected e.g. 8000000, '8M', '800k')")
    number, suffix = match.groups()
    scale = {"": 1, "k": 1_000, "K": 1_000, "m": 1_000_000, "M": 1_000_000}[suffix]
    return int(float(number) * scale)


def normalize_encoder_name(name: str) -> str:
    """Map config aliases (``sw``, ``nvenc``) onto libavcodec encoder names."""

    return _ENCODER_ALIASES.get(name.strip().lower(), name.strip())


# --- Annex-B helpers ---------------------------------------------------------


def _start_codes(buf: bytes) -> List[int]:
    positions: List[int] = []
    i = 0
    n = len(buf)
    while i < n - 2:
        if buf[i] == 0 and buf[i + 1] == 0:
            if buf[i + 2] == 1:
                positions.append(i + 3)
                i += 3
                continue
            if i + 3 < n and buf[i + 2] == 0 and buf[i + 3] == 1:
                positions.append(i + 4)
                i += 4
                continue
        i += 1
    return positions


def annexb_nal_types(access_unit: bytes) -> List[int]:
    """NAL unit types in an Annex-B access unit, in stream order."""

    return [access_unit[pos] & 0x1F for pos in _start_codes(access_unit) if pos < len(access_unit)]


def is_idr_with_headers(access_unit: bytes) -> bool:
    """True when the access unit carries SPS, PPS and an IDR slice."""

    types = set(annexb_nal_types(access_unit))
    return NAL_IDR in types and NAL_SPS in types and NAL_PPS in types


# --- encoder -----------------------------------------------------------------


@dataclass(frozen=True)
class EncodedAccessUnit:
    """One encoded frame: Annex-B bytes plus the caller's opaque ``user`` value."""

    data: bytes
    is_key: bool
    pts: int
    user: Any = None


def _import_av():
    try:
        import av  # noqa: WPS433 (lazy: the base package must import without PyAV)
    except ImportError as error:  # pragma: no cover - environment dependent
        raise ImportError("PyAV is required for H.264 encoding: install the 'video' extra (pip install 'vr-link-schema[video]')") from error
    return av


class H264Encoder:
    """Synchronous Annex-B H.264 encoder for one fixed-size video stream."""

    def __init__(
        self,
        width: int,
        height: int,
        fps: float,
        *,
        codec: str = SW_ENCODER,
        bitrate: "int | str" = "8M",
        gop: int = 30,
        options: Optional[Dict[str, str]] = None,
    ) -> None:
        if width <= 0 or height <= 0 or width % 2 or height % 2:
            raise ValueError(f"width/height must be positive and even for yuv420p, got {width}x{height}")
        if fps <= 0:
            raise ValueError(f"fps must be positive, got {fps}")
        self._av = _import_av()
        self.width = int(width)
        self.height = int(height)
        self.fps = float(fps)
        self.codec_name = normalize_encoder_name(codec)
        self.bitrate = parse_bitrate(bitrate)
        self.gop = int(gop)
        self._options = dict(_CODEC_OPTIONS.get(self.codec_name, {}))
        if options:
            self._options.update(options)

        self._ctx = None
        self._pts = 0
        self._pending: Dict[int, Any] = {}
        self._keyframe_pending = True  # the first frame is an IDR by construction
        self._verify_forced_idr = False
        self._recreate_on_keyframe = False
        self._open()

    # -- lifecycle ---------------------------------------------------------

    def _open(self) -> None:
        av = self._av
        ctx = av.CodecContext.create(self.codec_name, "w")
        ctx.width = self.width
        ctx.height = self.height
        ctx.pix_fmt = "yuv420p"
        rate = Fraction(self.fps).limit_denominator(1000)
        ctx.framerate = rate
        ctx.time_base = 1 / rate
        ctx.bit_rate = self.bitrate
        ctx.gop_size = self.gop
        ctx.max_b_frames = 0
        ctx.options = dict(self._options)
        # Never request a global header: SPS/PPS must travel in-band (Annex-B).
        ctx.open()
        self._ctx = ctx

    def close(self) -> None:
        if self._ctx is not None:
            try:
                self._ctx.close()
            except Exception:  # pragma: no cover - best effort
                pass
            self._ctx = None
        self._pending.clear()

    def __enter__(self) -> "H264Encoder":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- encoding ------------------------------------------------------------

    def request_keyframe(self) -> None:
        """Make the next submitted frame an IDR with SPS/PPS."""

        self._keyframe_pending = True

    def encode(self, pixels: np.ndarray, pixel_format: str = "rgb24", *, user: Any = None) -> List[EncodedAccessUnit]:
        """Encode one frame; returns the access units it produced (normally exactly one)."""

        if self._ctx is None:
            raise RuntimeError("encoder is closed")
        if pixels.shape[:2] != (self.height, self.width) or pixels.ndim != 3 or pixels.shape[2] != 3:
            raise ValueError(f"expected a ({self.height}, {self.width}, 3) frame, got {pixels.shape}")
        if pixels.dtype != np.uint8:
            raise ValueError(f"expected uint8 pixels, got {pixels.dtype}")

        force_key = self._keyframe_pending
        if force_key and self._recreate_on_keyframe:
            self.close()
            self._open()

        av = self._av
        frame = av.VideoFrame.from_ndarray(np.ascontiguousarray(pixels), format=pixel_format)
        frame = frame.reformat(format="yuv420p")
        frame.pts = self._pts
        frame.time_base = self._ctx.time_base
        if force_key:
            frame.pict_type = self._picture_type_i()
        self._pending[self._pts] = user
        self._pts += 1

        units = self._collect(self._ctx.encode(frame))
        if force_key and units:
            self._keyframe_pending = False
            if not self._recreate_on_keyframe and not is_idr_with_headers(units[0].data):
                logger.warning(
                    "%s ignored the forced IDR request (NAL types %s); recreating the codec context on every keyframe request from now on.",
                    self.codec_name,
                    annexb_nal_types(units[0].data),
                )
                self._recreate_on_keyframe = True
        return units

    def _picture_type_i(self):
        """``AV_PICTURE_TYPE_I`` in whatever form this PyAV accepts (enum since 14, int before)."""

        try:
            return self._av.video.frame.PictureType.I
        except AttributeError:  # pragma: no cover - older PyAV
            return 1

    def flush(self) -> List[EncodedAccessUnit]:
        """Drain buffered packets (none with zero-latency settings, kept for completeness)."""

        if self._ctx is None:
            return []
        return self._collect(self._ctx.encode(None))

    def _collect(self, packets) -> List[EncodedAccessUnit]:
        units: List[EncodedAccessUnit] = []
        for packet in packets:
            data = bytes(packet)
            if not data:
                continue
            pts = int(packet.pts) if packet.pts is not None else -1
            user = self._pending.pop(pts, None)
            units.append(EncodedAccessUnit(data=data, is_key=bool(packet.is_keyframe), pts=pts, user=user))
        return units


# --- probe ---------------------------------------------------------------------


def probe_h264_encoder(name: str, width: int = 64, height: int = 64) -> bool:
    """Encode one frame with ``name``; False if the encoder is missing or unusable."""

    try:
        encoder = H264Encoder(width, height, 30.0, codec=name, bitrate="200k", gop=30)
    except Exception as error:
        logger.debug("encoder %s unavailable: %s", name, error)
        return False
    try:
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        units = encoder.encode(frame) + encoder.flush()
        return bool(units)
    except Exception:
        logger.debug("encoder %s failed to encode a probe frame", name, exc_info=True)
        return False
    finally:
        encoder.close()


def resolve_h264_encoder(requested: str, *, fallback: Tuple[str, ...] = (SW_ENCODER,)) -> str:
    """Return a working encoder name: ``requested`` if it works, else the first working fallback."""

    requested = normalize_encoder_name(requested)
    if probe_h264_encoder(requested):
        return requested
    for name in fallback:
        if name != requested and probe_h264_encoder(name):
            logger.warning("H.264 encoder '%s' is unavailable here; falling back to %s.", requested, name)
            return name
    raise RuntimeError(f"No usable H.264 encoder: '{requested}' failed and so did the fallbacks {fallback}.")


# --- worker --------------------------------------------------------------------


class EncoderWorker:
    """Runs an :class:`H264Encoder` on its own thread with a bounded input queue.

    ``submit`` never blocks: when ``max_pending`` frames are already waiting,
    the oldest waiting frame is discarded. Access units reach ``on_output`` on
    the worker thread, in submission order. A keyframe request applies to the
    next *submitted* frame (and survives that frame being dropped), so the
    caller knows which frame becomes the IDR.
    """

    def __init__(self, encoder: H264Encoder, on_output: Callable[[EncodedAccessUnit], None], *, max_pending: int = 2) -> None:
        if max_pending < 1:
            raise ValueError("max_pending must be >= 1")
        self._encoder = encoder
        self._on_output = on_output
        self._queue: "queue.Queue[Optional[Tuple[np.ndarray, str, Any, bool]]]" = queue.Queue(maxsize=max_pending)
        self._keyframe_pending = False
        self.dropped = 0
        self._thread = threading.Thread(target=self._run, name=f"h264-{encoder.codec_name}", daemon=True)
        self._thread.start()

    @property
    def encoder(self) -> H264Encoder:
        return self._encoder

    def request_keyframe(self) -> None:
        """Make the next submitted frame an IDR (call from the submitting thread)."""

        self._keyframe_pending = True

    def submit(self, pixels: np.ndarray, pixel_format: str = "rgb24", *, user: Any = None) -> bool:
        """Queue one frame; returns False when an older queued frame had to be dropped."""

        force_key = self._keyframe_pending
        self._keyframe_pending = False
        dropped = False
        while True:
            try:
                self._queue.put_nowait((pixels, pixel_format, user, force_key))
                return not dropped
            except queue.Full:
                try:
                    _, _, _, dropped_key = self._queue.get_nowait()
                except queue.Empty:
                    continue
                self.dropped += 1
                dropped = True
                force_key = force_key or dropped_key  # the request must not be lost with the frame

    def close(self) -> None:
        self._queue.put(None)
        self._thread.join(timeout=5.0)
        self._encoder.close()

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                return
            pixels, pixel_format, user, force_key = item
            if force_key:
                self._encoder.request_keyframe()
            try:
                units = self._encoder.encode(pixels, pixel_format, user=user)
            except Exception:
                logger.exception("H.264 encode failed; frame dropped")
                continue
            for unit in units:
                try:
                    self._on_output(unit)
                except Exception:
                    logger.exception("encoder output callback failed")


__all__ = [
    "EncodedAccessUnit",
    "EncoderWorker",
    "H264Encoder",
    "NAL_IDR",
    "NAL_PPS",
    "NAL_SPS",
    "SW_ENCODER",
    "annexb_nal_types",
    "is_idr_with_headers",
    "normalize_encoder_name",
    "parse_bitrate",
    "probe_h264_encoder",
    "resolve_h264_encoder",
]
