"""Low-latency stereo H.264 encoding via system ffmpeg subprocesses.

Each eye owns an ffmpeg process that consumes raw RGB frames on stdin and emits
an Annex-B H.264 elementary stream on stdout. A reader thread splits that byte
stream into access units (one encoded frame each) and reports them through a
callback with a key/delta flag, so the transport layer can tag frames for
WebCodecs on the client.

The system ffmpeg is used directly (no extra Python dependency); it provides
both ``libx264`` (CPU) and ``h264_nvenc`` (GPU) encoders.
"""

from __future__ import annotations

import logging
import subprocess
import threading
from collections.abc import Callable

logger = logging.getLogger(__name__)

# H.264 NAL unit types (H.264 spec Table 7-1).
_NAL_IDR = 5  # coded slice of an IDR picture
_NAL_SPS = 7  # sequence parameter set
_VCL_TYPES = frozenset({1, 2, 3, 4, 5})  # coded-slice (picture data) NAL units


def _encoder_works(name: str) -> bool:
    """Probe whether this machine's ffmpeg can actually encode with ``name``.

    Encodes one synthetic frame end-to-end, which catches both a missing encoder
    and one that is listed but non-functional (e.g. h264_nvenc with no NVIDIA GPU
    or driver). Returns False on any failure instead of raising.
    """

    try:
        # 640x480 + yuv420p mirrors the real stream; a tiny size trips NVENC's
        # minimum-dimension check and yields a false negative.
        result = subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "lavfi",
                "-i",
                "color=c=black:s=640x480:d=1",
                "-frames:v",
                "1",
                "-pix_fmt",
                "yuv420p",
                "-c:v",
                name,
                "-f",
                "null",
                "-",
            ],
            capture_output=True,
            timeout=20,
        )
        return result.returncode == 0
    except (subprocess.SubprocessError, OSError):
        return False


def resolve_h264_encoder(requested: str) -> str:
    """Return a working H.264 encoder name, falling back to libx264 if needed.

    ``requested`` (e.g. the GPU ``h264_nvenc``) is used when it actually works on
    this machine; otherwise this falls back to the software ``libx264`` encoder,
    which ships with ffmpeg. Raises RuntimeError only if no encoder is usable.
    """

    if _encoder_works(requested):
        return requested
    if requested != "libx264" and _encoder_works("libx264"):
        logger.warning("H.264 encoder '%s' is unavailable here; falling back to libx264 (CPU).", requested)
        return "libx264"
    raise RuntimeError(
        f"No usable H.264 encoder: '{requested}' failed and the libx264 fallback is unavailable. "
        "Install an ffmpeg build with libx264, or run with --codec mjpeg."
    )


class _AnnexBSplitter:
    """Splits an Annex-B byte stream into access units (one per encoded frame).

    Assumes a single slice per frame (the encoder is configured without
    slicing), so each VCL NAL unit completes the current access unit.
    """

    def __init__(self) -> None:
        self._buffer = bytearray()
        self._au_nals: list[bytes] = []

    @staticmethod
    def _nal_type(nal: bytes) -> int:
        # nal starts with a 3-byte (00 00 01) or 4-byte (00 00 00 01) start code;
        # the NAL header byte follows it.
        header = 3 if nal[2] == 1 else 4
        return nal[header] & 0x1F

    def feed(self, chunk: bytes) -> list[tuple[bytes, bool]]:
        """Append bytes and return any completed (access_unit, is_keyframe) pairs."""

        self._buffer.extend(chunk)
        results: list[tuple[bytes, bool]] = []

        # Find start-code positions; keep the final (possibly incomplete) NAL buffered.
        starts = self._find_start_codes(self._buffer)
        if len(starts) < 2:
            return results
        for i in range(len(starts) - 1):
            nal = bytes(self._buffer[starts[i] : starts[i + 1]])
            au = self._push_nal(nal)
            if au is not None:
                results.append(au)
        del self._buffer[: starts[-1]]
        return results

    def _push_nal(self, nal: bytes) -> tuple[bytes, bool] | None:
        self._au_nals.append(nal)
        if self._nal_type(nal) in _VCL_TYPES:
            au = b"".join(self._au_nals)
            types = {self._nal_type(n) for n in self._au_nals}
            self._au_nals = []
            is_key = _NAL_IDR in types or _NAL_SPS in types
            return au, is_key
        return None

    @staticmethod
    def _find_start_codes(buffer: bytearray) -> list[int]:
        positions = []
        i = 0
        n = len(buffer)
        while i < n - 3:
            if buffer[i] == 0 and buffer[i + 1] == 0:
                if buffer[i + 2] == 1:
                    positions.append(i)
                    i += 3
                    continue
                if buffer[i + 2] == 0 and buffer[i + 3] == 1:
                    positions.append(i)
                    i += 4
                    continue
            i += 1
        return positions


class _EyeEncoder:
    """One ffmpeg encoding process plus a reader thread for a single eye."""

    def __init__(self, eye: str, command: list[str], on_output: Callable[[str, bytes, bool], None]) -> None:
        self._eye = eye
        self._on_output = on_output
        self._process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        self._splitter = _AnnexBSplitter()
        self._reader = threading.Thread(target=self._read_loop, name=f"h264-reader-{eye}", daemon=True)
        self._reader.start()

    def submit(self, frame_bytes: bytes) -> None:
        try:
            self._process.stdin.write(frame_bytes)
            self._process.stdin.flush()
        except (BrokenPipeError, ValueError):
            pass

    def _read_loop(self) -> None:
        stdout = self._process.stdout
        while True:
            # read1() returns whatever a single underlying read yields instead of
            # blocking until a full 64 KiB buffer fills; read() would batch several
            # frames together and release them in bursts (choppy playback).
            chunk = stdout.read1(65536)
            if not chunk:
                break
            for au, is_key in self._splitter.feed(chunk):
                self._on_output(self._eye, au, is_key)

    def close(self) -> None:
        try:
            if self._process.stdin is not None:
                self._process.stdin.close()
        except (BrokenPipeError, ValueError):
            pass
        self._process.terminate()
        try:
            self._process.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            self._process.kill()


class H264SBSEncoder:
    """Encodes one RGB stream (e.g. a side-by-side stereo frame) to Annex-B H.264.

    Single-stream counterpart of :class:`H264StereoEncoder` for transports whose
    receiver expects the left|right eyes combined in one video stream, such as
    the Unreal ``TCPActor`` wire protocol.
    """

    def __init__(
        self,
        width: int,
        height: int,
        fps: float,
        on_output: Callable[[bytes, bool], None],
        encoder: str = "h264_nvenc",
        bitrate: str = "8M",
        gop: int = 30,
    ) -> None:
        self._frame_nbytes = width * height * 3
        self._on_output = on_output
        self._command = H264StereoEncoder._build_command(width, height, fps, encoder, bitrate, gop)
        self._worker = _EyeEncoder("SBS", self._command, self._forward)

    def _forward(self, _eye: str, au: bytes, is_key: bool) -> None:
        self._on_output(au, is_key)

    def request_keyframe(self) -> None:
        """Restart the encoder so the next emitted frame is a fresh IDR.

        Frames already submitted but not yet emitted are discarded with the old
        process; callers tracking per-frame metadata must drop their pending
        entries alongside this call.
        """

        self._worker.close()
        self._worker = _EyeEncoder("SBS", self._command, self._forward)

    def submit(self, rgb_bytes: bytes) -> None:
        """Submit one contiguous RGB24 frame (``height*width*3`` bytes)."""

        if len(rgb_bytes) != self._frame_nbytes:
            raise ValueError(f"Expected {self._frame_nbytes} RGB bytes, got {len(rgb_bytes)}.")
        self._worker.submit(rgb_bytes)

    def close(self) -> None:
        self._worker.close()


class H264StereoEncoder:
    """Encodes left/right RGB frames to Annex-B H.264 with low latency."""

    def __init__(
        self,
        width: int,
        height: int,
        fps: float,
        on_output: Callable[[str, bytes, bool], None],
        encoder: str = "h264_nvenc",
        bitrate: str = "4M",
        gop: int = 30,
    ) -> None:
        self._width = width
        self._height = height
        self._frame_nbytes = width * height * 3
        self._on_output = on_output
        self._command = self._build_command(width, height, fps, encoder, bitrate, gop)
        self._left = _EyeEncoder("L", self._command, on_output)
        self._right = _EyeEncoder("R", self._command, on_output)

    def request_keyframe(self) -> None:
        """Restart both encoders so the next emitted frame is a fresh IDR.

        ffmpeg has no on-demand keyframe signal for a live raw pipe, so a fresh
        process is the way to force one. Called when a client connects, this lets
        a large GOP stay (few periodic keyframes, smoother) while a newly joined
        headset still starts decoding immediately instead of waiting a full GOP.
        Called from the simulation-loop thread, the same thread as :meth:`submit`.
        """

        self._left.close()
        self._right.close()
        self._left = _EyeEncoder("L", self._command, self._on_output)
        self._right = _EyeEncoder("R", self._command, self._on_output)

    @staticmethod
    def _build_command(width: int, height: int, fps: float, encoder: str, bitrate: str, gop: int) -> list[str]:
        command = [
            "ffmpeg",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{width}x{height}",
            "-framerate",
            f"{fps:g}",
            "-i",
            "-",
            "-an",
            "-c:v",
            encoder,
            "-profile:v",
            "baseline",
            "-pix_fmt",
            "yuv420p",
            "-g",
            str(gop),
            "-b:v",
            bitrate,
        ]
        if encoder == "libx264":
            # ultrafast + zerolatency: no lookahead/B-frames; repeat SPS/PPS at each IDR.
            # sliced-threads=0 keeps one slice (one NAL) per frame so each access unit
            # maps to a single WebCodecs chunk. CBR (nal-hrd) smooths keyframe spikes.
            command += [
                "-preset",
                "ultrafast",
                "-tune",
                "zerolatency",
                "-x264-params",
                "repeat-headers=1:sliced-threads=0:nal-hrd=cbr",
                "-maxrate",
                bitrate,
                "-bufsize",
                bitrate,
            ]
        elif encoder == "h264_nvenc":
            # Ultra-low-latency NVENC: no B-frames, no output delay (default is INT_MAX),
            # CBR to spread keyframe bits and avoid periodic stutter.
            command += [
                "-preset",
                "p1",
                "-tune",
                "ull",
                "-rc",
                "cbr",
                "-zerolatency",
                "1",
                "-delay",
                "0",
                "-bf",
                "0",
                "-maxrate",
                bitrate,
                "-bufsize",
                bitrate,
            ]
        # Flush each encoded frame to the pipe immediately instead of buffering.
        command += ["-flush_packets", "1", "-f", "h264", "-"]
        return command

    @staticmethod
    def video_codec_string() -> str:
        """WebCodecs codec id for baseline H.264 level 3.1."""

        return "avc1.42e01f"

    def submit(self, eye: str, rgb_bytes: bytes) -> None:
        """Submit one contiguous RGB24 frame (``height*width*3`` bytes) for an eye."""

        if len(rgb_bytes) != self._frame_nbytes:
            raise ValueError(f"Expected {self._frame_nbytes} RGB bytes, got {len(rgb_bytes)}.")
        (self._left if eye == "L" else self._right).submit(rgb_bytes)

    def close(self) -> None:
        self._left.close()
        self._right.close()
