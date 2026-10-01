"""``vr-link-focus-check``: live per-eye sharpness of the SBS eyes_stereo camera, terminal only.

For adjusting a lens focus ring on a headless robot NUC over SSH, where a forwarded
window lags by seconds and a terminal line does not. Each eye's sharpness is the
variance of the Laplacian over a centre window of that half (``--roi``, fraction of
the half's width/height): higher is sharper, and the number is only comparable
between the two eyes when they look at the same target at the same distance. Put a
printed sheet (checkerboard, text) in the centre of BOTH eyes and turn the ring until
the lagging eye reaches the other eye's figure. The absolute value depends on the
scene, so there is no fixed "good" threshold.

Reads the camera section of ``calibration_tools.yaml`` (device, mode, which half is
which eye) like the capture tools. ``--save-every`` additionally writes lossless
``NNNN_sbs.png`` snapshots into ``<output-dir>/<yymmdd_HHMMSS>/`` for a before/after
comparison on the desktop.

The camera must be free: on the robot, stop ``igris-c-camera.service`` first (it holds
``/dev/eyes_stereo``; a plain kill is restarted by systemd) and start it again after.
"""

from __future__ import annotations

import argparse
import signal
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence

from ._common import default_captures_root, load_tools_config, save_frame, split_sbs
from .capture_checkerboard import _UsbCapture


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Live per-eye sharpness (Laplacian variance) of the SBS camera, no window; for SSH use")
    p.add_argument("--config", default=None, metavar="PATH", help="calibration_tools.yaml (default: packaged)")
    p.add_argument("--roi", type=float, default=0.5, help="centre window per eye as a fraction of the half's width/height, 0 < roi <= 1 (default 0.5)")
    p.add_argument("--rate", type=float, default=5.0, metavar="HZ", help="status line updates per second (default 5)")
    p.add_argument("--smooth", type=int, default=5, metavar="N", help="average the last N frames so the figure does not jitter (default 5)")
    p.add_argument("--save-every", type=float, default=0.0, metavar="SEC", help="also save a lossless SBS snapshot every SEC seconds (default: never)")
    p.add_argument("--output-dir", type=Path, default=None, help="root of the timestamped snapshot folders (default local:captures)")
    return p.parse_args(argv)


def centre_roi(image, fraction: float):
    height, width = image.shape[:2]
    roi_w = max(8, int(width * fraction))
    roi_h = max(8, int(height * fraction))
    x0 = (width - roi_w) // 2
    y0 = (height - roi_h) // 2
    return image[y0 : y0 + roi_h, x0 : x0 + roi_w]


def sharpness(image) -> float:
    """Variance of the Laplacian: the usual no-reference focus measure."""

    import cv2

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def main(argv: Optional[Sequence[str]] = None) -> None:
    import cv2

    args = parse_args(argv)
    if not 0.0 < args.roi <= 1.0:
        raise ValueError("--roi must be in (0, 1]")
    if args.smooth < 1:
        raise ValueError("--smooth must be >= 1")

    config, config_path = load_tools_config(args.config)
    camera_config = config.get("camera", {})
    left_half = str(camera_config.get("left_half", "right")).lower()

    output_dir: Optional[Path] = None
    if args.save_every > 0:
        output_root = args.output_dir if args.output_dir is not None else default_captures_root()
        output_dir = output_root / datetime.now().strftime("%y%m%d_%H%M%S")
        output_dir.mkdir(parents=True, exist_ok=False)

    capture = _UsbCapture(camera_config)
    capture.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # the figure must describe the frame being shown now, not a queued one

    print(f"Config: {config_path}")
    print(f"ROI: centre {args.roi:.0%} of each eye.  Same target in both eyes, then turn the ring until L and R agree. Ctrl+C to stop.")
    if output_dir is not None:
        print(f"Snapshots: {output_dir} every {args.save_every:.1f} s")

    left_hist: list = []
    right_hist: list = []
    saved_count = 0
    last_print = 0.0
    last_save = time.monotonic()
    period = 1.0 / args.rate if args.rate > 0 else 0.0

    try:
        while True:
            frame = capture.read()
            if frame is None:
                time.sleep(0.01)
                continue

            left, right = split_sbs(frame, left_half)
            left_hist.append(sharpness(centre_roi(left, args.roi)))
            right_hist.append(sharpness(centre_roi(right, args.roi)))
            del left_hist[: -args.smooth], right_hist[: -args.smooth]

            now = time.monotonic()
            if now - last_print >= period:
                last_print = now
                left_value = sum(left_hist) / len(left_hist)
                right_value = sum(right_hist) / len(right_hist)
                ratio = right_value / left_value if left_value > 0 else float("nan")
                line = f"L {left_value:8.1f}   R {right_value:8.1f}   R/L {ratio:5.2f}"
                if output_dir is not None:
                    line += f"   saved {saved_count}"
                sys.stdout.write("\r" + line.ljust(70))
                sys.stdout.flush()

            if output_dir is not None and now - last_save >= args.save_every:
                last_save = now
                saved_count += 1
                save_frame(output_dir, frame, saved_count, "png")
    except KeyboardInterrupt:
        # A second Ctrl+C (or a forwarded one from a task runner) must not abort the cleanup below.
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        sys.stdout.write("\n")
    finally:
        capture.release()
        if output_dir is not None:
            print(f"Saved {saved_count} snapshot(s) to {output_dir}")


if __name__ == "__main__":
    main()
