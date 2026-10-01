"""``vr-link-capture-checkerboard-headless``: SBS checkerboard capture without a window.

For the robot NUC over SSH. ``vr-link-capture-checkerboard`` drives ``cv2.imshow`` /
``cv2.waitKey`` every loop, and over X forwarding each of those is a network round
trip: the loop drops to a few fps, the frame read lags the scene, and the saved shots
carry motion blur from a board that was still moving when the frame was taken.

This tool never opens a window. It reads the same ``calibration_tools.yaml`` (camera
mode, checkerboard, capture) and writes the same ``<output-dir>/<yymmdd_HHMMSS>/
NNNN_sbs.png`` + ``capture_metadata.json`` layout, so ``vr-link-calibrate-cams
--input-dir`` consumes the folder unchanged (copy it to the desktop with scp).

A shot is saved only when BOTH halves detect the board on two consecutive frames and
no corner moved more than ``--max-motion-px`` between them: hold the board still,
watch the terminal status line, and move on after ``SAVED``. Terminal output is
instant over SSH where a forwarded window is not.

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

from ._common import default_captures_root, detect_checkerboard, load_tools_config, save_frame, split_sbs, write_metadata
from .capture_checkerboard import _UsbCapture

SOURCE_NAME = "usb-headless"


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Headless stereo checkerboard capture (USB camera, no window; for SSH use)")
    p.add_argument("--config", default=None, metavar="PATH", help="calibration_tools.yaml (default: packaged)")
    p.add_argument("--output-dir", type=Path, default=None, help="root of the timestamped capture folders (default local:captures)")
    p.add_argument("--target-count", type=int, default=None, help="override capture.target_count")
    p.add_argument("--interval", type=float, default=None, metavar="SEC", help="override capture.auto_interval_sec (min gap between saves)")
    p.add_argument(
        "--max-motion-px",
        type=float,
        default=1.0,
        help="save only when no corner moved more than this between two consecutive detections (default 1.0)",
    )
    return p.parse_args(argv)


def _max_corner_motion(previous, current) -> float:
    import numpy as np

    return float(max(np.abs(current[0] - previous[0]).max(), np.abs(current[1] - previous[1]).max()))


def main(argv: Optional[Sequence[str]] = None) -> None:
    import cv2

    args = parse_args(argv)
    config, config_path = load_tools_config(args.config)
    camera_config = config.get("camera", {})
    board_config = config.get("checkerboard", {})
    capture_config = config.get("capture", {})

    width = int(camera_config.get("width", 2560))
    height = int(camera_config.get("height", 720))
    left_half = str(camera_config.get("left_half", "right")).lower()
    pattern_size = (int(board_config.get("board_cols", 9)), int(board_config.get("board_rows", 6)))

    target_count = args.target_count if args.target_count is not None else int(capture_config.get("target_count", 20))
    interval = args.interval if args.interval is not None else float(capture_config.get("auto_interval_sec", 1.0))
    image_format = str(capture_config.get("image_format", "png")).lower().lstrip(".")
    max_motion = float(args.max_motion_px)

    output_root = args.output_dir if args.output_dir is not None else default_captures_root()
    output_dir = output_root / datetime.now().strftime("%y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=False)

    capture = _UsbCapture(camera_config)
    # One driver buffer: the detection step takes tens of ms, and a deeper queue would hand
    # back a frame older than the status line the operator is reacting to.
    capture.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    print(f"Config: {config_path}")
    print(f"Output: {output_dir}")
    print(f"Target: {target_count} shots, >= {interval:.1f} s apart, board still within {max_motion:.1f} px. Ctrl+C to stop.")

    saved_count = 0
    last_save = 0.0
    previous_corners = None
    warned_size = False

    try:
        while saved_count < target_count:
            frame = capture.read()
            if frame is None:
                time.sleep(0.01)
                continue

            frame_height, frame_width = frame.shape[:2]
            if not warned_size and (frame_width != width or frame_height != height):
                print(f"Warning: actual frame size is {frame_width}x{frame_height}, expected {width}x{height}")
                warned_size = True

            left, right = split_sbs(frame, left_half)
            left_found, left_corners = detect_checkerboard(left, pattern_size)
            right_found, right_corners = detect_checkerboard(right, pattern_size)
            both = left_found and right_found

            motion = None
            if both and previous_corners is not None:
                motion = _max_corner_motion(previous_corners, (left_corners, right_corners))
            previous_corners = (left_corners, right_corners) if both else None

            now = time.monotonic()
            status = f"[{saved_count}/{target_count}] L {'OK' if left_found else '--'}  R {'OK' if right_found else '--'}"
            if motion is not None:
                status += f"  motion {motion:5.1f} px"
            sys.stdout.write("\r" + status.ljust(60))
            sys.stdout.flush()

            if motion is not None and motion <= max_motion and now - last_save >= interval:
                saved_count += 1
                last_save = now
                saved_path = save_frame(output_dir, frame, saved_count, image_format)
                sys.stdout.write("\n")
                print(f"[{saved_count}/{target_count}] SAVED {saved_path.name}")
                previous_corners = None  # require a fresh still pair for the next shot
    except KeyboardInterrupt:
        # A second Ctrl+C (or a forwarded one from a task runner) must not abort the cleanup below.
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        sys.stdout.write("\n")
        print("Interrupted.")
    finally:
        capture.release()
        write_metadata(output_dir, config, config_path, saved_count, SOURCE_NAME)
        print(f"Saved {saved_count} image(s) to {output_dir}")


if __name__ == "__main__":
    main()
