"""``vr-link-capture-checkerboard``: save SBS checkerboard shots for stereo calibration.

Frames come from the USB eyes_stereo camera (``--source usb``) or from the
robot's DDS JPEG topic (``--source dds``, needs igris_c_sdk). Both halves are
checked for the checkerboard live; a shot is saved manually (Space/Enter) or
automatically (A toggles) into ``<output-dir>/<yymmdd_HHMMSS>/NNNN_sbs.png``
together with ``capture_metadata.json``. Frames are saved as-is so the
calibration sees faithful pixels.
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence

from ._common import camera_api_from_config, default_captures_root, detect_checkerboard, load_tools_config, put_label, save_frame, split_sbs, write_metadata


class _UsbCapture:
    def __init__(self, camera_config: dict) -> None:
        import cv2

        raw_index = camera_config.get("index", 0)
        is_index = str(raw_index).isdigit()
        camera_index = int(raw_index) if is_index else str(raw_index)
        api = camera_api_from_config(camera_config.get("api", "CAP_V4L2"))
        self.cap = cv2.VideoCapture(camera_index, api)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera index {camera_index}")
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(camera_config.get("width", 2560)))
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(camera_config.get("height", 720)))
        self.cap.set(cv2.CAP_PROP_FPS, int(camera_config.get("fps", 30)))
        print(
            "Camera: "
            f"{self.cap.get(cv2.CAP_PROP_FRAME_WIDTH):.0f}x{self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT):.0f} @ "
            f"{self.cap.get(cv2.CAP_PROP_FPS):.0f} fps"
        )

    def read(self):
        ok, frame = self.cap.read()
        return frame if ok else None

    def release(self) -> None:
        self.cap.release()


class _DdsCapture:
    def __init__(self, dds_config: dict, args: argparse.Namespace, base_dir: Path) -> None:
        from vr_link.network.dds import DdsChannel, resolve_config_uri
        from vr_link.sources.dds_camera import DEFAULT_DDS_CAMERA_TOPIC, DdsCameraSource

        domain_id = args.domain_id if args.domain_id is not None else int(dds_config.get("domain_id", 1))
        namespace = args.namespace if args.namespace is not None else str(dds_config.get("namespace", ""))
        topic = args.topic if args.topic is not None else str(dds_config.get("camera_topic") or dds_config.get("topic") or DEFAULT_DDS_CAMERA_TOPIC)
        raw_uri = args.dds_config if args.dds_config is not None else (dds_config.get("config_uri") or dds_config.get("config"))
        self.channel = DdsChannel.create(
            domain_id=domain_id, namespace=namespace, topic_naming=str(dds_config.get("topic_naming", "native")),
            config_uri=resolve_config_uri(raw_uri, base_dir=base_dir),
        )
        self.source = DdsCameraSource(self.channel, topic)
        self.source.open()
        print(f"Source: DDS topic {self.channel.resolve(topic)}")

    def read(self):
        frame = self.source.read(timeout=0.03)
        return None if frame is None else frame.pixels

    def release(self) -> None:
        self.source.close()
        self.channel.release()


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stereo checkerboard capture (USB camera or DDS topic)")
    p.add_argument("--config", default=None, metavar="PATH", help="calibration_tools.yaml (default: packaged)")
    p.add_argument("--source", choices=("usb", "dds"), default="usb")
    p.add_argument("--output-dir", type=Path, default=None, help="root of the timestamped capture folders (default local:captures, i.e. <checkout>/local/captures)")
    p.add_argument("--domain-id", type=int, default=None, help="dds: override dds.domain_id")
    p.add_argument("--namespace", default=None, help="dds: override dds.namespace")
    p.add_argument("--topic", default=None, help="dds: override dds.camera_topic")
    p.add_argument("--dds-config", default=None, metavar="PATH_OR_URI", help="dds: override dds.config_uri")
    return p.parse_args(argv)


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

    target_count = int(capture_config.get("target_count", 20))
    mode = str(capture_config.get("mode", "both")).lower()
    if mode not in ("manual", "auto", "both"):
        raise ValueError("capture.mode must be one of: manual, auto, both")
    auto_enabled = mode in ("auto", "both")
    manual_enabled = mode in ("manual", "both")
    auto_active = bool(capture_config.get("auto_start", True)) and auto_enabled
    auto_interval = float(capture_config.get("auto_interval_sec", 1.0))
    require_both_detected = bool(capture_config.get("require_both_detected", True))
    image_format = str(capture_config.get("image_format", "png")).lower().lstrip(".")
    show_saved_feedback = bool(capture_config.get("show_saved_feedback", True))
    saved_feedback_sec = float(capture_config.get("saved_feedback_sec", 0.5))
    window_name = str(capture_config.get("window_name", "Stereo Checkerboard Capture"))

    output_root = args.output_dir if args.output_dir is not None else default_captures_root()
    output_dir = output_root / datetime.now().strftime("%y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=False)

    capture = _UsbCapture(camera_config) if args.source == "usb" else _DdsCapture(config.get("dds", {}), args, config_path.parent)

    print(f"Config: {config_path}")
    print(f"Output: {output_dir}")
    print("Keys: Space/Enter save, A toggle auto, Q/Esc quit")

    saved_count = 0
    last_auto_save = 0.0
    feedback_until = 0.0
    feedback_text = ""
    warned_size = False
    warned_waiting = False

    try:
        while saved_count < target_count:
            frame = capture.read()
            if frame is None:
                if not warned_waiting:
                    print("Waiting for frames...")
                    warned_waiting = True
                if (cv2.waitKey(30) & 0xFF) in (27, ord("q")):
                    break
                continue
            warned_waiting = False

            frame_height, frame_width = frame.shape[:2]
            if not warned_size and (frame_width != width or frame_height != height):
                print(f"Warning: actual frame size is {frame_width}x{frame_height}, expected {width}x{height}")
                warned_size = True

            left, right = split_sbs(frame, left_half)
            left_found, left_corners = detect_checkerboard(left, pattern_size)
            right_found, right_corners = detect_checkerboard(right, pattern_size)
            can_save = (left_found and right_found) or not require_both_detected
            now = time.monotonic()

            if auto_enabled and auto_active and can_save and saved_count < target_count and now - last_auto_save >= auto_interval:
                saved_count += 1
                saved_path = save_frame(output_dir, frame, saved_count, image_format)
                last_auto_save = now
                feedback_text = f"SAVED {saved_path.name}"
                feedback_until = now + saved_feedback_sec
                print(f"[{saved_count}/{target_count}] saved {saved_path}")

            preview = frame.copy()
            draw_left, draw_right = split_sbs(preview, left_half)
            if left_found:
                cv2.drawChessboardCorners(draw_left, pattern_size, left_corners, left_found)
            if right_found:
                cv2.drawChessboardCorners(draw_right, pattern_size, right_corners, right_found)
            midpoint = preview.shape[1] // 2
            cv2.line(preview, (midpoint, 0), (midpoint, preview.shape[0]), (255, 255, 255), 1)

            put_label(preview, f"Captured: {saved_count} / {target_count}", (12, 30))
            put_label(preview, f"Left: {'OK' if left_found else 'NO'}", (12, 62), (0, 220, 0) if left_found else (0, 0, 255))
            put_label(preview, f"Right: {'OK' if right_found else 'NO'}", (12, 94), (0, 220, 0) if right_found else (0, 0, 255))
            put_label(preview, f"Mode: {mode}", (12, 126))
            if auto_enabled:
                put_label(preview, f"Auto: {'ON' if auto_active else 'OFF'}", (12, 158), (0, 220, 0) if auto_active else (0, 180, 255))
            if show_saved_feedback and feedback_text and now < feedback_until:
                put_label(preview, feedback_text, (midpoint - 120, 40), (0, 255, 255), 0.8)
            cv2.imshow(window_name, preview)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord("a") and auto_enabled:
                auto_active = not auto_active
                feedback_text = f"AUTO {'ON' if auto_active else 'OFF'}"
                feedback_until = time.monotonic() + saved_feedback_sec
                print(feedback_text)
            if key in (13, 10, 32) and manual_enabled and saved_count < target_count:
                if can_save:
                    saved_count += 1
                    saved_path = save_frame(output_dir, frame, saved_count, image_format)
                    feedback_text = f"SAVED {saved_path.name}"
                    print(f"[{saved_count}/{target_count}] saved {saved_path}")
                else:
                    feedback_text = "NOT SAVED: checkerboard missing"
                    print(feedback_text)
                feedback_until = time.monotonic() + saved_feedback_sec
    finally:
        capture.release()
        cv2.destroyAllWindows()
        write_metadata(output_dir, config, config_path, saved_count, args.source)
        print(f"Saved {saved_count} image(s) to {output_dir}")


if __name__ == "__main__":
    main()
