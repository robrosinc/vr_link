"""``vr-link-make-calibration``: write the ideal stereo calibration YAML of a simulated rig.

Defaults match igris_c_description/mujoco/igris_c_v2_vision.xml.xacro (fovy 90,
14 deg toe-out, 2 * 0.026076 m baseline) and the igris_c_mujoco vision segment
(640x480 per eye). The simulator no longer needs this file at runtime (the
stream pump computes the same values from the live camera model); it remains
useful for offline consumers and as the golden reference in tests.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence

from vr_link.calibration.ideal import ideal_stereo_from_fovy
from vr_link.calibration.yaml_io import IDEAL_YAML_COMMENT, dump_unreal_yaml

DEFAULT_OUTPUT = Path("mujoco_stereo_calibration.yaml")


def render_ideal_yaml(cal: dict, *, generated_at: Optional[str] = None) -> str:
    """The legacy ``make_mujoco_calibration.py`` file layout."""

    meta = {
        "generated_at": generated_at or datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "source_input_dir": "synthetic (MuJoCo pinhole model, no capture)",
        "camera_model": "pinhole_as_fisheye_tan_series",
        # Same SBS order as the hardware: the left half of the frame is the RIGHT eye.
        "left_half": "right",
        "valid_pair_count": 0,
        "image_width": cal["image_width"],
        "image_height": cal["image_height"],
        "fovy_deg": cal["fovy_deg"],
        "toe_out_deg": cal["toe_out_deg"],
        "rms_left": 0.0,
        "rms_right": 0.0,
        "rms_stereo": 0.0,
        "baseline": cal["baseline_mm"],
        "baseline_unit": "mm",
    }
    return dump_unreal_yaml(cal, meta=meta, comments=IDEAL_YAML_COMMENT)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Write the ideal stereo calibration YAML of a simulated rig")
    p.add_argument("--width", type=int, default=640, help="per-eye render width in px (default 640)")
    p.add_argument("--height", type=int, default=480, help="per-eye render height in px (default 480)")
    p.add_argument("--fovy", type=float, default=90.0, help="camera fovy in degrees (default 90)")
    p.add_argument("--toe-out-deg", type=float, default=14.0, help="outward yaw of each eye in degrees (default 14)")
    p.add_argument("--baseline-mm", type=float, default=2 * 26.076, help="eye separation in mm (default 52.152)")
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help=f"output path (default ./{DEFAULT_OUTPUT})")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    cal = ideal_stereo_from_fovy(args.width, args.height, args.fovy, args.toe_out_deg, args.baseline_mm)
    args.output.write_text(render_ideal_yaml(cal), encoding="utf-8")
    print(f"wrote {args.output} ({args.width}x{args.height} per eye, fovy {args.fovy}, toe-out {args.toe_out_deg} deg)")


if __name__ == "__main__":
    main()
