"""``vr-link-calibration-fov``: print the field of view implied by a calibration YAML/JSON."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

from vr_link.calibration.fov import format_fov_summary
from vr_link.calibration.yaml_io import calibration_from_json, load_unreal_yaml


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Field of view implied by a stereo calibration file")
    p.add_argument("path", type=Path, help="flat calibration YAML (left_fx: ...) or calibration JSON")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    cal = calibration_from_json(args.path) if args.path.suffix.lower() == ".json" else load_unreal_yaml(args.path)
    print(format_fov_summary(cal))


if __name__ == "__main__":
    main()
