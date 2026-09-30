"""Clock helpers shared by senders, sources and the DDS layer.

All wire and SHM stamps are nanoseconds. ``wall_ns`` is CLOCK_REALTIME, the
domain the robot bridge, igris_c_gmr and the cameras stamp their samples in,
so frame/state matching compares like with like.
"""

from __future__ import annotations

import time
from typing import Tuple

NS_PER_S = 1_000_000_000
NS_PER_MS = 1_000_000


def wall_ns() -> int:
    """CLOCK_REALTIME in nanoseconds."""

    return time.time_ns()


def stamp_to_ns(sec: int, nanosec: int) -> int:
    """ROS-style (sec, nanosec) header stamp to nanoseconds."""

    return int(sec) * NS_PER_S + int(nanosec)


def split_ns(ns: int) -> Tuple[int, int]:
    """Nanoseconds to a (sec, nanosec) pair."""

    ns = int(ns)
    return ns // NS_PER_S, ns % NS_PER_S


def ms_to_ns(milliseconds: float) -> int:
    return int(round(float(milliseconds) * NS_PER_MS))
