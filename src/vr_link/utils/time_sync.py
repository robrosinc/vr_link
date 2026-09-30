"""Timestamped ring buffers and nearest-sample matching.

The headset used to match DDS robot-state samples to each video frame by
timestamp; that logic now lives on the sender. ``StampedRingBuffer`` is the
shared structure: producers (DDS callbacks on their own thread) push
``(stamp_ns, value)``; the pump asks for the sample nearest to a frame's
capture time within a tolerance and gets ``None`` when there is none, so a
stale sample is never mistaken for a matched one.
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Generic, Optional, TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class Stamped(Generic[T]):
    stamp_ns: int
    value: T


@dataclass
class MatchStats:
    pushed: int = 0
    hits: int = 0
    misses: int = 0
    last_error_ns: int = 0
    max_error_ns: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0


class StampedRingBuffer(Generic[T]):
    """Bounded, thread-safe history of stamped samples.

    Stamps need not arrive in order (DDS may reorder); ``nearest`` scans the
    whole buffer, which is a few hundred entries at most.
    """

    def __init__(self, capacity: int = 512, *, name: str = "") -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.name = name
        self._items: Deque[Stamped[T]] = deque(maxlen=capacity)
        self._lock = threading.Lock()
        self._stats = MatchStats()

    @property
    def capacity(self) -> int:
        return self._items.maxlen or 0

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)

    def push(self, stamp_ns: int, value: T) -> None:
        with self._lock:
            self._items.append(Stamped(int(stamp_ns), value))
            self._stats.pushed += 1

    def clear(self) -> None:
        with self._lock:
            self._items.clear()

    def latest(self) -> Optional[Stamped[T]]:
        """Newest sample by stamp (not by arrival)."""

        with self._lock:
            if not self._items:
                return None
            return max(self._items, key=lambda item: item.stamp_ns)

    def latest_within(self, now_ns: int, max_age_ns: int) -> Optional[Stamped[T]]:
        """Newest sample whose stamp is within ``max_age_ns`` of ``now_ns``."""

        item = self.latest()
        if item is None or abs(int(now_ns) - item.stamp_ns) > int(max_age_ns):
            return None
        return item

    def nearest(self, target_ns: int, tolerance_ns: int) -> Optional[Stamped[T]]:
        """Sample with the smallest |stamp - target| that is <= tolerance.

        Ties resolve to the newer sample. Updates the match statistics.
        """

        target_ns = int(target_ns)
        tolerance_ns = int(tolerance_ns)
        with self._lock:
            best: Optional[Stamped[T]] = None
            best_error = tolerance_ns + 1
            for item in self._items:
                error = abs(item.stamp_ns - target_ns)
                if error < best_error or (error == best_error and best is not None and item.stamp_ns > best.stamp_ns):
                    best, best_error = item, error
            if best is None:
                self._stats.misses += 1
                return None
            self._stats.hits += 1
            self._stats.last_error_ns = best_error
            self._stats.max_error_ns = max(self._stats.max_error_ns, best_error)
            return best

    def stats(self) -> MatchStats:
        with self._lock:
            return MatchStats(**vars(self._stats))
