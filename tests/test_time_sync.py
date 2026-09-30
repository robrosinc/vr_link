"""StampedRingBuffer: nearest-within-tolerance matching semantics."""
import threading

from vr_link.utils.time_sync import StampedRingBuffer


def test_nearest_picks_closest_not_newest_and_respects_tolerance():
    buf = StampedRingBuffer(8)
    for stamp in (100, 200, 300):
        buf.push(stamp, f"s{stamp}")
    assert buf.nearest(190, 15).value == "s200"
    assert buf.nearest(215, 15).value == "s200"
    assert buf.nearest(216, 15) is None  # boundary is inclusive at exactly tolerance
    assert buf.nearest(215, 14) is None
    stats = buf.stats()
    assert (stats.hits, stats.misses, stats.max_error_ns) == (2, 2, 15)


def test_ties_resolve_to_the_newer_sample_and_order_is_irrelevant():
    buf = StampedRingBuffer(8)
    buf.push(300, "late")
    buf.push(100, "early")
    assert buf.nearest(200, 100).value == "late"
    assert buf.latest().value == "late"


def test_capacity_evicts_oldest_arrivals():
    buf = StampedRingBuffer(3)
    for stamp in range(5):
        buf.push(stamp, stamp)
    assert len(buf) == 3 and buf.nearest(0, 0) is None and buf.nearest(2, 0).value == 2


def test_latest_within_age_and_clear():
    buf = StampedRingBuffer(4)
    buf.push(1_000, "a")
    assert buf.latest_within(1_050, 100).value == "a"
    assert buf.latest_within(2_000, 100) is None
    buf.clear()
    assert buf.latest() is None and len(buf) == 0


def test_concurrent_push_and_nearest_do_not_raise():
    buf = StampedRingBuffer(64)
    stop = threading.Event()

    def producer():
        i = 0
        while not stop.is_set():
            buf.push(i, i)
            i += 1

    thread = threading.Thread(target=producer)
    thread.start()
    for _ in range(2000):
        buf.nearest(500, 1_000_000)
    stop.set()
    thread.join()
    assert buf.stats().pushed > 0
