"""State-only sender: RobotState/Signals at a fixed rate, no camera (headset "NO CAMERA")."""

from __future__ import annotations

import logging
import threading
import time
from typing import Optional

from ..calibration.schema import CalibrationError
from ..network.calibration_server import CalibrationProvider
from ..network.tcp_server import EnvelopeTcpServer
from ..sources.shm import ShmSource
from ..utils.clocks import ms_to_ns, wall_ns
from .robot_state import RobotStateAssembler, SignalsProvider

logger = logging.getLogger(__name__)

STATS_INTERVAL_SEC = 5.0


class StateOnlyPump:
    """Emit the newest robot state ``rate_hz`` times per second while a headset is attached."""

    def __init__(
        self,
        server: EnvelopeTcpServer,
        assembler: RobotStateAssembler,
        signals: SignalsProvider,
        *,
        rate_hz: float = 50.0,
        max_age_ms: float = 100.0,
        shm: Optional[ShmSource] = None,
        calibration: Optional[CalibrationProvider] = None,
    ) -> None:
        if rate_hz <= 0:
            raise ValueError("rate_hz must be positive")
        self.server = server
        self.assembler = assembler
        self.signals = signals
        self.period = 1.0 / float(rate_hz)
        self.max_age_ns = ms_to_ns(max_age_ms)
        self.shm = shm
        self.calibration = calibration
        self.records_sent = 0
        self._calibration_set = False

    def _maybe_set_calibration(self) -> None:
        if self.calibration is None or self._calibration_set:
            return
        try:
            self.server.set_calibration(self.calibration())
            self._calibration_set = True
        except CalibrationError as error:
            logger.debug("calibration not ready: %s", error)
        except Exception:
            logger.exception("calibration provider failed; no in-band calibration")
            self._calibration_set = True

    def run(self, stop: threading.Event) -> None:
        if self.shm is not None:
            self.shm.open()
        next_deadline = time.monotonic()
        last_stats = time.monotonic()
        try:
            while not stop.is_set():
                next_deadline += self.period
                if not self.server.client_connected:
                    time.sleep(self.period)
                    next_deadline = time.monotonic()
                    continue
                self._maybe_set_calibration()
                block = self.shm.read_state(0.0) if self.shm is not None else None
                now = wall_ns()
                sample = self.assembler.assemble_latest(now, self.max_age_ns, block)
                sig = self.signals.sample(block)
                if block is not None and block.state_stamp_realtime_ns:
                    stamp, sim_ns = block.state_stamp_realtime_ns, block.state_sim_time_ns
                else:
                    stamp, sim_ns = now, 0
                self.server.send_state(wall_ns=stamp, sim_ns=sim_ns, robot_state=sample, signals=sig)
                self.records_sent += 1
                if time.monotonic() - last_stats >= STATS_INTERVAL_SEC:
                    logger.info("state-only: %d records sent; match %s", self.records_sent, self.assembler.format_stats())
                    last_stats = time.monotonic()
                delay = next_deadline - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
                else:
                    next_deadline = time.monotonic()
        finally:
            if self.shm is not None:
                self.shm.close()


__all__ = ["StateOnlyPump"]
