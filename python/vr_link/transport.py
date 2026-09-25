"""Abstract video stream transport interface.

The interface is intentionally transport-agnostic so the raw-TCP (VSTR), and
future wired or DDS implementations can be exchanged without changing the
simulation loop. It sits at the semantic level - "deliver one encoded video
frame with its metadata" - so each implementation owns its wire framing
(headers, topics, packetization).

Pose/controller/tracker *input* from the VR device is deliberately not part of
this interface: that is the :class:`vr_link.types.VRDevice` seam. The two
channels have opposite direction and requirements and must stay independently
replaceable; implementations may still share a physical connection internally.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class StreamTransport(ABC):
    """Abstract one-way video stream transport driven by the simulation loop.

    Lifecycle: ``start()`` before the loop, ``stop()`` after it. All other
    methods are called from the simulation loop (or the encoder's reader
    thread) and must never block on the network or on a slow receiver.
    """

    @abstractmethod
    def start(self) -> None:
        """Open the transport and begin accepting/serving receivers."""
        raise NotImplementedError

    @abstractmethod
    def stop(self) -> None:
        """Shut the transport down and release its resources."""
        raise NotImplementedError

    @property
    @abstractmethod
    def client_connected(self) -> bool:
        """Whether at least one receiver is currently attached."""
        raise NotImplementedError

    @abstractmethod
    def consume_keyframe_request(self) -> bool:
        """Return True once when a receiver attached since the last check.

        The caller responds by restarting the encoder so the next submitted
        frame is an IDR; the transport is responsible for discarding delta
        frames a fresh receiver cannot decode.
        """
        raise NotImplementedError

    def set_camera_params(self, params: dict, width: int, height: int) -> None:
        """Provide camera parameters for receivers that need them.

        ``width``/``height`` are the per-eye image size. The default does
        nothing so transports without a parameter channel remain valid.
        """

    @abstractmethod
    def send_video(
        self,
        access_unit: bytes,
        *,
        is_key: bool,
        width: int,
        height: int,
        wall_ns: int,
        sim_ns: int,
        head_pose: tuple[float, ...] | None = None,
    ) -> None:
        """Deliver one encoded H.264 access unit without blocking.

        ``width``/``height`` describe the encoded (side-by-side) frame.
        ``wall_ns`` is CLOCK_REALTIME at frame acquisition and ``sim_ns`` is
        the simulation time of the control step that rendered the frame; how
        (or whether) they reach the receiver is up to the implementation.

        ``head_pose`` is the pose that rendered this frame — 7 floats
        (position xyz in meters, quaternion wxyz) in the pose-uplink command frame —
        or ``None`` when unavailable. Implementations without a pose channel
        may ignore it; the receiver-side latency reprojection tolerates its
        absence.
        """
        raise NotImplementedError
