"""Downlink messages (PC -> headset): framed FlatBuffers ``robros.vr.Envelope``.

Every record on the 9100 stream and on the 9101 calibration channel is
``frame_payload(<Envelope bytes>)``. The per-message modules wrap the
generated ``vr_link_generated`` accessors so the rest of the package (and
tests) never touch FlatBuffers directly:

* :mod:`.envelope`    framing constants, ``parse_envelope``, ``RecordReader``
* :mod:`.video`       ``VideoFrame`` (one H.264 access unit)
* :mod:`.calibration` ``Calibration`` / ``CalibrationRequest``
* :mod:`.robot_state` ``RobotState`` (frame-matched joint state)
* :mod:`.signals`     ``Signals`` (recording flag, source kind)
"""

from .envelope import (  # noqa: F401
    SCHEMA_VERSION,
    VRL2_HEADER_FORMAT,
    VRL2_HEADER_SIZE,
    VRL2_MAGIC,
    VRL2_MAX_PAYLOAD_BYTES,
    DecodedEnvelope,
    FramingError,
    MessageKind,
    RecordReader,
    frame_payload,
    parse_envelope,
)
from .calibration import build_calibration_payload, build_calibration_request_payload, decode_calibration  # noqa: F401
from .robot_state import (  # noqa: F401
    BODY_JOINT_COUNT,
    BODY_SERIAL_JOINT_NAMES,
    HAND_MOTOR_COUNT,
    HAND_MOTOR_IDS,
    RobotStateSample,
    StampedVector,
    build_robot_state_payload,
    decode_robot_state,
)
from .signals import SignalsSample, SourceKind, build_signals_payload, decode_signals  # noqa: F401
from .video import VideoFrameMsg, build_video_frame_payload, decode_video_frame  # noqa: F401
