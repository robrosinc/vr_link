"""PC side of the headset link: VRS1 pose uplink, VSTR video downlink, calibration.

``vr_link.types`` holds the decoded data model and the ``VRDevice`` interface;
``vr_link.tcp_device`` holds the VRS1 framing, the record builder/decoder and
the TCP server-side receiver. The flatc binding lives beside this package as
the flat module ``vr_link_generated``. Body-joint orientation conventions are
the consumer's concern: pass a ``body_joint_adapter`` (see ``tcp_device``).

The video side (``wire``, ``stream_server``, ``encoder``, ``sources``,
``pump``) and the stereo calibration package (``calibration``) are imported
explicitly by their users; importing this package stays dependency-light
(flatbuffers + numpy). See the repository README for the channel layout.
"""

from .tcp_device import (
    BODY_JOINT_COUNT,
    HEAD_VALID_BIT,
    VRS1_HEADER_FORMAT,
    VRS1_HEADER_SIZE,
    VRS1_MAGIC,
    VRS1_MAX_PAYLOAD_BYTES,
    BodyJointAdapter,
    Quat,
    TcpVRDevice,
    build_vr_state_record,
    decode_vr_state_payload,
)
from .types import (
    BODY_JOINT_NAMES,
    HAND_KEYPOINT_COUNT,
    XROBOT_BODY_JOINT_NAMES,
    ControllerInput,
    HandTrackingState,
    TrackedPose,
    VRDevice,
    VRInputState,
)

__all__ = [
    "BODY_JOINT_COUNT",
    "BODY_JOINT_NAMES",
    "BodyJointAdapter",
    "ControllerInput",
    "HAND_KEYPOINT_COUNT",
    "HEAD_VALID_BIT",
    "HandTrackingState",
    "Quat",
    "TcpVRDevice",
    "TrackedPose",
    "VRDevice",
    "VRInputState",
    "VRS1_HEADER_FORMAT",
    "VRS1_HEADER_SIZE",
    "VRS1_MAGIC",
    "VRS1_MAX_PAYLOAD_BYTES",
    "XROBOT_BODY_JOINT_NAMES",
    "build_vr_state_record",
    "decode_vr_state_payload",
]
