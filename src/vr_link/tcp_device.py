"""Compatibility shim (2.0): the VRS1 uplink receiver lives in :mod:`vr_link.network.tcp_device`.

igris_c_gmr and hit-bench import from this path; keep it stable.
"""

from .network.tcp_device import *  # noqa: F401,F403
from .network.tcp_device import (  # noqa: F401
    BODY_JOINT_COUNT,
    HEAD_VALID_BIT,
    VRS1_HEADER_FORMAT,
    VRS1_HEADER_SIZE,
    VRS1_MAGIC,
    VRS1_MAX_PAYLOAD_BYTES,
    BodyJointAdapter,
    Pose,
    Quat,
    TcpVRDevice,
    build_vr_state_record,
    decode_vr_state_payload,
)
