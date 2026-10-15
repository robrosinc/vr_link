"""igris_c_sdk (CycloneDDS) access: import, channel singleton, subscribers and signal publishers.

Robot-state inputs are subscriptions (lowstate, handstate, handcmd,
gmr/refstate, the camera image topic). :mod:`.teleop` adds the teleop signal
topics (``vr/controller``, ``recording/state``) as ``Joy_`` publishers and
subscribers that any process holding an SDK participant can host. Robot
control topics (lowcmd, handcmd, services) are never written here. The SDK
Python binding is not on PyPI and vr_link does not vendor it (2.1): the host
repository that runs vr_link (igris_c_gmr, hit-tasks) builds the wheel from
its own ``igris_c_sdk`` submodule pin and installs it into the shared env, or
``IGRIS_C_SDK_ROOT`` points at a checkout. Nothing in this package imports it
at module load.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

WHEEL_HINT = (
    "Install the igris_c_sdk Python 3.12 wheel into this env from the host repository's igris_c_sdk submodule "
    "(e.g. igris_c_gmr: `task wheel` there, then `pixi run install-sdk`), or point IGRIS_C_SDK_ROOT at a checkout "
    "that contains igris_c_sdk/__init__.py."
)


def import_sdk():
    """Return the ``igris_c_sdk`` module or raise ``ImportError`` with a hint."""

    try:
        import igris_c_sdk as sdk
    except ImportError as error:
        sdk = _import_from_env_root()
        if sdk is None:
            raise ImportError(f"igris_c_sdk Python binding not importable ({error}). {WHEEL_HINT}") from error
    if not hasattr(sdk, "ChannelFactory"):
        raise ImportError(
            f"'igris_c_sdk' resolved to {getattr(sdk, '__file__', '?')} which has no ChannelFactory: this is the rosidl "
            "message package from a ROS overlay shadowing the pybind wheel. Remove that overlay from PYTHONPATH."
        )
    return sdk


def _import_from_env_root():
    env_root = os.environ.get("IGRIS_C_SDK_ROOT")
    if not env_root:
        return None
    for root in (Path(env_root), Path(env_root) / "python"):
        if (root / "igris_c_sdk" / "__init__.py").exists():
            sys.path.insert(0, str(root))
            try:
                import igris_c_sdk  # noqa: WPS433
            except ImportError:
                sys.path.remove(str(root))
            else:
                return igris_c_sdk
    return None


def sdk_type(sdk, name: str):
    """Message class lookup accepting both ``LowState_`` (current) and ``LowState`` (older) spellings."""

    for candidate in (f"{name}_", name):
        found = getattr(sdk, candidate, None)
        if found is not None:
            return found
    raise ImportError(f"igris_c_sdk exposes neither {name!r} nor {name + '_'!r}; binding version mismatch.")


from .channel import TOPIC_NAMINGS, DdsChannel, resolve_config_uri  # noqa: E402,F401
from .subscribers import (  # noqa: E402,F401
    DdsStateTopics,
    StampedSubscription,
    header_stamp_ns,
    joint_state_q,
    motor_cmd_q_by_id,
    motor_state_q,
    start_state_subscriptions,
)
from .teleop import (  # noqa: E402,F401
    ControllerStatePublisher,
    ControllerStateSubscriber,
    RecordingStatePublisher,
    RecordingStateSubscriber,
)
