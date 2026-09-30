"""Process-wide ``ChannelFactory`` wrapper (init once, release last)."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

from . import import_sdk

logger = logging.getLogger(__name__)

TOPIC_NAMINGS = ("native", "ros_compatible")


def resolve_config_uri(raw: Optional[str], *, base_dir: Optional[Path] = None) -> str:
    """``""``, a ``file://`` URI, or a path (relative to ``base_dir``) -> CycloneDDS URI."""

    if not raw:
        return ""
    raw = str(raw)
    if raw.startswith("file://"):
        return raw
    path = Path(raw)
    if not path.is_absolute() and base_dir is not None:
        path = Path(base_dir) / path
    return f"file://{os.path.abspath(path)}"


class DdsChannel:
    """The SDK's singleton participant. Create once per process; ``release`` after every subscriber stopped."""

    def __init__(self, sdk, channel, *, domain_id: int, namespace: str, topic_naming: str) -> None:
        self.sdk = sdk
        self._channel = channel
        self.domain_id = int(domain_id)
        self.namespace = str(namespace)
        self.topic_naming = topic_naming
        self._released = False

    @classmethod
    def create(cls, *, domain_id: int, namespace: str, topic_naming: str = "native", config_uri: str = "", sdk=None) -> "DdsChannel":
        if topic_naming not in TOPIC_NAMINGS:
            raise ValueError(f"topic_naming must be one of {TOPIC_NAMINGS}, got {topic_naming!r}")
        if not namespace:
            raise ValueError("dds.namespace must not be empty: a wrong namespace fails silently on the wire")
        if sdk is None:
            sdk = import_sdk()
        channel = sdk.ChannelFactory.Instance()
        parse_naming = getattr(sdk, "parse_topic_naming", None)
        if parse_naming is not None:
            channel.Init(int(domain_id), str(namespace), str(config_uri or ""), parse_naming(topic_naming))
        else:
            if topic_naming != "native":
                raise RuntimeError("this igris_c_sdk binding predates topic naming; only 'native' is available")
            channel.Init(int(domain_id), str(namespace), str(config_uri or ""))
        if hasattr(channel, "IsInitialized") and not channel.IsInitialized():
            raise RuntimeError(f"igris_c_sdk ChannelFactory failed to initialize (domain {domain_id}, namespace {namespace!r})")
        logger.info(
            "igris_c_sdk %s from %s (domain %d, namespace %r, naming %s)",
            getattr(sdk, "__version__", "?"), getattr(sdk, "__file__", "?"), int(domain_id), namespace, topic_naming,
        )
        return cls(sdk, channel, domain_id=domain_id, namespace=namespace, topic_naming=topic_naming)

    @property
    def raw(self):
        return self._channel

    def resolve(self, topic: str) -> str:
        """Wire name of a bare topic under the configured namespace/naming."""

        resolver = getattr(self._channel, "resolve", None)
        if resolver is not None:
            return str(resolver(topic))
        bare = topic[1:] if topic.startswith("/") else topic
        if self.topic_naming == "ros_compatible":
            return f"rt/{self.namespace}/{bare}"
        return f"{self.namespace}/{bare}"

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        try:
            self._channel.Release()
        except Exception as error:  # teardown must not mask the run's outcome
            logger.warning("igris_c_sdk ChannelFactory release failed: %s", error)
