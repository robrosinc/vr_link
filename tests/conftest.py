"""Test this checkout's package, not whichever vr_link_schema is pip-installed.

Consumers install the package editable from their own submodule checkout, so
an environment may already expose a different revision of ``vr_link`` /
``vr_link_generated``. Putting this repository's source roots first on
``sys.path`` makes the tests exercise the code under test.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for root in (REPO_ROOT / "python", REPO_ROOT / "generated" / "python"):
    root_str = str(root)
    if root_str in sys.path:
        sys.path.remove(root_str)
    sys.path.insert(0, root_str)

import vr_link  # noqa: E402  (after the sys.path setup on purpose)

assert Path(vr_link.__file__).resolve().is_relative_to(REPO_ROOT), (
    f"vr_link resolved outside this checkout: {vr_link.__file__}"
)
