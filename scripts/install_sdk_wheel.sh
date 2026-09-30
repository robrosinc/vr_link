#!/usr/bin/env bash
# Install the igris_c_sdk Python binding wheel into the active pixi environment.
# The wheel is produced by a human with `task wheel` inside thirdparty/igris_c_sdk
# (checked out at the pinned commit); its filename carries a setuptools-scm
# version, so glob it. vr_link only subscribes with it (no command path).
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
dist="${root}/thirdparty/igris_c_sdk/dist"
shopt -s nullglob
wheels=("${dist}"/igris_c_sdk-*.whl)
if [ ${#wheels[@]} -eq 0 ]; then
  echo "No wheel in ${dist}. Build it first: (cd thirdparty/igris_c_sdk && task wheel)" >&2
  exit 1
fi
if [ ${#wheels[@]} -gt 1 ]; then
  echo "Several wheels in ${dist}; keep exactly one:" >&2
  printf '  %s\n' "${wheels[@]}" >&2
  exit 1
fi
python -m pip install --force-reinstall --no-deps "${wheels[0]}"
python - <<'PY'
import igris_c_sdk
assert hasattr(igris_c_sdk, "ChannelFactory"), igris_c_sdk.__file__
print("igris_c_sdk binding OK:", igris_c_sdk.__file__, getattr(igris_c_sdk, "__version__", "?"))
PY
