#!/usr/bin/env bash
# Regenerates the committed language bindings under generated/ from
# vr_link.fbs. Run after every schema change and commit vr_link.fbs together
# with generated/ in the same commit; consumers (vr_system, hit-bench) only
# bump their submodule pin and never run flatc themselves.
#
# flatc is version-pinned: vr_system vendors the flatbuffers C++ runtime at
# the same tag and its generated header static_asserts the version, so a
# mismatched flatc produces bindings that fail to compile there. Get flatc
# from the conda-forge flatbuffers package or
# https://github.com/google/flatbuffers/releases, or point FLATC at a local
# build.
set -euo pipefail

FLATC_VERSION="25.12.19"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FLATC="${FLATC:-flatc}"

version_output="$("$FLATC" --version)"
if [[ "$version_output" != *"$FLATC_VERSION"* ]]; then
    echo "error: need flatc $FLATC_VERSION, got: $version_output" >&2
    echo "       (must match the flatbuffers runtime pinned by consumers)" >&2
    exit 1
fi

"$FLATC" --cpp \
    -o "$REPO_ROOT/generated/cpp" \
    "$REPO_ROOT/vr_link.fbs"
"$FLATC" --python --gen-onefile \
    -o "$REPO_ROOT/generated/python" \
    "$REPO_ROOT/vr_link.fbs"
echo "Regenerated generated/cpp/vr_link_generated.h and generated/python/vr_link_generated.py"
