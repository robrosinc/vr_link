#!/usr/bin/env bash
# One-shot igris_c_sdk setup for a fresh clone:
#   1. git submodule update --init --recursive   (SDK + its cyclonedds/cyclonedds-cxx/pybind11)
#   2. task wheel in thirdparty/igris_c_sdk      (CMake build + wheel under the SDK's own pixi env)
#   3. pixi run install-sdk                       (pip install the wheel into the vr_link env)
#
# Run it from a plain shell with ROS 2 Jazzy sourced, NOT under `pixi run`/`pixi shell`
# of this repo: step 2 configures CMake inside the SDK's own pixi env, and an already
# activated vr_link env can leak its interpreter into that configure (see the --fresh
# note in thirdparty/igris_c_sdk/taskfile.yml). Only step 3 enters the vr_link env.
#
# Host prerequisites (not installed here): ROS 2 Jazzy (ament_cmake,
# rosidl_default_generators, std_msgs, sensor_msgs), OpenSSL dev headers, a C++17
# compiler, go-task (`task`) and `pixi`.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
sdk="${root}/thirdparty/igris_c_sdk"

die() { echo "setup_sdk: $*" >&2; exit 1; }

if [ "${PIXI_PROJECT_ROOT:-}" = "${root}" ]; then
  die "run from a plain shell, not inside this repo's pixi env (pixi run / pixi shell)"
fi
for tool in git task pixi; do
  command -v "${tool}" >/dev/null 2>&1 || die "'${tool}' not found on PATH"
done
if [ -z "${AMENT_PREFIX_PATH:-}" ]; then
  die "ROS 2 not sourced (AMENT_PREFIX_PATH unset); e.g. 'source /opt/ros/jazzy/setup.bash'"
fi

echo "== [1/3] submodules"
git -C "${root}" submodule update --init --recursive

echo "== [2/3] igris_c_sdk wheel (task wheel, SDK pixi env)"
(cd "${sdk}" && task wheel)

echo "== [3/3] install wheel into the vr_link pixi env"
pixi run --manifest-path "${root}/pyproject.toml" install-sdk
