#!/bin/bash
# Launch the annotator (or any command) with galbot_sdk on the path.
#
# galbot_sdk is a compiled pybind extension; its shared-library dependencies
# (libgalbot_sdk.so, the *_proto.so bundle, opencv/pcl/protobuf, ...) are not
# baked in with an rpath, and libgalbot_sdk.so itself is missing a few NEEDED
# entries for its own *_proto.so dependencies (they resolve only if already
# loaded into the process's global symbol table) - so both LD_LIBRARY_PATH and
# LD_PRELOAD must be set before the interpreter starts.
#
# Usage:
#   scripts/run_with_galbot_sdk.sh                       # runs the annotator
#   scripts/run_with_galbot_sdk.sh python3 some_script.py # runs anything else
#
# Prefers a vendored copy at third_party/galbot_sdk/{python,lib} (gitignored -
# it's a compiled, arch/Python-version-specific blob, not source) so this
# script has no dependency on the galbot_g1_sdk_source checkout that built it.
# Falls back to building the paths from GALBOT_SDK_SOURCE_DIR (default
# ~/workspace/galbot_g1_sdk_source) if the vendored copy isn't there.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENDORED_DIR="${REPO_ROOT}/third_party/galbot_sdk"

if [[ -d "${VENDORED_DIR}/lib" ]]; then
    CORE_LIB="${VENDORED_DIR}/lib"
    THIRDPARTY_LIB=""
    export PYTHONPATH="${VENDORED_DIR}/python:${PYTHONPATH:-}"
else
    GALBOT_SDK_SOURCE_DIR="${GALBOT_SDK_SOURCE_DIR:-$HOME/workspace/galbot_g1_sdk_source}"
    CORE_LIB="${GALBOT_SDK_SOURCE_DIR}/output/galbot_sdk/linux-x86_64-gcc940/lib"
    THIRDPARTY_LIB="${GALBOT_SDK_SOURCE_DIR}/submodules/galbot_sdk_common/submodules/ThirdParty/gcc940-x86_64-ubuntu2004-gnu/lib"
fi

if [[ ! -d "${CORE_LIB}" ]]; then
    echo "[ERROR] ${CORE_LIB} 不存在，请先编译/放置 x86_64 版 galbot_sdk（见 scripts/run_with_galbot_sdk.sh 顶部注释）" >&2
    exit 1
fi

PRELOAD="$(ls "${CORE_LIB}"/*_proto.so 2>/dev/null | paste -sd: -)"

export LD_LIBRARY_PATH="${CORE_LIB}${THIRDPARTY_LIB:+:${THIRDPARTY_LIB}}:${LD_LIBRARY_PATH:-}"
export LD_PRELOAD="${PRELOAD}${LD_PRELOAD:+:${LD_PRELOAD}}"

cd "${REPO_ROOT}"

if [[ $# -eq 0 ]]; then
    exec uv run galbot-motion-obstacle-annotator
else
    exec uv run "$@"
fi
