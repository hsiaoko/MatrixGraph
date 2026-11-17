#!/usr/bin/env bash
# Header hygiene gate.
#
# Every header claimed portable must compile two ways:
#   1. as a plain C++ TU with no CUDA include paths (proves the header is
#      CUDA-free, so data-only reusers never link nvcc's world);
#   2. as an nvcc TU (proves it is includable from .cu files, which also
#      proves it is TBB-free: nvcc's EDG frontend cannot parse TBB headers,
#      see .audit/engine-grounding.md).
#
# Add headers to PUBLIC_HEADERS as the data layer is split; a violation of
# either rule is a build break here, not a later surprise in a consumer.
set -uo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)

PUBLIC_HEADERS=(
    core/util/error.h
    core/util/parallel.h
    core/engine/engine.h
    core/common/consts.h
    core/common/types.h
)

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
fail=0

for h in "${PUBLIC_HEADERS[@]}"; do
    printf '#include "%s"\n' "$h" >"$tmp/probe.cpp"
    if ! g++ -std=c++20 -fsyntax-only -I"$ROOT" "$tmp/probe.cpp" 2>"$tmp/cxx.err"; then
        echo "FAIL cxx  $h"
        head -3 "$tmp/cxx.err"
        fail=1
        continue
    fi
    cp "$tmp/probe.cpp" "$tmp/probe.cu"
    if ! nvcc -std=c++20 -I"$ROOT" -c "$tmp/probe.cu" -o "$tmp/probe.o" 2>"$tmp/cu.err"; then
        echo "FAIL cuda $h"
        head -3 "$tmp/cu.err"
        fail=1
        continue
    fi
    echo "OK        $h"
done

if [ "$fail" -ne 0 ]; then
    echo "header hygiene: FAIL"
    exit 1
fi
echo "header hygiene: PASS (${#PUBLIC_HEADERS[@]} headers)"