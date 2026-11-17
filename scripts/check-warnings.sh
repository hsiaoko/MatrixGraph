#!/usr/bin/env bash
# Warning gate for the cleanup refactor.
#
# Rebuilds a configured build directory, extracts warnings from project code
# (core/, apps/, tools/ -- third_party/ and bundled deps are exempt), dedupes
# by file:line:column, and fails when the unique count exceeds the allowed
# baseline. The baseline only ratchets down: regenerate it after each unit
# that removes warnings, never after one that adds any.
#
# NOTE: counts only warnings emitted during THIS build. Incremental builds
# recompile few TUs, so a passing gate here does not re-prove untouched files;
# run against a fresh build directory for a full measurement.
#
# Usage: scripts/check-warnings.sh <build-dir> [max-unique-warnings]
# Default cap: 173 (Debug+Release union measured at HEAD 4cec1d5, see
# .audit/warnings-baseline*.txt).
set -uo pipefail

BUILD_DIR=${1:?usage: check-warnings.sh <build-dir> [max-unique-warnings]}
CAP=${2:-173}

LOG=$(mktemp /tmp/mg-warnings-XXXXXX.log)
trap 'rm -f "$LOG" "$LOG.uniq"' EXIT

cmake --build "$BUILD_DIR" -j"$(nproc)" >"$LOG" 2>&1
BUILD_RC=$?
if [ "$BUILD_RC" -ne 0 ]; then
    echo "BUILD FAILED (rc=$BUILD_RC); last 30 lines:"
    tail -30 "$LOG"
    exit "$BUILD_RC"
fi

grep 'warning:' "$LOG" |
    grep -E '/(core|apps|tools)/' |
    grep -v '/third_party/' |
    awk -F'warning:' '{split($1,a,":"); print a[1]":"a[2]":"a[3]}' |
    sort -u >"$LOG.uniq"

COUNT=$(wc -l <"$LOG.uniq")
echo "unique project warnings: $COUNT (cap: $CAP)"

if [ "$COUNT" -gt "$CAP" ]; then
    echo "NEW WARNINGS above cap. Offending locations by category:"
    grep 'warning:' "$LOG" | grep -E '/(core|apps|tools)/' | grep -v '/third_party/' |
        grep -oE '\[-W[a-z-]+\]' | sort | uniq -c | sort -rn | head -10
    exit 1
fi

echo "warning gate: PASS"
