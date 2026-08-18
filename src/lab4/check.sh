#!/bin/bash
# check.sh — validate Lab 4 output against golden results.
# Usage: ./check.sh [RESULT_DIR] [GOLDEN_DIR]
# See `python3 scripts/check_result.py --help` for details.
set -euo pipefail

ROOT_DIR="$(pwd)"

# The lab4g10/lab4g5 node images preset AMSS_OUTPUT_ROOT to /workspace/lab4,
# the container's ephemeral volume that is invisible from the shared home /
# the devpod. Treat any such /workspace/* preset as unset (mirrors run.sh) so
# results written to the shared home are checked instead.
case "${AMSS_OUTPUT_ROOT:-}" in
  /workspace/*) unset AMSS_OUTPUT_ROOT ;;
esac

exec "${PYTHON:-python3}" "$ROOT_DIR/scripts/check_result.py" \
    --time-tolerance "${TIME_TOLERANCE:-1e-8}" \
    "$@"
