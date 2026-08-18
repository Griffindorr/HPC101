#!/bin/bash
# check.sh — validate Lab 4 output against golden results.
# Usage: ./check.sh [RESULT_DIR] [GOLDEN_DIR]
# See `python3 scripts/check_result.py --help` for details.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
PWD_PHYS="$(pwd -P)"
if [[ -x "$PWD_PHYS/run.sh" ]]; then
    ROOT_DIR="$PWD_PHYS"
else
    ROOT_DIR="$SCRIPT_DIR"
fi
export AMSS_LAB_DIR="${AMSS_LAB_DIR:-$ROOT_DIR}"
export AMSS_OUTPUT_ROOT="${AMSS_OUTPUT_ROOT:-$ROOT_DIR}"

exec "${PYTHON:-python3}" "$ROOT_DIR/scripts/check_result.py" \
    --time-tolerance "${TIME_TOLERANCE:-1e-8}" \
    "$@"
