#!/bin/bash
# Large-scale test: every clip in a directory (finite / length / clipping
# / determinism) plus parity with the official demo on a random subset.
#   scripts/workflows/stress.sh clip_dir [long_file ...]
source "$(dirname "$0")/common.sh"
[ $# -gt 0 ] || { echo "usage: $0 clip_dir [long_file ...]" >&2; exit 2; }
CLIPS=$1; shift
PY tools/stress_test.py --engine-dir "${STRITON_ENGINE_DIR}" "${TS_ARGS[@]}" \
  --clip-dir "${CLIPS}" --long-files "$@" --output "${STRITON_ENGINE_DIR}/stress.json"
