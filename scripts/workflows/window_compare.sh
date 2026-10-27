#!/bin/bash
# 96 s demo chunking vs 20 s sliding windows: speed and listening pairs.
#   scripts/workflows/window_compare.sh output_dir input.wav [...]
source "$(dirname "$0")/common.sh"
[ $# -gt 1 ] || { echo "usage: $0 output_dir input.wav [...]" >&2; exit 2; }
OUT=$1; shift
PY tools/make_window_compare.py --engine-dir "${STRITON_ENGINE_DIR}" "${TS_ARGS[@]}" \
  --output-dir "${OUT}" --speed-wav "$1" "$@"
