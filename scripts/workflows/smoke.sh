#!/bin/bash
# Quick end-to-end check of built engines: CUDA tail kernels, benchmark,
# file restoration, parity with the official demo.
#   scripts/workflows/smoke.sh input.wav [input2.wav ...]
source "$(dirname "$0")/common.sh"
[ $# -gt 0 ] || { echo "usage: $0 input.wav [...]" >&2; exit 2; }
if [ -f "${STRITON_ENGINE_DIR}/decoder_tail.pt" ]; then
  PY tools/test_cuda_tail.py --bundle "${STRITON_BUNDLE}" | grep -E "^tail|OK"
fi
PY scripts/bench_e2e.py --engine-dir "${STRITON_ENGINE_DIR}" --seconds 15,96 \
  --clips 32 --wav "$1" | grep -E "^e2e|^clips|OK"
PY scripts/run.py --engine-dir "${STRITON_ENGINE_DIR}" \
  --output-dir "${STRITON_WORK}/smoke-out" --format flac "$@"
PY tools/compare_demo.py --engine-dir "${STRITON_ENGINE_DIR}" "${TS_ARGS[@]}" \
  --output-dir "${STRITON_WORK}/smoke-compare" "$@"
echo "SMOKE_OK"
