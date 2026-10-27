#!/bin/bash
# Speed of the official Sidon path (TorchScript fp32 + CPU fbank).
#   scripts/workflows/bench_baseline.sh [input.wav] [output.json]
source "$(dirname "$0")/common.sh"
unset NVIDIA_TF32_OVERRIDE  # the official path as users run it
WAV_ARGS=()
if [ $# -gt 0 ]; then WAV_ARGS=(--wav "$1"); fi
PY scripts/bench_baseline.py "${TS_ARGS[@]}" "${WAV_ARGS[@]}" \
  --output "${2:-${STRITON_WORK}/baseline.json}"
echo "BENCH_BASELINE_OK"
