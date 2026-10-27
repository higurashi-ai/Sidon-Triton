#!/bin/bash
# Convert (once), build the engines, benchmark, and check parity with the
# official demo output.
#   scripts/workflows/build_and_bench.sh [eval.wav ...]
# STRITON_DECODER=triton|cuda|trt  STRITON_ENCODER_PRECISION=fp16|bf16|fp32
source "$(dirname "$0")/common.sh"
if [ ! -f "${STRITON_BUNDLE}/encoder.safetensors" ]; then
  PY tools/convert_torchscript.py "${TS_ARGS[@]}" --output "${STRITON_BUNDLE}"
fi
WAV_ARGS=()
if [ $# -gt 0 ]; then WAV_ARGS=(--wav "$1"); fi
PY scripts/build_engines.py --bundle "${STRITON_BUNDLE}" \
  --engine-dir "${STRITON_ENGINE_DIR}" --onnx-dir "${STRITON_WORK}/onnx" "${WAV_ARGS[@]}" \
  --encoder-precision "${STRITON_ENCODER_PRECISION:-fp16}" --decoder "${STRITON_DECODER:-triton}"
PY scripts/bench_e2e.py --engine-dir "${STRITON_ENGINE_DIR}" "${WAV_ARGS[@]}" \
  --output "${STRITON_ENGINE_DIR}/bench_e2e.json"
if [ $# -gt 0 ]; then
  PY tools/compare_demo.py --engine-dir "${STRITON_ENGINE_DIR}" "${TS_ARGS[@]}" \
    --output-dir "${STRITON_ENGINE_DIR}/compare" "$@"
fi
echo "BUILD_AND_BENCH_OK"
