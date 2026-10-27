#!/bin/bash
# --decoder trt precision variants (--decoder-fp32), one per GPU: build,
# benchmark, parity with the official demo on the given inputs.
#   scripts/workflows/decoder_precision_sweep.sh eval.wav [eval2.wav ...]
# STRITON_VARIANTS (space separated, default below)
source "$(dirname "$0")/common.sh"
[ $# -gt 0 ] || { echo "usage: $0 eval.wav [...]" >&2; exit 2; }
VARIANTS=${STRITON_VARIANTS:-"weak none final final,block4 final,residual"}
GPUS=$(nvidia-smi -L | wc -l)
gpu=0; pids=()
for parts in ${VARIANTS}; do
  dir="${STRITON_ENGINE_DIR}-trt-${parts//,/-}"
  (
    export CUDA_VISIBLE_DEVICES=$((gpu % GPUS))
    export TRITON_CACHE_DIR="${STRITON_WORK}/triton-${gpu}"
    PY scripts/build_engines.py --bundle "${STRITON_BUNDLE}" --engine-dir "${dir}" \
      --onnx-dir "${STRITON_WORK}/onnx-${gpu}" --wav "$1" \
      --decoder trt --decoder-fp32 "${parts}"
    PY scripts/bench_e2e.py --engine-dir "${dir}" --wav "$1" \
      --output "${dir}/bench_e2e.json"
    PY tools/compare_demo.py --engine-dir "${dir}" "${TS_ARGS[@]}" \
      --output-dir "${dir}/compare" "$@"
  ) > "${dir}.log" 2>&1 && echo "OK ${parts}" || echo "FAILED ${parts} (${dir}.log)" &
  pids+=($!)
  gpu=$((gpu + 1))
  if [ $((gpu % GPUS)) -eq 0 ]; then wait "${pids[@]}"; pids=(); fi
done
wait
