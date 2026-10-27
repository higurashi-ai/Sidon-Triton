#!/bin/bash
# Disk-to-disk throughput of scripts/run.py (overlapped decode / restore /
# encode) on a directory of audio files, WAV and FLAC output.
#   scripts/workflows/stream_bench.sh input_dir [glob]
source "$(dirname "$0")/common.sh"
[ $# -gt 0 ] || { echo "usage: $0 input_dir [glob]" >&2; exit 2; }
for fmt in wav flac; do
  rm -rf "${STRITON_WORK}/stream-${fmt}"
  PY scripts/run.py --engine-dir "${STRITON_ENGINE_DIR}" --input-dir "$1" \
    --glob "${2:-**/*}" --output-dir "${STRITON_WORK}/stream-${fmt}" \
    --format "${fmt}" --report "${STRITON_WORK}/stream-${fmt}.json"
done
echo "STREAM_BENCH_OK"
