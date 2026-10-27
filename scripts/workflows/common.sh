# shellcheck shell=bash
# Sourced by every workflow: repo root, environment, shared defaults.
# Override any STRITON_* variable from the environment.
set -euo pipefail
STRITON_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
if [ -f "${STRITON_ROOT}/configs/env.local.sh" ]; then
  source "${STRITON_ROOT}/configs/env.local.sh"
else
  source "${STRITON_ROOT}/configs/env.example.sh"
fi
STRITON_BUNDLE=${STRITON_BUNDLE:-${STRITON_ROOT}/bundle}              # converted weights
STRITON_ENGINE_DIR=${STRITON_ENGINE_DIR:-${STRITON_ROOT}/engines}     # TensorRT engines
STRITON_WORK=${STRITON_WORK:-${TMPDIR:-/tmp}/sidon-triton-${USER:-user}}  # scratch outputs
# local sidon-v0.1 TorchScript files; unset = download from the Hub
TS_ARGS=()
if [ -n "${STRITON_TORCHSCRIPT:-}" ]; then TS_ARGS=(--torchscript-dir "${STRITON_TORCHSCRIPT}"); fi
mkdir -p "${STRITON_WORK}"
cd "${STRITON_ROOT}"
PY() { "${STRITON_PYTHON}" -W ignore "$@"; }
