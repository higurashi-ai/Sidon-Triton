# Sidon-Triton runtime environment — source this before any script.
# Copy to env.local.sh and adjust paths for your site.

# REQUIRED: TF32 must be off. The fbank front-end GEMM must be true fp32
# (TF32 noise turns quiet-bin log-mel chaotic) and TensorRT refuses to run
# engines built under a different setting.
export NVIDIA_TF32_OVERRIDE=0

# Python with this package installed (see docs/SETUP.md); default: python3 on PATH
export STRITON_PYTHON="${STRITON_PYTHON:-$(command -v python3)}"

# Optional TensorRT back-end ([trt] extra): tensorrt-cu12 pip wheels put
# libnvinfer under site-packages/tensorrt_libs. Skipped when not installed.
TRT_LIBS="$("${STRITON_PYTHON}" - 2>/dev/null <<'PY' || true
import pathlib, tensorrt_libs
print(pathlib.Path(tensorrt_libs.__file__).parent)
PY
)"
if [ -n "${TRT_LIBS}" ]; then
  export LD_LIBRARY_PATH="${TRT_LIBS}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

# Triton kernel cache off shared filesystems (NFS quota / lock contention)
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/$USER/triton-cache}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/tmp/$USER/torchinductor}"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
