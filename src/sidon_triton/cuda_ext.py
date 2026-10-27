"""Build (nvcc, no ninja) and load the fused DAC CUDA kernels.

    from sidon_triton import cuda_ext
    cuda_ext.load()            # registers torch.ops.sidon_triton.*

The shared library is cached next to the sources under ``build/`` keyed by
a hash of the source and flags, so a changed .cu triggers a rebuild.
Targets: ``STRITON_CUDA_ARCHS`` (e.g. "90" or "90;120"), else the visible
GPU's arch, else a portable list.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "csrc" / "dac_kernels.cu"
BUILD = ROOT / "build"
DEFAULT_ARCHS = "80;89;90;100;120"
_loaded: set[bool] = set()


def _nvcc() -> str:
    for candidate in (
        os.environ.get("CUDA_HOME", "") + "/bin/nvcc",
        shutil.which("nvcc") or "",
    ):
        if candidate and Path(candidate).exists():
            return candidate
    raise RuntimeError("nvcc not found: put the CUDA toolkit on PATH or set CUDA_HOME")


def _flags(bf16: bool = False) -> list[str]:
    from torch.utils.cpp_extension import include_paths, library_paths

    archs = os.environ.get("STRITON_CUDA_ARCHS", "")
    if not archs and torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        archs = f"{major}{minor}"
    archs = (archs or DEFAULT_ARCHS).split(";")
    gencode = []
    for arch in archs:
        gencode += ["-gencode", f"arch=compute_{arch},code=sm_{arch}"]
    gencode += ["-gencode", f"arch=compute_{archs[-1]},code=compute_{archs[-1]}"]
    abi = int(torch._C._GLIBCXX_USE_CXX11_ABI)
    flags = [
        "-O3", "-std=c++17", "--use_fast_math", "-lineinfo",
        "-shared", "-Xcompiler", "-fPIC",
        f"-D_GLIBCXX_USE_CXX11_ABI={abi}",
        "-DTORCH_API_INCLUDE_EXTENSION_H",
        *gencode,
    ]
    if bf16:
        flags.append("-DSTRITON_BF16")
    flags += [f"-I{path}" for path in include_paths(device_type="cuda")]
    flags += [f"-L{path}" for path in library_paths(device_type="cuda")]
    flags += ["-lc10", "-lc10_cuda", "-ltorch", "-ltorch_cpu", "-ltorch_cuda"]
    lib_dir = library_paths(device_type="cuda")[0]
    flags += [f"-Xlinker", f"-rpath={lib_dir}"]
    return flags


def library_path(bf16: bool = False) -> Path:
    flags = _flags(bf16)
    digest = hashlib.sha256(
        SOURCE.read_bytes() + "\0".join(flags).encode()
        + torch.__version__.encode()
    ).hexdigest()[:16]
    suffix = "_bf16" if bf16 else ""
    return BUILD / f"sidon_triton_dac{suffix}_{digest}.so"


def build(verbose: bool = True, bf16: bool = False) -> Path:
    """bf16=True builds the -DSTRITON_BF16 twin (ops sidon_triton_bf16::*)."""
    target = library_path(bf16)
    if target.exists():
        return target
    BUILD.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(f".{os.getpid()}.tmp.so")
    command = [_nvcc(), str(SOURCE), "-o", str(tmp), *_flags(bf16)]
    if verbose:
        print("[cuda_ext] building", target.name, flush=True)
    subprocess.run(command, check=True)
    tmp.replace(target)  # atomic: concurrent jobs never load half a file
    return target


def load(bf16: bool = False) -> None:
    if bf16 in _loaded:
        return
    torch.ops.load_library(str(build(bf16=bf16)))
    _loaded.add(bf16)


def ops(dtype: torch.dtype = torch.float16):
    """torch.ops namespace of the kernels for `dtype` (fp16 / bf16)."""
    bf16 = dtype == torch.bfloat16
    load(bf16)
    return torch.ops.sidon_triton_bf16 if bf16 else torch.ops.sidon_triton
