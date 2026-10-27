"""Fused Snake activation as a TensorRT AOT plugin (``sidon::Snake``).

``y = x + inv_alpha[c] * sin(alpha[c] * x) ** 2`` over x [B, C, T].

TensorRT 10.16 does not fuse the exported Snake chain (Mul, Sin, Pow,
Mul, Add with per-channel constants) — each op is its own full pass over
the activation, which at the 24/48 kHz DAC stages (~276 M elements per
120 s window) costs ~0.75 ms per op. This plugin does one read and one
write per element, with fp32 math.

No ``from __future__ import annotations`` here: tensorrt.plugin inspects
the real TensorDesc annotation classes at registration time.
"""

from typing import Tuple, Union

import torch
import triton
import triton.language as tl

from .flexrel_triton import _aot_blob

_PLUGIN_OP = "sidon::Snake"
_registered = False
BLOCK = 2048
WARPS = 8


@triton.jit
def _snake_kernel(X, A, INV, T, C, Y, BLOCK: tl.constexpr):
    # TRT AOT order: inputs, extra scalar args, outputs
    pid_t = tl.program_id(0)
    row = tl.program_id(1)  # b * C + c
    channel = row % C
    alpha = tl.load(A + channel)
    inv = tl.load(INV + channel)
    offsets = pid_t * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < T
    base = row.to(tl.int64) * T
    x = tl.load(X + base + offsets, mask=mask).to(tl.float32)
    s = tl.sin(alpha * x)
    y = x + inv * s * s
    tl.store(Y + base + offsets, y.to(Y.dtype.element_ty), mask=mask)


def snake(x: torch.Tensor, alpha: torch.Tensor, inv: torch.Tensor):
    """Eager launch (tests / torch path)."""
    x = x.contiguous()
    out = torch.empty_like(x)
    batch, channels, length = x.shape
    grid = (triton.cdiv(length, BLOCK), batch * channels)
    _snake_kernel[grid](
        x, alpha.float().contiguous(), inv.float().contiguous(),
        length, channels, out, BLOCK=BLOCK, num_warps=WARPS,
    )
    return out


def register_trt_plugin() -> None:
    global _registered
    if _registered:
        return
    import tensorrt.plugin as trtp

    @trtp.register(_PLUGIN_OP)
    def _snake_desc(
        x: trtp.TensorDesc, alpha: trtp.TensorDesc, inv: trtp.TensorDesc
    ) -> trtp.TensorDesc:
        return x.like()

    @trtp.autotune(_PLUGIN_OP)
    def _snake_autotune(x, alpha, inv, outputs):
        return [
            trtp.AutoTuneCombination("FP16, FP32, FP32, FP16", "LINEAR"),
            trtp.AutoTuneCombination("BF16, FP32, FP32, BF16", "LINEAR"),
            trtp.AutoTuneCombination("FP32, FP32, FP32, FP32", "LINEAR"),
        ]

    @trtp.aot_impl(_PLUGIN_OP)
    def _snake_aot(
        x, alpha, inv, outputs, tactic
    ) -> Tuple[
        Union[str, bytes],
        Union[str, bytes],
        trtp.KernelLaunchParams,
        trtp.SymExprs,
    ]:
        import tensorrt as trt

        element = {
            trt.float16: "*fp16", trt.bfloat16: "*bf16", trt.float32: "*fp32"
        }[x.dtype]
        src = triton.compiler.ASTSource(
            fn=_snake_kernel,
            signature={
                "X": element, "A": "*fp32", "INV": "*fp32",
                "T": "i32", "C": "i32", "Y": element, "BLOCK": "constexpr",
            },
            constexprs={"BLOCK": BLOCK},
            attrs={(index,): [["tt.divisibility", 16]] for index in (0, 5)},
        )
        compiled = triton.compile(src, options={"num_warps": WARPS})
        length = x.shape_expr[2]
        channels = x.shape_expr[1]
        launch = trtp.KernelLaunchParams()
        launch.grid_x = trtp.cdiv(length, BLOCK)
        launch.grid_y = x.shape_expr[0] * channels
        launch.block_x = WARPS * 32
        launch.shared_mem = compiled.metadata.shared
        extra_args = trtp.SymIntExprs(2)
        extra_args[0] = trtp.SymInt32(length)
        extra_args[1] = trtp.SymInt32(channels)
        return (
            compiled.metadata.name, _aot_blob(compiled), launch, extra_args
        )

    _registered = True


class SnakeOp(torch.autograd.Function):
    """Traces as the real math; exports as the sidon::Snake plugin node."""

    @staticmethod
    def forward(ctx, x, alpha, inv):  # noqa: ARG004
        a = alpha.view(1, -1, 1)
        i = inv.view(1, -1, 1)
        return (x + i * torch.sin(a * x).pow(2)).to(x.dtype)

    @staticmethod
    def symbolic(g, x, alpha, inv):
        out = g.op(
            "sidon::Snake",
            x, alpha, inv,
            plugin_namespace_s="sidon",
            plugin_version_s="1",
        )
        out.setType(x.type())
        return out
