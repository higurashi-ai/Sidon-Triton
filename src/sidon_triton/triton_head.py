"""DAC decoder head on Triton GEMM kernels, time-major [B, T, C].

Replaces the TensorRT head engine (input conv, blocks < TAIL_FROM_BLOCK,
and block TAIL_FROM_BLOCK's Snake + upsample) -> z [B, T, 192] fp16, the
CUDA tail's input. Every conv is one GEMM kernel whose epilogue adds the
bias (+ residual) and writes the raw output and/or the Snake of it for
the next consumer, so each Snake is computed once and no layout change or
standalone elementwise pass remains (TensorRT spent ~6 of 11.5 ms of the
96 s head on NCHW<->NHWC reformats, Snake and pointwise kernels).

Rows past each item's valid length are read as zeros and written as zeros
(exact-length semantics, like the CUDA tail).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import triton
import triton.language as tl

from . import cuda_tail

F16 = torch.float16


@triton.jit
def _sin(x):
    return tl.inline_asm_elementwise(
        "sin.approx.f32 $0, $1;", "=r,r", [x], dtype=tl.float32,
        is_pure=True, pack=1,
    )


@triton.jit
def _epilogue(acc, orow, ns, b, L, T, COUT: tl.constexpr, bias, res, y, ys,
              alpha, inv, HAS_RES: tl.constexpr, STORE_Y: tl.constexpr,
              STORE_S: tl.constexpr):
    acc += tl.load(bias + ns)[None, :]
    off = b.to(tl.int64) * T * COUT + orow[:, None] * COUT + ns[None, :]
    valid = (orow < L)[:, None]
    if HAS_RES:
        acc += tl.load(res + off, mask=valid, other=0.0).to(tl.float32)
    acc = tl.where(valid, acc, 0.0)
    inside = (orow < T)[:, None]
    if STORE_Y:
        tl.store(y + off, acc.to(tl.float16), mask=inside)
    if STORE_S:
        a = tl.load(alpha + ns)[None, :]
        s = _sin(acc * a)
        out = acc + tl.load(inv + ns)[None, :] * s * s
        tl.store(ys + off, out.to(tl.float16), mask=inside)


# (BM, BN, BK, warps, stages) per GEMM shape: the H100 autotuning winners
# over 4 s..96 s windows, fixed so that no tuning runs inside CUDA-graph
# capture and every bucket length gets the same kernels.
_SMALL = (64, 128, 64, 4, 4)


def conv_config(rows: int, cout: int, taps: int) -> tuple:
    if taps == 1:
        return _SMALL
    if cout == 1536:  # input conv (K = 7 x 1024)
        return (128, 256, 64, 8, 3) if rows >= 1500 else _SMALL
    if cout == 768:
        return (128, 256, 64, 8, 3) if rows >= 12000 else (128, 128, 64, 8, 3)
    return (128, 128, 64, 8, 3)


def upsample_config(rows_in: int, cout: int) -> tuple:
    if cout == 768:
        return (128, 256, 64, 8, 3) if rows_in >= 1500 else _SMALL
    if cout == 384:
        return (256, 128, 64, 8, 3) if rows_in >= 6000 else _SMALL
    return (128, 64, 64, 4, 4)


@triton.jit
def conv_gemm_kernel(a, w, bias, res, y, ys, alpha, inv, lens, T,
                     CIN: tl.constexpr, COUT: tl.constexpr,
                     KT: tl.constexpr, DIL: tl.constexpr,
                     HAS_RES: tl.constexpr, STORE_Y: tl.constexpr,
                     STORE_S: tl.constexpr, BM: tl.constexpr,
                     BN: tl.constexpr, BK: tl.constexpr):
    """out[t] = sum_tap a[t + (tap - KT//2) DIL] @ w[tap] (+ epilogue)."""
    NT: tl.constexpr = COUT // BN
    pid = tl.program_id(0)
    pid_n = pid % NT
    pid_m = pid // NT
    b = tl.program_id(1)
    L = tl.load(lens + b)
    rows = (pid_m * BM + tl.arange(0, BM)).to(tl.int64)
    ns = pid_n * BN + tl.arange(0, BN)
    ks = tl.arange(0, BK)
    ab = a + b.to(tl.int64) * T * CIN
    acc = tl.zeros([BM, BN], dtype=tl.float32)
    for tap in tl.static_range(KT):
        r = rows + (tap - KT // 2) * DIL
        rm = ((r >= 0) & (r < L))[:, None]
        for k0 in range(0, CIN, BK):
            kk = k0 + ks
            av = tl.load(ab + r[:, None] * CIN + kk[None, :], mask=rm,
                         other=0.0)
            wv = tl.load(w + (tap * CIN + kk)[:, None] * COUT + ns[None, :])
            acc = tl.dot(av, wv, acc)
    _epilogue(acc, rows, ns, b, L, T, COUT, bias, res, y, ys, alpha, inv,
              HAS_RES, STORE_Y, STORE_S)


@triton.jit
def upsample_gemm_kernel(a, w, bias, y, ys, alpha, inv, lens_in, lens_out,
                         TIN, TOUT, CIN: tl.constexpr, COUT: tl.constexpr,
                         S: tl.constexpr, P: tl.constexpr,
                         STORE_Y: tl.constexpr, STORE_S: tl.constexpr,
                         BM: tl.constexpr, BN: tl.constexpr,
                         BK: tl.constexpr):
    """ConvTranspose1d(K = 2S, stride S, pad P) of a (already Snake'd), one
    output phase r per program row: out[qS + r] =
    a[q + (r+P)//S] @ w[(r+P)%S] + a[q + (r+P)//S - 1] @ w[(r+P)%S + S]."""
    NT: tl.constexpr = COUT // BN
    pid = tl.program_id(0)
    pid_n = pid % NT
    pid_m = pid // NT
    r_ph = tl.program_id(1)
    b = tl.program_id(2)
    Lin = tl.load(lens_in + b)
    Lout = tl.load(lens_out + b)
    q = (pid_m * BM + tl.arange(0, BM)).to(tl.int64)
    ns = pid_n * BN + tl.arange(0, BN)
    ks = tl.arange(0, BK)
    base = q + (r_ph + P) // S
    tap0 = (r_ph + P) % S
    ab = a + b.to(tl.int64) * TIN * CIN
    acc = tl.zeros([BM, BN], dtype=tl.float32)
    for j in tl.static_range(2):
        src = base - j
        sm = ((src >= 0) & (src < Lin))[:, None]
        tap = tap0 + j * S
        for k0 in range(0, CIN, BK):
            kk = k0 + ks
            av = tl.load(ab + src[:, None] * CIN + kk[None, :], mask=sm,
                         other=0.0)
            wv = tl.load(w + (tap * CIN + kk)[:, None] * COUT + ns[None, :])
            acc = tl.dot(av, wv, acc)
    _epilogue(acc, q * S + r_ph, ns, b, Lout, TOUT, COUT, bias, bias, y, ys,
              alpha, inv, False, STORE_Y, STORE_S)


def _vec(t: torch.Tensor) -> torch.Tensor:
    return t.detach().float().reshape(-1).contiguous()


class _Conv(nn.Module):
    """nn.Conv1d -> conv_gemm_kernel operands; `snake_out` = the Snake
    applied to its output by the next consumer (None: raw only)."""

    def __init__(self, conv: nn.Conv1d, snake_out=None) -> None:
        super().__init__()
        weight = conv.weight.detach().float()  # [Cout, Cin, K]
        self.cout, self.cin, self.kt = weight.shape
        self.dil = int(conv.dilation[0])
        assert int(conv.padding[0]) == self.kt // 2 * self.dil
        self.register_buffer("w", weight.permute(2, 1, 0).contiguous().to(F16))
        self.register_buffer("bias", _vec(conv.bias))
        _snake_buffers(self, snake_out, self.cout, conv.weight.device)

    def forward(self, a, lens, res=None, store_y=False, store_s=True):
        B, T, _ = a.shape
        y = torch.empty(B, T, self.cout, device=a.device, dtype=F16) \
            if store_y else None
        ys = torch.empty(B, T, self.cout, device=a.device, dtype=F16) \
            if store_s else None
        bm, bn, bk, warps, stages = conv_config(T, self.cout, self.kt)
        grid = (triton.cdiv(T, bm) * (self.cout // bn), B)
        conv_gemm_kernel[grid](
            a, self.w, self.bias, res if res is not None else a,
            y if y is not None else a, ys if ys is not None else a,
            self.alpha, self.inv, lens, T,
            CIN=self.cin, COUT=self.cout, KT=self.kt, DIL=self.dil,
            HAS_RES=res is not None, STORE_Y=store_y, STORE_S=store_s,
            BM=bm, BN=bn, BK=bk, num_warps=warps, num_stages=stages,
        )
        return y, ys


class _Upsample(nn.Module):
    def __init__(self, up: nn.ConvTranspose1d, snake_out=None) -> None:
        super().__init__()
        weight = up.weight.detach().float()  # [Cin, Cout, 2s]
        self.cin, self.cout, taps = weight.shape
        self.stride = int(up.stride[0])
        self.pad = int(up.padding[0])
        assert taps == 2 * self.stride
        self.register_buffer("w", weight.permute(2, 0, 1).contiguous().to(F16))
        self.register_buffer("bias", _vec(up.bias))
        _snake_buffers(self, snake_out, self.cout, up.weight.device)

    def forward(self, a, lens, store_y=True, store_s=True):
        B, T, _ = a.shape
        tout = cuda_tail.upsampled(T, self.stride)
        lens_out = cuda_tail.upsampled(lens, self.stride).to(torch.int32)
        y = torch.empty(B, tout, self.cout, device=a.device, dtype=F16) \
            if store_y else None
        ys = torch.empty(B, tout, self.cout, device=a.device, dtype=F16) \
            if store_s else None
        nq = -(-tout // self.stride)
        bm, bn, bk, warps, stages = upsample_config(T, self.cout)
        grid = (triton.cdiv(nq, bm) * (self.cout // bn), self.stride, B)
        upsample_gemm_kernel[grid](
            a, self.w, self.bias, y if y is not None else a,
            ys if ys is not None else a, self.alpha, self.inv, lens,
            lens_out, T, tout, CIN=self.cin, COUT=self.cout, S=self.stride,
            P=self.pad, STORE_Y=store_y, STORE_S=store_s,
            BM=bm, BN=bn, BK=bk, num_warps=warps, num_stages=stages,
        )
        return y, ys, lens_out


def _snake_buffers(module, snake, channels, device):
    if snake is None:
        alpha = inv = torch.zeros(channels, device=device)
    else:
        alpha, inv = _vec(snake.alpha), _vec(snake.inv_alpha)
    module.register_buffer("alpha", alpha)
    module.register_buffer("inv", inv)


class TritonDecoderHead(nn.Module):
    """hidden [B, L, 1024] (+ valid frames [B]) -> z [B, T, C] fp16
    time-major = block `from_block` upsample output (CUDA tail input)."""

    def __init__(self, decoder: nn.Module,
                 from_block: int = cuda_tail.TAIL_FROM_BLOCK) -> None:
        super().__init__()
        blocks = decoder.blocks
        self.from_block = from_block
        self.input_conv = _Conv(decoder.input_conv, blocks[0].snake)
        self.ups = nn.ModuleList()
        self.units = nn.ModuleList()  # per block: [conv1, conv2] x 3
        for bi in range(from_block):
            block = blocks[bi]
            self.ups.append(_Upsample(block.upsample, block.res[0].snake1))
            convs = nn.ModuleList()
            for ui, unit in enumerate(block.res):
                nxt = (block.res[ui + 1].snake1 if ui + 1 < len(block.res)
                       else blocks[bi + 1].snake)
                convs.append(_Conv(unit.conv1, unit.snake2))
                convs.append(_Conv(unit.conv2, nxt))
            self.units.append(convs)
        # also writes snake1 of the tail's first residual unit (Triton
        # tail input; the CUDA tail applies it itself)
        self.last_up = _Upsample(blocks[from_block].upsample,
                                 blocks[from_block].res[0].snake1)

    @torch.no_grad()
    def forward(self, hidden: torch.Tensor,
                frames: torch.Tensor | None = None,
                with_snake: bool = False):
        """-> z, or (z, snake1(z)) with `with_snake` (Triton tail)."""
        B, L, _ = hidden.shape
        a = hidden.to(F16).contiguous()
        if frames is None:
            lens = torch.full((B,), L, dtype=torch.int32, device=a.device)
        else:
            lens = frames.to(torch.int32)
        _, s = self.input_conv(a, lens)
        for up, convs in zip(self.ups, self.units):
            y, s, lens = up(s, lens)
            for ui in range(0, len(convs), 2):
                last = ui + 2 == len(convs)
                _, h = convs[ui](s, lens)
                y, s = convs[ui + 1](h, lens, res=y, store_y=not last)
        z, zs, _ = self.last_up(s, lens, store_y=True, store_s=with_snake)
        return (z, zs) if with_snake else z


HEAD_FILE = "decoder_head.pt"


def save_head(head: TritonDecoderHead, decoder_config: dict, path) -> None:
    torch.save(
        {
            "state": {k: v.cpu() for k, v in head.state_dict().items()},
            "from_block": head.from_block,
            "decoder_config": decoder_config,
        },
        str(path),
    )


def load_head(path, device: str = "cuda") -> TritonDecoderHead:
    from .modeling import build_decoder

    saved = torch.load(str(path), map_location="cpu", weights_only=False)
    head = TritonDecoderHead(
        build_decoder(saved["decoder_config"]), saved["from_block"]
    )
    head.load_state_dict(saved["state"], strict=True)
    return head.to(device).eval()
