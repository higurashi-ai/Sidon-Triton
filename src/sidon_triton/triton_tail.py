"""DAC decoder tail (blocks 2-4, 8 -> 48 kHz) on Triton kernels: the
no-CUDA-toolkit twin of the fused CUDA tail (csrc/dac_kernels.cu).

Activations are time-major [B, T, C] fp16 with C = 192 / 96 / 48; every
GEMM accumulates in fp32. Each kernel's epilogue writes exactly what its
consumers need, so each Snake is computed once:

- residual unit: y = x + conv1x1(snake2(conv7_d(s) + b1)) + b2 where
  s = snake1(x) was written by the producer of x; writes y (the next
  residual) and/or s' = (next Snake)(y). The dilated conv7 is ONE GEMM
  over a flattened (tap, channel) K axis (K = 7 C, padded to the block),
  and N is padded to a power of two in registers only (C = 3 * 2^k does
  not fit Triton's power-of-two tiles; splitting C in three gave dots too
  small to be efficient). The last unit of block 4 instead reduces
  final_snake(y) against the 7 final-conv taps -> p [B, T, 8] fp32.
- upsample (Snake'd input -> ConvTranspose1d(2S, S)): one output phase
  per program, all output channels in one tile; writes y and s'.
- final: audio[t] = tanh(b + sum_k p[t + k - 3, k]).

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
def _snake(v, a_ptr, i_ptr, n):
    a = tl.load(a_ptr + n)[None, :]
    s = _sin(v * a)
    return v + tl.load(i_ptr + n)[None, :] * s * s


@triton.jit
def _unit_epilogue(out, rows, n, b, L, T, C: tl.constexpr, NP: tl.constexpr,
                   y, s_out, ao, io, p, fw, fa, fi,
                   STORE_Y: tl.constexpr, STORE_S: tl.constexpr,
                   FINAL_P: tl.constexpr):
    off = b.to(tl.int64) * T * C + rows[:, None] * C + n[None, :]
    smask = (rows < T)[:, None] & (n < C)[None, :]
    if STORE_Y:
        tl.store(y + off, out.to(tl.float16), mask=smask)
    if STORE_S:
        tl.store(s_out + off, _snake(out, ao, io, n).to(tl.float16),
                 mask=smask)
    if FINAL_P:
        # final_snake in fp32 (no fp16 rounding before the last conv)
        sn = _snake(out, fa, fi, n)
        pb = p + b.to(tl.int64) * T * 8 + rows * 8
        for k in tl.static_range(7):
            pk = tl.sum(sn * tl.load(fw + k * NP + n)[None, :], axis=1)
            tl.store(pb + k, pk, mask=rows < T)


@triton.jit
def resunit_flat_kernel(
    x, s_in, w1, b1, a2, i2, w2, b2,
    y, s_out, ao, io, p, fw, fa, fi, lens, T,
    C: tl.constexpr, NP: tl.constexpr, DIL: tl.constexpr,
    STORE_Y: tl.constexpr, STORE_S: tl.constexpr, FINAL_P: tl.constexpr,
    BM: tl.constexpr, BK: tl.constexpr,
):
    """One residual unit; conv7 as a GEMM over K = (tap, channel) of the
    Snake'd input s_in (Snake on the load instead: 1.1-1.2x slower)."""
    pid_m = tl.program_id(0)
    b = tl.program_id(1)
    L = tl.load(lens + b)
    rows = (pid_m * BM + tl.arange(0, BM)).to(tl.int64)
    n = tl.arange(0, NP)
    kk = tl.arange(0, BK)
    base = b.to(tl.int64) * T * C
    acc = tl.zeros([BM, NP], dtype=tl.float32)
    for k0 in range(0, 7 * C, BK):
        f = k0 + kk
        tap = f // C
        c = f - tap * C
        r = rows[:, None] + ((tap - 3) * DIL)[None, :]
        m = (f < 7 * C)[None, :] & (r >= 0) & (r < L)
        a = tl.load(s_in + base + r * C + c[None, :], mask=m, other=0.0)
        wv = tl.load(w1 + f[:, None] * NP + n[None, :])
        acc = tl.dot(a, wv, acc)
    h = _snake(acc + tl.load(b1 + n)[None, :], a2, i2, n).to(tl.float16)
    out = tl.dot(h, tl.load(w2 + n[:, None] * NP + n[None, :]))
    out += tl.load(b2 + n)[None, :]
    valid = rows < L
    out += tl.load(x + base + rows[:, None] * C + n[None, :],
                   mask=valid[:, None] & (n < C)[None, :],
                   other=0.0).to(tl.float32)
    out = tl.where(valid[:, None], out, 0.0)
    _unit_epilogue(out, rows, n, b, L, T, C, NP, y, s_out, ao, io, p, fw,
                   fa, fi, STORE_Y, STORE_S, FINAL_P)


@triton.jit
def upsample_kernel(
    a, w, bias, y, s_out, ao, io, lens_in, lens_out, TIN, TOUT,
    CIN: tl.constexpr, C: tl.constexpr, NP: tl.constexpr, S: tl.constexpr,
    P: tl.constexpr, BM: tl.constexpr, BK: tl.constexpr,
):
    """ConvTranspose1d(K = 2S, stride S, pad P) of a Snake'd input, output
    phase r = program_id(1): out[qS + r] = a[q + (r+P)//S] @ w[(r+P)%S]
    + a[q + (r+P)//S - 1] @ w[(r+P)%S + S]; writes y and snake(y)."""
    pid_m = tl.program_id(0)
    r_ph = tl.program_id(1)
    b = tl.program_id(2)
    Lin = tl.load(lens_in + b)
    Lout = tl.load(lens_out + b)
    q = (pid_m * BM + tl.arange(0, BM)).to(tl.int64)
    n = tl.arange(0, NP)
    kk = tl.arange(0, BK)
    start = q + (r_ph + P) // S
    tap0 = (r_ph + P) % S
    ab = a + b.to(tl.int64) * TIN * CIN
    acc = tl.zeros([BM, NP], dtype=tl.float32)
    for j in tl.static_range(2):
        src = start - j
        sm = ((src >= 0) & (src < Lin))[:, None]
        tap = tap0 + j * S
        for k0 in range(0, CIN, BK):
            kc = k0 + kk
            av = tl.load(ab + src[:, None] * CIN + kc[None, :], mask=sm,
                         other=0.0)
            wv = tl.load(w + (tap * CIN + kc)[:, None] * NP + n[None, :])
            acc = tl.dot(av, wv, acc)
    acc += tl.load(bias + n)[None, :]
    o = q * S + r_ph
    acc = tl.where((o < Lout)[:, None], acc, 0.0)
    off = b.to(tl.int64) * TOUT * C + o[:, None] * C + n[None, :]
    smask = (o < TOUT)[:, None] & (n < C)[None, :]
    tl.store(y + off, acc.to(tl.float16), mask=smask)
    tl.store(s_out + off, _snake(acc, ao, io, n).to(tl.float16), mask=smask)


@triton.jit
def final_kernel(p, out, bias, lens, T, BT: tl.constexpr):
    """audio[t] = tanh(bias + sum_k p[t + k - 3, k]); 0 past lens[b]."""
    b = tl.program_id(1)
    L = tl.load(lens + b)
    t = (tl.program_id(0) * BT + tl.arange(0, BT)).to(tl.int64)
    pb = p + b.to(tl.int64) * T * 8
    acc = tl.zeros([BT], dtype=tl.float32) + bias
    for k in tl.static_range(7):
        src = t + (k - 3)
        acc += tl.load(pb + src * 8 + k, mask=(src >= 0) & (src < L),
                       other=0.0)
    e = tl.exp(-2.0 * tl.abs(acc))
    th = (1.0 - e) / (1.0 + e)
    th = tl.where(acc < 0, -th, th)
    th = tl.where(t < L, th, 0.0)
    tl.store(out + b.to(tl.int64) * T + t, th, mask=t < T)


# ---- modules ----------------------------------------------------------------


def _np2(c: int) -> int:
    return 1 << (int(c) - 1).bit_length()


def _padvec(t: torch.Tensor, size: int) -> torch.Tensor:
    v = t.detach().float().reshape(-1)
    out = torch.zeros(size, device=v.device)
    out[: v.numel()] = v
    return out


# (BM, BK, warps, stages) per output channels: the best of a sweep over
# BM 64/128, BK 32/64, 4/8 warps, 1-3 stages on H100 at 96 s
UNIT_CONFIG = {
    192: (128, 64, 8, 3), 96: (128, 64, 4, 3), 48: (128, 32, 4, 3),
}
UP_CONFIG = {96: (128, 32, 4, 3), 48: (128, 32, 4, 3)}
FINAL_BT = 1024


class _Unit(nn.Module):
    def __init__(self, unit: nn.Module, next_snake=None,
                 final=None) -> None:
        """next_snake: Snake applied to y for the next consumer (written as
        s'); final: decoder (last unit of block 4: write p instead)."""
        super().__init__()
        conv1, conv2 = unit.conv1, unit.conv2
        c = conv1.weight.shape[0]
        self.c, self.np = c, _np2(c)
        self.dil = int(conv1.dilation[0])
        bm, bk, _, _ = UNIT_CONFIG[c]
        kp = -(-7 * c // bk) * bk
        w1 = torch.zeros(kp, self.np, device=conv1.weight.device)
        w1[: 7 * c, :c] = conv1.weight.detach().float().permute(2, 1, 0) \
            .reshape(7 * c, c)
        w2 = torch.zeros(self.np, self.np, device=conv1.weight.device)
        w2[:c, :c] = conv2.weight.detach().float()[:, :, 0].t()
        self.register_buffer("w1", w1.to(F16))
        self.register_buffer("w2", w2.to(F16))
        self.register_buffer("b1", _padvec(conv1.bias, self.np))
        self.register_buffer("b2", _padvec(conv2.bias, self.np))
        for name, src in (("a2", unit.snake2.alpha),
                          ("i2", unit.snake2.inv_alpha)):
            self.register_buffer(name, _padvec(src, self.np))
        zeros = torch.zeros(self.np, device=conv1.weight.device)
        self.store_s = next_snake is not None
        self.register_buffer(
            "ao", _padvec(next_snake.alpha, self.np) if self.store_s
            else zeros)
        self.register_buffer(
            "io", _padvec(next_snake.inv_alpha, self.np) if self.store_s
            else zeros)
        self.final = final is not None
        if self.final:
            fw = torch.zeros(7, self.np, device=conv1.weight.device)
            fw[:, :c] = final.final_conv.weight.detach().float()[0].t()
            self.register_buffer("fw", fw.reshape(-1).contiguous())
            self.register_buffer("fa", _padvec(final.final_snake.alpha,
                                               self.np))
            self.register_buffer("fi", _padvec(final.final_snake.inv_alpha,
                                               self.np))
        else:
            self.register_buffer("fw", zeros)
            self.register_buffer("fa", zeros)
            self.register_buffer("fi", zeros)

    def forward(self, x, s, lens, store_y=True):
        """x [B, T, C] residual input, s = snake1(x) -> (y, s', p)."""
        B, T, _ = x.shape
        y = torch.empty_like(x) if store_y else None
        s_out = torch.empty_like(x) if self.store_s else None
        p = (torch.empty(B, T, 8, device=x.device) if self.final else None)
        bm, bk, warps, stages = UNIT_CONFIG[self.c]
        resunit_flat_kernel[(triton.cdiv(T, bm), B)](
            x, s, self.w1, self.b1, self.a2, self.i2, self.w2, self.b2,
            y if y is not None else x, s_out if s_out is not None else x,
            self.ao, self.io, p if p is not None else self.fw, self.fw,
            self.fa, self.fi, lens, T,
            C=self.c, NP=self.np, DIL=self.dil, STORE_Y=store_y,
            STORE_S=self.store_s, FINAL_P=self.final,
            BM=bm, BK=bk, num_warps=warps, num_stages=stages,
        )
        return y, s_out, p


class _Upsample(nn.Module):
    def __init__(self, block: nn.Module) -> None:
        super().__init__()
        up = block.upsample
        weight = up.weight.detach().float()  # [Cin, Cout, 2s]
        self.cin, self.c, taps = weight.shape
        self.np = _np2(self.c)
        self.stride = int(up.stride[0])
        self.pad = int(up.padding[0])
        w = torch.zeros(taps, self.cin, self.np, device=weight.device)
        w[:, :, : self.c] = weight.permute(2, 0, 1)
        self.register_buffer("w", w.to(F16))
        self.register_buffer("bias", _padvec(up.bias, self.np))
        snake = block.res[0].snake1
        self.register_buffer("ao", _padvec(snake.alpha, self.np))
        self.register_buffer("io", _padvec(snake.inv_alpha, self.np))

    def forward(self, s, lens):
        B, T, _ = s.shape
        tout = cuda_tail.upsampled(T, self.stride)
        lens_out = cuda_tail.upsampled(lens, self.stride).to(torch.int32)
        y = torch.empty(B, tout, self.c, device=s.device, dtype=F16)
        s_out = torch.empty_like(y)
        bm, bk, warps, stages = UP_CONFIG[self.c]
        assert self.cin % bk == 0, (self.cin, bk)
        nq = -(-tout // self.stride)
        upsample_kernel[(triton.cdiv(nq, bm), self.stride, B)](
            s, self.w, self.bias, y, s_out, self.ao, self.io, lens,
            lens_out, T, tout, CIN=self.cin, C=self.c, NP=self.np,
            S=self.stride, P=self.pad, BM=bm, BK=bk, num_warps=warps,
            num_stages=stages,
        )
        return y, s_out, lens_out


class TritonDecoderTail(nn.Module):
    """(z, s) [B, T, 192] time-major (head output and its snake1 for the
    first residual unit) -> audio [B, 1, T48] fp32."""

    def __init__(self, decoder: nn.Module,
                 from_block: int = cuda_tail.TAIL_FROM_BLOCK) -> None:
        super().__init__()
        self.from_block = from_block
        blocks = decoder.blocks
        stages = []
        for bi in range(from_block, len(blocks)):
            block = blocks[bi]
            if bi > from_block:
                stages.append(_Upsample(block))
            for ui, unit in enumerate(block.res):
                last_block = bi + 1 == len(blocks)
                if ui + 1 < len(block.res):
                    stages.append(_Unit(unit, block.res[ui + 1].snake1))
                elif not last_block:
                    stages.append(_Unit(unit, blocks[bi + 1].snake))
                else:
                    stages.append(_Unit(unit, final=decoder))
        self.stages = nn.ModuleList(stages)
        self.final_bias = float(decoder.final_conv.bias.item())

    @torch.no_grad()
    def forward(self, z: torch.Tensor, s: torch.Tensor,
                frames: torch.Tensor | None = None) -> torch.Tensor:
        B, T, _ = z.shape
        if frames is None:
            valid = torch.full((B,), T, dtype=torch.int64, device=z.device)
        else:
            valid = cuda_tail.head_samples(frames.to(torch.int64),
                                           self.from_block)
        lens = valid.to(torch.int32)
        x = z
        p = None
        for i, stage in enumerate(self.stages):
            if isinstance(stage, _Upsample):
                x, s, lens = stage(s, lens)
                continue
            last_in_block = (i + 1 == len(self.stages)
                             or isinstance(self.stages[i + 1], _Upsample))
            x, s, p = stage(x, s, lens, store_y=not last_in_block)
        T48 = p.shape[1]
        audio = torch.empty(B, 1, T48, device=z.device)
        final_kernel[(triton.cdiv(T48, FINAL_BT), B)](
            p, audio, self.final_bias, lens, T48, BT=FINAL_BT, num_warps=4,
        )
        return audio


TAIL_FILE = "decoder_tail_triton.pt"


def save_tail(tail: TritonDecoderTail, decoder_config: dict, path) -> None:
    torch.save(
        {
            "state": {k: v.cpu() for k, v in tail.state_dict().items()},
            "final_bias": tail.final_bias,
            "from_block": tail.from_block,
            "decoder_config": decoder_config,
        },
        str(path),
    )


def load_tail(path, device: str = "cuda") -> TritonDecoderTail:
    from .modeling import build_decoder

    saved = torch.load(str(path), map_location="cpu", weights_only=False)
    tail = TritonDecoderTail(
        build_decoder(saved["decoder_config"]), saved["from_block"]
    )
    tail.load_state_dict(saved["state"], strict=True)
    tail.final_bias = float(saved["final_bias"])
    return tail.to(device).eval()
