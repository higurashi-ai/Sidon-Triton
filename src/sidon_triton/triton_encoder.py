"""w2v-BERT encoder (8 layers, relative_key) on Triton kernels + cuBLAS.

Token-major ``[rows, C]`` activations (rows = batch * frames). GEMM operands
fp16 with fp32 accumulation; the residual stream, LayerNorms, softmax and
the depthwise conv run in fp32. The kernels fuse the memory-bound glue
between GEMMs (ported from FastDialogueSidon's Triton backend):

``res_norm``        x += y (+ bias); optional LayerNorm of x in place;
                    LayerNorm of x -> fp16 GEMM operand.
``glu``             a * sigmoid(b) of the conformer pointwise-conv1 output.
``causal_dw_norm``  conformer causal depthwise conv (k=31) + LayerNorm + SiLU.
``flexrel_attention`` flash attention with the relative_key bias generated
                    in-kernel (Q.R^T per row tile, clamp gather); keys past
                    each item's valid length are masked (bucket padding).

Residual GEMMs accumulate straight into the fp32 stream (cuBLAS beta=1);
their biases ride along as a running vector added in the next norm. The
feature projection runs in fp32 (fp16 there is off by up to 14 dB on short
clips: silent frames give LayerNorm variances ~0).

Padding: attention is the only op mixing frames other than the causal
depthwise conv, which only looks back, so with keys masked the valid frames
of a bucket-padded run equal an exact-length run.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

F16 = torch.float16
LOG2E = tl.constexpr(1.4426950408889634)
LEFT, RIGHT = 64, 8
NUM_POS = LEFT + RIGHT + 1  # 73
NUM_POS_PAD = 128


def _pow2(n: int) -> int:
    return 1 << (int(n) - 1).bit_length()


# ---- residual + LayerNorm ---------------------------------------------------


@triton.jit
def _layernorm(x, w_ptr, b_ptr, c, cmask, eps, C: tl.constexpr):
    """Row LayerNorm of fp32 x [R, CP] (columns >= C are zero) -> fp32."""
    mean = tl.sum(x, axis=1) / C
    d = tl.where(cmask[None, :], x - mean[:, None], 0.0)
    var = tl.sum(d * d, axis=1) / C
    y = d * tl.math.rsqrt(var + eps)[:, None]
    w = tl.load(w_ptr + c, mask=cmask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + c, mask=cmask, other=0.0).to(tl.float32)
    return y * w[None, :] + b[None, :]


@triton.jit
def _res_norm_kernel(
    x_ptr, y_ptr, out_x_ptr, xn_ptr, bias_ptr, n1_w, n1_b, n2_w, n2_b,
    R, eps1, eps2,
    C: tl.constexpr, CP: tl.constexpr, BR: tl.constexpr,
    HAS_Y: tl.constexpr, NORM1: tl.constexpr, WRITE_X: tl.constexpr,
    NORM2: tl.constexpr, HAS_BIAS: tl.constexpr,
):
    r = tl.program_id(0).to(tl.int64) * BR + tl.arange(0, BR)
    c = tl.arange(0, CP)
    cmask = c < C
    m = (r < R)[:, None] & cmask[None, :]
    offs = r[:, None] * C + c[None, :]
    x = tl.load(x_ptr + offs, mask=m, other=0.0).to(tl.float32)
    if HAS_BIAS:
        x += tl.load(bias_ptr + c, mask=cmask, other=0.0)[None, :]
    if HAS_Y:
        x += tl.load(y_ptr + offs, mask=m, other=0.0).to(tl.float32)
    if NORM1:
        x = _layernorm(x, n1_w, n1_b, c, cmask, eps1, C)
    if WRITE_X:
        tl.store(out_x_ptr + offs, x, mask=m)
    if NORM2:
        xn = _layernorm(x, n2_w, n2_b, c, cmask, eps2, C)
        tl.store(xn_ptr + offs, xn.to(xn_ptr.dtype.element_ty), mask=m)


def res_norm(x, y=None, norm1=None, write_x=True, norm2=None, bias=None,
             xn_dtype=F16):
    """x [R, C] fp32 (in place): x += bias + y; x = LN1(x); returns LN2(x) in
    ``xn_dtype``. norm = (weight, bias, eps)."""
    R, C = x.shape
    cp = _pow2(C)
    br = max(1, 4096 // cp)
    xn = (torch.empty(R, C, device=x.device, dtype=xn_dtype)
          if norm2 is not None else None)
    n1 = norm1 or (x, x, 0.0)
    n2 = norm2 or (x, x, 0.0)
    _res_norm_kernel[(triton.cdiv(R, br),)](
        x, y if y is not None else x, x, xn if xn is not None else x,
        bias if bias is not None else x, n1[0], n1[1], n2[0], n2[1],
        R, float(n1[2]), float(n2[2]),
        C=C, CP=cp, BR=br, HAS_Y=y is not None, NORM1=norm1 is not None,
        WRITE_X=write_x, NORM2=norm2 is not None, HAS_BIAS=bias is not None,
        num_warps=4 if cp <= 512 else 8,
    )
    return xn


# ---- conformer conv module --------------------------------------------------


@triton.jit
def _glu_kernel(h_ptr, g_ptr, N, C: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    m = i < N
    row = i // C
    col = i % C
    a = tl.load(h_ptr + row * (2 * C) + col, mask=m, other=0.0).to(tl.float32)
    b = tl.load(h_ptr + row * (2 * C) + C + col, mask=m,
                other=0.0).to(tl.float32)
    tl.store(g_ptr + i, (a * tl.sigmoid(b)).to(g_ptr.dtype.element_ty),
             mask=m)


def glu(h: torch.Tensor) -> torch.Tensor:
    R, C2 = h.shape
    C = C2 // 2
    g = torch.empty(R, C, device=h.device, dtype=h.dtype)
    _glu_kernel[(triton.cdiv(R * C, 2048),)](h, g, R * C, C=C, BLOCK=2048,
                                              num_warps=4)
    return g


@triton.jit
def _causal_dw_norm_kernel(
    g_ptr, dwt_ptr, ln_w, ln_b, out_ptr, T, eps,
    C: tl.constexpr, K: tl.constexpr, BT: tl.constexpr,
):
    """out[t] = silu(LN(sum_k w[k] g[t + k - (K-1)])) per item (zero left
    pad)."""
    b = tl.program_id(1).to(tl.int64)
    t = tl.program_id(0) * BT + tl.arange(0, BT)
    c = tl.arange(0, C)
    base = g_ptr + b * T * C
    acc = tl.zeros([BT, C], tl.float32)
    for k in tl.static_range(K):
        src = t + (k - (K - 1))
        valid = (src >= 0) & (src < T)
        row = tl.load(base + src[:, None] * C + c[None, :],
                      mask=valid[:, None], other=0.0)
        acc += row.to(tl.float32) * tl.load(dwt_ptr + k * C + c)[None, :]
    y = _layernorm(acc, ln_w, ln_b, c, c < C, eps, C)
    y = y * tl.sigmoid(y)
    tl.store(out_ptr + b * T * C + t[:, None] * C + c[None, :],
             y.to(out_ptr.dtype.element_ty), mask=(t < T)[:, None])


_SMS: dict = {}


def causal_dw_norm(g, batch, dwt, ln_w, ln_b, eps):
    """g [B*T, C] fp16, dwt [K, C] fp32 -> silu(LN(causal dwconv)). Fixed
    BT=8 / 4 warps (BT >= 16 spills); shorter BT when the grid would not
    cover 2x the SMs."""
    R, C = g.shape
    T = R // batch
    out = torch.empty_like(g)
    key = g.device.index or 0
    if key not in _SMS:
        _SMS[key] = torch.cuda.get_device_properties(key).multi_processor_count
    bt = 2
    for cand in (8, 4):
        if triton.cdiv(T, cand) * batch >= 2 * _SMS[key]:
            bt = cand
            break
    _causal_dw_norm_kernel[(triton.cdiv(T, bt), batch)](
        g, dwt, ln_w, ln_b, out, T, eps, C=C, K=dwt.shape[0], BT=bt,
        num_warps=4,
    )
    return out


# ---- FlexRel attention (relative_key bias in-kernel, key-length mask) -------


@triton.jit
def _softmax_update(s, m_i, l_i, acc, v_tile):
    """s is in log2 units."""
    m_new = tl.maximum(m_i, tl.max(s, 1))
    alpha = tl.math.exp2(m_i - m_new)
    p = tl.math.exp2(s - m_new[:, None])
    l_i = l_i * alpha + tl.sum(p, 1)
    acc = acc * alpha[:, None] + tl.dot(p.to(v_tile.dtype), v_tile)
    return m_new, l_i, acc


def _attn_configs():
    return [
        triton.Config({"BLOCK_M": bm, "BLOCK_N": bn}, num_warps=w,
                      num_stages=s)
        for bm, bn, w, s in (
            (64, 64, 4, 2), (64, 64, 4, 3), (64, 64, 4, 4),
            (128, 64, 4, 3), (128, 64, 8, 3), (128, 64, 8, 4),
            (64, 128, 4, 3), (128, 128, 8, 3), (128, 32, 4, 3),
        )
    ]


@triton.jit
def _flexrel_body(
    Q, K, V, R, O, LENS,
    stride_b, stride_m, stride_ob, stride_om,
    H, L, qk_scale,
    HEAD_DIM: tl.constexpr, NPOS: tl.constexpr, NPOS_PAD: tl.constexpr,
    CLAMP_LEFT: tl.constexpr, CLAMP_RIGHT: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    """scores = (Q.K^T + Q.R[clamp(j-i)]^T) * scale over keys j < LENS[b];
    qk_scale = scale * log2(e). Q/K/V rows [b, t, h*D + d], row stride
    ``stride_m`` (column slices of one fused qkv GEMM output)."""
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b = (pid_bh // H).to(tl.int64)
    h = pid_bh % H
    LK = tl.load(LENS + b)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    m_valid = offs_m < L
    head = b * stride_b + h * HEAD_DIM

    q = tl.load(Q + head + offs_m[:, None] * stride_m + offs_d[None, :],
                mask=m_valid[:, None], other=0.0)
    offs_p = tl.arange(0, NPOS_PAD)
    r = tl.load(R + offs_p[:, None] * HEAD_DIM + offs_d[None, :],
                mask=offs_p[:, None] < NPOS, other=0.0)
    qr = tl.dot(q, tl.trans(r)) * qk_scale
    qr_left = tl.sum(tl.where(offs_p[None, :] == 0, qr, 0.0), 1)
    qr_right = tl.sum(tl.where(offs_p[None, :] == NPOS - 1, qr, 0.0), 1)

    m_i = tl.full([BLOCK_M], float("-inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], tl.float32)

    k_base = K + head
    v_base = V + head
    m_lo = pid_m * BLOCK_M
    m_hi = m_lo + BLOCK_M - 1
    # keys: [0, n_left) pure-left clamp | [n_left, band_end) band (gather)
    # [n_right, n_full) pure-right clamp | [n_tail, LK) right tail (masked).
    # Rows past LK (padding queries) may start their band early: the band
    # is exact for any distance, so only speed differs there.
    n_full = (LK // BLOCK_N) * BLOCK_N
    n_left = tl.minimum(
        tl.maximum((m_lo - CLAMP_LEFT + 1) // BLOCK_N, 0) * BLOCK_N, n_full)
    n_right = ((m_hi + CLAMP_RIGHT + BLOCK_N - 1) // BLOCK_N) * BLOCK_N
    band_end = tl.minimum(n_right, LK)
    n_tail = tl.maximum(n_right, n_full)

    # clamp-saturated regions add a per-row constant (qr_left / qr_right):
    # run them bias-free and shift the running max afterwards (exact)
    for n0 in tl.range(0, n_left, BLOCK_N):
        offs_n = n0 + tl.arange(0, BLOCK_N)
        k_tile = tl.load(k_base + offs_n[:, None] * stride_m
                         + offs_d[None, :])
        s = tl.dot(q, tl.trans(k_tile)) * qk_scale
        v_tile = tl.load(v_base + offs_n[:, None] * stride_m
                         + offs_d[None, :])
        m_i, l_i, acc = _softmax_update(s, m_i, l_i, acc, v_tile)
    m_i += qr_left

    for n0 in tl.range(n_left, band_end, BLOCK_N):
        offs_n = n0 + tl.arange(0, BLOCK_N)
        n_valid = offs_n < LK
        k_tile = tl.load(k_base + offs_n[:, None] * stride_m
                         + offs_d[None, :], mask=n_valid[:, None], other=0.0)
        rel = tl.minimum(
            tl.maximum(offs_n[None, :] - offs_m[:, None], -CLAMP_LEFT),
            CLAMP_RIGHT) + CLAMP_LEFT
        s = tl.dot(q, tl.trans(k_tile)) * qk_scale + tl.gather(qr, rel,
                                                               axis=1)
        s = tl.where(n_valid[None, :], s, float("-inf"))
        v_tile = tl.load(v_base + offs_n[:, None] * stride_m
                         + offs_d[None, :], mask=n_valid[:, None], other=0.0)
        m_i, l_i, acc = _softmax_update(s, m_i, l_i, acc, v_tile)

    m_i -= qr_right
    for n0 in tl.range(n_right, n_full, BLOCK_N):
        offs_n = n0 + tl.arange(0, BLOCK_N)
        k_tile = tl.load(k_base + offs_n[:, None] * stride_m
                         + offs_d[None, :])
        s = tl.dot(q, tl.trans(k_tile)) * qk_scale
        v_tile = tl.load(v_base + offs_n[:, None] * stride_m
                         + offs_d[None, :])
        m_i, l_i, acc = _softmax_update(s, m_i, l_i, acc, v_tile)

    for n0 in tl.range(n_tail, LK, BLOCK_N):
        offs_n = n0 + tl.arange(0, BLOCK_N)
        n_valid = offs_n < LK
        k_tile = tl.load(k_base + offs_n[:, None] * stride_m
                         + offs_d[None, :], mask=n_valid[:, None], other=0.0)
        s = tl.dot(q, tl.trans(k_tile)) * qk_scale
        s = tl.where(n_valid[None, :], s, float("-inf"))
        v_tile = tl.load(v_base + offs_n[:, None] * stride_m
                         + offs_d[None, :], mask=n_valid[:, None], other=0.0)
        m_i, l_i, acc = _softmax_update(s, m_i, l_i, acc, v_tile)

    acc = acc / l_i[:, None]
    tl.store(O + b * stride_ob + h * HEAD_DIM + offs_m[:, None] * stride_om
             + offs_d[None, :], acc.to(O.dtype.element_ty),
             mask=m_valid[:, None])


@triton.autotune(configs=_attn_configs(), key=["H", "L_BUCKET"])
@triton.jit
def _flexrel_tuned(Q, K, V, R, O, LENS, stride_b, stride_m, stride_ob,
                   stride_om, H, L, L_BUCKET, qk_scale,
                   HEAD_DIM: tl.constexpr, NPOS: tl.constexpr,
                   NPOS_PAD: tl.constexpr, CLAMP_LEFT: tl.constexpr,
                   CLAMP_RIGHT: tl.constexpr, BLOCK_M: tl.constexpr,
                   BLOCK_N: tl.constexpr):
    _flexrel_body(Q, K, V, R, O, LENS, stride_b, stride_m, stride_ob,
                  stride_om, H, L, qk_scale, HEAD_DIM, NPOS, NPOS_PAD,
                  CLAMP_LEFT, CLAMP_RIGHT, BLOCK_M, BLOCK_N)


@triton.jit
def _flexrel_fixed(Q, K, V, R, O, LENS, stride_b, stride_m, stride_ob,
                   stride_om, H, L, qk_scale,
                   HEAD_DIM: tl.constexpr, NPOS: tl.constexpr,
                   NPOS_PAD: tl.constexpr, CLAMP_LEFT: tl.constexpr,
                   CLAMP_RIGHT: tl.constexpr, BLOCK_M: tl.constexpr,
                   BLOCK_N: tl.constexpr):
    _flexrel_body(Q, K, V, R, O, LENS, stride_b, stride_m, stride_ob,
                  stride_om, H, L, qk_scale, HEAD_DIM, NPOS, NPOS_PAD,
                  CLAMP_LEFT, CLAMP_RIGHT, BLOCK_M, BLOCK_N)


# (BLOCK_M, BLOCK_N, warps, stages): the H100 autotuning winner at every
# length (2.3-96 s, stages 2-4 within noise), fixed so no tuning runs during
# CUDA-graph capture; None = autotune (exploration)
ATTN_CONFIG: tuple | None = (64, 64, 4, 3)


def flexrel_attention(qkv, batch, heads, rel, lens):
    """qkv [B*L, 3*H*D] fp16 (q | k | v column blocks), rel [73, D] fp16,
    lens int32 [B] valid keys -> context [B*L, H*D] fp16."""
    rows, width = qkv.shape
    hidden = width // 3
    dim = hidden // heads
    length = rows // batch
    assert dim == 64 and rel.shape == (NUM_POS, dim)
    out = torch.empty(rows, hidden, device=qkv.device, dtype=qkv.dtype)
    q, k, v = qkv[:, :hidden], qkv[:, hidden:2 * hidden], qkv[:, 2 * hidden:]
    args = (q, k, v, rel, out, lens, length * width, width, length * hidden,
            hidden, heads, length)
    meta = dict(HEAD_DIM=dim, NPOS=NUM_POS, NPOS_PAD=NUM_POS_PAD,
                CLAMP_LEFT=LEFT, CLAMP_RIGHT=RIGHT)
    scale = LOG2E.value / math.sqrt(dim)
    if ATTN_CONFIG is None:
        grid = lambda m: (triton.cdiv(length, m["BLOCK_M"]),  # noqa: E731
                          batch * heads)
        _flexrel_tuned[grid](*args, _pow2(length), scale, **meta)
    else:
        bm, bn, warps, stages = ATTN_CONFIG
        _flexrel_fixed[(triton.cdiv(length, bm), batch * heads)](
            *args, scale, **meta, BLOCK_M=bm, BLOCK_N=bn, num_warps=warps,
            num_stages=stages,
        )
    return out


# ---- model ------------------------------------------------------------------


def _f32(t: torch.Tensor) -> torch.Tensor:
    return t.detach().float().contiguous()


def _h(t: torch.Tensor) -> torch.Tensor:
    return t.detach().to(F16).contiguous()


def _ln(module) -> tuple:
    return (_f32(module.weight), _f32(module.bias), float(module.eps))


ENCODER_FILE = "encoder_triton.pt"


def pack_encoder(encoder: torch.nn.Module) -> dict:
    """SidonEncoder (HF Wav2Vec2BertModel, relative_key) -> the re-laid-out
    weights TritonEncoder runs on (on the encoder's device)."""
    ssl = encoder.model
    cfg = ssl.config
    assert cfg.position_embeddings_type == "relative_key"
    assert int(cfg.left_max_position_embeddings) == LEFT
    assert int(cfg.right_max_position_embeddings) == RIGHT
    fp = ssl.feature_projection
    packed = {
        "heads": int(cfg.num_attention_heads),
        "fp_ln": _ln(fp.layer_norm),
        "fp_w": _f32(fp.projection.weight),
        "fp_b": _f32(fp.projection.bias),
        "layers": [],
    }
    with torch.no_grad():
        for layer in ssl.encoder.layers:
            att, conv = layer.self_attn, layer.conv_module
            p = {
                "ffn1_ln": _ln(layer.ffn1_layer_norm),
                "ffn1_w1": _h(layer.ffn1.intermediate_dense.weight),
                "ffn1_b1": _h(layer.ffn1.intermediate_dense.bias),
                "ffn1_w2": _h(layer.ffn1.output_dense.weight * 0.5),
                "attn_ln": _ln(layer.self_attn_layer_norm),
                "qkv_w": _h(torch.cat([att.linear_q.weight,
                                       att.linear_k.weight,
                                       att.linear_v.weight])),
                "qkv_b": _h(torch.cat([att.linear_q.bias, att.linear_k.bias,
                                       att.linear_v.bias])),
                "rel": _h(att.distance_embedding.weight),
                "out_w": _h(att.linear_out.weight),
                "conv_ln": _ln(conv.layer_norm),
                "pw1_w": _h(conv.pointwise_conv1.weight[:, :, 0]),
                "dwt": _f32(conv.depthwise_conv.weight[:, 0, :].t()),
                "dw_ln": _ln(conv.depthwise_layer_norm),
                "pw2_w": _h(conv.pointwise_conv2.weight[:, :, 0]),
                "ffn2_ln": _ln(layer.ffn2_layer_norm),
                "ffn2_w1": _h(layer.ffn2.intermediate_dense.weight),
                "ffn2_b1": _h(layer.ffn2.intermediate_dense.bias),
                "ffn2_w2": _h(layer.ffn2.output_dense.weight * 0.5),
                "final_ln": _ln(layer.final_layer_norm),
            }
            assert conv.pointwise_conv1.bias is None
            assert conv.depthwise_conv.bias is None
            assert conv.pointwise_conv2.bias is None
            # biases skipped by the accumulating residual GEMMs, as running
            # sums (fp64) added inside the next norm
            b1 = layer.ffn1.output_dense.bias.double() * 0.5
            b2 = b1 + att.linear_out.bias.double()
            b4 = b2 + layer.ffn2.output_dense.bias.double() * 0.5
            p["cum_bias"] = (_f32(b1), _f32(b2), _f32(b4))
            packed["layers"].append(p)
    return packed


def _to(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, (tuple, list)):
        return type(value)(_to(v, device) for v in value)
    if isinstance(value, dict):
        return {k: _to(v, device) for k, v in value.items()}
    return value


class TritonEncoder:
    """fp32 features [B, L, 160] (+ valid frames [B]) -> hidden [B, L, 1024]
    fp32, on the packed weights of :func:`pack_encoder`."""

    def __init__(self, packed: dict, device="cuda") -> None:
        packed = _to(packed, torch.device(device))
        self.heads = packed["heads"]
        self.fp_ln = packed["fp_ln"]
        self.fp_w, self.fp_b = packed["fp_w"], packed["fp_b"]
        self.layers = packed["layers"]

    @torch.no_grad()
    def __call__(self, features: torch.Tensor,
                 frames: torch.Tensor | None = None,
                 out: torch.Tensor | None = None) -> torch.Tensor:
        """out: optional fp32 [B, L, 1024] buffer the residual stream (and
        so the result) lives in."""
        batch, length, _ = features.shape
        rows = batch * length
        if frames is None:
            lens = torch.full((batch,), length, dtype=torch.int32,
                              device=features.device)
        else:
            lens = frames.to(torch.int32)
        feats = features.reshape(rows, -1).float().contiguous()
        fn = res_norm(feats, norm2=self.fp_ln, write_x=False,
                      xn_dtype=torch.float32)
        x = torch.empty(rows, self.fp_w.shape[0], device=feats.device) \
            if out is None else out.view(rows, -1)
        torch.addmm(self.fp_b, fn, self.fp_w.t(), out=x)  # fp32 stream
        layers = self.layers
        xn = res_norm(x, norm2=layers[0]["ffn1_ln"], write_x=False)

        def res(a, w, bias, **kw):
            torch.addmm(x, a, w.t(), out_dtype=torch.float32, out=x)
            return res_norm(x, bias=bias, **kw)

        for i, p in enumerate(layers):
            b1, b2, b4 = p["cum_bias"]
            last = i + 1 == len(layers)
            h = F.linear(xn, p["ffn1_w1"], p["ffn1_b1"])
            F.silu(h, inplace=True)
            xn = res(h, p["ffn1_w2"], b1, norm2=p["attn_ln"], write_x=False)
            qkv = F.linear(xn, p["qkv_w"], p["qkv_b"])
            ctx = flexrel_attention(qkv, batch, self.heads, p["rel"], lens)
            xn = res(ctx, p["out_w"], b2, norm2=p["conv_ln"], write_x=False)
            g = glu(F.linear(xn, p["pw1_w"]))
            s = causal_dw_norm(g, batch, p["dwt"], *p["dw_ln"])
            xn = res(s, p["pw2_w"], b2, norm2=p["ffn2_ln"], write_x=False)
            h = F.linear(xn, p["ffn2_w1"], p["ffn2_b1"])
            F.silu(h, inplace=True)
            xn = res(h, p["ffn2_w2"], b4, norm1=p["final_ln"],
                     norm2=None if last else layers[i + 1]["ffn1_ln"])
        return x.view(batch, length, -1)


def save_encoder(packed: dict, max_batch: int, max_seconds: float,
                 path) -> None:
    torch.save({"packed": _to(packed, torch.device("cpu")),
                "max_batch": max_batch, "max_seconds": max_seconds},
               str(path))


def load_encoder(path, device: str = "cuda"):
    """-> (TritonEncoder, max_batch, max_seconds)."""
    saved = torch.load(str(path), map_location="cpu", weights_only=False)
    return (TritonEncoder(saved["packed"], device), saved["max_batch"],
            saved["max_seconds"])
