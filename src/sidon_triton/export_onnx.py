"""ONNX export wrappers for the two TensorRT engines.

- Encoder: (fbank features [B, L, 160] fp32, lens [B]) -> hidden
  [B, L, 1024] fp32. Strongly typed: the w2v-BERT student runs in fp16
  (bf16 / fp32 selectable) with an fp32 feature-projection island, and its
  relative_key attention on the sidon::FlexRelAttnQRD plugin op (QR bias
  table from a plain MatMul node, [B, L, H, D] layout, keys past lens[b]
  masked). The fbank front-end stays in torch (fp32, see frontend.py).
- Decoder head: (hidden [B, L, 1024], four validity masks) -> z [B, T, 192]
  time-major fp16 — DAC input conv, blocks 0-1 and block 2's Snake
  (sidon::Snake plugin) + ConvTranspose (as GEMM + phase interleave, also
  required on sm_120: no BF16 ConvTranspose tactic in TRT 10.16). The rest
  of the decoder is the fused CUDA tail (cuda_tail.py).
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import flexrel_triton
from .modeling import DACDecoder, FlexRelSelfAttention, SidonEncoder


# ---- encoder ----------------------------------------------------------------


class ExportFlexRelSelfAttention(nn.Module):
    """FlexRelSelfAttention twin whose core is the TRT plugin custom op."""

    def __init__(
        self, source: FlexRelSelfAttention, fold_qkv_bias: bool
    ) -> None:
        super().__init__()
        self.num_heads = source.num_heads
        self.head_dim = source.head_dim
        self.fold_qkv_bias = fold_qkv_bias
        self.linear_out = source.linear_out
        self.distance_embedding = source.distance_embedding
        if fold_qkv_bias:
            # Blackwell Myelin can segfault at create_execution_context()
            # when a GEMM+bias epilogue feeds the AOT plugin. Fold each qkv
            # bias into its GEMM (ones column appended to the activations).
            for name, linear in (
                ("wq_aug", source.linear_q),
                ("wk_aug", source.linear_k),
                ("wv_aug", source.linear_v),
            ):
                self.register_buffer(
                    name,
                    torch.cat(
                        [linear.weight.t(), linear.bias[None, :]], dim=0
                    ).detach().clone(),
                )
        else:
            self.linear_q = source.linear_q
            self.linear_k = source.linear_k
            self.linear_v = source.linear_v

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        relative_position_embeddings: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        **_: object,
    ) -> tuple[torch.Tensor, None]:
        batch, length, hidden = hidden_states.shape
        shape = (batch, length, self.num_heads, self.head_dim)
        if self.fold_qkv_bias:
            augmented = torch.cat(
                [hidden_states, hidden_states.new_ones(batch, length, 1)],
                dim=-1,
            )
            query = torch.matmul(augmented, self.wq_aug).view(shape)
            key = torch.matmul(augmented, self.wk_aug).view(shape)
            value = torch.matmul(augmented, self.wv_aug).view(shape)
        else:
            query = self.linear_q(hidden_states).view(shape)
            key = self.linear_k(hidden_states).view(shape)
            value = self.linear_v(hidden_states).view(shape)
        query_rel = torch.matmul(query, self.distance_embedding.weight.t())
        # self._lens: valid key length per batch item (set by EncoderWrapper)
        context = flexrel_triton.FlexRelAttnQRDynOp.apply(
            query, key, value, query_rel, self._lens
        )
        return self.linear_out(context.reshape(batch, length, hidden)), None


def fold_qkv_bias_for_capability(capability: tuple[int, int]) -> bool:
    """Use the Blackwell-safe QKV graph on sm_100 and newer GPUs."""
    return capability >= (10, 0)


def swap_flex_for_export(encoder: SidonEncoder, fold_qkv_bias: bool) -> int:
    count = 0
    for layer in encoder.model.encoder.layers:
        assert isinstance(layer.self_attn, FlexRelSelfAttention)
        layer.self_attn = ExportFlexRelSelfAttention(
            layer.self_attn, fold_qkv_bias=fold_qkv_bias
        )
        count += 1
    return count


class FP32Projection(nn.Module):
    """w2v-BERT feature projection (LayerNorm(160) + Linear) kept in fp32
    on fp32 features, output cast to the encoder dtype.

    In silent frames (e.g. the demo's 1.5 s zero tail, which dominates
    short clips) the 160 normalized fbank values are nearly equal, so the
    LayerNorm divides by a tiny variance and amplifies the fp16 rounding
    of its input: 15-30 dB hidden SNR on ~1-4 s clips (up to 14 dB log-mel
    off the official output). fp32 here restores 55-60 dB."""

    def __init__(self, projection: nn.Module, dtype: torch.dtype) -> None:
        super().__init__()
        self.projection = projection.float()
        self._dtype = dtype

    def forward(self, features: torch.Tensor):
        hidden, normed = self.projection(features.float())
        return hidden.to(self._dtype), normed.to(self._dtype)


class EncoderWrapper(nn.Module):
    """fp32 IO, reduced-precision student (strongly typed).

    lens (int32 [B]): valid frames per item. Rows past it are bucket
    padding: attention ignores them as keys (the conformer's depthwise conv
    is left-padded / causal and every other op is per-frame, so valid rows
    are exactly the exact-length result)."""

    def __init__(self, student: nn.Module, dtype: torch.dtype) -> None:
        super().__init__()
        self.student = student
        self._dtype = dtype

    def forward(self, features: torch.Tensor, lens: torch.Tensor) -> torch.Tensor:
        for layer in self.student.model.encoder.layers:
            layer.self_attn._lens = lens
        # features stay fp32: the projection is an fp32 island
        return self.student(features).float()


def export_encoder(
    encoder: SidonEncoder,
    features_example: torch.Tensor,
    onnx_path: Path,
    dtype: torch.dtype = torch.bfloat16,
) -> None:
    lens_example = torch.full(
        (features_example.shape[0],), features_example.shape[1],
        dtype=torch.int32, device=features_example.device,
    )
    """encoder must be a throwaway copy with FlexRelSelfAttention layers —
    attentions are swapped in place and it is cast to `dtype`."""
    capability = torch.cuda.get_device_capability(features_example.device)
    fold_qkv_bias = fold_qkv_bias_for_capability(capability)
    print(
        f"encoder qkv export: sm_{capability[0]}{capability[1]} uses "
        f"{'bias-folded' if fold_qkv_bias else 'linear+bias'}",
        flush=True,
    )
    swap_flex_for_export(encoder, fold_qkv_bias=fold_qkv_bias)
    projection = copy.deepcopy(encoder.model.feature_projection)
    encoder = encoder.to(dtype)
    encoder.model.feature_projection = FP32Projection(projection, dtype)
    wrapper = EncoderWrapper(encoder, dtype).eval()
    torch.onnx.export(
        wrapper,
        (features_example, lens_example),
        str(onnx_path),
        input_names=["features", "lens"],
        output_names=["hidden"],
        dynamic_axes={
            "features": {0: "batch", 1: "seq"},
            "lens": {0: "batch"},
            "hidden": {0: "batch", 1: "seq"},
        },
        opset_version=17,
        dynamo=False,
    )


# ---- decoder ----------------------------------------------------------------


class GemmConvTranspose1d(nn.Module):
    """ConvTranspose1d(C_in, C_out, K=2s, stride=s, pad=p) as one GEMM +
    phase interleave ([B, C_in, L] -> [B, C_out, (L-1)s - 2p + 2s]).

    Each input frame i contributes kernel taps k = half*s + phase to output
    i*s + k - p, i.e. a "low" half landing on frame i and a "high" half on
    frame i+1, which overlap-add after a one-frame shift.
    """

    def __init__(self, conv: nn.ConvTranspose1d) -> None:
        super().__init__()
        weight = conv.weight.detach()  # [C_in, C_out, K]
        self.stride = int(conv.stride[0])
        self.pad = int(conv.padding[0])
        assert weight.shape[-1] == 2 * self.stride, weight.shape
        assert 0 < self.pad <= self.stride
        self.register_buffer(
            "weight",
            weight.permute(0, 2, 1).reshape(weight.shape[0], -1).clone(),
        )  # [C_in, 2*s*C_out]
        self.register_buffer("bias", conv.bias.detach().clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_time_major(x).transpose(1, 2)

    def forward_time_major(self, x: torch.Tensor) -> torch.Tensor:
        """[B, C_in, L] -> [B, L_out, C_out] (the GEMM's native layout)."""
        batch, _, num_frame = x.shape
        stride = self.stride
        mixed = torch.matmul(x.transpose(1, 2), self.weight).reshape(
            batch, num_frame, 2, stride, -1
        )
        low = mixed[:, :, 0].reshape(batch, num_frame * stride, -1)
        high = mixed[:, :, 1].reshape(batch, num_frame * stride, -1)
        overlapped = F.pad(low, (0, 0, 0, stride)) + F.pad(
            high, (0, 0, stride, 0)
        )
        # end-relative crop only (start:start+len bakes the length in)
        cropped = overlapped[:, self.pad : -self.pad]
        return cropped + self.bias


class PluginSnake(nn.Module):
    """Snake exported as the fused sidon::Snake TRT plugin."""

    def __init__(self, source: nn.Module) -> None:
        super().__init__()
        self.register_buffer("alpha", source.alpha.detach().float().view(-1))
        self.register_buffer(
            "inv_alpha", source.inv_alpha.detach().float().view(-1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from .snake_triton import SnakeOp

        return SnakeOp.apply(x, self.alpha, self.inv_alpha)


def _swap_snakes(module: nn.Module) -> int:
    from .modeling import Snake

    count = 0
    for name, child in list(module.named_children()):
        if isinstance(child, Snake):
            setattr(module, name, PluginSnake(child))
            count += 1
        else:
            count += _swap_snakes(child)
    return count


class ExportableDACDecoder(nn.Module):
    """hidden [B, L, 1024] -> audio [B, 1, samples]."""

    def __init__(
        self,
        decoder: DACDecoder,
        gemm_upsample: bool,
        snake_plugin: bool = False,
    ) -> None:
        super().__init__()
        self.decoder = decoder
        if gemm_upsample:
            for block in decoder.blocks:
                block.upsample = GemmConvTranspose1d(block.upsample)
        if snake_plugin:
            _swap_snakes(decoder)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.decoder(hidden.transpose(1, 2))


HEAD_MASKS = ("mask_in", "mask_b0", "mask_b1", "mask_out")  # CUDA-tail head
FULL_MASKS = tuple(["mask_in"] + [f"mask_b{i}" for i in range(5)])  # TRT-only


def head_mask_lengths(frames, from_block: int = 2):
    """valid lengths at the head's mask points for `frames` decoder frames
    (int or integer tensor): input, after block 0..from_block-1, output."""
    from .cuda_tail import RATES, upsampled

    lengths = [frames]
    for stride in RATES[: from_block + 1]:
        lengths.append(upsampled(lengths[-1], stride))
    return lengths


class ExportableDACHead(nn.Module):
    """(hidden [B, L, 1024], mask_in [B, L], mask_b0 [B, T0], mask_b1
    [B, T1], mask_out [B, T]) -> z [B, T, C] fp16 TIME-MAJOR: the decoder
    up to block TAIL_FROM_BLOCK's upsample (TensorRT part of the CUDA-tail
    split; see cuda_tail.py). Requires the GEMM upsample (native [B, T, C]).

    The masks (1 = valid, 0 = bucket padding; built by the runtime) are
    applied after every conv at their resolution, so the valid region
    equals an exact-length run whose convs see zero padding there — this
    lets bucket-padded inputs share CUDA graphs and batch together. They
    are inputs rather than built in-graph from a length tensor: TensorRT
    10.16 (fp16) mis-evaluated in-graph Range/Less masks (all zeros).
    """

    def __init__(self, exportable: ExportableDACDecoder) -> None:
        super().__init__()
        self.decoder = exportable.decoder

    def forward(self, hidden, mask_in, mask_b0, mask_b1, mask_out):
        from .cuda_tail import TAIL_FROM_BLOCK

        assert TAIL_FROM_BLOCK == 2
        decoder = self.decoder
        masks = [mask_b0, mask_b1]
        x = hidden.transpose(1, 2) * mask_in[:, None, :]
        x = decoder.input_conv(x) * mask_in[:, None, :]
        for block, mask in zip(decoder.blocks[:TAIL_FROM_BLOCK], masks):
            m = mask[:, None, :]
            x = block.upsample(block.snake(x)) * m  # snake(0) = 0
            for unit in block.res:
                # conv2 is 1x1 (pointwise): padding garbage after conv1 never
                # reaches valid positions, so one mask per unit suffices —
                # before the next time-mixing conv (conv1 / upsample)
                y = unit.conv2(unit.snake2(unit.conv1(unit.snake1(x))))
                x = (x + y) * m
        block = decoder.blocks[TAIL_FROM_BLOCK]
        assert isinstance(block.upsample, GemmConvTranspose1d)
        z = block.upsample.forward_time_major(block.snake(x))  # [B, T, C]
        return (z * mask_out[..., None]).half()


def head_masks(frames: torch.Tensor, lengths: list,
               from_block: int = 2) -> list:
    """float masks [B, T_r] for buffers of `lengths` (bucket) and valid
    `frames` (int [B])."""
    valid = head_mask_lengths(frames.to(torch.int64), from_block)
    out = []
    for length, v in zip(lengths, valid):
        index = torch.arange(length, device=frames.device)[None, :]
        out.append((index < v[:, None]).float())
    return out


class ExportableDACMasked(nn.Module):
    """(hidden [B, L, 1024], mask_in, mask_b0 .. mask_b4) -> audio
    [B, 1, 960 L - 26]: the whole DAC decoder in TensorRT (the TRT-only
    fallback to the CUDA tail; ``build_engines.py --decoder trt``). Same
    masking contract as :class:`ExportableDACHead`, at all six
    resolutions."""

    def __init__(self, exportable: ExportableDACDecoder,
                 fp32_parts: Optional[tuple] = None) -> None:
        super().__init__()
        self.decoder = exportable.decoder
        # None: weakly typed export (fp32 graph, TRT fp16 flag). A tuple:
        # strongly typed fp16 graph with fp32 islands — "final" (final
        # Snake + conv + tanh), "block<i>" (whole DAC block i),
        # "residual" (residual stream / adds in fp32, convs fp16).
        self.fp32_parts = None if fp32_parts is None else set(fp32_parts)
        if self.fp32_parts is not None:
            unknown = self.fp32_parts - {
                "final", "residual", *(f"block{i}" for i in range(5))
            }
            assert not unknown, unknown
            self._cast_parts()

    def _dtype(self, part: str) -> torch.dtype:
        return torch.float32 if part in self.fp32_parts else torch.float16

    def _cast_parts(self) -> None:
        decoder = self.decoder
        decoder.input_conv.half()
        for index, block in enumerate(decoder.blocks):
            block.to(self._dtype(f"block{index}"))
        decoder.final_snake.to(self._dtype("final"))
        decoder.final_conv.to(self._dtype("final"))
        for module in decoder.modules():  # plugin per-channel params: fp32
            if isinstance(module, PluginSnake):
                module.alpha.data = module.alpha.data.float()
                module.inv_alpha.data = module.inv_alpha.data.float()

    def forward(self, hidden, *masks):
        if self.fp32_parts is not None:
            return self._forward_typed(hidden, *masks)
        decoder = self.decoder
        x = hidden.transpose(1, 2) * masks[0][:, None, :]
        x = decoder.input_conv(x) * masks[0][:, None, :]
        for block, mask in zip(decoder.blocks, masks[1:]):
            m = mask[:, None, :]
            x = block.upsample(block.snake(x)) * m
            for unit in block.res:
                y = unit.conv2(unit.snake2(unit.conv1(unit.snake1(x))))
                x = (x + y) * m
        audio = torch.tanh(decoder.final_conv(decoder.final_snake(x)))
        return audio * masks[-1][:, None, :]

    def _forward_typed(self, hidden, *masks):
        decoder = self.decoder
        half = torch.float16
        residual32 = "residual" in self.fp32_parts
        m = masks[0][:, None, :].to(half)
        x = decoder.input_conv(hidden.transpose(1, 2).to(half) * m) * m
        for index, (block, mask) in enumerate(zip(decoder.blocks, masks[1:])):
            dtype = self._dtype(f"block{index}")
            stream = torch.float32 if residual32 else dtype
            m = mask[:, None, :]
            x = block.upsample(block.snake(x.to(dtype))).to(stream)
            x = x * m.to(stream)
            for unit in block.res:
                y = unit.conv2(unit.snake2(unit.conv1(unit.snake1(
                    x.to(dtype)))))
                x = (x + y.to(stream)) * m.to(stream)
        dtype = self._dtype("final")
        audio = torch.tanh(decoder.final_conv(decoder.final_snake(
            x.to(dtype))))
        return audio.float() * masks[-1][:, None, :]


def export_head(
    head: nn.Module, hidden_example: torch.Tensor, onnx_path: Path
) -> None:
    """ExportableDACHead (-> z) or ExportableDACMasked (-> audio)."""
    full = isinstance(head, ExportableDACMasked)
    names = FULL_MASKS if full else HEAD_MASKS
    from_block = 4 if full else 2
    output = "audio" if full else "z"
    batch, frames = hidden_example.shape[:2]
    lengths = head_mask_lengths(frames, from_block)
    masks = head_masks(
        torch.full((batch,), frames, device=hidden_example.device), lengths,
        from_block,
    )
    torch.onnx.export(
        head,
        (hidden_example, *masks),
        str(onnx_path),
        input_names=["hidden", *names],
        output_names=[output],
        dynamic_axes={
            "hidden": {0: "batch", 1: "seq"},
            **{name: {0: "batch", 1: f"t_{name}"} for name in names},
            output: {0: "batch", 1 if not full else 2: "samples"},
        },
        opset_version=17,
        dynamo=False,
    )
