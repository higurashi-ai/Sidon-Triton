"""DAC decoder tail on the fused CUDA kernels (csrc/dac_kernels.cu).

The decoder is split before block ``TAIL_FROM_BLOCK``'s residual units:

- head (TensorRT engine): input conv, blocks < TAIL_FROM_BLOCK, and block
  TAIL_FROM_BLOCK's Snake + upsample -> z [B, T, C] fp16, TIME-MAJOR (the
  GEMM upsample produces this layout natively, so no transpose is paid);
- tail (this module, torch ops in the same CUDA graph): for each block from
  TAIL_FROM_BLOCK on, its residual units (fused mma.sync kernel; C=192
  streams weights per tap) and the next block's Snake + ConvTranspose
  (fused per-phase kernel); finally Snake + conv(48 -> 1) + tanh.
  Everything stays time-major [B, T, C].

With TAIL_FROM_BLOCK = 2 (default) the head ends at 8 kHz / 192 channels.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from . import cuda_ext

TAIL_FROM_BLOCK = 2
TAIL_FILE = "decoder_tail.pt"
_HEAD_CHANNELS = {2: 192, 3: 96}
TAIL_CHANNELS = _HEAD_CHANNELS[TAIL_FROM_BLOCK]


RATES = (8, 5, 4, 3, 2)


def upsampled(length, stride: int):
    """ConvTranspose1d(K=2s, stride s, pad ceil(s/2)) output length (int or
    integer tensor)."""
    pad = (stride + 1) // 2
    return (length - 1) * stride - 2 * pad + 2 * stride


def head_samples(frames, from_block: int = TAIL_FROM_BLOCK):
    """Head output length (after block `from_block`'s upsample); int or
    integer tensor (per-item valid lengths)."""
    length = frames
    for stride in RATES[: from_block + 1]:
        length = upsampled(length, stride)
    return length


def time_major(z: torch.Tensor) -> torch.Tensor:
    return z.transpose(1, 2).contiguous()


def _vec(t: torch.Tensor) -> torch.Tensor:
    return t.detach().float().reshape(-1).contiguous()


class FusedResUnit(nn.Module):
    """modeling.ResidualUnit -> packed weights for sidon_triton::resunit_mma
    (C = 48 / 96 / 192)."""

    def __init__(self, unit: nn.Module,
                 dtype: torch.dtype = torch.float16) -> None:
        super().__init__()
        self.ops = cuda_ext.ops(dtype)
        conv1, conv2 = unit.conv1, unit.conv2
        channels = conv1.weight.shape[0]
        ld = channels + 8  # smem row pitch used by the kernels
        w1 = torch.zeros(7, channels, ld, dtype=dtype)
        w1[:, :, :channels] = (
            conv1.weight.detach().float().permute(2, 0, 1).to(dtype).cpu()
        )
        w2 = torch.zeros(channels, ld, dtype=dtype)
        w2[:, :channels] = conv2.weight.detach().float()[:, :, 0].to(dtype).cpu()
        device = conv1.weight.device
        self.register_buffer("w1", w1.to(device))
        self.register_buffer("w2", w2.to(device))
        self.register_buffer("b1", _vec(conv1.bias))
        self.register_buffer("a1", _vec(unit.snake1.alpha))
        self.register_buffer("i1", _vec(unit.snake1.inv_alpha))
        self.register_buffer("a2", _vec(unit.snake2.alpha))
        self.register_buffer("i2", _vec(unit.snake2.inv_alpha))
        self.register_buffer("b2", _vec(conv2.bias))
        self.dilation = int(conv1.dilation[0])

    def forward(self, x: torch.Tensor, lens: torch.Tensor) -> torch.Tensor:
        """x [B, T, C]; lens int32 [B]: valid rows (rest = bucket padding)"""
        return self.ops.resunit_mma(
            x, self.w1, self.w2, self.b1, self.a1, self.i1,
            self.a2, self.i2, self.b2, self.dilation, lens,
        )


class FusedUpsample(nn.Module):
    """DecoderBlock's Snake + ConvTranspose1d -> sidon_triton::upsample."""

    def __init__(self, block: nn.Module,
                 dtype: torch.dtype = torch.float16) -> None:
        super().__init__()
        self.ops = cuda_ext.ops(dtype)
        up = block.upsample
        weight = up.weight.detach()  # [C_in, C_out, K = 2s]
        self.stride = int(up.stride[0])
        assert weight.shape[-1] == 2 * self.stride
        assert int(up.padding[0]) == (self.stride + 1) // 2
        c_in, c_out, taps = weight.shape
        packed = torch.zeros(taps, c_out, c_in + 8)
        packed[:, :, :c_in] = weight.float().permute(2, 1, 0).cpu()
        self.register_buffer("w", packed.to(dtype).to(weight.device))
        self.register_buffer("bias", _vec(up.bias))
        self.register_buffer("alpha", _vec(block.snake.alpha))
        self.register_buffer("inv", _vec(block.snake.inv_alpha))

    def forward(self, z: torch.Tensor, lens: torch.Tensor) -> torch.Tensor:
        return self.ops.upsample(
            z, self.w, self.bias, self.alpha, self.inv, self.stride, lens
        )


class CudaDecoderTail(nn.Module):
    """z [B, T, C] time-major (block TAIL_FROM_BLOCK upsample output)
    -> audio [B, 1, T48] fp32. `dtype` (fp16 default / bf16) selects the
    kernel build; z is cast to it on entry."""

    def __init__(self, decoder: nn.Module,
                 from_block: int = TAIL_FROM_BLOCK,
                 dtype: torch.dtype = torch.float16) -> None:
        super().__init__()
        self.ops = cuda_ext.ops(dtype)
        self.dtype = dtype
        self.from_block = from_block
        stages = []
        for index in range(from_block, len(decoder.blocks)):
            block = decoder.blocks[index]
            if index > from_block:
                stages.append(FusedUpsample(block, dtype))
            stages.extend(FusedResUnit(unit, dtype) for unit in block.res)
        self.stages = nn.ModuleList(stages)
        self.register_buffer("final_alpha", _vec(decoder.final_snake.alpha))
        self.register_buffer("final_inv", _vec(decoder.final_snake.inv_alpha))
        self.register_buffer("final_weight", _vec(decoder.final_conv.weight))
        self.final_bias = float(decoder.final_conv.bias.item())

    def forward(
        self, z: torch.Tensor, frames: torch.Tensor | None = None
    ) -> torch.Tensor:
        """frames (int [B], optional): valid decoder frames per item when z
        is bucket-padded; positions past each item's valid length are
        treated as zero padding at every stage (exact-length semantics) and
        output as zeros. None = every row valid."""
        z = z.to(self.dtype).contiguous()
        if frames is None:
            valid = torch.full(
                (z.shape[0],), z.shape[1], dtype=torch.int64, device=z.device
            )
        else:
            valid = head_samples(frames.to(torch.int64), self.from_block)
        lens = valid.to(torch.int32)
        for stage in self.stages:
            z = stage(z, lens)
            if isinstance(stage, FusedUpsample):
                valid = upsampled(valid, stage.stride)
                lens = valid.to(torch.int32)
        return self.ops.final_head(
            z, self.final_weight, self.final_alpha, self.final_inv,
            self.final_bias, lens,
        )


def save_tail(tail: CudaDecoderTail, decoder_config: dict, path) -> None:
    torch.save(
        {
            "state": {k: v.cpu() for k, v in tail.state_dict().items()},
            "final_bias": tail.final_bias,
            "from_block": tail.from_block,
            "decoder_config": decoder_config,
        },
        str(path),
    )


def load_tail(path, device: str = "cuda") -> CudaDecoderTail:
    from .modeling import build_decoder

    saved = torch.load(str(path), map_location="cpu", weights_only=False)
    tail = CudaDecoderTail(
        build_decoder(saved["decoder_config"]), saved["from_block"]
    )
    tail.load_state_dict(saved["state"], strict=True)
    tail.final_bias = float(saved["final_bias"])
    return tail.to(device).eval()


@torch.no_grad()
def reference_head(decoder: nn.Module, hidden: torch.Tensor,
                   from_block: int = TAIL_FROM_BLOCK) -> torch.Tensor:
    """hidden [B, L, 1024] -> head output fp32, channel-major [B, C, T]."""
    z = decoder.input_conv(hidden.transpose(1, 2).float())
    for block in decoder.blocks[:from_block]:
        z = block(z)
    block = decoder.blocks[from_block]
    return block.upsample(block.snake(z))


@torch.no_grad()
def reference_tail(decoder: nn.Module, z: torch.Tensor,
                   from_block: int = TAIL_FROM_BLOCK) -> torch.Tensor:
    """fp32 torch twin of CudaDecoderTail; z channel-major [B, C, T]."""
    z = z.float()
    for unit in decoder.blocks[from_block].res:
        z = unit(z)
    for block in decoder.blocks[from_block + 1:]:
        z = block(z)
    return torch.tanh(decoder.final_conv(decoder.final_snake(z)))
