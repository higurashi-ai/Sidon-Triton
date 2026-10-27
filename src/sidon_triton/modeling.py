"""Slim inference-only Sidon v0.1 model definitions + bundle loader.

Weight-compatible rebuilds of the two released TorchScript modules
(``sarulab-speech/sidon-v0.1``):

- ``encoder``: w2v-BERT 2.0 truncated to 8 layers (``relative_key``
  attention), LoRA on ``output_dense`` merged into the base weights.
  Input: SeamlessM4T fbank [B, L, 160]; output last_hidden_state
  [B, L, 1024] at 50 Hz.
- ``decoder``: DAC decoder (channels 1536, rates 8/5/4/3/2 -> hop 960),
  weight-norm already folded, Snake activations. [B, 1024, L] ->
  [B, 1, 960 * L - 26] at 48 kHz.

No peft / dac / Lightning dependency. ``tools/convert_torchscript.py``
produces the bundle loaded by :func:`load_bundle`.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

ENCODER_WEIGHTS = "encoder.safetensors"
ENCODER_CONFIG = "encoder_config.json"
DECODER_WEIGHTS = "decoder.safetensors"
DECODER_CONFIG = "decoder_config.json"


# ---- encoder ----------------------------------------------------------------


class FlexRelSelfAttention(nn.Module):
    """flex_attention drop-in for ``Wav2Vec2BertSelfAttention``.

    Keeps the ``relative_key`` position term exactly:
    ``scores += Q.R[clamp(j-i,-left,right)]^T / sqrt(d)`` with the 73-entry
    ``distance_embedding``. The per-pair bias is never materialized.
    """

    _flex_attention = None

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        left_max_position: int = 64,
        right_max_position: int = 8,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.left_max_position = left_max_position
        self.right_max_position = right_max_position
        self.linear_q = nn.Linear(hidden_size, hidden_size)
        self.linear_k = nn.Linear(hidden_size, hidden_size)
        self.linear_v = nn.Linear(hidden_size, hidden_size)
        self.linear_out = nn.Linear(hidden_size, hidden_size)
        self.distance_embedding = nn.Embedding(
            left_max_position + right_max_position + 1, self.head_dim
        )

    @classmethod
    def _flex(cls):
        if cls._flex_attention is None:
            import os

            from torch.nn.attention.flex_attention import flex_attention

            if (
                os.environ.get("STRITON_FLEX_NO_COMPILE") == "1"
                or not torch.cuda.is_available()
            ):
                cls._flex_attention = flex_attention
            else:
                cls._flex_attention = torch.compile(
                    flex_attention, dynamic=False
                )
        return cls._flex_attention

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        relative_position_embeddings: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        **_: object,
    ) -> tuple[torch.Tensor, None]:
        assert attention_mask is None, "Sidon runs unmasked"
        batch, length, hidden = hidden_states.shape
        shape = (batch, length, self.num_heads, self.head_dim)
        query = self.linear_q(hidden_states).view(shape).transpose(1, 2)
        key = self.linear_k(hidden_states).view(shape).transpose(1, 2)
        value = self.linear_v(hidden_states).view(shape).transpose(1, 2)
        query_rel = torch.einsum(
            "bhld,rd->bhlr", query, self.distance_embedding.weight
        ) / math.sqrt(self.head_dim)
        left, right = self.left_max_position, self.right_max_position

        def score_mod(score, b, h, q_idx, kv_idx):
            return score + query_rel[
                b, h, q_idx,
                torch.clamp(kv_idx - q_idx, -left, right) + left,
            ]

        context = self._flex()(query, key, value, score_mod=score_mod)
        context = context.transpose(1, 2).reshape(batch, length, hidden)
        return self.linear_out(context), None


def replace_attention_with_flex(ssl_model: nn.Module) -> int:
    """Swap every HF self-attention for FlexRelSelfAttention (same weights)."""
    config = ssl_model.config
    count = 0
    for layer in ssl_model.encoder.layers:
        source = layer.self_attn
        flex = FlexRelSelfAttention(
            int(config.hidden_size),
            int(config.num_attention_heads),
            int(config.left_max_position_embeddings),
            int(config.right_max_position_embeddings),
        ).to(source.linear_q.weight.device, source.linear_q.weight.dtype)
        flex.load_state_dict(source.state_dict(), strict=True)
        layer.self_attn = flex
        count += 1
    return count


class SidonEncoder(nn.Module):
    """[B, L, 160] fbank -> [B, L, 1024] last_hidden_state."""

    def __init__(self, ssl_model: nn.Module) -> None:
        super().__init__()
        self.model = ssl_model

    def forward(self, input_features: torch.Tensor) -> torch.Tensor:
        return self.model(input_features=input_features).last_hidden_state


def build_encoder(config: dict) -> SidonEncoder:
    from transformers import Wav2Vec2BertConfig, Wav2Vec2BertModel

    hf_config = Wav2Vec2BertConfig(**config)
    hf_config._attn_implementation = "eager"
    return SidonEncoder(Wav2Vec2BertModel(hf_config).eval())


# ---- decoder ----------------------------------------------------------------


class Snake(nn.Module):
    """DAC Snake1d with the reciprocal precomputed (as in the TorchScript):
    ``x + inv_alpha * sin(alpha * x) ** 2``."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.register_buffer("alpha", torch.ones(1, channels, 1))
        self.register_buffer("inv_alpha", torch.ones(1, channels, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.inv_alpha * torch.sin(self.alpha * x).pow(2)


class ResidualUnit(nn.Module):
    def __init__(self, dim: int, dilation: int) -> None:
        super().__init__()
        self.snake1 = Snake(dim)
        self.conv1 = nn.Conv1d(
            dim, dim, 7, dilation=dilation, padding=3 * dilation
        )
        self.snake2 = Snake(dim)
        self.conv2 = nn.Conv1d(dim, dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv2(self.snake2(self.conv1(self.snake1(x))))
        return x + y


class DecoderBlock(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, stride: int) -> None:
        super().__init__()
        self.snake = Snake(input_dim)
        self.upsample = nn.ConvTranspose1d(
            input_dim,
            output_dim,
            kernel_size=2 * stride,
            stride=stride,
            padding=math.ceil(stride / 2),
        )
        self.res = nn.ModuleList(
            [ResidualUnit(output_dim, d) for d in (1, 3, 9)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.upsample(self.snake(x))
        for unit in self.res:
            x = unit(x)
        return x


class DACDecoder(nn.Module):
    """[B, input_channel, L] -> [B, 1, prod(rates) * L - trim] in [-1, 1]."""

    def __init__(
        self, input_channel: int, channels: int, rates: list[int]
    ) -> None:
        super().__init__()
        self.rates = list(rates)
        self.input_conv = nn.Conv1d(input_channel, channels, 7, padding=3)
        blocks = []
        for index, stride in enumerate(rates):
            blocks.append(
                DecoderBlock(
                    channels // 2**index, channels // 2 ** (index + 1), stride
                )
            )
        self.blocks = nn.ModuleList(blocks)
        out_dim = channels // 2 ** len(rates)
        self.final_snake = Snake(out_dim)
        self.final_conv = nn.Conv1d(out_dim, 1, 7, padding=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_conv(x)
        for block in self.blocks:
            x = block(x)
        return torch.tanh(self.final_conv(self.final_snake(x)))

    def output_length(self, frames: int) -> int:
        length = frames
        for stride in self.rates:
            # ConvTranspose1d(K=2s, stride s, pad ceil(s/2))
            length = (
                (length - 1) * stride
                - 2 * math.ceil(stride / 2)
                + 2 * stride
            )
        return length


def build_decoder(config: dict) -> DACDecoder:
    return DACDecoder(
        int(config["input_channel"]),
        int(config["channels"]),
        list(config["rates"]),
    ).eval()


# ---- bundle -------------------------------------------------------------------


class SidonBundle:
    """encoder (HF eager attention) + decoder, fp32, loaded from a bundle."""

    def __init__(self, bundle_dir: Path, device: str = "cuda") -> None:
        from safetensors.torch import load_file

        bundle_dir = Path(bundle_dir)
        self.encoder_config = json.loads(
            (bundle_dir / ENCODER_CONFIG).read_text()
        )
        self.decoder_config = json.loads(
            (bundle_dir / DECODER_CONFIG).read_text()
        )
        self.encoder = build_encoder(self.encoder_config)
        self.encoder.model.load_state_dict(
            load_file(str(bundle_dir / ENCODER_WEIGHTS)), strict=True
        )
        self.decoder = build_decoder(self.decoder_config)
        self.decoder.load_state_dict(
            load_file(str(bundle_dir / DECODER_WEIGHTS)), strict=True
        )
        self.encoder.to(device).eval()
        self.decoder.to(device).eval()
        self.hop = math.prod(self.decoder.rates)
        self.output_rate = int(self.decoder_config["sample_rate"])

    def use_flex_attention(self) -> int:
        return replace_attention_with_flex(self.encoder.model)


def load_bundle(bundle_dir, device: str = "cuda") -> SidonBundle:
    return SidonBundle(Path(bundle_dir), device=device)
