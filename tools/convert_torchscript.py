#!/usr/bin/env python3
"""Convert the released Sidon v0.1 TorchScript modules into a Sidon-Triton bundle.

    convert_torchscript.py --torchscript-dir sidon-v0.1/ --output bundle/

The TorchScript files are frozen: every weight is an anonymous constant
``CONSTANTS.cN``. The constants are mapped onto named parameters by their
order of first use in the generated code (one fixed op sequence per
w2v-BERT conformer layer / DAC block), with shape asserts on every tensor
and value asserts on the shared scalars. LoRA (r=64, alpha=16, on every
``output_dense``) is merged in float64: ``W + 0.25 * (A @ B)``.

Parity is checked against the TorchScript itself (fp32, CPU) on real or
random audio before anything is written.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sidon_triton import hub  # noqa: E402
from sidon_triton import frontend  # noqa: E402
from sidon_triton.audio_io import read_wav  # noqa: E402
from sidon_triton.modeling import (  # noqa: E402
    DECODER_CONFIG,
    DECODER_WEIGHTS,
    ENCODER_CONFIG,
    ENCODER_WEIGHTS,
    build_decoder,
    build_encoder,
)

# facebook/w2v-bert-2.0 config.json, truncated to the 8 layers Sidon uses;
# training-only knobs zeroed (mask_time_prob > 0 would add masked_spec_embed)
ENCODER_CONFIG_DICT = {
    "hidden_size": 1024,
    "num_hidden_layers": 8,
    "num_attention_heads": 16,
    "intermediate_size": 4096,
    "feature_projection_input_dim": 160,
    "hidden_act": "swish",
    "layer_norm_eps": 1e-5,
    "position_embeddings_type": "relative_key",
    "left_max_position_embeddings": 64,
    "right_max_position_embeddings": 8,
    "max_source_positions": 5000,
    "conv_depthwise_kernel_size": 31,
    "rotary_embedding_base": 10000,
    "add_adapter": False,
    "use_intermediate_ffn_before_adapter": False,
    "use_weighted_layer_sum": False,
    "apply_spec_augment": False,
    "mask_time_prob": 0.0,
    "mask_feature_prob": 0.0,
    "layerdrop": 0.0,
    "hidden_dropout": 0.0,
    "activation_dropout": 0.0,
    "attention_dropout": 0.0,
    "feat_proj_dropout": 0.0,
    "conformer_conv_dropout": 0.0,
}
DECODER_CONFIG_DICT = {
    "input_channel": 1024,
    "channels": 1536,
    "rates": [8, 5, 4, 3, 2],
    "sample_rate": 48_000,
}

LORA_SCALE = 16 / 64
FFN_RESIDUAL = 0.5
ATTN_SCALE = math.sqrt(64)

T = "linear_t"  # TorchScript stores Linear weights as W^T (x @ c)


def _ffn(prefix: str) -> list[tuple[str, str]]:
    return [
        (f"{prefix}.intermediate_dense.weight", T),
        (f"{prefix}.intermediate_dense.bias", ""),
        (f"{prefix}.output_dense.weight", T),
        (f"{prefix}.output_dense.bias", ""),
        (f"{prefix}.output_dense.lora_A", ""),
        (f"{prefix}.output_dense.lora_B", ""),
    ]


def encoder_layout(num_layers: int) -> list[tuple[str, str]]:
    """Named tensors in TorchScript first-use order."""
    names = [
        ("feature_projection.layer_norm.weight", ""),
        ("feature_projection.layer_norm.bias", ""),
        ("feature_projection.projection.weight", T),
        ("feature_projection.projection.bias", ""),
    ]
    for index in range(num_layers):
        p = f"encoder.layers.{index}"
        names += [
            (f"{p}.ffn1_layer_norm.weight", ""),
            (f"{p}.ffn1_layer_norm.bias", ""),
            *_ffn(f"{p}.ffn1"),
            (f"{p}.self_attn_layer_norm.weight", ""),
            (f"{p}.self_attn_layer_norm.bias", ""),
        ]
        for proj in ("q", "k", "v"):
            names += [
                (f"{p}.self_attn.linear_{proj}.weight", T),
                (f"{p}.self_attn.linear_{proj}.bias", ""),
            ]
        names += [
            (f"{p}.self_attn.distance_embedding.weight", ""),
            (f"{p}.self_attn.linear_out.weight", T),
            (f"{p}.self_attn.linear_out.bias", ""),
            (f"{p}.conv_module.layer_norm.weight", ""),
            (f"{p}.conv_module.layer_norm.bias", ""),
            (f"{p}.conv_module.pointwise_conv1.weight", ""),
            (f"{p}.conv_module.depthwise_conv.weight", ""),
            (f"{p}.conv_module.depthwise_layer_norm.weight", ""),
            (f"{p}.conv_module.depthwise_layer_norm.bias", ""),
            (f"{p}.conv_module.pointwise_conv2.weight", ""),
            (f"{p}.ffn2_layer_norm.weight", ""),
            (f"{p}.ffn2_layer_norm.bias", ""),
            *_ffn(f"{p}.ffn2"),
            (f"{p}.final_layer_norm.weight", ""),
            (f"{p}.final_layer_norm.bias", ""),
        ]
    return names


def decoder_layout(num_blocks: int) -> list[str]:
    names = ["input_conv.weight", "input_conv.bias"]
    for index in range(num_blocks):
        p = f"blocks.{index}"
        names += [
            f"{p}.snake.alpha", f"{p}.snake.inv_alpha",
            f"{p}.upsample.weight", f"{p}.upsample.bias",
        ]
        for unit in range(3):
            u = f"{p}.res.{unit}"
            names += [
                f"{u}.snake1.alpha", f"{u}.snake1.inv_alpha",
                f"{u}.conv1.weight", f"{u}.conv1.bias",
                f"{u}.snake2.alpha", f"{u}.snake2.inv_alpha",
                f"{u}.conv2.weight", f"{u}.conv2.bias",
            ]
    names += [
        "final_snake.alpha", "final_snake.inv_alpha",
        "final_conv.weight", "final_conv.bias",
    ]
    return names


def ordered_constants(module) -> tuple[list[torch.Tensor], dict[str, object]]:
    """Tensor constants in first-use order + all scalar constants."""
    code, constants = module.code_with_constants
    mapping = constants.const_mapping
    seen: list[str] = []
    for match in re.finditer(r"CONSTANTS\.(c\d+)", code):
        if match.group(1) not in seen:
            seen.append(match.group(1))
    tensors, scalars = [], {}
    for name in seen:
        value = mapping[name]
        if isinstance(value, torch.Tensor) and value.ndim > 0:
            tensors.append(value)
        else:
            scalars[name] = value
    return tensors, scalars


def convert_encoder(module) -> dict[str, torch.Tensor]:
    tensors, scalars = ordered_constants(module)
    values = sorted(
        float(v) for v in scalars.values() if isinstance(v, torch.Tensor)
    )
    # LoRA scale, FFN half-step residual, sqrt(head_dim), left offset
    assert values == sorted(
        [LORA_SCALE, FFN_RESIDUAL, ATTN_SCALE, 64.0]
    ), scalars
    layout = encoder_layout(ENCODER_CONFIG_DICT["num_hidden_layers"])
    assert len(tensors) == len(layout), (len(tensors), len(layout))
    raw = {}
    for (name, kind), tensor in zip(layout, tensors):
        raw[name] = tensor.t() if kind == T else tensor
    out: dict[str, torch.Tensor] = {}
    for name, tensor in raw.items():
        if name.endswith((".lora_A", ".lora_B")):
            continue
        if name.endswith("output_dense.weight"):
            base = name[: -len(".weight")]
            lora_a = raw[f"{base}.lora_A"].double()  # [4096, 64]  (x @ A)
            lora_b = raw[f"{base}.lora_B"].double()  # [64, 1024]
            assert lora_a.shape[1] == lora_b.shape[0] == 64
            merged = tensor.double() + LORA_SCALE * (lora_a @ lora_b).t()
            tensor = merged
        out[name] = tensor.float().contiguous()
    return out


def convert_decoder(module) -> dict[str, torch.Tensor]:
    tensors, scalars = ordered_constants(module)
    assert not scalars, scalars
    layout = decoder_layout(len(DECODER_CONFIG_DICT["rates"]))
    assert len(tensors) == len(layout), (len(tensors), len(layout))
    return {
        name: tensor.float().contiguous()
        for name, tensor in zip(layout, tensors)
    }


def load_audio(path: Path | None, seconds: float) -> torch.Tensor:
    if path is None:
        torch.manual_seed(0)
        return torch.randn(int(seconds * 16_000)) * 0.1
    data, rate = read_wav(path)
    wav16 = frontend.preprocess(torch.from_numpy(data), rate)
    return wav16[: int(seconds * 16_000)]


def snr_db(reference: torch.Tensor, actual: torch.Tensor) -> float:
    noise = (reference.double() - actual.double()).square().mean().sqrt()
    signal = reference.double().square().mean().sqrt()
    return (20 * torch.log10(signal / noise.clamp_min(1e-30))).item()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--torchscript-dir", type=Path, default=None,
                        help=hub.HELP)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wav", type=Path, default=None)
    parser.add_argument("--seconds", type=float, default=6.0)
    args = parser.parse_args()
    args.torchscript_dir = hub.torchscript_dir(args.torchscript_dir)
    torch.set_grad_enabled(False)

    feature_script = torch.jit.load(
        str(args.torchscript_dir / "feature_extractor_cpu.pt"),
        map_location="cpu",
    ).eval()
    decoder_script = torch.jit.load(
        str(args.torchscript_dir / "decoder_cpu.pt"), map_location="cpu"
    ).eval()

    encoder_state = convert_encoder(feature_script)
    decoder_state = convert_decoder(decoder_script)

    encoder = build_encoder(ENCODER_CONFIG_DICT)
    missing = set(encoder.model.state_dict()) ^ set(encoder_state)
    assert not missing, sorted(missing)
    encoder.model.load_state_dict(encoder_state, strict=True)
    decoder = build_decoder(DECODER_CONFIG_DICT)
    decoder.load_state_dict(decoder_state, strict=True)

    # parity vs the TorchScript (fp32 CPU) on the demo's own extractor output
    chunk = load_audio(args.wav, args.seconds)
    features = frontend.reference_features(chunk)
    ours_features = frontend.SeamlessFbank()(chunk.unsqueeze(0))
    report = {
        "fbank_snr_db": snr_db(features, ours_features),
        "fbank_max_abs": (features - ours_features).abs().max().item(),
    }
    reference_hidden = feature_script(features)["last_hidden_state"]
    hidden = encoder(features)
    report["encoder_snr_db"] = snr_db(reference_hidden, hidden)
    report["encoder_max_abs"] = (reference_hidden - hidden).abs().max().item()
    reference_audio = decoder_script(reference_hidden.transpose(1, 2))
    audio = decoder(reference_hidden.transpose(1, 2))
    assert audio.shape == reference_audio.shape, (
        audio.shape, reference_audio.shape
    )
    assert audio.shape[-1] == decoder.output_length(hidden.shape[1])
    report["decoder_snr_db"] = snr_db(reference_audio, audio)
    report["decoder_max_abs"] = (reference_audio - audio).abs().max().item()
    report["frames"] = int(hidden.shape[1])
    report["audio_samples"] = int(audio.shape[-1])
    print(json.dumps(report, indent=2))
    assert report["encoder_snr_db"] > 80, "encoder conversion mismatch"
    assert report["decoder_snr_db"] > 80, "decoder conversion mismatch"
    assert report["fbank_snr_db"] > 60, "fbank front-end mismatch"

    from safetensors.torch import save_file

    args.output.mkdir(parents=True, exist_ok=True)
    save_file(encoder_state, str(args.output / ENCODER_WEIGHTS))
    save_file(decoder_state, str(args.output / DECODER_WEIGHTS))
    (args.output / ENCODER_CONFIG).write_text(
        json.dumps(ENCODER_CONFIG_DICT, indent=2) + "\n"
    )
    (args.output / DECODER_CONFIG).write_text(
        json.dumps(DECODER_CONFIG_DICT, indent=2) + "\n"
    )
    (args.output / "conversion_report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(f"CONVERT_OK bundle={args.output}")


if __name__ == "__main__":
    main()
