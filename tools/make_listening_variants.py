#!/usr/bin/env python3
"""Localize audible differences: official Sidon vs Sidon-Triton stage by stage.

For each (single-chunk, <= 94 s) input, writes to <output>/<item>/:
    0_input.wav            the input
    1_official_sidon.wav   official demo (TorchScript fp32 + CPU extractor)
    2_A_fp32.wav           our torch fp32 encoder + fp32 decoder (should match 1)
    3_B_enc_trt.wav       TensorRT encoder (bf16 or fp16 engine) + fp32 torch decoder
    4_C_dec_fast.wav       fp32 torch encoder + fast decoder (TRT head + CUDA tail)
    5_D_sidon_triton.wav      Sidon-Triton (TRT encoder + fast decoder)
    6_E_sidon_triton_fp32.wav Sidon-Triton all-fp32 engines (--engine-dir-fp32)
    7_F_sidon_triton_bf16tail.wav  Sidon-Triton with the bf16 CUDA tail (--bf16-tail)
    official_then_sidon_triton.wav   1 then 0.7 s silence then 5
Multi-channel inputs are processed channel by channel (2-channel dialogue:
one speaker per channel) and keep their channel layout.
and metrics.json with log-mel L1 / SNR of every variant vs the official.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from compare_demo import log_mel_distance  # noqa: E402

from sidon_triton import hub  # noqa: E402
from sidon_triton import frontend  # noqa: E402
from sidon_triton.audio_io import read_wav, write_wav  # noqa: E402
from sidon_triton.demo import DemoSidon  # noqa: E402
from sidon_triton.engines import errors  # noqa: E402
from sidon_triton.modeling import load_bundle  # noqa: E402
from sidon_triton.runtime import (  # noqa: E402
    DEMO_CHUNK_SECONDS,
    DEMO_TAIL_PAD,
    HOP,
    OUT_RATE,
    TRIM,
    SidonPipeline,
)


@torch.no_grad()
def fast_decoder(
    pipeline: SidonPipeline, hidden: torch.Tensor, tail=None
) -> torch.Tensor:
    """TRT head + CUDA tail on hidden [1, L, 1024] -> audio [N] (GPU).
    `tail` overrides the pipeline's tail (e.g. the bf16 kernel build)."""
    engines = pipeline.long
    if tail is None:
        audio = engines.decode(hidden)
    else:
        saved, engines.tail = engines.tail, tail
        try:
            audio = engines.decode(hidden)
        finally:
            engines.tail = saved
    torch.cuda.synchronize()
    return audio.view(-1)


@torch.no_grad()
def trt_encoder(pipeline: SidonPipeline, features: torch.Tensor) -> torch.Tensor:
    hidden = pipeline.long.encode(features)
    torch.cuda.synchronize()
    return hidden


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine-dir", type=Path, required=True)
    parser.add_argument("--torchscript-dir", type=Path, default=None,
                        help=hub.HELP)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--bf16-tail", action="store_true",
        help="add 7_F_sidon_triton_bf16tail (CUDA tail kernels built in bf16)",
    )
    parser.add_argument(
        "--engine-dir-fp32", type=Path, default=None,
        help="optional all-fp32 Sidon-Triton engines -> 6_E_sidon_triton_fp32",
    )
    parser.add_argument("wavs", type=Path, nargs="+")
    args = parser.parse_args()
    torch.set_float32_matmul_precision("highest")

    demo = DemoSidon(args.torchscript_dir, device="cuda")
    pipeline = SidonPipeline(args.engine_dir, use_graph=False)
    pipeline32 = (
        SidonPipeline(args.engine_dir_fp32, use_graph=False)
        if args.engine_dir_fp32 else None
    )
    bundle = load_bundle(args.bundle, device="cuda")
    front = frontend.SeamlessFbank().cuda().eval()
    tail_bf16 = None
    if args.bf16_tail:
        from sidon_triton.cuda_tail import CudaDecoderTail

        tail_bf16 = CudaDecoderTail(
            bundle.decoder, dtype=torch.bfloat16
        ).cuda().eval()
    metrics = {}
    def variants(channel: torch.Tensor, rate: int) -> dict:
        """one channel [samples] -> {variant name: 48 kHz audio}"""
        target = int(OUT_RATE / rate * channel.shape[-1])
        wav16 = frontend.preprocess_gpu(channel.cuda(), rate)
        wav16 = torch.nn.functional.pad(wav16, (0, DEMO_TAIL_PAD))
        assert wav16.numel() <= DEMO_CHUNK_SECONDS * 16_000, (
            "longer than one demo chunk"
        )
        with torch.no_grad():
            features = front(wav16.unsqueeze(0))
            hidden32 = bundle.encoder(features).float()
            hidden_trt = trt_encoder(pipeline, features)

            def finish(audio: torch.Tensor) -> torch.Tensor:
                return audio.view(-1)[:-HOP][:target].float()

            def dec32(hidden):
                return finish(bundle.decoder(hidden.transpose(1, 2)))

            extra = {}
            if tail_bf16 is not None:
                extra["7_F_sidon_triton_bf16tail"] = finish(
                    fast_decoder(pipeline, hidden_trt, tail=tail_bf16)
                )
            if pipeline32 is not None:
                extra["6_E_sidon_triton_fp32"] = finish(
                    fast_decoder(pipeline32, trt_encoder(pipeline32, features))
                )
            return {
                **extra,
                "1_official_sidon": demo.restore(channel, rate).float().cuda(),
                "2_A_fp32": dec32(hidden32),
                "3_B_enc_trt": dec32(hidden_trt),
                "4_C_dec_fast": finish(fast_decoder(pipeline, hidden32)),
                "5_D_sidon_triton": finish(fast_decoder(pipeline, hidden_trt)),
            }

    for path in args.wavs:
        item = path.stem
        data, rate = read_wav(path)
        # each channel separately (2-channel dialogue: one speaker each)
        per_channel = [variants(torch.from_numpy(c), rate) for c in data]
        outputs = {
            name: torch.stack([v[name] for v in per_channel])
            for name in per_channel[0]
        }  # [channels, samples]
        official = outputs["1_official_sidon"]
        folder = args.output_dir / item
        folder.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, folder / "0_input.wav")
        entry = {}
        for name, audio in outputs.items():
            write_wav(folder / f"{name}.wav", audio.cpu().numpy(), OUT_RATE)
            if name != "1_official_sidon":
                e = errors(official, audio)
                entry[name] = {
                    "log_mel_l1_db": log_mel_distance(official, audio),
                    "snr_db": e["snr_db"],
                }
        silence = np.zeros(
            (official.shape[0], int(0.7 * OUT_RATE)), dtype=np.float32
        )
        write_wav(
            folder / "official_then_sidon_triton.wav",
            np.concatenate(
                [official.cpu().numpy(), silence,
                 outputs["5_D_sidon_triton"].cpu().numpy()],
                axis=1,
            ),
            OUT_RATE,
        )
        metrics[item] = entry
        print(item, json.dumps(entry), flush=True)
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n"
    )
    print("LISTENING_VARIANTS_OK", flush=True)


if __name__ == "__main__":
    main()
