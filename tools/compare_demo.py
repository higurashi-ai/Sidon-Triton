#!/usr/bin/env python3
"""End-to-end parity: official demo path vs Sidon-Triton ``restore`` on real
files. Reports waveform SNR / log-mel distance and writes both outputs
for listening.

    compare_demo.py --engine-dir E --torchscript-dir sidon-v0.1 \
        --output-dir out/ a.wav b.wav ...
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sidon_triton import hub  # noqa: E402
from sidon_triton.audio_io import read_wav, write_wav  # noqa: E402
from sidon_triton.demo import DemoSidon  # noqa: E402
from sidon_triton.engines import errors  # noqa: E402
from sidon_triton.runtime import SidonPipeline  # noqa: E402


def log_mel_distance(
    reference: torch.Tensor, actual: torch.Tensor, floor_db: float = 50.0
) -> float:
    """Mean |log-mel| difference (dB) at 48 kHz over ACTIVE frames — those
    within `floor_db` of the reference's loudest frame. Tolerant to phase;
    ignores the fine structure of a -70 dB silence floor, which would
    otherwise dominate on per-channel dialogue audio (each channel is
    silent while the other speaker talks)."""
    import torchaudio

    mel = torchaudio.transforms.MelSpectrogram(
        48_000, n_fft=2048, hop_length=480, n_mels=128
    ).to(reference.device)
    power_ref = mel(reference.float())
    power_act = mel(actual.float())
    a = 10 * torch.log10(power_ref + 1e-8)
    b = 10 * torch.log10(power_act + 1e-8)
    frame_db = 10 * torch.log10(power_ref.mean(dim=-2) + 1e-12)
    active = frame_db > frame_db.max() - floor_db  # [..., frames]
    diff = (a - b).abs().mean(dim=-2)  # per frame
    return diff[active].mean().item()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine-dir", type=Path, required=True)
    parser.add_argument("--torchscript-dir", type=Path, default=None,
                        help=hub.HELP)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repeat-to-seconds", type=float, default=0.0)
    parser.add_argument("wavs", type=Path, nargs="+")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    demo = DemoSidon(args.torchscript_dir, device="cuda")
    pipeline = SidonPipeline(args.engine_dir, use_graph=True)
    report = {}
    for path in args.wavs:
        data, rate = read_wav(path)
        data = torch.from_numpy(data)
        if args.repeat_to_seconds > 0:
            total = int(args.repeat_to_seconds * rate)
            data = data.repeat(1, -(-total // data.shape[-1]))[:, :total]
        reference = demo.restore(data, rate).float().cuda()
        actual = pipeline.restore(data, rate).float()
        assert reference.shape == actual.shape, (reference.shape, actual.shape)
        entry = errors(reference, actual)
        entry["log_mel_l1_db"] = log_mel_distance(reference, actual)
        entry["seconds"] = data.shape[-1] / rate
        stem = f"{path.parent.name}_{path.stem}"
        report[stem] = entry
        print(f"{stem}: {json.dumps(entry)}", flush=True)
        write_wav(
            args.output_dir / f"{stem}_demo.wav", reference.cpu().numpy(), 48_000
        )
        write_wav(
            args.output_dir / f"{stem}_sidon_triton.wav",
            actual.cpu().numpy(),
            48_000,
        )
    (args.output_dir / "compare_demo.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print("COMPARE_DEMO_OK", flush=True)


if __name__ == "__main__":
    main()
