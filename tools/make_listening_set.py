#!/usr/bin/env python3
"""Listening set: official demo vs Sidon-Triton on the same inputs.

    make_listening_set.py --engine-dir E --torchscript-dir sidon-v0.1 \
        --output-dir listening/ inputs/*.wav

Layout:
    named/<item>/input.wav, official.wav, sidon_triton.wav
    blind/<item>_A.wav, <item>_B.wav   (random order, key.json reveals)
Multi-channel inputs (e.g. 2-channel dialogue) are restored per
channel and written with the same channel layout.
    metrics.json                        (waveform SNR, log-mel L1 vs official)
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from compare_demo import log_mel_distance  # noqa: E402

from sidon_triton import hub  # noqa: E402
from sidon_triton.audio_io import read_wav, write_wav  # noqa: E402
from sidon_triton.demo import DemoSidon  # noqa: E402
from sidon_triton.engines import errors  # noqa: E402
from sidon_triton.runtime import OUT_RATE, SidonPipeline  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine-dir", type=Path, required=True)
    parser.add_argument("--torchscript-dir", type=Path, default=None,
                        help=hub.HELP)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("wavs", type=Path, nargs="+")
    args = parser.parse_args()

    demo = DemoSidon(args.torchscript_dir, device="cuda")
    pipeline = SidonPipeline(args.engine_dir, use_graph=True)
    rng = random.Random(args.seed)
    (args.output_dir / "blind").mkdir(parents=True, exist_ok=True)
    key, metrics = {}, {}
    for path in args.wavs:
        item = path.stem
        data, rate = read_wav(path)
        waveform = torch.from_numpy(data)
        # 2-channel dialogue (one speaker per channel): restore each channel
        per_channel = data.shape[0] > 1
        official = demo.restore(waveform, rate, per_channel=per_channel)
        official = official.float().cuda()
        fast = pipeline.restore(waveform, rate, per_channel=per_channel)
        fast = fast.float()
        entry = errors(official, fast)
        entry["log_mel_l1_db"] = log_mel_distance(official, fast)
        entry["seconds"] = data.shape[-1] / rate
        entry["input_rate"] = rate
        entry["channels"] = int(data.shape[0])
        metrics[item] = entry
        print(f"{item}: {json.dumps(entry)}", flush=True)

        named = args.output_dir / "named" / item
        named.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, named / "input.wav")
        outputs = {
            "official": official.cpu().numpy(),
            "sidon_triton": fast.cpu().numpy(),
        }
        for name, audio in outputs.items():
            write_wav(named / f"{name}.wav", audio, OUT_RATE)
        order = ["official", "sidon_triton"]
        rng.shuffle(order)
        key[item] = {"A": order[0], "B": order[1]}
        for label, name in zip("AB", order):
            write_wav(
                args.output_dir / "blind" / f"{item}_{label}.wav",
                outputs[name],
                OUT_RATE,
            )
    (args.output_dir / "blind" / "key.json").write_text(
        json.dumps(key, indent=2) + "\n"
    )
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n"
    )
    print("LISTENING_SET_OK", flush=True)


if __name__ == "__main__":
    main()
