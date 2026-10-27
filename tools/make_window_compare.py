#!/usr/bin/env python3
"""96 s one-shot chunks vs 20 s overlapping sliding windows.

For each input writes <output>/<item>/:
    0_input.wav
    1_official_96s.wav          official demo (96 s chunks)
    2_sidon_triton_96s.wav         Sidon-Triton, demo chunking (96 s, 1-frame carry)
    3_sidon_triton_slide20.wav     Sidon-Triton, 20 s windows / 1 s overlap
    96s_then_slide20.wav        2 then 0.7 s silence then 3
Multi-channel inputs are restored per channel. metrics.json holds log-mel
L1 (active frames) of 2 and 3 vs the official and of 3 vs 2, plus a speed
comparison of both modes on a long tiled input.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from compare_demo import log_mel_distance  # noqa: E402

from sidon_triton import hub  # noqa: E402
from sidon_triton.audio_io import read_wav, write_wav  # noqa: E402
from sidon_triton.demo import DemoSidon  # noqa: E402
from sidon_triton.runtime import OUT_RATE, SidonPipeline  # noqa: E402


def timed(fn, runs: int = 5) -> float:
    fn()  # warm-up (graph capture)
    walls = []
    for _ in range(runs):
        torch.cuda.synchronize()
        started = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        walls.append((time.perf_counter() - started) * 1e3)
    return sorted(walls)[len(walls) // 2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine-dir", type=Path, required=True)
    parser.add_argument("--torchscript-dir", type=Path, default=None,
                        help=hub.HELP)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--window-seconds", type=float, default=20.0)
    parser.add_argument("--overlap-seconds", type=float, default=1.0)
    parser.add_argument("--speed-wav", type=Path, default=None)
    parser.add_argument("--speed-seconds", type=float, default=600.0)
    parser.add_argument("wavs", type=Path, nargs="+")
    args = parser.parse_args()

    demo = DemoSidon(args.torchscript_dir, device="cuda")
    pipeline = SidonPipeline(args.engine_dir, use_graph=True)
    slide = dict(
        window_seconds=args.window_seconds,
        overlap_seconds=args.overlap_seconds,
    )
    report = {"items": {}, "speed": {}}
    for path in args.wavs:
        item = path.stem
        data, rate = read_wav(path)
        waveform = torch.from_numpy(data)
        per_channel = data.shape[0] > 1
        official = demo.restore(waveform, rate, per_channel=per_channel)
        official = official.float().cuda()
        one_shot = pipeline.restore(waveform, rate, per_channel=per_channel)
        sliding = pipeline.restore_sliding(
            waveform, rate, per_channel=per_channel, **slide
        )
        assert one_shot.shape == sliding.shape == official.shape
        entry = {
            "seconds": data.shape[-1] / rate,
            "channels": int(data.shape[0]),
            "fast96_vs_official": log_mel_distance(official, one_shot),
            "slide20_vs_official": log_mel_distance(official, sliding),
            "slide20_vs_fast96": log_mel_distance(one_shot, sliding),
        }
        report["items"][item] = entry
        print(item, json.dumps(entry), flush=True)
        folder = args.output_dir / item
        folder.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, folder / "0_input.wav")
        outputs = {
            "1_official_96s": official,
            "2_sidon_triton_96s": one_shot,
            "3_sidon_triton_slide20": sliding,
        }
        for name, audio in outputs.items():
            write_wav(folder / f"{name}.wav", audio.cpu().numpy(), OUT_RATE)
        a, b = one_shot.cpu().numpy(), sliding.cpu().numpy()
        if a.ndim == 1:
            a, b = a[None], b[None]
        silence = np.zeros((a.shape[0], int(0.7 * OUT_RATE)), np.float32)
        write_wav(
            folder / "96s_then_slide20.wav",
            np.concatenate([a, silence, b], axis=1),
            OUT_RATE,
        )

    if args.speed_wav is not None:
        data, rate = read_wav(args.speed_wav)
        mono = torch.from_numpy(data[:1])
        total = int(args.speed_seconds * rate)
        long = mono.repeat(1, -(-total // mono.shape[-1]))[:, :total]
        audio_seconds = long.shape[-1] / rate
        for name, fn in (
            ("fast96", lambda: pipeline.restore(long, rate)),
            ("slide20", lambda: pipeline.restore_sliding(long, rate, **slide)),
            ("official96", None),
        ):
            if fn is None:
                continue
            ms = timed(fn)
            report["speed"][name] = {
                "audio_seconds": audio_seconds,
                "wall_ms": ms,
                "rtf": ms / 1e3 / audio_seconds,
            }
            print(name, json.dumps(report["speed"][name]), flush=True)
    (args.output_dir / "metrics.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print("WINDOW_COMPARE_OK", flush=True)


if __name__ == "__main__":
    main()
