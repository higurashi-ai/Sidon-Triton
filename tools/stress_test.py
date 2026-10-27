#!/usr/bin/env python3
"""Large-scale robustness / throughput / parity test.

- every clip in --clip-dir through ``restore``: finite, correct length,
  |x| <= 1, device memory over time, throughput, graph count;
- determinism: a subset twice;
- parity: random clips vs the official demo (log-mel L1 on active frames);
- long multi-chunk 2-channel recordings (--long-files), per channel.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
import warnings
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from compare_demo import log_mel_distance  # noqa: E402

from sidon_triton import hub  # noqa: E402
from sidon_triton.demo import DemoSidon  # noqa: E402
from sidon_triton.runtime import OUT_RATE, SidonPipeline  # noqa: E402

warnings.filterwarnings("ignore")


def load(path: Path):
    import torchaudio

    waveform, rate = torchaudio.load(str(path))
    return waveform, rate


def used_mb() -> float:
    free, total = torch.cuda.mem_get_info()
    return (total - free) / 1e6


def check(audio: torch.Tensor, samples: int, rate: int) -> list:
    problems = []
    expected = int(OUT_RATE / rate * samples)
    if audio.shape[-1] != expected:
        problems.append(f"length {audio.shape[-1]} != {expected}")
    if not torch.isfinite(audio).all():
        problems.append("non-finite")
    if audio.abs().max() > 1.0 + 1e-6:
        problems.append(f"peak {audio.abs().max().item():.3f}")
    return problems


def stats(values: list) -> dict:
    values = sorted(values)
    pick = lambda q: values[min(len(values) - 1, int(q * len(values)))]  # noqa
    return {
        "n": len(values),
        "mean": sum(values) / len(values),
        "p50": pick(0.5),
        "p95": pick(0.95),
        "max": values[-1],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine-dir", type=Path, required=True)
    parser.add_argument("--torchscript-dir", type=Path, default=None,
                        help=hub.HELP)
    parser.add_argument("--clip-dir", type=Path, required=True)
    parser.add_argument("--long-files", type=Path, nargs="*", default=[])
    parser.add_argument("--max-clips", type=int, default=0)
    parser.add_argument("--parity", type=int, default=200)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rng = random.Random(0)

    paths = sorted(args.clip_dir.glob("*.flac")) + sorted(
        args.clip_dir.glob("*.wav")
    )
    if args.max_clips:
        paths = paths[: args.max_clips]
    clips = []
    for path in paths:
        waveform, rate = load(path)
        clips.append((path.name, waveform[0].contiguous(), rate))
    total_seconds = sum(w.numel() / r for _, w, r in clips)
    print(f"{len(clips)} clips, {total_seconds / 3600:.2f} h", flush=True)

    pipeline = SidonPipeline(args.engine_dir)
    report = {"clips": len(clips), "audio_hours": total_seconds / 3600}

    # -- all clips -------------------------------------------------------------
    problems = {}
    memory = []
    outputs = {}
    started = time.perf_counter()
    for index, (name, waveform, rate) in enumerate(clips):
        audio = pipeline.restore(waveform, rate)
        issue = check(audio, waveform.numel(), rate)
        if issue:
            problems[name] = issue
        outputs[name] = audio
        if (index + 1) % 500 == 0 or index + 1 == len(clips):
            memory.append(used_mb())
            print(f"  {index + 1}/{len(clips)} memory {memory[-1]:.0f} MB "
                  f"graphs {len(pipeline._graphs)}", flush=True)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    # second pass: every bucket's CUDA graph already captured
    started = time.perf_counter()
    for name, waveform, rate in clips:
        pipeline.restore(waveform, rate)
    torch.cuda.synchronize()
    warm = time.perf_counter() - started
    report["all_clips"] = {
        "wall_s": elapsed,
        "rtf": elapsed / total_seconds,
        "x_realtime": total_seconds / elapsed,
        "warm_wall_s": warm,
        "warm_rtf": warm / total_seconds,
        "warm_x_realtime": total_seconds / warm,
        "problems": problems,
        "memory_mb_first_last_max": [memory[0], memory[-1], max(memory)],
        "graphs": len(pipeline._graphs),
    }
    print("all", json.dumps(report["all_clips"])[:400], flush=True)

    # -- determinism ---------------------------------------------------------
    part = rng.sample(clips, min(64, len(clips)))
    report["determinism_max_abs"] = max(
        (pipeline.restore(w, r) - outputs[n]).abs().max().item()
        for n, w, r in part
    )
    print("determinism", report["determinism_max_abs"], flush=True)

    # -- parity vs the official demo ----------------------------------------
    demo = DemoSidon(args.torchscript_dir, device="cuda")
    parity = []
    per_clip = {}
    for name, waveform, rate in rng.sample(clips, min(args.parity, len(clips))):
        official = demo.restore(waveform, rate).float().cuda()
        distance = log_mel_distance(official, outputs[name])
        parity.append(distance)
        per_clip[name] = {
            "logmel": distance,
            "seconds": waveform.numel() / rate,
            "input_rms_db": 20 * torch.log10(
                waveform.float().pow(2).mean().sqrt() + 1e-9
            ).item(),
            "output_rms_db": 20 * torch.log10(
                official.pow(2).mean().sqrt() + 1e-9
            ).item(),
        }
    report["parity_logmel_vs_official"] = stats(parity)
    report["parity_per_clip"] = per_clip
    print("parity", json.dumps(report["parity_logmel_vs_official"]), flush=True)

    # -- long multi-chunk recordings, per channel ----------------------------
    report["long"] = {}
    for path in args.long_files:
        waveform, rate = load(path)
        seconds = waveform.shape[-1] / rate
        torch.cuda.synchronize()
        started = time.perf_counter()
        audio = pipeline.restore(waveform, rate, per_channel=True)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        official = demo.restore(waveform, rate, per_channel=True)
        entry = {
            "seconds": seconds,
            "channels": int(waveform.shape[0]),
            "wall_ms": elapsed * 1e3,
            "rtf": elapsed / (seconds * waveform.shape[0]),
            "problems": check(audio, waveform.shape[-1], rate),
            "logmel_vs_official": log_mel_distance(
                official.float().cuda(), audio
            ),
        }
        report["long"][path.name] = entry
        print(path.name, json.dumps(entry), flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print("STRESS_OK", flush=True)


if __name__ == "__main__":
    main()
