#!/usr/bin/env python3
"""Speed of the official Sidon v0.1 path (TorchScript fp32 + CPU extractor).

Per window length: CPU fbank (transformers), TorchScript encoder and
decoder on GPU (CUDA-event timed). Plus the demo's full ``restore`` on a
long input (tiled from --wav) with its wall-clock breakdown, the same
random short clips as ``bench_e2e.py --clips`` and the device memory used
after loading.

Run WITHOUT NVIDIA_TF32_OVERRIDE to measure the official path as users run
it (PyTorch defaults: fp32 matmul, TF32 cuDNN convolutions).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sidon_triton import hub  # noqa: E402
from sidon_triton import frontend  # noqa: E402
from sidon_triton.audio_io import read_wav  # noqa: E402
from sidon_triton.demo import DemoSidon  # noqa: E402
from sidon_triton.engines import timed_cuda  # noqa: E402


def tiled(wav: Path | None, seconds: float):
    if wav is None:
        torch.manual_seed(0)
        return torch.randn(int(seconds * 16_000)) * 0.1, 16_000
    data, rate = read_wav(wav)
    data = torch.from_numpy(data)
    total = int(seconds * rate)
    repeats = -(-total // data.shape[-1])
    return data.repeat(1, repeats)[:, :total].contiguous(), rate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--torchscript-dir", type=Path, default=None,
                        help=hub.HELP)
    parser.add_argument("--wav", type=Path, default=None)
    parser.add_argument(
        "--seconds",
        type=lambda s: [int(v) for v in s.split(",")],
        default=[15, 30, 60, 96, 120],
    )
    parser.add_argument("--e2e-seconds", type=float, default=600.0)
    parser.add_argument("--clips", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    def device_used_mb() -> float:
        free, total = torch.cuda.mem_get_info()
        return (total - free) / 1e6

    baseline_mb = device_used_mb()
    demo = DemoSidon(args.torchscript_dir, device="cuda")
    after_init_mb = device_used_mb()
    data, rate = tiled(args.wav, max(args.seconds))
    wav16 = frontend.preprocess(data, rate)
    result = {
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "tf32_cudnn": torch.backends.cudnn.allow_tf32,
        "baseline_device_memory_mb": baseline_mb,
        "after_init_device_memory_mb": after_init_mb,
        "lengths": {},
    }
    with torch.inference_mode():
        for seconds in args.seconds:
            chunk = wav16[: seconds * 16_000]
            padded = torch.nn.functional.pad(chunk, (160, 160)).numpy()
            started = time.perf_counter()
            for _ in range(args.runs):
                features = demo.preprocessor(
                    padded, sampling_rate=16_000, return_tensors="pt"
                )["input_features"]
            fbank_ms = (time.perf_counter() - started) * 1e3 / args.runs
            features = features.cuda()
            hidden = demo.fe(features)["last_hidden_state"]
            entry = {
                "fbank_cpu_ms": fbank_ms,
                "encoder": timed_cuda(
                    lambda: demo.fe(features), args.warmup, args.runs
                ),
                "decoder": timed_cuda(
                    lambda: demo.decoder(hidden.transpose(1, 2)),
                    args.warmup, args.runs,
                ),
                "peak_memory_gb": torch.cuda.max_memory_allocated() / 1e9,
            }
            entry["gpu_ms"] = (
                entry["encoder"]["median_ms"] + entry["decoder"]["median_ms"]
            )
            entry["window_ms"] = entry["gpu_ms"] + fbank_ms
            entry["gpu_rtf"] = entry["gpu_ms"] / 1e3 / seconds
            result["lengths"][f"{seconds}s"] = entry
            print(f"{seconds}s: {json.dumps(entry)}", flush=True)
            torch.cuda.reset_peak_memory_stats()

    data, rate = tiled(args.wav, args.e2e_seconds)
    demo.restore(data[..., : rate * 30], rate)  # warm-up
    started = time.perf_counter()
    demo.restore(data, rate)
    torch.cuda.synchronize()
    wall = time.perf_counter() - started
    result["e2e"] = {
        "audio_seconds": data.shape[-1] / rate,
        "input_rate": rate,
        "wall_ms": wall * 1e3,
        "rtf": wall / (data.shape[-1] / rate),
        "breakdown_ms": {k: v * 1e3 for k, v in demo.timings.items()},
    }
    # as the demo runs: no per-stage sync, CPU fbank overlaps the GPU
    demo.profile = False
    walls = []
    for _ in range(3):
        torch.cuda.synchronize()
        started = time.perf_counter()
        demo.restore(data, rate)
        torch.cuda.synchronize()
        walls.append(time.perf_counter() - started)
    wall = sorted(walls)[1]
    result["e2e_overlapped"] = {
        "wall_ms": wall * 1e3,
        "rtf": wall / (data.shape[-1] / rate),
    }
    if args.clips:  # same clips as bench_e2e.py (seed 0), demo restore each
        rng = random.Random(0)
        source, rate = tiled(args.wav, 64.0)
        clips = []
        for _ in range(args.clips):
            seconds = rng.uniform(1.5, 25.0)
            start = rng.randint(0, source.shape[-1] - int(seconds * rate) - 1)
            clips.append(source[0, start : start + int(seconds * rate)].clone())
        total = sum(c.shape[-1] for c in clips) / rate
        walls = []
        for _ in range(3):  # first pass is warm-up
            torch.cuda.synchronize()
            started = time.perf_counter()
            for clip in clips:
                demo.restore(clip, rate)
            torch.cuda.synchronize()
            walls.append(time.perf_counter() - started)
        wall = sorted(walls[1:])[0]
        result["clips"] = {
            "count": len(clips),
            "audio_seconds": total,
            "wall_ms": wall * 1e3,
            "rtf": wall / total,
            "device_memory_mb": device_used_mb(),
        }
    print(json.dumps(result, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print("BENCH_BASELINE_OK", flush=True)


if __name__ == "__main__":
    main()
