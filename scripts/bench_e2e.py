#!/usr/bin/env python3
"""Benchmark the pipeline.

- per window length: stage times (fbank front-end, encoder, decoder head,
  decoder tail) and the bucketed CUDA-graph chunk;
- the demo-parity ``restore`` on a long input;
- many short clips of random length: throughput with bucketed graphs.
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

from sidon_triton import frontend  # noqa: E402
from sidon_triton.audio_io import read_wav  # noqa: E402
from sidon_triton.engines import timed_cuda  # noqa: E402
from sidon_triton.runtime import SidonPipeline  # noqa: E402


def tiled(wav: Path | None, seconds: float):
    if wav is None:
        torch.manual_seed(0)
        return torch.randn(1, int(seconds * 16_000)) * 0.1, 16_000
    data, rate = read_wav(wav)
    data = torch.from_numpy(data[:1])
    total = int(seconds * rate)
    repeats = -(-total // data.shape[-1])
    return data.repeat(1, repeats)[:, :total].contiguous(), rate


def wall(fn, runs: int = 3) -> float:
    fn()
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
    parser.add_argument("--wav", type=Path, default=None)
    parser.add_argument(
        "--seconds",
        type=lambda s: [int(v) for v in s.split(",")],
        default=[15, 30, 60, 96],
    )
    parser.add_argument("--e2e-seconds", type=float, default=600.0)
    parser.add_argument("--clips", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    def device_used_mb() -> float:
        free, total = torch.cuda.mem_get_info()
        return (total - free) / 1e6

    baseline_mb = device_used_mb()
    pipeline = SidonPipeline(args.engine_dir, use_graph=True)
    engines = pipeline.long
    data, rate = tiled(args.wav, max(args.seconds) + 2)
    wav16 = frontend.preprocess(data, rate).cuda()
    result = {
        "gpu": torch.cuda.get_device_name(0),
        "baseline_device_memory_mb": baseline_mb,
        "after_init_device_memory_mb": device_used_mb(),
        "lengths": {},
    }
    for seconds in args.seconds:
        samples = seconds * 16_000
        seq = frontend.num_features(samples)
        entry = {"seq": seq}
        with torch.cuda.stream(pipeline.compute), torch.no_grad():
            engines.wave[:samples].copy_(wav16[:samples])
            engines.wave_valid[:1].fill_(samples)
            engines.enqueue(1, samples, cached=False)  # fills every buffer
            current = torch.cuda.current_stream
            valid = engines.wave_valid[:1]
            wave = engines.wave_view(1, samples)
            entry["stage_frontend"] = timed_cuda(
                lambda: engines.front(wave, valid), args.warmup, args.runs
            )
            entry["stage_encoder"] = timed_cuda(
                lambda: engines.run_encoder(1, seq, current()),
                args.warmup, args.runs,
            )
            dec_input = engines.dec_input[: seq * 1024].view(1, seq, 1024)
            frames = engines.dec_lens[:1]
            entry["stage_decoder_head"] = timed_cuda(
                lambda: engines.run_head(dec_input, frames, current()),
                args.warmup, args.runs,
            )
            if not engines.trt_decoder:
                z = engines.run_head(dec_input, frames, current())
                entry["stage_decoder_tail"] = timed_cuda(
                    lambda: engines.run_tail(z, frames),
                    args.warmup, args.runs,
                )

            def chunk():
                engines.wave[:samples].copy_(wav16[:samples])
                pipeline.run_chunk(samples, cached=False)

            entry["chunk_graph"] = timed_cuda(chunk, args.warmup, args.runs)
        torch.cuda.synchronize()
        entry["rtf"] = entry["chunk_graph"]["median_ms"] / 1e3 / seconds
        entry["device_memory_mb"] = device_used_mb()
        result["lengths"][f"{seconds}s"] = entry
        print(f"{seconds}s: {json.dumps(entry)}", flush=True)

    data, rate = tiled(args.wav, args.e2e_seconds)
    audio_seconds = data.shape[-1] / rate
    ms = wall(lambda: pipeline.restore(data, rate))
    result["e2e"] = {
        "audio_seconds": audio_seconds,
        "input_rate": rate,
        "wall_ms": ms,
        "rtf": ms / 1e3 / audio_seconds,
    }
    print("e2e", json.dumps(result["e2e"]), flush=True)

    if args.clips:  # many short clips of random length (bucketed graphs)
        rng = random.Random(0)
        source, rate = tiled(args.wav, 64.0)
        clips = []
        for _ in range(args.clips):
            seconds = rng.uniform(1.5, 25.0)
            start = rng.randint(0, source.shape[-1] - int(seconds * rate) - 1)
            clips.append(source[0, start : start + int(seconds * rate)].clone())
        total = sum(c.numel() for c in clips) / rate
        ms = wall(lambda: [pipeline.restore(c, rate) for c in clips], 2)
        result["clips"] = {
            "count": len(clips),
            "audio_seconds": total,
            "wall_ms": ms,
            "rtf": ms / 1e3 / total,
            "device_memory_mb": device_used_mb(),
            "graphs": len(pipeline._graphs),
        }
        print("clips", json.dumps(result["clips"]), flush=True)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print("BENCH_E2E_OK", flush=True)


if __name__ == "__main__":
    main()
