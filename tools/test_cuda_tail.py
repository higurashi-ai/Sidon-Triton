#!/usr/bin/env python3
"""GPU check of the fused CUDA decoder tail on real weights/activations.

- every tail stage (residual units, fused upsamples) vs its fp32 torch
  module on real activations (SNR), and its time at the full window;
- the whole tail vs the fp32 torch tail (SNR) and its time.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sidon_triton import cuda_tail  # noqa: E402
from sidon_triton.engines import errors, timed_cuda  # noqa: E402
from sidon_triton.modeling import load_bundle  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=120.0)
    parser.add_argument("--probe-frames", type=int, default=200)
    parser.add_argument("--dtype", choices=["fp16", "bf16"], default="fp16")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    torch.manual_seed(0)
    torch.set_float32_matmul_precision("highest")
    bundle = load_bundle(args.bundle, device="cuda")
    decoder = bundle.decoder
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    tail = cuda_tail.CudaDecoderTail(decoder, dtype=dtype).cuda().eval()
    from_block = tail.from_block
    frames = int(args.seconds * 50)
    result = {"from_block": from_block, "stages": {}}

    with torch.no_grad():
        hidden = torch.randn(1, args.probe_frames, 1024, device="cuda")
        z = cuda_tail.reference_head(decoder, hidden)
        # torch twins of the tail stages, in order
        twins = []
        for index in range(from_block, len(decoder.blocks)):
            block = decoder.blocks[index]
            if index > from_block:
                twins.append(
                    (f"block{index}.upsample",
                     lambda t, b=block: b.upsample(b.snake(t)))
                )
            for unit_index, unit in enumerate(block.res):
                twins.append((f"block{index}.res{unit_index}", unit))
        x = z
        for (name, twin), stage in zip(twins, tail.stages):
            reference = twin(x)
            xin = cuda_tail.time_major(x.to(dtype))
            lens = torch.full((1,), xin.shape[1], dtype=torch.int32,
                              device="cuda")
            actual = stage(xin, lens).float()
            entry = errors(reference, actual.transpose(1, 2))
            scale = x.shape[-1] / args.probe_frames
            big = torch.randn(
                1, int(frames * scale), x.shape[1], device="cuda"
            ).to(dtype)
            entry["T"] = big.shape[1]
            big_lens = torch.full((1,), big.shape[1], dtype=torch.int32,
                                  device="cuda")
            entry["ms"] = timed_cuda(lambda: stage(big, big_lens), 3, 20)[
                "median_ms"
            ]
            result["stages"][name] = entry
            print(name, json.dumps(entry), flush=True)
            x = reference
            del big

        reference = cuda_tail.reference_tail(decoder, z)
        actual = tail(cuda_tail.time_major(z.to(dtype)))
        assert actual.shape == reference.shape, (actual.shape, reference.shape)
        result["tail"] = errors(reference, actual)
        assert z.shape[-1] == cuda_tail.head_samples(args.probe_frames)
        big = torch.randn(
            1, cuda_tail.head_samples(frames), z.shape[1], device="cuda"
        ).to(dtype)
        result["tail"]["ms"] = timed_cuda(lambda: tail(big), 3, 20)[
            "median_ms"
        ]
        print("tail", json.dumps(result["tail"]), flush=True)
    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print("TEST_CUDA_TAIL_OK", flush=True)


if __name__ == "__main__":
    main()
