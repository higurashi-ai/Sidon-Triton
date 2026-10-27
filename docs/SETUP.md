# Setup and pitfalls

## Environment

Tested with Python 3.12, torch 2.8 (cu128), triton 3.4, transformers
4.56.1, tensorrt-cu12 10.16, onnx 1.22 on H100, with the CUDA 12.9
toolkit. TensorRT and onnx are only used by `--encoder trt` /
`--decoder cuda|trt`; the default build and runtime never import them.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .        # pip install -e ".[trt]" for the TensorRT back-ends
cp configs/env.example.sh configs/env.local.sh   # optional: site overrides
source configs/env.local.sh                      # or env.example.sh
```

`configs/env.example.sh` sets `NVIDIA_TF32_OVERRIDE=0` (required, see
pitfall 1), puts the pip TensorRT libraries on `LD_LIBRARY_PATH` and
chooses the Python (`STRITON_PYTHON`, default `python3` on `PATH`). The
workflow scripts in `scripts/workflows/` source `env.local.sh` if present,
else `env.example.sh`.

`--decoder triton-cuda` and `--decoder cuda` compile the CUDA-tail
kernels with nvcc on first use
(`build/sidon_triton_dac_<hash>.so`, keyed by source + flags; the visible
GPU's arch unless `STRITON_CUDA_ARCHS` is set, e.g. `90`). nvcc must be on
`PATH` or under `CUDA_HOME`; no ninja needed. The default (Triton) and
`--decoder trt` need no CUDA toolkit.

## Weights

`tools/convert_torchscript.py --output bundle/` downloads the released
TorchScript files (`sarulab-speech/sidon-v0.1`, pinned revision, see
`sidon_triton/hub.py`) — or reads them from `--torchscript-dir` — and writes
`encoder.safetensors`, `decoder.safetensors` and their configs after a
parity check against the TorchScript (asserts > 80 dB). Every tool that
runs the official model as a reference takes the same optional
`--torchscript-dir`.

## Engines

`scripts/build_engines.py --bundle bundle/ --engine-dir E --onnx-dir /tmp/x`
(GPU required) writes, by default, `encoder_triton.pt` (re-laid-out
encoder weights), `decoder_head.pt` and `decoder_tail_triton.pt`
(Triton head / tail weights) — no ONNX, no engines. `--decoder
triton-cuda` writes `decoder_tail.pt` (packed CUDA-tail weights) instead
of `decoder_tail_triton.pt`.
`--encoder trt` builds `encoder.plan` instead of `encoder_triton.pt`,
`--decoder cuda` builds the head as `decoder.plan`. The longest
chunk is the demo's 96 s (`--max-seconds`). `build_report.json` holds
accuracy vs the fp32 torch reference and the padding-invariance check.
`--decoder trt` builds the whole decoder in TensorRT (no custom kernels;
strongly typed fp16 with fp32 islands from `--decoder-fp32`, default
`final` — the last conv + tanh in fp16 was the whole accuracy gap);
`--encoder-precision` (with `--encoder trt`) fp16 (default) / bf16 /
fp32. Engines are specific to the GPU architecture and TensorRT version
they were built with.

## Workflows

`scripts/workflows/*.sh` chain the tools; paths come from arguments and
`STRITON_*` variables (`STRITON_BUNDLE`, `STRITON_ENGINE_DIR`, `STRITON_WORK`,
`STRITON_TORCHSCRIPT`, `STRITON_DECODER`, ...; defaults in `common.sh`):

| script | |
|---|---|
| `build_and_bench.sh [eval.wav ...]` | convert (once), build, benchmark, parity with the official demo |
| `bench_baseline.sh [input.wav]` | speed of the official path |
| `smoke.sh input.wav ...` | kernel test, benchmark, file restoration, parity |
| `stress.sh clip_dir [long_file ...]` | every clip of a corpus + parity subset |
| `stream_bench.sh input_dir` | disk-to-disk throughput of `scripts/run.py` |
| `decoder_precision_sweep.sh eval.wav ...` | `--decoder trt` precision variants, one per GPU |
| `listening.sh out_dir input.wav ...` | A/B listening pairs and stage-by-stage variants |
| `window_compare.sh out_dir input.wav ...` | 96 s chunks vs 20 s sliding windows |

`scripts/examples/build_and_bench.pbs` shows how to wrap one in a batch
job.

## Pitfalls

1. `NVIDIA_TF32_OVERRIDE=0` is required: the fbank GEMM must be true fp32
   and TensorRT refuses engines built under another TF32 setting.
2. Register the TensorRT plugins (`flexrel_triton.register_*`,
   `snake_triton.register_trt_plugin`) before creating any TensorRT
   builder or runtime (it segfaults otherwise), and enqueue engines on a
   non-default CUDA stream.
3. TensorRT does not fuse the exported Snake (`x + inv * sin(a x)^2`):
   5 full passes per activation. TRT parts use the `sidon::Snake` plugin;
   the CUDA tail fuses it into its kernels.
4. TensorRT bf16 (weakly typed) is as slow as fp32 for this decoder, and a
   strongly-typed bf16 decoder fails in Myelin (`stride_order` CHECK). The
   decoder engines are fp16.
5. Keep the encoder's feature projection fp32 (done in export): on silent
   frames its LayerNorm divides by a ~0 variance and amplifies fp16
   rounding (up to 14 dB log-mel off the official output on short clips).
6. Bucket masks are engine **inputs**: building them in-graph
   (Range/Less on a length tensor) made TRT 10.16 fp16 output all zeros.
7. TensorRT AOT plugins (tensorrt.plugin) cap shared memory at 48 KiB —
   that is why the fused residual units are CUDA torch ops, not plugins.
   Also: no `from __future__ import annotations` in plugin modules, and
   int plugin attributes break on numpy 2.x (`int(f.data)`).
8. CUDA graphs share one memory pool (`SidonPipeline._pool`); the tail
   allocates inside the graphs, so memory grows with the number of
   distinct length buckets seen (~13 GB with all of them).
9. torchaudio's decode/encode hold the GIL: `FileStreamer` uses spawned
   worker processes, so scripts using it need an `if __name__ ==
   "__main__":` guard.
10. Parity with the official demo needs its exact preprocessing: the HF
    `SeamlessM4TFeatureExtractor` fbank (waveform × 2^15, `var(ddof=1) +
    1e-7`, an odd frame count padded with 1.0 — the w2v-bert-2.0
    `padding_value`), 1.5 s tail pad, 96 s chunks normalized
    independently, previous chunk's last feature frame prepended, last
    960 samples of each chunk dropped.
