# Sidon-Triton

Sidon-Triton is a fast inference implementation of
[Sidon v0.1](https://huggingface.co/sarulab-speech/sidon-v0.1) speech
restoration (any-rate noisy speech in, 48 kHz restored speech out),
implemented by Rikuto Kotoge. It is built by **converting the released
weights — no retraining** — and matches the official demo's output to
within ~0.03 dB log-mel.

Everything runs on Triton kernels + cuBLAS (encoder, decoder head and
48 kHz decoder tail): only torch + Triton are needed — no TensorRT, no
CUDA toolkit. Optional TensorRT / hand-written CUDA back-ends are kept for
comparison (see [Other back-ends](#other-back-ends)).

## Official Sidon vs Sidon-Triton (H100 SXM, batch 1)

Official = the released TorchScript (fp32) with the demo's CPU fbank, run
as the demo runs it — no per-stage synchronization, so the CPU fbank of
the next chunk overlaps the GPU work of the previous one
(`scripts/bench_baseline.py`; with a sync after every stage, for the
breakdown, it takes 3462 ms). Sidon-Triton = `scripts/bench_e2e.py`, one
CUDA graph per chunk (front-end + encoder + decoder).

| | official | Sidon-Triton |
|---|---|---|
| 600 s input, end to end | 2677 ms | **202 ms (13.3x)** |
| 128 clips 1.5-25 s (29 min) | 5333 ms | **828 ms (6.4x)** |
| device memory after init | 1.7 GB | 1.3 GB |
| device memory after the 128 clips | 16.2 GB | 6.6 GB |
| log-mel L1 vs official (5 noisy mixtures) | 0 | **0.02-0.03 dB** |

Per chunk length (milliseconds; official: GPU stages + CPU fbank of that
chunk, serial; Sidon-Triton encoder times are CUDA-graph replays):

| chunk | official enc / dec / CPU fbank | **official total** | Sidon-Triton enc / dec head / dec tail | **Sidon-Triton total** | RTF |
|---|---|---|---|---|---|
| 15 s | 13.8 / 17.8 / 23.7 | **55** | 1.3 / 1.5 / 2.7 | **5.5** | 3.6e-4 |
| 30 s | 33.8 / 32.7 / 45.2 | **112** | 2.2 / 2.6 / 5.1 | **9.8** | 3.3e-4 |
| 60 s | 102.7 / 63.9 / 91.1 | **258** | 4.4 / 5.0 / 9.9 | **19.2** | 3.2e-4 |
| 96 s | 234.3 / 109.9 / 144.3 | **489** | 7.6 / 7.7 / 16.0 | **31.1** | 3.2e-4 |

(The front-end — fbank on GPU — is 0.33 ms per chunk. Official
preprocessing of a 600 s file adds 410 ms CPU; Sidon-Triton's GPU
preprocessing takes 3 ms.)

File throughput with `scripts/run.py` (decode / encode in worker
processes): ~1040x realtime disk to disk over a few thousand speech files
(FLAC, ~9 h in total), GPU busy 98% — measured with an earlier, ~15% slower
TensorRT build.

**Why batch 1.** From 15 s up, a single chunk already gives every kernel
far more parallel work than the GPU has SMs (the 48 kHz decoder stages
alone are millions of rows) and CUDA graphs remove the launch overhead,
so there is nothing left for batching to fill: the cost per audio second
is flat from 30 s to 96 s chunks (RTF 3.3e-4 → 3.2e-4 above). Batching
mixed lengths also pads every item to the longest one. Measured with a
batched set: no gain on mixed 1.5-25 s clips or long files, ~1.3x only on
uniformly short clips (~5.8 s average), for +14 GB of memory — so it was
removed.

**Chunks longer than 96 s.** Longer chunks are accepted
(`build_engines.py --max-seconds`, `run.py --chunk-seconds`), but this is
not validated in the current version and gains little: with the cost per
second already flat, a longer chunk only removes a few chunk boundaries,
while moving the output further from the official 96 s chunking.

## Accuracy

- Conversion is exact: encoder 126 dB, decoder bit-identical vs the
  released TorchScript (fp32), `tools/convert_torchscript.py`.
- vs the fp32 torch reference: encoder hidden 66.6-67.1 dB (fp16 GEMM
  operands, fp32 residual stream / norms / softmax, fp32 feature
  projection), decoder head 61 dB, tail 50 dB; audio waveform SNR 43 dB
  (fp16 phase noise; the log-mel distance is the meaningful number).
- vs the official demo output (log-mel L1 on speech frames):
  **mean 0.026 / p95 0.042 / max 0.072 dB** over 200 random speech clips;
  0.02 dB on long 2-channel dialogue recordings (`--per-channel`).
- For scale: re-chunking the official model itself (20 s instead of 96 s
  windows) moves its output by 1.4-3.2 dB.
- Large-scale run (thousands of clips + long 2-channel recordings): no non-finite,
  clipped or wrong-length outputs; bit-identical across repeated runs.
- Listening: `tools/make_listening_set.py` writes named and blind A/B pairs
  (official vs Sidon-Triton, also as stereo L=official / R=Sidon-Triton);
  `tools/make_listening_variants.py` adds fp32 / bf16 variants.

## How

```
wav (any rate/channels)
 └ GPU preprocess: peak 0.9, 50 Hz high-pass (fp64 FFT conv), 16 kHz resample
 └ demo chunking: +1.5 s tail, 96 s chunks, 1-frame feature carry-over
   └ fp32 SeamlessM4T fbank (torch)                         ┐
   └ encoder — Triton + cuBLAS: w2v-BERT 8 layers, LoRA     │ one CUDA graph
       merged; fp16 GEMMs accumulating into an fp32         │ per length bucket
       residual stream, fused norm / GLU / causal dwconv,   │ (1 s steps <= 32 s,
       flash attention with in-kernel relative_key bias and │
       key-length mask                                      │
   └ decoder:                                               │  4 s above)
       head — Triton GEMM kernels, time-major: input conv,  │
         blocks 0-1, block 2 upsample; each epilogue adds   │
         bias (+ residual) and writes the Snake of its      │
         output for the next conv                           │
       tail — Triton: each residual unit of blocks 2-4 is   │
         one kernel (dilated conv as one GEMM over a        │
         (tap, channel) K axis -> Snake -> 1x1 -> residual; │
         writes the next Snake'd input too), one kernel per │
         upsample, the final conv reduced in the last       │
         unit's epilogue (fp32) + tanh                      ┘
```

- **Exact bucketing**: a chunk zero-padded to its bucket gives the
  exact-length output — fbank statistics over valid frames only, attention
  keys past the valid length masked, every decoder conv re-zeroed past it
  (valid lengths are kernel inputs). Any mix of input lengths reuses
  ~30-50 captured graphs.
- **File throughput**: `sidon_triton.stream.FileStreamer` decodes and encodes
  in spawned worker processes overlapped with the GPU.

## GPU support

Built and measured on H100 only. The Triton kernels JIT-compile on first
use; their tile sizes were tuned on H100 and the largest (C = 192 tail
units: 128 x 256 tiles, 3 stages) assume Hopper-class shared memory, so
other GPUs may need smaller tiles (`UNIT_CONFIG` / `UP_CONFIG` in
`triton_tail.py`, `conv_config` in `triton_head.py`).

## Quick start

```bash
git clone <this repo> Sidon-Triton && cd Sidon-Triton
python -m venv .venv && source .venv/bin/activate && pip install -e .
source configs/env.example.sh       # NVIDIA_TF32_OVERRIDE=0 (required)
python tools/convert_torchscript.py --output bundle/   # downloads sidon-v0.1 from the Hub
python scripts/build_engines.py --bundle bundle/ --engine-dir engines/ \
    --onnx-dir /tmp/sidon-triton-onnx  # packs the kernel weights + validates
python scripts/run.py --engine-dir engines/ --input-dir corpus/ \
    --output-dir restored/ --format flac [--per-channel]
```

Or in one go, with a parity check against the official model on your own
inputs: `scripts/workflows/build_and_bench.sh eval1.wav eval2.wav`.
See `docs/SETUP.md` for the environment, the other workflows and pitfalls,
and `docs/DESIGN.md` for what was measured and dropped.

The runtime detects the back-end from the files in the engine directory.

```python
from sidon_triton.runtime import SidonPipeline
pipeline = SidonPipeline("engines/")
audio48k = pipeline.restore(waveform, sample_rate)              # demo chunking
stereo = pipeline.restore(two_channel, sr, per_channel=True)    # 1 speaker / channel
sliding = pipeline.restore_sliding(waveform, sr, window_seconds=20)

from sidon_triton.stream import FileStreamer                       # many files
with FileStreamer(pipeline, readers=8, writers=8) as streamer:  # spawned workers:
    stats = streamer.run(paths, "restored/", fmt="flac")        # use a __main__ guard
```

`--per-channel` restores each channel separately (2-channel dialogue,
one speaker per channel, channel layout kept); the default downmixes like
the official demo. See `docs/SETUP.md` for setup details and pitfalls.

## Other back-ends

Kept for comparison, selected at build time (`pip install -e ".[trt]"`
for TensorRT; nvcc for the CUDA tail). H100, batch 1:

| build flags | encoder / decoder head / tail | 600 s | log-mel L1 (200 clips mean / max) |
|---|---|---|---|
| default | Triton / Triton / Triton | **202 ms** | **0.026 / 0.072 dB** |
| `--decoder triton-cuda` | Triton / Triton / CUDA (`csrc/dac_kernels.cu`, mma.sync, Hopper smem) | 203 ms | 0.029 / 0.088 dB |
| `--encoder trt --decoder cuda` | TensorRT fp16 / TensorRT / CUDA | 238 ms | 0.11 / 0.42 dB |
| `--encoder trt --decoder trt` | TensorRT fp16 / TensorRT (whole decoder) | 404 ms | 0.08-0.13 dB (5 mixtures) |

The TensorRT encoder is 52 dB from the fp32 reference (Triton: 67 dB) and
was the main deviation from the official output. TensorRT engines are
architecture- and version-specific; see `docs/SETUP.md`.

## Repository layout

| path | |
|---|---|
| `src/sidon_triton/frontend.py` | demo preprocessing (CPU + GPU) and fbank, incl. bucket masking |
| `src/sidon_triton/modeling.py` | encoder / DAC decoder rebuilt as torch modules |
| `src/sidon_triton/export_onnx.py` | ONNX export for the TensorRT back-ends |
| `src/sidon_triton/runtime.py` | `SidonPipeline`: bucketed CUDA graphs, demo chunking, sliding windows |
| `src/sidon_triton/triton_encoder.py` | w2v-BERT encoder (Triton + cuBLAS) |
| `src/sidon_triton/triton_head.py`, `triton_tail.py` | DAC decoder head / tail (Triton) |
| `src/sidon_triton/stream.py`, `fileio.py`, `audio_io.py` | overlapped decode / restore / encode |
| `src/sidon_triton/cuda_tail.py`, `cuda_ext.py`, `csrc/dac_kernels.cu` | optional CUDA tail (nvcc build, no ninja) |
| `src/sidon_triton/engines.py`, `flexrel_triton.py`, `snake_triton.py` | optional TensorRT engines and plugins |
| `src/sidon_triton/demo.py` | the official demo pipeline (baseline + parity reference) |
| `tools/convert_torchscript.py` | TorchScript -> bundle with parity check |
| `tools/compare_demo.py`, `make_listening_*.py`, `make_window_compare.py` | parity and listening sets |
| `tools/stress_test.py`, `test_cuda_tail.py` | large-scale and kernel tests |
| `scripts/` | `build_engines.py` (weights / engines), `run.py`, `bench_e2e.py`, `bench_baseline.py` |
| `scripts/workflows/`, `scripts/examples/` | shell workflows (build, benchmarks, tests, listening) and a batch-job template |
| `src/sidon_triton/hub.py` | download of the released TorchScript files (pinned revision) |

## Acknowledgements

Sidon-Triton is an inference port of **Sidon** by sarulab-speech
(Wataru Nakata, Yuki Saito, Yota Ueda, Hiroshi Saruwatari); all model weights and the method are
theirs, and this repository only converts and accelerates the released
v0.1 model without retraining. If you use it, please cite the Sidon paper:

- Paper: [Sidon: Fast and Robust Open-Source Multilingual Speech
  Restoration for Large-scale Dataset Cleansing (ICASSP 2026, arXiv:2509.17052)](https://arxiv.org/abs/2509.17052)
- Model: [sarulab-speech/sidon-v0.1](https://huggingface.co/sarulab-speech/sidon-v0.1) (MIT)
- Demo whose preprocessing and chunking are reproduced here:
  [sarulab-speech/sidon_demo_beta](https://huggingface.co/spaces/sarulab-speech/sidon_demo_beta)

```bibtex
@inproceedings{nakata2026sidon,
  title={Sidon: Fast and robust open-source multilingual speech restoration for large-scale dataset cleansing},
  author={Nakata, Wataru and Saito, Yuki and Ueda, Yota and Saruwatari, Hiroshi},
  booktitle={ICASSP 2026-2026 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={17617--17621},
  year={2026},
  organization={IEEE}
}
```

Sidon builds on [w2v-BERT 2.0](https://huggingface.co/facebook/w2v-bert-2.0)
(Meta) and the [DAC](https://github.com/descriptinc/descript-audio-codec)
decoder architecture (Descript).
<!-- The encoder kernels are shared with FastDialogueSidon. -->

## License

MIT License, Copyright (c) 2026 Rikuto Kotoge (see `LICENSE`). The Sidon
model weights are released by sarulab-speech under their own MIT license.
