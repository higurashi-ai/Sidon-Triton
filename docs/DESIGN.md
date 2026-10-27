# Design notes (measured and dropped)

Measured on H100, 96 s chunks unless noted.

| tried | outcome |
|---|---|
| TensorRT encoder (fp16 engine) | 52 dB from fp32, 1/3 of its time in pointwise / layout kernels -> Triton encoder (fp32 residual stream, kernels from FastDialogueSidon): 67 dB, 1.2-1.9x faster |
| TensorRT decoder head | half of its 11.7 ms / 96 s in NCHW<->NHWC reformats, Snake and pointwise kernels -> Triton head 7.5 ms |
| Triton convs with Snake applied on the GEMM input load | 2.6x the cuBLAS time -> each Snake computed once, in the producer's epilogue |
| Triton tail units with C split in 3 pow2 chunks | C = 96 / 48 1.5x / 1.9x slower than the CUDA tail -> one GEMM over K = (tap, channel), N padded in registers only: on par with CUDA (16.0 vs 15.7 ms / 96 s) |
| bf16 encoder | 0.4-1.1 dB off official |
| fp16 feature projection | up to 14 dB off on short clips (silent frames: LayerNorm variance ~0) -> fp32 |
| bf16 tail | 18 dB less accurate than fp16 at the same speed |
| batching short clips | ~1.3x on real short clips, none on mixed lengths, +14 GB; removed |
| 20 s sliding windows (training length) | 16% slower; differs from 96 s chunking by 1.4-3.2 dB (window effect) |

Earlier TensorRT / CUDA-kernel iterations:

| tried | outcome |
|---|---|
| TensorRT decoder with Snake exported as ops | TRT does not fuse Snake: 57% of decoder time -> `sidon::Snake` Triton TRT plugin |
| fused residual unit as Triton TRT AOT plugin | slower than TRT convs (pow2 padding, 48 KiB AOT smem cap) |
| CUDA residual units, WMMA, channel-major | 8-10 ms/unit -> time-major + mma.sync (1-2 ms/unit) |
| TensorRT bf16 decoder | as slow as fp32 (no fast bf16 conv tactics); strongly typed fails in Myelin |
| TensorRT decoder in fp16 | 0.19-0.32 dB off official -> fp32 final Snake/conv/tanh: 0.08-0.13 dB (+1%); more fp32 (block 4, residual stream) gains nothing, +20-30% |
| all-fp32 engines | 0.01-0.03 dB off official, 6.3x slower than fp16 |
| in-graph length masks in TensorRT | TRT 10.16 fp16 produced all-zero output -> masks are inputs |
| wgmma/TMA CUDA tail kernels | not done: est. ~2x on the tail |
