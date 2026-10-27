// Fused CUDA kernels for the DAC decoder tail of Sidon v0.1 (blocks 2-4:
// 8 -> 48 kHz). Activations are TIME-MAJOR [B, T, C] (channels contiguous),
// fp16 (or bf16 with -DSTRITON_BF16); accumulation and Snake math in fp32.
// Every kernel takes lens [B]: per-item valid rows of a bucket-padded batch
// (rows past it are read as zeros and written as zeros), so a padded run
// equals an exact-length run on the valid region.
//
// resunit_mma: one DAC residual unit
//     y = x + conv1x1(snake2(conv7_dilated(snake1(x)) + b1)) + b2
//   persistent CTAs, 128-row time tiles, raw mma.sync m16n8k16 + ldmatrix.
//   C = 48 / 96: W1 (7 taps) and W2 resident in shared memory; snake2 is
//   applied to the accumulators in registers and repacked directly as the
//   A operand of the 1x1 GEMM. C = 192: the 16 weight chunks (taps x
//   channel halves + W2) stream through a cp.async double buffer. The next
//   tile's x slab is prefetched into registers during the GEMMs. HBM
//   traffic per unit: x read once (+halo), y written once.
// upsample:    Snake + ConvTranspose1d(K = 2s, stride s, pad ceil(s/2)):
//   all s output phases of a 128-row input tile per CTA, taps streamed via
//   cp.async, the s*128 output rows written as one contiguous slab.
// final_head:  tanh(conv1d(snake(x), w[1, C, 7], pad 3) + b), C -> 1.
//
// Shared memory: up to ~210 KB per CTA (C = 96 resunit, 192->96 upsample)
// -> Hopper / datacenter Blackwell class GPUs (A100 / Ada do not fit).
//
// Registered as torch ops sidon_triton::{resunit_mma, upsample, final_head}
// (sidon_triton_bf16::* for the bf16 build).

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <torch/library.h>

#include <cuda_bf16.h>

// Element type: fp16 by default; -DSTRITON_BF16 builds the bf16 twin
// (registered as torch ops sidon_triton_bf16::*). Accumulation and Snake math
// are fp32 in both.
#ifdef STRITON_BF16
using elem = __nv_bfloat16;
using elem2 = __nv_bfloat162;
#define TO_F2 __bfloat1622float2
#define TO_F __bfloat162float
#define FROM_F2 __floats2bfloat162_rn
#define FROM_F __float2bfloat16
#define ELEM_AT at::BFloat16
#define ELEM_KIND at::kBFloat16
#define STRITON_MMA_T "bf16"
#else
using elem = __half;
using elem2 = __half2;
#define TO_F2 __half22float2
#define TO_F __half2float
#define FROM_F2 __floats2half2_rn
#define FROM_F __float2half
#define ELEM_AT at::Half
#define ELEM_KIND at::kHalf
#define STRITON_MMA_T "f16"
#endif

namespace {

constexpr int kWarps = 8;
constexpr int kMaxBatch = 64;  // lens cached in smem per kernel
constexpr int kThreads = kWarps * 32;

__host__ __device__ constexpr int align_up(int v, int a) {
  return (v + a - 1) / a * a;
}

__device__ __forceinline__ float snake_f(float x, float a, float inv) {
  const float s = __sinf(a * x);
  return x + inv * s * s;
}

union Vec8 {
  int4 raw;
  elem2 h2[4];
};

void check_time_major(const at::Tensor& x) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == ELEM_KIND && x.dim() == 3 &&
                  x.is_contiguous(),
              "x must be contiguous fp16/bf16 (per build) time-major [B, T, C]");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0,
              "x must be 16-byte aligned");
}

void check_lens(const at::Tensor& lens, const at::Tensor& x) {
  TORCH_CHECK(lens.is_cuda() && lens.scalar_type() == at::kInt &&
                  lens.dim() == 1 && lens.size(0) == x.size(0) &&
                  lens.is_contiguous(),
              "lens must be a contiguous int32 CUDA tensor [B]");
  TORCH_CHECK(x.size(0) <= kMaxBatch, "batch > ", kMaxBatch);
}


// ---- final head: tanh(conv(snake(x), w[1,C,7], pad 3) + b) ----------------

template <int C, int NT>
__global__ void __launch_bounds__(256)
final_head_kernel(const elem* __restrict__ x, const float* __restrict__ w,
                  const float* __restrict__ alpha,
                  const float* __restrict__ inv, float bias,
                  float* __restrict__ y, const int* __restrict__ lens, int T) {
  // per-item valid lengths cached in smem: a global load per tile sat on
  // the critical path of the next tile's prefetch (~0.3 ms per kernel)
  __shared__ int lens_s[kMaxBatch];
  for (int i = threadIdx.x; i < static_cast<int>(gridDim.y); i += blockDim.x)
    lens_s[i] = lens[i];
  __syncthreads();
  constexpr int ROWS = NT + 6;
  constexpr int CV = C / 8;
  __shared__ __align__(16) float hs[ROWS][C + 1];
  __shared__ float ws[7][C];
  for (int i = threadIdx.x; i < C * 7; i += blockDim.x)
    ws[i % 7][i / 7] = w[i];  // w is [1, C, 7]
  const int b = blockIdx.y;
  const int t0 = blockIdx.x * NT;
  const elem* xb = x + static_cast<long long>(b) * T * C;
  const int Tv = lens_s[b];  // valid samples of this item
  for (int v = threadIdx.x; v < ROWS * CV; v += blockDim.x) {
    const int r = v / CV;
    const int c0 = (v % CV) * 8;
    const int t = t0 - 3 + r;
    Vec8 data;
    data.raw = make_int4(0, 0, 0, 0);
    if (t >= 0 && t < Tv)
      data.raw = __ldg(reinterpret_cast<const int4*>(
          xb + static_cast<long long>(t) * C + c0));
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int c = c0 + 2 * j;
      const float2 f = TO_F2(data.h2[j]);
      const bool valid = t >= 0 && t < Tv;
      hs[r][c] = valid ? snake_f(f.x, __ldg(alpha + c), __ldg(inv + c)) : 0.f;
      hs[r][c + 1] =
          valid ? snake_f(f.y, __ldg(alpha + c + 1), __ldg(inv + c + 1)) : 0.f;
    }
  }
  __syncthreads();
  // 8 threads per output sample, each summing C/8 channels x 7 taps
  const int lane8 = threadIdx.x % 8;
  for (int n = threadIdx.x / 8; n < NT; n += blockDim.x / 8) {
    float acc = 0.f;
#pragma unroll
    for (int k = 0; k < 7; ++k) {
#pragma unroll
      for (int c = lane8; c < C; c += 8) acc += ws[k][c] * hs[n + k][c];
    }
#pragma unroll
    for (int offset = 4; offset > 0; offset >>= 1)
      acc += __shfl_down_sync(0xffffffff, acc, offset, 8);
    const int t = t0 + n;
    if (lane8 == 0 && t < T)
      y[static_cast<long long>(b) * T + t] = t < Tv ? tanhf(acc + bias) : 0.f;
  }
}

at::Tensor final_head(const at::Tensor& x, const at::Tensor& w,
                      const at::Tensor& alpha, const at::Tensor& inv,
                      double bias, const at::Tensor& lens) {
  check_time_major(x);
  check_lens(lens, x);
  TORCH_CHECK(x.size(2) == 48, "final_head expects C=48");
  const c10::cuda::CUDAGuard guard(x.device());
  const int batch = static_cast<int>(x.size(0));
  const int T = static_cast<int>(x.size(1));
  auto y = at::empty({x.size(0), 1, x.size(1)}, x.options().dtype(at::kFloat));
  constexpr int NT = 128;
  dim3 grid((T + NT - 1) / NT, batch);
  final_head_kernel<48, NT><<<grid, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const elem*>(x.data_ptr<ELEM_AT>()),
      w.data_ptr<float>(), alpha.data_ptr<float>(), inv.data_ptr<float>(),
      static_cast<float>(bias), y.data_ptr<float>(), lens.data_ptr<int>(), T);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}

// ---- resunit v2: raw mma.sync m16n8k16 with register-resident snake2 -------
// Each warp owns 16 time rows x all C output channels (C <= 96). GEMM1
// accumulators are biased + snake2'd in registers and repacked directly as
// the A operand of GEMM2 (the m16n8 accumulator layout equals the A-fragment
// k-halves), so u never round-trips through shared memory. Operands come
// from smem via ldmatrix (rows padded to C+8 -> 16B-aligned, conflict-free).

__device__ __forceinline__ void ldsm_x4(uint32_t (&r)[4], const void* p) {
  const uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(p));
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
      : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
      : "r"(a));
}

__device__ __forceinline__ void mma16816(float (&c)[4], const uint32_t (&a)[4],
                                         uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32." STRITON_MMA_T
      "." STRITON_MMA_T ".f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint32_t pack_half2(float lo, float hi) {
  const elem2 h = FROM_F2(lo, hi);
  return *reinterpret_cast<const uint32_t*>(&h);
}

__device__ __forceinline__ void cp_async16(void* smem_ptr, const void* gmem) {
  const uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(a),
               "l"(gmem));
}
__device__ __forceinline__ void cp_async_commit() {
  asm volatile("cp.async.commit_group;\n" ::);
}
__device__ __forceinline__ void cp_async_wait_all() {
  asm volatile("cp.async.wait_group 0;\n" ::: "memory");
}

// Stage snake1(x) for one tile from prefetched 16-byte vectors.
template <int C, int ROWS, int LD, int PER>
__device__ __forceinline__ void stage_snake(const int4 (&pre)[PER],
                                            elem* hs, const float* as,
                                            const float* is) {
  constexpr int CV = C / 8;
#pragma unroll
  for (int p = 0; p < PER; ++p) {
    const int v = threadIdx.x + p * kThreads;
    if (v < ROWS * CV) {
      const int r = v / CV;
      const int cv = v % CV;
      Vec8 data;
      data.raw = pre[p];
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const int c = cv * 8 + 2 * j;
        const float2 f = TO_F2(data.h2[j]);
        data.h2[j] = FROM_F2(snake_f(f.x, as[c], is[c]),
                                       snake_f(f.y, as[c + 1], is[c + 1]));
      }
      *reinterpret_cast<int4*>(hs + r * LD + cv * 8) = data.raw;
    }
  }
}

// Load one tile's x slab (+halo) into registers (zeros outside [0, T)).
template <int C, int ROWS, int HALO, int PER>
__device__ __forceinline__ void prefetch_tile(int4 (&pre)[PER],
                                              const elem* xb, int t0,
                                              int T) {
  // T: valid length of this batch item (rows >= T read as zeros)
  constexpr int CV = C / 8;
#pragma unroll
  for (int p = 0; p < PER; ++p) {
    const int v = threadIdx.x + p * kThreads;
    pre[p] = make_int4(0, 0, 0, 0);
    if (v < ROWS * CV) {
      const int t = t0 - HALO + v / CV;
      if (t >= 0 && t < T)
        pre[p] = __ldg(reinterpret_cast<const int4*>(
            xb + static_cast<long long>(t) * C + (v % CV) * 8));
    }
  }
}

// y = x + staged (16-byte vectors over the tile rows)
template <int C, int NT, int LD>
__device__ __forceinline__ void residual_store(const elem* xb, elem* yb,
                                               const elem* staged, int t0,
                                               int T, int Tv) {
  // rows in [Tv, T): bucket padding, written as zeros
  constexpr int CV = C / 8;
  for (int v = threadIdx.x; v < NT * CV; v += kThreads) {
    const int n = v / CV;
    const int cv = v % CV;
    const int t = t0 + n;
    if (t >= T) continue;
    const long long gi = static_cast<long long>(t) * C + cv * 8;
    if (t >= Tv) {
      *reinterpret_cast<int4*>(yb + gi) = make_int4(0, 0, 0, 0);
      continue;
    }
    Vec8 xr, o;
    xr.raw = __ldg(reinterpret_cast<const int4*>(xb + gi));
    o.raw = *reinterpret_cast<const int4*>(staged + n * LD + cv * 8);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const float2 a = TO_F2(xr.h2[j]);
      const float2 bb = TO_F2(o.h2[j]);
      xr.h2[j] = FROM_F2(a.x + bb.x, a.y + bb.y);
    }
    *reinterpret_cast<int4*>(yb + gi) = xr.raw;
  }
}

template <int C>
__device__ __forceinline__ void load_params(float* ps, const float* a1,
                                            const float* i1, const float* a2,
                                            const float* i2, const float* b1,
                                            const float* b2) {
  for (int c = threadIdx.x; c < C; c += kThreads) {
    ps[c] = a1[c];
    ps[C + c] = i1[c];
    ps[2 * C + c] = a2[c];
    ps[3 * C + c] = i2[c];
    ps[4 * C + c] = b1[c];
    ps[5 * C + c] = b2[c];
  }
}

template <int C, int NT, int D>
struct MmaLayout {
  static constexpr int LD = C + 8;
  static constexpr int HALO = 3 * D;
  static constexpr int ROWS = NT + 2 * HALO;
  static constexpr int N8 = C / 8;
  static constexpr int K16 = C / 16;
  static constexpr int PER = (ROWS * (C / 8) + kThreads - 1) / kThreads;
  static constexpr int W1_BYTES = 7 * C * LD * 2;
  static constexpr int W2_BYTES = C * LD * 2;
  static constexpr int H_BYTES = ROWS * LD * 2;
  static constexpr int P_BYTES = 6 * C * 4;
  static constexpr int W1_OFF = 0;
  static constexpr int W2_OFF = align_up(W1_OFF + W1_BYTES, 128);
  static constexpr int H_OFF = align_up(W2_OFF + W2_BYTES, 128);
  static constexpr int P_OFF = align_up(H_OFF + H_BYTES, 128);
  static constexpr int SMEM = align_up(P_OFF + P_BYTES, 128);
  static_assert(NT == 16 * kWarps, "one 16-row strip per warp");
  static_assert(C % 16 == 0, "k16 steps");
};

template <int C, int NT, int D>
__global__ void __launch_bounds__(kThreads)
resunit_mma_kernel(const elem* __restrict__ x, const elem* __restrict__ w1,
                   const elem* __restrict__ w2, const float* __restrict__ b1,
                   const float* __restrict__ a1, const float* __restrict__ i1,
                   const float* __restrict__ a2, const float* __restrict__ i2,
                   const float* __restrict__ b2, elem* __restrict__ y,
                   const int* __restrict__ lens, int batch, int T) {
  // per-item valid lengths cached in smem: a global load per tile sat on
  // the critical path of the next tile's prefetch (~0.3 ms per kernel)
  __shared__ int lens_s[kMaxBatch];
  for (int i = threadIdx.x; i < batch; i += blockDim.x)
    lens_s[i] = lens[i];
  __syncthreads();
  using L = MmaLayout<C, NT, D>;
  extern __shared__ __align__(128) unsigned char smem[];
  elem* w1s = reinterpret_cast<elem*>(smem + L::W1_OFF);
  elem* w2s = reinterpret_cast<elem*>(smem + L::W2_OFF);
  elem* hs = reinterpret_cast<elem*>(smem + L::H_OFF);
  float* ps = reinterpret_cast<float*>(smem + L::P_OFF);
  const float* a1s = ps;
  const float* i1s = ps + C;
  const float* a2s = ps + 2 * C;
  const float* i2s = ps + 3 * C;
  const float* b1s = ps + 4 * C;
  const float* b2s = ps + 5 * C;
  {  // packed weights are [rows][C + 8] already (same pitch as smem)
    const int4* src = reinterpret_cast<const int4*>(w1);
    int4* dst = reinterpret_cast<int4*>(w1s);
    for (int i = threadIdx.x; i < L::W1_BYTES / 16; i += kThreads)
      dst[i] = src[i];
    src = reinterpret_cast<const int4*>(w2);
    dst = reinterpret_cast<int4*>(w2s);
    for (int i = threadIdx.x; i < L::W2_BYTES / 16; i += kThreads)
      dst[i] = src[i];
    load_params<C>(ps, a1, i1, a2, i2, b1, b2);
  }

  const int lane = threadIdx.x % 32;
  const int warp = threadIdx.x / 32;
  const int row0 = warp * 16;
  const int g = lane / 4;
  const int q = (lane % 4) * 2;
  const int a_row = lane % 16;
  const int a_col = (lane / 16) * 8;
  const int b_row = (lane % 8) + (lane / 16) * 8;
  const int b_col = ((lane / 8) % 2) * 8;
  const int tiles_per_row = (T + NT - 1) / NT;
  const int total_tiles = tiles_per_row * batch;

  int4 pre[L::PER];
  if (blockIdx.x < total_tiles) {
    const int tile = blockIdx.x;
    prefetch_tile<C, L::ROWS, L::HALO, L::PER>(
        pre, x + static_cast<long long>(tile / tiles_per_row) * T * C,
        (tile % tiles_per_row) * NT, lens_s[tile / tiles_per_row]);
  }

  for (int tile = blockIdx.x; tile < total_tiles; tile += gridDim.x) {
    const int b = tile / tiles_per_row;
    const int t0 = (tile % tiles_per_row) * NT;
    const elem* xb = x + static_cast<long long>(b) * T * C;
    elem* yb = y + static_cast<long long>(b) * T * C;

    __syncthreads();  // weights ready / previous tile's staging consumed
    stage_snake<C, L::ROWS, L::LD, L::PER>(pre, hs, a1s, i1s);
    const int next = tile + gridDim.x;
    if (next < total_tiles)
      prefetch_tile<C, L::ROWS, L::HALO, L::PER>(
          pre, x + static_cast<long long>(next / tiles_per_row) * T * C,
          (next % tiles_per_row) * NT, lens_s[next / tiles_per_row]);
    __syncthreads();

    float acc[L::N8][4];
#pragma unroll
    for (int j = 0; j < L::N8; ++j)
      acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.f;
#pragma unroll
    for (int k = 0; k < 7; ++k) {
#pragma unroll
      for (int kk = 0; kk < L::K16; ++kk) {
        uint32_t af[4];
        ldsm_x4(af, hs + (row0 + k * D + a_row) * L::LD + kk * 16 + a_col);
#pragma unroll
        for (int jp = 0; jp < L::N8 / 2; ++jp) {
          uint32_t bf[4];
          ldsm_x4(bf, w1s + (k * C + jp * 16 + b_row) * L::LD + kk * 16 +
                          b_col);
          mma16816(acc[2 * jp], af, bf[0], bf[1]);
          mma16816(acc[2 * jp + 1], af, bf[2], bf[3]);
        }
      }
    }

    // vf[kk] = {row g k0-7, row g+8 k0-7, row g k8-15, row g+8 k8-15}
    //        = the m16n8k16 A-fragment register order {a0a1,a2a3,a4a5,a6a7}
    uint32_t vf[L::K16][4];
#pragma unroll
    for (int kk = 0; kk < L::K16; ++kk) {
#pragma unroll
      for (int half = 0; half < 2; ++half) {
        const int j = 2 * kk + half;
        const int c = j * 8 + q;
        vf[kk][2 * half + 0] = pack_half2(
            snake_f(acc[j][0] + b1s[c], a2s[c], i2s[c]),
            snake_f(acc[j][1] + b1s[c + 1], a2s[c + 1], i2s[c + 1]));
        vf[kk][2 * half + 1] = pack_half2(
            snake_f(acc[j][2] + b1s[c], a2s[c], i2s[c]),
            snake_f(acc[j][3] + b1s[c + 1], a2s[c + 1], i2s[c + 1]));
      }
    }
#pragma unroll
    for (int j = 0; j < L::N8; ++j)
      acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.f;
#pragma unroll
    for (int kk = 0; kk < L::K16; ++kk) {
#pragma unroll
      for (int jp = 0; jp < L::N8 / 2; ++jp) {
        uint32_t bf[4];
        ldsm_x4(bf, w2s + (jp * 16 + b_row) * L::LD + kk * 16 + b_col);
        mma16816(acc[2 * jp], vf[kk], bf[0], bf[1]);
        mma16816(acc[2 * jp + 1], vf[kk], bf[2], bf[3]);
      }
    }

    __syncthreads();  // all warps done reading hs (GEMM1 halo overlaps)
#pragma unroll
    for (int j = 0; j < L::N8; ++j) {
      const int c = j * 8 + q;
      *reinterpret_cast<elem2*>(hs + (row0 + g) * L::LD + c) =
          FROM_F2(acc[j][0] + b2s[c], acc[j][1] + b2s[c + 1]);
      *reinterpret_cast<elem2*>(hs + (row0 + g + 8) * L::LD + c) =
          FROM_F2(acc[j][2] + b2s[c], acc[j][3] + b2s[c + 1]);
    }
    __syncthreads();
    residual_store<C, NT, L::LD>(xb, yb, hs, t0, T, lens_s[b]);
  }
}

template <typename Layout, typename Kernel>
int prepare_kernel(Kernel kernel, int& per_sm) {
  if (per_sm < 0) {
    TORCH_CHECK(cudaFuncSetAttribute(
                    kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                    Layout::SMEM) == cudaSuccess,
                "cudaFuncSetAttribute failed (smem ", Layout::SMEM, ")");
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, kernel, kThreads,
                                                  Layout::SMEM);
    TORCH_CHECK(per_sm > 0, "kernel does not fit (smem ", Layout::SMEM, ")");
  }
  return per_sm;
}

template <typename Layout, int NT, typename Kernel>
void launch_unit(Kernel kernel, int per_sm, const at::Tensor& x,
                 const at::Tensor& w1, const at::Tensor& w2,
                 const at::Tensor& b1, const at::Tensor& a1,
                 const at::Tensor& i1, const at::Tensor& a2,
                 const at::Tensor& i2, const at::Tensor& b2,
                 const at::Tensor& lens, at::Tensor& y) {
  const int sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  const int batch = static_cast<int>(x.size(0));
  const int T = static_cast<int>(x.size(1));
  const int tiles = (T + NT - 1) / NT * batch;
  const int grid = std::min(tiles, sms * per_sm);
  kernel<<<grid, kThreads, Layout::SMEM, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const elem*>(x.data_ptr<ELEM_AT>()),
      reinterpret_cast<const elem*>(w1.data_ptr<ELEM_AT>()),
      reinterpret_cast<const elem*>(w2.data_ptr<ELEM_AT>()),
      b1.data_ptr<float>(), a1.data_ptr<float>(), i1.data_ptr<float>(),
      a2.data_ptr<float>(), i2.data_ptr<float>(), b2.data_ptr<float>(),
      reinterpret_cast<elem*>(y.data_ptr<ELEM_AT>()), lens.data_ptr<int>(),
      batch, T);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <int C, int D>
void launch_resunit_mma(const at::Tensor& x, const at::Tensor& w1,
                        const at::Tensor& w2, const at::Tensor& b1,
                        const at::Tensor& a1, const at::Tensor& i1,
                        const at::Tensor& a2, const at::Tensor& i2,
                        const at::Tensor& b2, const at::Tensor& lens,
                    at::Tensor& y) {
  constexpr int NT = 16 * kWarps;
  using L = MmaLayout<C, NT, D>;
  auto kernel = resunit_mma_kernel<C, NT, D>;
  static int per_sm = -1;
  prepare_kernel<L>(kernel, per_sm);
  launch_unit<L, NT>(kernel, per_sm, x, w1, w2, b1, a1, i1, a2, i2, b2, lens,
                     y);
}

// ---- resunit for wide units, v3: 128-row tiles, half-width weight chunks --
// The C=192 weights (7 taps + W2, 77 KB each) are streamed from L2 for every
// tile, so the tile height sets the weight traffic. Streaming each weight
// as two output-channel halves (38 KB) through a cp.async double buffer
// frees enough shared memory for 128-row tiles (half the L2 traffic of the
// 64-row v1). Each warp owns a 16-row strip x all C output channels, so
// GEMM2 only reads the warp's own staged rows (no block barrier needed).

template <int C, int NT, int D>
struct Stream2Layout {
  static constexpr int LD = C + 8;
  static constexpr int HALO = 3 * D;
  static constexpr int ROWS = NT + 2 * HALO;
  static constexpr int CH = C / 2;                 // channels per chunk
  static constexpr int N8 = C / 8;                 // accumulator n8 tiles
  static constexpr int CN8 = CH / 8;               // n8 tiles per chunk
  static constexpr int K16 = C / 16;
  static constexpr int PER = (ROWS * (C / 8) + kThreads - 1) / kThreads;
  static constexpr int CHUNK_BYTES = CH * LD * 2;
  static constexpr int H_BYTES = ROWS * LD * 2;
  static constexpr int P_BYTES = 6 * C * 4;
  static constexpr int WB0_OFF = 0;
  static constexpr int WB1_OFF = align_up(WB0_OFF + CHUNK_BYTES, 128);
  static constexpr int H_OFF = align_up(WB1_OFF + CHUNK_BYTES, 128);
  static constexpr int P_OFF = align_up(H_OFF + H_BYTES, 128);
  static constexpr int SMEM = align_up(P_OFF + P_BYTES, 128);
  static constexpr int LOADS = 16;                 // 7 taps x 2 + W2 x 2
  static_assert(NT == 16 * kWarps, "one strip per warp");
  static_assert(CN8 % 2 == 0, "n8 pairs");
};

template <typename L, int HALF>
__device__ __forceinline__ void chunk_mma(float (&acc)[L::N8][4],
                                          const elem* hs, const elem* w,
                                          int row, int a_row, int a_col,
                                          int b_row, int b_col) {
#pragma unroll
  for (int kk = 0; kk < L::K16; ++kk) {
    uint32_t af[4];
    ldsm_x4(af, hs + (row + a_row) * L::LD + kk * 16 + a_col);
#pragma unroll
    for (int jp = 0; jp < L::CN8 / 2; ++jp) {
      uint32_t bf[4];
      ldsm_x4(bf, w + (jp * 16 + b_row) * L::LD + kk * 16 + b_col);
      mma16816(acc[HALF * L::CN8 + 2 * jp], af, bf[0], bf[1]);
      mma16816(acc[HALF * L::CN8 + 2 * jp + 1], af, bf[2], bf[3]);
    }
  }
}

template <int C, int NT, int D>
__global__ void __launch_bounds__(kThreads)
resunit_stream2_kernel(const elem* __restrict__ x,
                       const elem* __restrict__ w1,
                       const elem* __restrict__ w2,
                       const float* __restrict__ b1,
                       const float* __restrict__ a1,
                       const float* __restrict__ i1,
                       const float* __restrict__ a2,
                       const float* __restrict__ i2,
                       const float* __restrict__ b2,
                       elem* __restrict__ y, const int* __restrict__ lens, int batch, int T) {
  // per-item valid lengths cached in smem: a global load per tile sat on
  // the critical path of the next tile's prefetch (~0.3 ms per kernel)
  __shared__ int lens_s[kMaxBatch];
  for (int i = threadIdx.x; i < batch; i += blockDim.x)
    lens_s[i] = lens[i];
  __syncthreads();
  using L = Stream2Layout<C, NT, D>;
  extern __shared__ __align__(128) unsigned char smem[];
  elem* wb0 = reinterpret_cast<elem*>(smem + L::WB0_OFF);
  elem* wb1 = reinterpret_cast<elem*>(smem + L::WB1_OFF);
  elem* hs = reinterpret_cast<elem*>(smem + L::H_OFF);
  float* ps = reinterpret_cast<float*>(smem + L::P_OFF);
  const float* a1s = ps;
  const float* i1s = ps + C;
  const float* a2s = ps + 2 * C;
  const float* i2s = ps + 3 * C;
  const float* b1s = ps + 4 * C;
  const float* b2s = ps + 5 * C;
  load_params<C>(ps, a1, i1, a2, i2, b1, b2);

  const int lane = threadIdx.x % 32;
  const int warp = threadIdx.x / 32;
  const int row0 = warp * 16;
  const int g = lane / 4;
  const int q = (lane % 4) * 2;
  const int a_row = lane % 16;
  const int a_col = (lane / 16) * 8;
  const int b_row = (lane % 8) + (lane / 16) * 8;
  const int b_col = ((lane / 8) % 2) * 8;
  const int tiles_per_row = (T + NT - 1) / NT;
  const int total_tiles = tiles_per_row * batch;

  // load id 0..13: W1 tap id/2, channel half id%2; 14,15: W2 halves
  auto issue_chunk = [&](int id, elem* dst) {
    const int half = id % 2;
    const elem* src =
        id < 14 ? w1 + (static_cast<long long>(id / 2) * C + half * L::CH) *
                           L::LD
                : w2 + static_cast<long long>(half) * L::CH * L::LD;
    for (int i = threadIdx.x; i < L::CHUNK_BYTES / 16; i += kThreads)
      cp_async16(dst + i * 8, src + i * 8);
    cp_async_commit();
  };

  int4 pre[L::PER];
  if (blockIdx.x < total_tiles) {
    issue_chunk(0, wb0);
    const int tile = blockIdx.x;
    prefetch_tile<C, L::ROWS, L::HALO, L::PER>(
        pre, x + static_cast<long long>(tile / tiles_per_row) * T * C,
        (tile % tiles_per_row) * NT, lens_s[tile / tiles_per_row]);
  }

  int stream = 0;  // global load counter: buffer = parity
  for (int tile = blockIdx.x; tile < total_tiles; tile += gridDim.x) {
    const int b = tile / tiles_per_row;
    const int t0 = (tile % tiles_per_row) * NT;
    const elem* xb = x + static_cast<long long>(b) * T * C;
    elem* yb = y + static_cast<long long>(b) * T * C;

    __syncthreads();  // previous tile's residual pass done with hs
    stage_snake<C, L::ROWS, L::LD, L::PER>(pre, hs, a1s, i1s);
    const int next = tile + gridDim.x;
    if (next < total_tiles)
      prefetch_tile<C, L::ROWS, L::HALO, L::PER>(
          pre, x + static_cast<long long>(next / tiles_per_row) * T * C,
          (next % tiles_per_row) * NT, lens_s[next / tiles_per_row]);

    float acc[L::N8][4];
#pragma unroll
    for (int j = 0; j < L::N8; ++j)
      acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.f;

#pragma unroll 1
    for (int id = 0; id < L::LOADS; ++id, ++stream) {
      cp_async_wait_all();
      __syncthreads();  // chunk `id` visible, other buffer free, hs ready
      if (id + 1 < L::LOADS || next < total_tiles)
        issue_chunk((id + 1) % L::LOADS, (stream % 2) ? wb0 : wb1);
      const elem* w = (stream % 2) ? wb1 : wb0;
      const int half = id % 2;
      if (id == 14) {
        // every warp is past GEMM1 (barrier above): v = snake2(u + b1)
        // into this warp's own rows; only this warp reads them back
#pragma unroll
        for (int j = 0; j < L::N8; ++j) {
          const int c = j * 8 + q;
          *reinterpret_cast<elem2*>(hs + (row0 + g) * L::LD + c) =
              FROM_F2(
                  snake_f(acc[j][0] + b1s[c], a2s[c], i2s[c]),
                  snake_f(acc[j][1] + b1s[c + 1], a2s[c + 1], i2s[c + 1]));
          *reinterpret_cast<elem2*>(hs + (row0 + g + 8) * L::LD + c) =
              FROM_F2(
                  snake_f(acc[j][2] + b1s[c], a2s[c], i2s[c]),
                  snake_f(acc[j][3] + b1s[c + 1], a2s[c + 1], i2s[c + 1]));
          acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.f;
        }
        __syncwarp();
      }
      const int shift = id < 14 ? (id / 2) * D : 0;
      // accumulator indices must be compile-time constants (a runtime
      // `half * CN8 + j` index demotes acc[] to local memory)
      if (half == 0)
        chunk_mma<L, 0>(acc, hs, w, row0 + shift, a_row, a_col, b_row, b_col);
      else
        chunk_mma<L, 1>(acc, hs, w, row0 + shift, a_row, a_col, b_row, b_col);
    }

    __syncwarp();  // own rows: GEMM2 reads done before the overwrite
#pragma unroll
    for (int j = 0; j < L::N8; ++j) {
      const int c = j * 8 + q;
      *reinterpret_cast<elem2*>(hs + (row0 + g) * L::LD + c) =
          FROM_F2(acc[j][0] + b2s[c], acc[j][1] + b2s[c + 1]);
      *reinterpret_cast<elem2*>(hs + (row0 + g + 8) * L::LD + c) =
          FROM_F2(acc[j][2] + b2s[c], acc[j][3] + b2s[c + 1]);
    }
    __syncthreads();
    residual_store<C, NT, L::LD>(xb, yb, hs, t0, T, lens_s[b]);
  }
  cp_async_wait_all();
}

template <int C, int D>
void launch_resunit_stream2(const at::Tensor& x, const at::Tensor& w1,
                            const at::Tensor& w2, const at::Tensor& b1,
                            const at::Tensor& a1, const at::Tensor& i1,
                            const at::Tensor& a2, const at::Tensor& i2,
                            const at::Tensor& b2, const at::Tensor& lens,
                    at::Tensor& y) {
  constexpr int NT = 16 * kWarps;
  using L = Stream2Layout<C, NT, D>;
  auto kernel = resunit_stream2_kernel<C, NT, D>;
  static int per_sm = -1;
  prepare_kernel<L>(kernel, per_sm);
  launch_unit<L, NT>(kernel, per_sm, x, w1, w2, b1, a1, i1, a2, i2, b2, lens,
                     y);
}

// ---- upsample: snake + ConvTranspose1d(CIN->COUT, K=2S, stride S,
//      padding P=ceil(S/2)), time-major, all phases per CTA --------------------
// Output row j = S*m + ph receives two taps: for d in {-1,0,1} with
// 0 <= k = S*d + ph + P < 2S, input row m - d through W_k. One CTA computes
// every phase of a 128-row input tile: the input slab is read once (+1 halo
// row each side), the 2S taps stream through a cp.async double buffer, the
// S*NT output rows are assembled in smem and written as one contiguous
// slab. Output length S*T + S - 2P.

template <int CIN, int COUT, int S, int NT>
struct UpStreamLayout {
  static constexpr int LDI = CIN + 8;
  static constexpr int LDO = COUT + 8;
  static constexpr int ROWS = NT + 2;
  static constexpr int N8 = COUT / 8;
  static constexpr int K16 = CIN / 16;
  static constexpr int PER = (ROWS * (CIN / 8) + kThreads - 1) / kThreads;
  static constexpr int TAP_BYTES = COUT * LDI * 2;
  static constexpr int H_BYTES = ROWS * LDI * 2;
  static constexpr int O_BYTES = S * NT * LDO * 2;
  static constexpr int WB0_OFF = 0;
  static constexpr int WB1_OFF = align_up(WB0_OFF + TAP_BYTES, 128);
  static constexpr int H_OFF = align_up(WB1_OFF + TAP_BYTES, 128);
  static constexpr int O_OFF = align_up(H_OFF + H_BYTES, 128);
  static constexpr int SMEM = align_up(O_OFF + O_BYTES, 128);
  static constexpr int LOADS = 2 * S;
  static_assert(NT == 16 * kWarps, "one strip per warp");
  static_assert(N8 % 2 == 0, "n8 pairs");
};

// tap (kernel index) and input-row shift (relative to m, +1 halo) of the
// q-th contribution to phase ph
template <int S>
__device__ __forceinline__ void phase_tap(int ph, int q, int& k, int& shift) {
  constexpr int P = (S + 1) / 2;
  int found = 0;
#pragma unroll
  for (int d = 1; d >= -1; --d) {
    const int kk = S * d + ph + P;
    if (kk >= 0 && kk < 2 * S) {
      if (found == q) {
        k = kk;
        shift = 1 - d;
      }
      ++found;
    }
  }
}

template <int CIN, int COUT, int S, int NT>
__global__ void __launch_bounds__(kThreads)
upsample_stream_kernel(const elem* __restrict__ x,
                       const elem* __restrict__ w,
                       const float* __restrict__ bias,
                       const float* __restrict__ alpha,
                       const float* __restrict__ inv,
                       elem* __restrict__ y,
                       const int* __restrict__ lens, int batch, int T) {
  // per-item valid lengths cached in smem: a global load per tile sat on
  // the critical path of the next tile's prefetch (~0.3 ms per kernel)
  __shared__ int lens_s[kMaxBatch];
  for (int i = threadIdx.x; i < batch; i += blockDim.x)
    lens_s[i] = lens[i];
  __syncthreads();
  using L = UpStreamLayout<CIN, COUT, S, NT>;
  constexpr int P = (S + 1) / 2;
  extern __shared__ __align__(128) unsigned char smem[];
  elem* wb0 = reinterpret_cast<elem*>(smem + L::WB0_OFF);
  elem* wb1 = reinterpret_cast<elem*>(smem + L::WB1_OFF);
  elem* hs = reinterpret_cast<elem*>(smem + L::H_OFF);
  elem* os = reinterpret_cast<elem*>(smem + L::O_OFF);

  const int lane = threadIdx.x % 32;
  const int warp = threadIdx.x / 32;
  const int row0 = warp * 16;
  const int g = lane / 4;
  const int q = (lane % 4) * 2;
  const int a_row = lane % 16;
  const int a_col = (lane / 16) * 8;
  const int b_row = (lane % 8) + (lane / 16) * 8;
  const int b_col = ((lane / 8) % 2) * 8;
  const int out_len = S * T + S - 2 * P;
  const int tiles_per_row = (T + NT - 1) / NT;
  const int total_tiles = tiles_per_row * batch;

  auto issue_tap = [&](int id, elem* dst) {  // id = 2*ph + q
    int k = 0, shift = 0;
    phase_tap<S>(id / 2, id % 2, k, shift);
    const elem* src = w + static_cast<long long>(k) * COUT * L::LDI;
    for (int i = threadIdx.x; i < L::TAP_BYTES / 16; i += kThreads)
      cp_async16(dst + i * 8, src + i * 8);
    cp_async_commit();
  };

  // prefetch (+ snake on staging): rows t0-1 .. t0+NT
  int4 pre[L::PER];
  auto prefetch = [&](int tile) {
    const int b = tile / tiles_per_row;
    const int t0 = (tile % tiles_per_row) * NT;
    prefetch_tile<CIN, L::ROWS, 1, L::PER>(
        pre, x + static_cast<long long>(b) * T * CIN, t0, lens_s[b]);
  };
  if (blockIdx.x < total_tiles) {
    issue_tap(0, wb0);
    prefetch(blockIdx.x);
  }

  int stream = 0;
  for (int tile = blockIdx.x; tile < total_tiles; tile += gridDim.x) {
    const int b = tile / tiles_per_row;
    const int t0 = (tile % tiles_per_row) * NT;
    elem* yb = y + static_cast<long long>(b) * out_len * COUT;

    __syncthreads();  // previous tile's output copy done with os / hs
    stage_snake<CIN, L::ROWS, L::LDI, L::PER>(pre, hs, alpha, inv);
    const int next = tile + gridDim.x;
    if (next < total_tiles) prefetch(next);

    float acc[L::N8][4];
#pragma unroll 1
    for (int id = 0; id < L::LOADS; ++id, ++stream) {
      cp_async_wait_all();
      __syncthreads();
      if (id + 1 < L::LOADS || next < total_tiles)
        issue_tap((id + 1) % L::LOADS, (stream % 2) ? wb0 : wb1);
      const elem* wt = (stream % 2) ? wb1 : wb0;
      const int ph = id / 2;
      int k = 0, shift = 0;
      phase_tap<S>(ph, id % 2, k, shift);
      if (id % 2 == 0) {
#pragma unroll
        for (int j = 0; j < L::N8; ++j)
          acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.f;
      }
#pragma unroll
      for (int kk = 0; kk < L::K16; ++kk) {
        uint32_t af[4];
        ldsm_x4(af, hs + (row0 + shift + a_row) * L::LDI + kk * 16 + a_col);
#pragma unroll
        for (int jp = 0; jp < L::N8 / 2; ++jp) {
          uint32_t bf[4];
          ldsm_x4(bf, wt + (jp * 16 + b_row) * L::LDI + kk * 16 + b_col);
          mma16816(acc[2 * jp], af, bf[0], bf[1]);
          mma16816(acc[2 * jp + 1], af, bf[2], bf[3]);
        }
      }
      if (id % 2 == 1) {  // phase complete: rows S*n + ph of the slab
#pragma unroll
        for (int j = 0; j < L::N8; ++j) {
          const int c = j * 8 + q;
          const float bb0 = __ldg(bias + c);
          const float bb1 = __ldg(bias + c + 1);
          *reinterpret_cast<elem2*>(
              os + (S * (row0 + g) + ph) * L::LDO + c) =
              FROM_F2(acc[j][0] + bb0, acc[j][1] + bb1);
          *reinterpret_cast<elem2*>(
              os + (S * (row0 + g + 8) + ph) * L::LDO + c) =
              FROM_F2(acc[j][2] + bb0, acc[j][3] + bb1);
        }
      }
    }
    __syncthreads();
    // contiguous output slab rows [S*t0, S*t0 + S*NT) (clipped to out_len;
    // rows past this item's valid output length are bucket padding -> 0)
    const long long base = static_cast<long long>(S) * t0;
    const int out_valid = S * lens_s[b] + S - 2 * P;
    for (int v = threadIdx.x; v < S * NT * (COUT / 8); v += kThreads) {
      const int r = v / (COUT / 8);
      const int cv = v % (COUT / 8);
      if (base + r >= out_len) continue;
      *reinterpret_cast<int4*>(yb + (base + r) * COUT + cv * 8) =
          base + r < out_valid
              ? *reinterpret_cast<const int4*>(os + r * L::LDO + cv * 8)
              : make_int4(0, 0, 0, 0);
    }
  }
  cp_async_wait_all();
}

template <int CIN, int COUT, int S>
at::Tensor launch_upsample_stream(const at::Tensor& x, const at::Tensor& w,
                                  const at::Tensor& bias,
                                  const at::Tensor& alpha,
                                  const at::Tensor& inv,
                                  const at::Tensor& lens) {
  constexpr int NT = 16 * kWarps;
  using L = UpStreamLayout<CIN, COUT, S, NT>;
  auto kernel = upsample_stream_kernel<CIN, COUT, S, NT>;
  static int per_sm = -1;
  prepare_kernel<L>(kernel, per_sm);
  constexpr int P = (S + 1) / 2;
  const int batch = static_cast<int>(x.size(0));
  const int T = static_cast<int>(x.size(1));
  auto y = at::empty({x.size(0), S * x.size(1) + S - 2 * P, COUT},
                     x.options());
  const int sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  const int tiles = (T + NT - 1) / NT * batch;
  const int grid = std::min(tiles, sms * per_sm);
  kernel<<<grid, kThreads, L::SMEM, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const elem*>(x.data_ptr<ELEM_AT>()),
      reinterpret_cast<const elem*>(w.data_ptr<ELEM_AT>()),
      bias.data_ptr<float>(), alpha.data_ptr<float>(), inv.data_ptr<float>(),
      reinterpret_cast<elem*>(y.data_ptr<ELEM_AT>()), lens.data_ptr<int>(),
      batch, T);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}

at::Tensor upsample(const at::Tensor& x, const at::Tensor& w,
                    const at::Tensor& bias, const at::Tensor& alpha,
                    const at::Tensor& inv, int64_t stride,
                    const at::Tensor& lens) {
  check_time_major(x);
  check_lens(lens, x);
  const c10::cuda::CUDAGuard guard(x.device());
  const int64_t cin = x.size(2);
  if (cin == 96 && stride == 2)
    return launch_upsample_stream<96, 48, 2>(x, w, bias, alpha, inv, lens);
  if (cin == 192 && stride == 3)
    return launch_upsample_stream<192, 96, 3>(x, w, bias, alpha, inv, lens);
  TORCH_CHECK(false, "upsample: unsupported C_in=", cin, " stride=", stride);
  return x;
}

at::Tensor resunit_mma(const at::Tensor& x, const at::Tensor& w1,
                       const at::Tensor& w2, const at::Tensor& b1,
                       const at::Tensor& a1, const at::Tensor& i1,
                       const at::Tensor& a2, const at::Tensor& i2,
                       const at::Tensor& b2, int64_t dilation,
                       const at::Tensor& lens) {
  check_time_major(x);
  check_lens(lens, x);
  const c10::cuda::CUDAGuard guard(x.device());
  auto y = at::empty_like(x);
  const int64_t C = x.size(2);
#define STRITON_CASE(LAUNCH, CC, DD)                                          \
  if (C == CC && dilation == DD) {                                        \
    LAUNCH<CC, DD>(x, w1, w2, b1, a1, i1, a2, i2, b2, lens, y);           \
    return y;                                                             \
  }
  STRITON_CASE(launch_resunit_mma, 48, 1)
  STRITON_CASE(launch_resunit_mma, 48, 3)
  STRITON_CASE(launch_resunit_mma, 48, 9)
  STRITON_CASE(launch_resunit_mma, 96, 1)
  STRITON_CASE(launch_resunit_mma, 96, 3)
  STRITON_CASE(launch_resunit_mma, 96, 9)
  STRITON_CASE(launch_resunit_stream2, 192, 1)
  STRITON_CASE(launch_resunit_stream2, 192, 3)
  STRITON_CASE(launch_resunit_stream2, 192, 9)
#undef STRITON_CASE
  TORCH_CHECK(false, "resunit_mma: unsupported C=", C, " dilation=", dilation);
  return y;
}

}  // namespace

#define STRITON_DEFS(m) \
  m.def( \
      "resunit_mma(Tensor x, Tensor w1, Tensor w2, Tensor b1, Tensor a1, " \
      "Tensor i1, Tensor a2, Tensor i2, Tensor b2, int dilation, " \
      "Tensor lens) -> Tensor"); \
  m.def( \
      "upsample(Tensor x, Tensor w, Tensor bias, Tensor alpha, Tensor inv, " \
      "int stride, Tensor lens) -> Tensor"); \
  m.def( \
      "final_head(Tensor x, Tensor w, Tensor alpha, Tensor inv, float bias, " \
      "Tensor lens) -> Tensor");

#define STRITON_IMPLS(m) \
  m.impl("resunit_mma", &resunit_mma); \
  m.impl("upsample", &upsample); \
  m.impl("final_head", &final_head);

#ifdef STRITON_BF16
TORCH_LIBRARY(sidon_triton_bf16, m) { STRITON_DEFS(m) }
TORCH_LIBRARY_IMPL(sidon_triton_bf16, CUDA, m) { STRITON_IMPLS(m) }
#else
TORCH_LIBRARY(sidon_triton, m) { STRITON_DEFS(m) }
TORCH_LIBRARY_IMPL(sidon_triton, CUDA, m) { STRITON_IMPLS(m) }
#endif
