// src/prefill/gemm_hfma.cuh - STRATA_PF_HFMA: the prompt's dense projections as FP16 HFMA2 GEMMs on GP100 (sm_60); see gemm.cu.
//
// Y[T, N] (FP32, row stride ldy) = X[T, K] . W[N, K]^T with X and W FP16 (rows of K values; X's row stride ldx8 uint4), the
// products summed in FP16 pairs by fma.rn.f16x2 - GP100 issues those at twice the FP32 rate - and every 64 products of a
// lane (kSteps steps of kBK values: each half of a pair takes every second value) widened into an FP32 accumulator, so an
// FP16 sum never holds more than 64 terms.  A block is a 128 x 128 tile of Y: 256 threads of 8 x 8 outputs each, double
// buffered through shared memory, one block per SM.  The loop is campaign fp16-work/kern_gemm.cu's gemm_v0_g64 as the
// compiler schedules it (4144 G MAC/s at G = 64 against cuBLAS FP32's 3070), unchanged; what is new is the edges:
//   - the rows of X and W past T and N are the last row's on load (they only feed outputs that are never stored),
//   - the stores are bounds-checked and use the row stride ldy,
//   - a K that is not a multiple of 128 has its last, shorter group flushed after the loop (K itself is a multiple of 32).
// X and W must be 16-byte aligned with ldx8 * 8 >= K.
#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace strata::prefill::hfma {

constexpr int kBM = 128, kBN = 128, kBK = 32, kThreads = 256;
constexpr int kSteps = 4;   // steps of kBK values per flush: 4 * 32 / 2 = 64 products in each FP16 lane

__device__ __forceinline__ uint32_t hfma2(const uint32_t a, const uint32_t b, const uint32_t c) {
    uint32_t d;
    asm("fma.rn.f16x2 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
    return d;
}
__device__ __forceinline__ void h2f(const uint32_t h, float& lo, float& hi) {
    asm("{\n\t.reg .f16 l, h;\n\tmov.b32 {l, h}, %2;\n\tcvt.f32.f16 %0, l;\n\tcvt.f32.f16 %1, h;\n\t}" : "=f"(lo), "=f"(hi) : "r"(h));
}
template <int W>
__device__ __forceinline__ void fma_tile(uint32_t (&acc)[8][8], const uint4 (&a)[8], const uint4 (&b)[8]) {
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const uint32_t av = W == 0 ? a[i].x : W == 1 ? a[i].y : W == 2 ? a[i].z : a[i].w;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const uint32_t bv = W == 0 ? b[j].x : W == 1 ? b[j].y : W == 2 ? b[j].z : b[j].w;
            acc[i][j] = hfma2(av, bv, acc[i][j]);
        }
    }
}
// the FP16 pair sums into the FP32 ones, and clear
__device__ __forceinline__ void flush(uint32_t (&acc)[8][8], float (&acc32)[8][8]) {
#pragma unroll
    for (int i = 0; i < 8; ++i)
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            float lo, hi;
            h2f(acc[i][j], lo, hi);
            acc32[i][j] += lo + hi;
            acc[i][j] = 0u;
        }
}

static __global__ void __launch_bounds__(kThreads, 1) gemm_kernel(const uint4* __restrict__ X, const uint4* __restrict__ Wt,
                                                                    float* __restrict__ Y, const int T, const int N, const int K,
                                                                    const int ldx8, const int64_t ldy) {
    __shared__ __align__(16) uint4 As[2][kBM * 5];
    __shared__ __align__(16) uint4 Bs[2][kBN * 5];
    const int tid = threadIdx.x, tx = tid & 15, ty = tid >> 4;
    const int m0 = blockIdx.y * kBM, n0 = blockIdx.x * kBN;
    const int K8 = K / 8;
    const uint4* gA[2]; const uint4* gB[2]; int so[2];
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int idx = tid + i * kThreads, row = idx >> 2, q = idx & 3;
        gA[i] = X + (size_t) min(m0 + row, T - 1) * ldx8 + q;
        gB[i] = Wt + (size_t) min(n0 + row, N - 1) * K8 + q;
        so[i] = row * 5 + q;
    }
    uint32_t acc[8][8];
    float acc32[8][8];
#pragma unroll
    for (int i = 0; i < 8; ++i)
#pragma unroll
        for (int j = 0; j < 8; ++j) { acc[i][j] = 0u; acc32[i][j] = 0.0f; }
    uint4 ra[2], rb[2];
#pragma unroll
    for (int i = 0; i < 2; ++i) { ra[i] = __ldg(gA[i]); rb[i] = __ldg(gB[i]); }
#pragma unroll
    for (int i = 0; i < 2; ++i) { As[0][so[i]] = ra[i]; Bs[0][so[i]] = rb[i]; }
    __syncthreads();
    const int S = K / kBK;
    for (int s = 0; s < S; ++s) {
        const int cur = s & 1;
        if (s + 1 < S) {
#pragma unroll
            for (int i = 0; i < 2; ++i) { ra[i] = __ldg(gA[i] + (size_t) (s + 1) * 4); rb[i] = __ldg(gB[i] + (size_t) (s + 1) * 4); }
        }
#pragma unroll
        for (int kq = 0; kq < 4; ++kq) {
            uint4 a[8], b[8];
#pragma unroll
            for (int i = 0; i < 8; ++i) a[i] = As[cur][(ty + 16 * i) * 5 + kq];
#pragma unroll
            for (int j = 0; j < 8; ++j) b[j] = Bs[cur][(tx + 16 * j) * 5 + kq];
            fma_tile<0>(acc, a, b);
            fma_tile<1>(acc, a, b);
            fma_tile<2>(acc, a, b);
            fma_tile<3>(acc, a, b);
        }
        if ((s + 1) % kSteps == 0) flush(acc, acc32);
        if (s + 1 < S) {
#pragma unroll
            for (int i = 0; i < 2; ++i) { As[cur ^ 1][so[i]] = ra[i]; Bs[cur ^ 1][so[i]] = rb[i]; }
        }
        __syncthreads();
    }
    if (S % kSteps != 0) flush(acc, acc32);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int row = m0 + ty + 16 * i;
        if (row < T) {
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const int col = n0 + tx + 16 * j;
                if (col < N) Y[(size_t) row * (size_t) ldy + col] = acc32[i][j];
            }
        }
    }
}

// X: T rows of K FP16 values, `ldx` apart (ldx >= K, a multiple of 8); W: N rows of K; Y: T rows of N floats, `ldy` apart
inline void gemm(const uint16_t* X, int64_t ldx, const uint16_t* W, float* Y, int64_t T, int64_t N, int64_t K, int64_t ldy,
                 cudaStream_t st) {
    const dim3 grid((unsigned) ((N + kBN - 1) / kBN), (unsigned) ((T + kBM - 1) / kBM));
    gemm_kernel<<<grid, kThreads, 0, st>>>(reinterpret_cast<const uint4*>(X), reinterpret_cast<const uint4*>(W), Y, (int) T,
                                           (int) N, (int) K, (int) (ldx / 8), ldy);
}

}  // namespace strata::prefill::hfma
