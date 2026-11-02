// src/prefill/gemm_hfma_hc.cuh - STRATA_PF_HFMA_HC: the hyper-connection read's BF16 projections (down: N 320, K 10240;
// up: N 10240, K 320) as FP16 HFMA2 GEMMs on GP100 (sm_60); see gemm.cu.  Reuses gemm_hfma.cuh's tile (the same inner loop,
// the same 64-product FP16 lane sums widened to FP32 every kSteps steps); what is new is around it:
//   - the weight is the BF16 tensor's FP16 copy (exact for every normal value of the FP16 range; the copy is refused when a
//     weight is outside it or not finite), kept resident like the dense projections' (gemm.cu's shadow_get_h);
//   - the activation is the BF16 (or FP16) image of the slice scaled by an exact power of two 2^-k and converted to FP16, k
//     picked per slice from its own max |x| (a max-reduce kernel one launch ahead in the stream) and the weight's max |w| so
//     that 64 * wmax * xmax 2^-k <= 2^15 and xmax 2^-k <= 2^15: no lane and no input can leave FP16's range whatever the
//     data.  The FP32 results are multiplied by 2^k;
//   - K is split over gridDim.z (the down's N 320 makes 3 tiles of 128 columns: too few blocks for 56 SMs), each part a
//     disjoint partial sum summed in a fixed order by reduce_kernel (deterministic).
// A non-finite input or output is counted (g_hc), never hidden.
#pragma once

#include "gemm_hfma.cuh"

namespace strata::prefill::hfma::hc {
namespace {

constexpr int KMAXH = 60;
struct HcState {
    unsigned cur;               // max |x| of the slice (FP32 bit pattern; NaN above inf); zeroed by a memset before each max-reduce
    unsigned bad_x;             // slices whose input held a non-finite value
    unsigned bad_y;             // outputs that came out non-finite
    unsigned xmax_all;          // max over the slices
    unsigned slices;
    unsigned khist[KMAXH + 1];  // the activation's shift k, per slice
};
__device__ HcState g_hc;

__device__ __forceinline__ float pow2i(const int e) { return __int_as_float((e + 127) << 23); }   // e in [-126, 127]

// the activation's shift: 2^k >= xmax wmax / 512 (64 products of < wmax * xmax 2^-k stay <= 2^15) and 2^k >= xmax / 2^15
__device__ __forceinline__ int shift_k(const unsigned cur_bits, const float wmax) {
    const float xm = __uint_as_float(cur_bits);
    const float need = fmaxf(xm * wmax * (1.0f / 512.0f), xm * (1.0f / 32768.0f));
    if (!(need <= 1e30f)) return KMAXH;
    if (need <= 1.0f) return 0;
    int e;
    (void) frexpf(need, &e);   // need = m 2^e, m in [0.5, 1): 2^e > need
    return e > KMAXH ? KMAXH : e;
}

// the 16-bit pattern as |FP32| bits
template <bool F16>
__device__ __forceinline__ unsigned abs_bits(const unsigned h) {
    if constexpr (F16) return __float_as_uint(fabsf(__half2float(__ushort_as_half((unsigned short) h))));
    else return (h & 0x7FFFu) << 16;
}
template <bool F16>
__device__ __forceinline__ float to_f32(const unsigned h) {
    if constexpr (F16) return __half2float(__ushort_as_half((unsigned short) h));
    else return __uint_as_float(h << 16);
}

// max |x| of the slice's T rows of K values (row stride ldx8 uint4); g_hc.cur was zeroed by a memset before
template <bool F16>
__global__ void __launch_bounds__(256) xmax_kernel(const uint4* __restrict__ x, const int64_t n8, const int64_t K8, const int64_t ldx8) {
    unsigned m = 0;
    for (int64_t i = (int64_t) blockIdx.x * 256 + threadIdx.x; i < n8; i += (int64_t) gridDim.x * 256) {
        const int64_t r = i / K8, c = i - r * K8;
        const uint4 v = __ldg(x + r * ldx8 + c);
        const unsigned ws[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            m = max(m, abs_bits<F16>(ws[j] & 0xFFFFu));
            m = max(m, abs_bits<F16>(ws[j] >> 16));
        }
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) m = max(m, (unsigned) __shfl_xor_sync(0xFFFFFFFFu, (int) m, o));
    if ((threadIdx.x & 31) == 0 && m != 0) atomicMax(&g_hc.cur, m);
}

// the slice as FP16 rows of K values (contiguous), scaled by 2^-k
template <bool F16>
__global__ void __launch_bounds__(256) cvt_kernel(const uint4* __restrict__ x, uint4* __restrict__ out, const int64_t n8, const int64_t K8,
                                                  const int64_t ldx8, const float wmax) {
    const unsigned cur = *(volatile const unsigned*) &g_hc.cur;
    const int k = shift_k(cur, wmax);
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        atomicAdd(&g_hc.slices, 1u);
        atomicMax(&g_hc.xmax_all, cur);
        atomicAdd(&g_hc.khist[k], 1u);
        if (cur >= 0x7F800000u) atomicAdd(&g_hc.bad_x, 1u);
    }
    const float sc = pow2i(-k);
    for (int64_t i = (int64_t) blockIdx.x * 256 + threadIdx.x; i < n8; i += (int64_t) gridDim.x * 256) {
        const int64_t r = i / K8, c = i - r * K8;
        const uint4 v = __ldg(x + r * ldx8 + c);
        const unsigned ws[4] = {v.x, v.y, v.z, v.w};
        unsigned o[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float lo = to_f32<F16>(ws[j] & 0xFFFFu) * sc, hi = to_f32<F16>(ws[j] >> 16) * sc;
            o[j] = (unsigned) __half_as_ushort(__float2half_rn(lo)) | ((unsigned) __half_as_ushort(__float2half_rn(hi)) << 16);
        }
        out[i] = make_uint4(o[0], o[1], o[2], o[3]);
    }
}

// Y part z [T, N] (row stride ldy) = X[T, K] . W[N, K]^T over the steps of part z of Z, times 2^k.  Z = 1: the whole sum into
// Y directly (and the non-finite outputs counted here); Z > 1: the part's sum into Y + z * zstride (reduce_kernel counts).
static __global__ void __launch_bounds__(kThreads, 1) gemm_sk_kernel(const uint4* __restrict__ X, const uint4* __restrict__ Wt,
                                                                        float* __restrict__ Y, const int T, const int N, const int K,
                                                                        const int ldx8, const int64_t ldy, const int64_t zstride, const int Z,
                                                                        const float wmax) {
    __shared__ __align__(16) uint4 As[2][kBM * 5];
    __shared__ __align__(16) uint4 Bs[2][kBN * 5];
    const int tid = threadIdx.x, tx = tid & 15, ty = tid >> 4;
    const int m0 = blockIdx.y * kBM, n0 = blockIdx.x * kBN;
    const int K8 = K / 8;
    const int S = K / kBK, G = (S + kSteps - 1) / kSteps, z = blockIdx.z;
    const int s0 = (int) ((int64_t) G * z / Z) * kSteps;
    const int s1 = min(S, (int) ((int64_t) G * (z + 1) / Z) * kSteps);
    const int R = s1 - s0;   // steps of this part (a multiple of kSteps except where the part ends at K)
    const int kshift = shift_k(*(volatile const unsigned*) &g_hc.cur, wmax);
    const float ysc = pow2i(kshift);
    const uint4* gA[2]; const uint4* gB[2]; int so[2];
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int idx = tid + i * kThreads, row = idx >> 2, q = idx & 3;
        gA[i] = X + (size_t) min(m0 + row, T - 1) * ldx8 + q + (size_t) s0 * 4;
        gB[i] = Wt + (size_t) min(n0 + row, N - 1) * K8 + q + (size_t) s0 * 4;
        so[i] = row * 5 + q;
    }
    uint32_t acc[8][8];
    float acc32[8][8];
#pragma unroll
    for (int i = 0; i < 8; ++i)
#pragma unroll
        for (int j = 0; j < 8; ++j) { acc[i][j] = 0u; acc32[i][j] = 0.0f; }
    if (R > 0) {
        uint4 ra[2], rb[2];
#pragma unroll
        for (int i = 0; i < 2; ++i) { ra[i] = __ldg(gA[i]); rb[i] = __ldg(gB[i]); }
#pragma unroll
        for (int i = 0; i < 2; ++i) { As[0][so[i]] = ra[i]; Bs[0][so[i]] = rb[i]; }
        __syncthreads();
        for (int s = 0; s < R; ++s) {
            const int cur = s & 1;
            if (s + 1 < R) {
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
            if (s + 1 < R) {
#pragma unroll
                for (int i = 0; i < 2; ++i) { As[cur ^ 1][so[i]] = ra[i]; Bs[cur ^ 1][so[i]] = rb[i]; }
            }
            __syncthreads();
        }
        if (R % kSteps != 0) flush(acc, acc32);
    }
    float* Yz = Y + (size_t) z * (size_t) zstride;
    int bad = 0;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int row = m0 + ty + 16 * i;
        if (row < T) {
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const int col = n0 + tx + 16 * j;
                if (col < N) {
                    const float v = acc32[i][j] * ysc;
                    Yz[(size_t) row * (size_t) ldy + col] = v;
                    bad += !(fabsf(v) <= 3.0e38f);
                }
            }
        }
    }
    if (Z == 1 && bad != 0) atomicAdd(&g_hc.bad_y, (unsigned) bad);
}

// Y[t][n] = the parts' sums in the order z = 0 .. Z-1; the non-finite outputs counted
__global__ void __launch_bounds__(256) reduce_kernel(const float* __restrict__ P, float* __restrict__ Y, const int64_t T, const int64_t N,
                                                      const int64_t ldy, const int Z, const int64_t zstride) {
    int bad = 0;
    for (int64_t i = (int64_t) blockIdx.x * 256 + threadIdx.x; i < T * N; i += (int64_t) gridDim.x * 256) {
        const int64_t t = i / N, n = i - t * N;
        float s = P[i];
        for (int z = 1; z < Z; ++z) s += P[(size_t) z * (size_t) zstride + i];
        Y[(size_t) t * (size_t) ldy + n] = s;
        bad += !(fabsf(s) <= 3.0e38f);
    }
    if (bad != 0) atomicAdd(&g_hc.bad_y, (unsigned) bad);
}

// the BF16 weight's scan before its FP16 copy is made: stats[0] max |w| over the finite values (FP32 bits), [1] non-finite
// values, [2] finite values outside the FP16 range
__global__ void __launch_bounds__(256) wscan_kernel(const uint16_t* __restrict__ w, const int64_t n, unsigned* __restrict__ stats) {
    unsigned m = 0, nonfin = 0, big = 0;
    for (int64_t i = (int64_t) blockIdx.x * 256 + threadIdx.x; i < n; i += (int64_t) gridDim.x * 256) {
        const unsigned a = ((unsigned) w[i] & 0x7FFFu) << 16;
        if (a >= 0x7F800000u) { ++nonfin; continue; }
        if (a > 0x477FE000u) ++big;   // 65504 = 0x477FE000: values above it round to infinity in FP16
        m = max(m, a);
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        m = max(m, (unsigned) __shfl_xor_sync(0xFFFFFFFFu, (int) m, o));
        nonfin += __shfl_xor_sync(0xFFFFFFFFu, nonfin, o);
        big += __shfl_xor_sync(0xFFFFFFFFu, big, o);
    }
    if ((threadIdx.x & 31) == 0) {
        atomicMax(&stats[0], m);
        if (nonfin != 0) atomicAdd(&stats[1], nonfin);
        if (big != 0) atomicAdd(&stats[2], big);
    }
}

}  // namespace
}  // namespace strata::prefill::hfma::hc
