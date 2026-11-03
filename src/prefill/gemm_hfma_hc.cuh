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

// STRATA_PF_SASS: the hand-scheduled hc kernels (gemm_sass.cuh) take the output scale 2^k as a float from a device word instead of
// working out shift_k themselves: this one-thread kernel (after cvt_kernel in the stream) writes the very value gemm_sk_kernel's
// `ysc` is, from the same g_hc.cur and the same shift_k.  g_hc_sink is where the split-K parts' non-finite counts go (reduce_kernel
// counts the sums; gemm_sk_kernel counts only when Z == 1).
__device__ float g_hc_ysc;
__device__ unsigned g_hc_sink;
__global__ void ysc_kernel(float* __restrict__ out, const float wmax) {
    *out = pow2i(shift_k(*(volatile const unsigned*) &g_hc.cur, wmax));
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

// ---- STRATA_PF_HFMA_SMALL ---------------------------------------------------------------------------------------------------------------
// xmax_kernel + cvt_kernel are two passes over the slice, the first dominated by its 64-bit divide per element (row = i / K8).  The pair
// below is one pass that takes the max |x| AND writes the FP16 image at shift 0 (scale 1), and a second kernel that does the work only for
// a slice whose shift is not 0 (xmax * wmax > 512, or xmax > 32768: not seen on any real read so far) - redoing the image from x at the
// slice's own shift, exactly cvt_kernel's - and counts the slice (g_hc.slices / xmax_all / khist / bad_x) either way, as cvt_kernel does.
// For a slice at shift 0 the image is cvt_kernel's bit for bit (to_f32 * 1.0f is exact), for any other it is cvt_kernel's again, so the
// outputs of the products are unchanged.  No division: a row of the slice is addressed straight when ldx8 == K8 and by stepping (row, col)
// otherwise (the grid-stride step is a constant).  n8 < 2^31 (the host checks).
template <bool F16>
__global__ void __launch_bounds__(256) xcvt_kernel(const uint4* __restrict__ x, uint4* __restrict__ out, const unsigned n8, const unsigned K8,
                                                   const unsigned ldx8) {
    unsigned m = 0;
    const unsigned step = gridDim.x * 256u;
    unsigned i = blockIdx.x * 256u + threadIdx.x;
    const bool straight = ldx8 == K8;
    unsigned r = 0, c = 0, sr = 0, sc = 0;
    if (!straight) {
        r = i / K8;
        c = i - r * K8;
        sr = step / K8;
        sc = step - sr * K8;
    }
    for (; i < n8; i += step) {
        const uint4 v = __ldg(x + (straight ? (size_t) i : (size_t) r * ldx8 + c));
        const unsigned ws[4] = {v.x, v.y, v.z, v.w};
        unsigned o[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            m = max(m, abs_bits<F16>(ws[j] & 0xFFFFu));
            m = max(m, abs_bits<F16>(ws[j] >> 16));
            o[j] = (unsigned) __half_as_ushort(__float2half_rn(to_f32<F16>(ws[j] & 0xFFFFu))) |
                   ((unsigned) __half_as_ushort(__float2half_rn(to_f32<F16>(ws[j] >> 16))) << 16);
        }
        out[i] = make_uint4(o[0], o[1], o[2], o[3]);
        if (!straight) {
            c += sc;
            r += sr;
            if (c >= K8) { c -= K8; ++r; }
        }
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) m = max(m, (unsigned) __shfl_xor_sync(0xFFFFFFFFu, (int) m, o));
    if ((threadIdx.x & 31) == 0 && m != 0) atomicMax(&g_hc.cur, m);
}

// cvt_kernel after xcvt_kernel: the slice's counts, and the image again only when the slice's shift is not 0
template <bool F16>
__global__ void __launch_bounds__(256) cvt_fix_kernel(const uint4* __restrict__ x, uint4* __restrict__ out, const unsigned n8, const unsigned K8,
                                                      const unsigned ldx8, const float wmax) {
    const unsigned cur = *(volatile const unsigned*) &g_hc.cur;
    const int k = shift_k(cur, wmax);
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        atomicAdd(&g_hc.slices, 1u);
        atomicMax(&g_hc.xmax_all, cur);
        atomicAdd(&g_hc.khist[k], 1u);
        if (cur >= 0x7F800000u) atomicAdd(&g_hc.bad_x, 1u);
    }
    if (k == 0) return;
    const float sc = pow2i(-k);
    const unsigned step = gridDim.x * 256u;
    unsigned i = blockIdx.x * 256u + threadIdx.x;
    const bool straight = ldx8 == K8;
    unsigned r = 0, c = 0, sr = 0, sc2 = 0;
    if (!straight) {
        r = i / K8;
        c = i - r * K8;
        sr = step / K8;
        sc2 = step - sr * K8;
    }
    for (; i < n8; i += step) {
        const uint4 v = __ldg(x + (straight ? (size_t) i : (size_t) r * ldx8 + c));
        const unsigned ws[4] = {v.x, v.y, v.z, v.w};
        unsigned o[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float lo = to_f32<F16>(ws[j] & 0xFFFFu) * sc, hi = to_f32<F16>(ws[j] >> 16) * sc;
            o[j] = (unsigned) __half_as_ushort(__float2half_rn(lo)) | ((unsigned) __half_as_ushort(__float2half_rn(hi)) << 16);
        }
        out[i] = make_uint4(o[0], o[1], o[2], o[3]);
        if (!straight) {
            c += sc2;
            r += sr;
            if (c >= K8) { c -= K8; ++r; }
        }
    }
}

// ---- STRATA_PF_HC_NOCVT ---------------------------------------------------------------------------------------------------------------
// The writers of the BF16 image (kernels.cu) recorded the slice's max in `slot` (abs_bits<false>: the max BF16 magnitude bits << 16, 0 for an all-zero
// slice) and wrote its FP16 image at shift 0 (half_rn(float(bf16)), xcvt_kernel<false>'s conversion), so this kernel is cvt_fix_kernel<false> with the
// max read from the slot: g_hc.cur gets it (what xcvt_kernel's atomicMax left there: the ysc kernel, the products and inj_kernel read it from g_hc),
// the slice is counted the same way, and a slice with a shift above 0 is rewritten from the BF16 image `x` into `out` by the same code (straight
// rows only: the host takes this route for ldx == K).
__global__ void __launch_bounds__(256) cvt_slot_kernel(const uint4* __restrict__ x, uint4* __restrict__ out, const unsigned n8, const float wmax,
                                                       const unsigned* __restrict__ slot) {
    const unsigned cur = *(volatile const unsigned*) slot;
    const int k = shift_k(cur, wmax);
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        *(volatile unsigned*) &g_hc.cur = cur;
        atomicAdd(&g_hc.slices, 1u);
        atomicMax(&g_hc.xmax_all, cur);
        atomicAdd(&g_hc.khist[k], 1u);
        if (cur >= 0x7F800000u) atomicAdd(&g_hc.bad_x, 1u);
    }
    if (k == 0) return;
    const float sc = pow2i(-k);
    const unsigned step = gridDim.x * 256u;
    for (unsigned i = blockIdx.x * 256u + threadIdx.x; i < n8; i += step) {
        const uint4 v = __ldg(x + i);
        const unsigned ws[4] = {v.x, v.y, v.z, v.w};
        unsigned o[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float lo = to_f32<false>(ws[j] & 0xFFFFu) * sc, hi = to_f32<false>(ws[j] >> 16) * sc;
            o[j] = (unsigned) __half_as_ushort(__float2half_rn(lo)) | ((unsigned) __half_as_ushort(__float2half_rn(hi)) << 16);
        }
        out[i] = make_uint4(o[0], o[1], o[2], o[3]);
    }
}
// STRATA_PF_HC_NOCVT_CHECK=1: the old route's image (xcvt_kernel + cvt_fix_kernel, into the scratch `ref`) and max (saved in g_nc_ref_cur by
// nc_save_kernel right after) against the new route's (`img`, g_hc.cur): the counts of slices, of slices with a different max, of different 16-bit
// words and of slices with a shift (every one of them compared)
__device__ unsigned g_nc_ref_cur;
__device__ unsigned g_nc_chk[4];   // slices, max differs, words differ, shifted slices
__global__ void nc_save_kernel() { g_nc_ref_cur = *(volatile unsigned*) &g_hc.cur; }
__global__ void __launch_bounds__(256) nc_cmp_kernel(const uint4* __restrict__ ref, const uint4* __restrict__ img, const unsigned n8, const float wmax) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        const unsigned cur = *(volatile unsigned*) &g_hc.cur;
        atomicAdd(&g_nc_chk[0], 1u);
        if (cur != g_nc_ref_cur) atomicAdd(&g_nc_chk[1], 1u);
        if (shift_k(cur, wmax) != 0) atomicAdd(&g_nc_chk[3], 1u);
    }
    unsigned bad = 0;
    for (unsigned i = blockIdx.x * 256u + threadIdx.x; i < n8; i += gridDim.x * 256u) {
        const uint4 a = ref[i], b = img[i];
        bad += (a.x != b.x) + (a.y != b.y) + (a.z != b.z) + (a.w != b.w);
    }
    if (bad != 0) atomicAdd(&g_nc_chk[2], bad);
}

// The inject product (N 4, K 10240: the 4 columns of one hyper-connection read) on FP16 HFMA2 pairs: Y[t][n] = 2^k * sum_j x16[t][j] w16[n][j],
// from the very FP16 image the down product reads (same slice, same shift: k from the pair's larger max |w|).  A 128-wide tile of the GEMM would
// compute 4 useful columns of 128, so this is the dot products themselves: a block of 256 threads keeps its 5 chunks of 8 columns of the 4 weight rows
// in registers and walks kInjRows rows; a thread's 40 products per output are summed in FP16 pairs (20 a half-lane: fewer than the GEMM's 64), widened
// to FP32 and added across the block in a fixed order (warp xor-shuffle tree, then the 8 warps in order): deterministic.
constexpr int kInjN = 4, kInjK8 = 1280, kInjCpt = kInjK8 / 256, kInjRows = 8;
__global__ void __launch_bounds__(256) inj_kernel(const uint4* __restrict__ X, const uint4* __restrict__ W, float* __restrict__ Y, const int rows,
                                                  const int64_t ldy, const float wmax) {
    __shared__ float part[kInjRows][8][kInjN];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int r0 = blockIdx.x * kInjRows;
    const int nr = min(kInjRows, rows - r0);
    const float ysc = pow2i(shift_k(*(volatile const unsigned*) &g_hc.cur, wmax));
    uint4 w[kInjN][kInjCpt];
#pragma unroll
    for (int n = 0; n < kInjN; ++n)
#pragma unroll
        for (int c = 0; c < kInjCpt; ++c) w[n][c] = __ldg(W + (size_t) n * kInjK8 + tid + 256 * c);
    uint4 xv[kInjCpt];
#pragma unroll
    for (int c = 0; c < kInjCpt; ++c) xv[c] = __ldg(X + (size_t) r0 * kInjK8 + tid + 256 * c);
    for (int r = 0; r < nr; ++r) {
        uint4 xn[kInjCpt];
#pragma unroll
        for (int c = 0; c < kInjCpt; ++c) xn[c] = xv[c];
        if (r + 1 < nr) {
#pragma unroll
            for (int c = 0; c < kInjCpt; ++c) xn[c] = __ldg(X + (size_t) (r0 + r + 1) * kInjK8 + tid + 256 * c);
        }
        uint32_t acc[kInjN] = {0u, 0u, 0u, 0u};
#pragma unroll
        for (int c = 0; c < kInjCpt; ++c)
#pragma unroll
            for (int n = 0; n < kInjN; ++n) {
                acc[n] = hfma2(xv[c].x, w[n][c].x, acc[n]);
                acc[n] = hfma2(xv[c].y, w[n][c].y, acc[n]);
                acc[n] = hfma2(xv[c].z, w[n][c].z, acc[n]);
                acc[n] = hfma2(xv[c].w, w[n][c].w, acc[n]);
            }
        float v[kInjN];
#pragma unroll
        for (int n = 0; n < kInjN; ++n) {
            float lo, hi;
            h2f(acc[n], lo, hi);
            v[n] = lo + hi;
        }
#pragma unroll
        for (int n = 0; n < kInjN; ++n)
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) v[n] += __shfl_xor_sync(0xFFFFFFFFu, v[n], o);
        if (lane == 0) {
#pragma unroll
            for (int n = 0; n < kInjN; ++n) part[r][warp][n] = v[n];
        }
#pragma unroll
        for (int c = 0; c < kInjCpt; ++c) xv[c] = xn[c];
    }
    __syncthreads();
    if (tid < nr * kInjN) {
        const int r = tid / kInjN, n = tid % kInjN;
        float s = 0.0f;
#pragma unroll
        for (int wi = 0; wi < 8; ++wi) s += part[r][wi][n];
        s *= ysc;
        Y[(size_t) (r0 + r) * (size_t) ldy + n] = s;
        if (!(fabsf(s) <= 3.0e38f)) atomicAdd(&g_hc.bad_y, 1u);
    }
}

}  // namespace
}  // namespace strata::prefill::hfma::hc
