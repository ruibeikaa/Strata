// src/prefill/moe_ws_hf.cuh - STRATA_PF_HFMA_EXP (moe_ws.hpp): the weight-stationary gate/up product of a layer whose
// experts are 4-bit codes in their slots (STRATA_EXPERT_NIBBLE; strata/core/expert_nibble.hpp) on FP16 HFMA2 pairs.
// Depends on CUDA, moe_ws.hpp's Chunk and expert_nibble.hpp only (the test program includes it alone); included by
// moe_ws_gu_hf.cu.  Everything here has internal linkage.
//
// WHAT IT MULTIPLIES.  The layer's input as FP16 rows (the prompt path's `mixed_h`, one row a TOKEN, read through the
// routed row -> token table: no q8_1 quantisation and no routed copy of the activations) against weights decoded in
// the kernel from the same 16-byte item codes moe_ws_gu_nib.cu reads.  A weight is the INTEGER the int8 kernels use
// (a grid value, |v| <= 62) times the item's odd integer scale S (2 ls + 1, or ls): |v S| <= 1922 < 2048, exact in
// FP16, and times a power of two 2^-k (also exact).  The fp16 d of the 256-value superblock is applied in FP32 at the
// flush:  y = sum over steps of fd * (lane_lo + lane_hi),  fd = d * c_fmt * 2^k,
//   c_fmt  IQ2_S / IQ2_XS  1/8  ((2 ls_h + 1) per 16 values: scales16's d(2 ls + 1)/8)
//          IQ2_XXS         1/8  (d ls / 8, ls odd)
//          IQ3_XXS         1/4  ((ls d + d/2)/2 = d (2 ls + 1)/4)
//          IQ3_S           1    (ls d)
// so the product is the same dequantised weight times the FP16 image of the activation, with FP16 lane sums of 64
// products (one step of 128 K values: 4 items x 16 half2) widened to FP32 at every step.  Not bitwise the int8
// kernels' GU (which carry the q8_1 activation's 7-bit rounding); closer to the exact product, which the test program
// measures.
//
// RANGE.  A lane holds at most 64 products |v S 2^-k| |x| <= 64 Wmax 2^-k xmax, Wmax = max |v S| of the format
// (IQ2_*: 43 * 31, IQ3_XXS: 62 * 31, IQ3_S: 15 * 31).  The shift k is picked per launch from the layer's own xmax (a
// max-reduce over the FP16 rows, `xmax_kernel`, one launch ahead in the stream) so that bound is <= 2^15: no lane of
// this launch can leave FP16's range, whatever the data.  A non-finite x makes the launch's outputs non-finite and is
// counted (g_hf), never hidden.
#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

#include "strata/core/expert_nibble.hpp"
#include "strata/prefill/moe_ws.hpp"

namespace strata::prefill::ws::hf {
namespace {

namespace nb = strata::core::nibble;

constexpr int GU_ROWS = 1280, GU_K = 2560, QKK = 256, NB = GU_K / QKK, NSTEP = 2 * NB, NT = 128, PIECES = GU_ROWS / NT;
constexpr int ITEMS = (int) nb::kItems;
constexpr int CS = 20;            // staged code words per row in shared memory (16 + 4: a quarter warp's LDS.128 are conflict free)
constexpr int KMIN = -4, KMAX = 20;   // the weights' shift: 2^-k in [2^-20, 2^4]
constexpr int XROW16 = GU_K / 8;  // uint4 per FP16 row (320)

// family of a format: the code and scale layout it shares
//   0  IQ2_S, IQ2_XS   nibble = {8, 25, 43, 0} | sign << 2, two 4-bit scales an item (first / second 16 values), S = 2 ls + 1, c 1/8
//   1  IQ2_XXS         the same nibbles, one odd scale an item, S = ls, c 1/8
//   2  IQ3_XXS         nibble = {4, 12, 20, 28, 36, 44, 52, 62}[c & 7] | sign << 3, S = 2 ls + 1 (ls 0..15), c 1/4
//   3  IQ3_S           nibble u, v = 2u - 15, S = ls (odd 1..31), c 1; a byte holds the values j and j + 4
__host__ __device__ constexpr int family(const int gu_type) {
    return gu_type == nb::kIQ2_S || gu_type == nb::kIQ2_XS ? 0 : gu_type == nb::kIQ2_XXS ? 1 : gu_type == nb::kIQ3_XXS ? 2 : gu_type == nb::kIQ3_S ? 3 : -1;
}
__host__ __device__ constexpr float wmax(const int fam) { return fam == 2 ? 1922.0f : fam == 3 ? 465.0f : 1333.0f; }
__host__ __device__ constexpr float cfmt(const int fam) { return fam == 2 ? 0.25f : fam == 3 ? 1.0f : 0.125f; }

// what the report reads back (per device)
struct HfState {
    unsigned cur;                 // |x| bit pattern (FP16, sign cleared) max of the launch about to run; zeroed by a memset before each max-reduce
    unsigned launches;            // GU launches
    unsigned bad_x;               // launches whose input held a non-finite value
    unsigned bad_y;               // (token, weight row) outputs that came out non-finite
    unsigned all_max;             // max |x| pattern over the launches
    unsigned layer_max[64];       // per layer
    unsigned khist[KMAX - KMIN + 1];
};
__device__ HfState g_hf;

__device__ __forceinline__ uint32_t hfma2(const uint32_t a, const uint32_t b, const uint32_t c) {
    uint32_t d;
    asm("fma.rn.f16x2 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
    return d;
}
__device__ __forceinline__ uint32_t hmul2(const uint32_t a, const uint32_t b) {
    uint32_t d;
    asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b));
    return d;
}
__device__ __forceinline__ void h2f(const uint32_t h, float& lo, float& hi) {
    asm("{\n\t.reg .f16 l, h;\n\tmov.b32 {l, h}, %2;\n\tcvt.f32.f16 %0, l;\n\tcvt.f32.f16 %1, h;\n\t}" : "=f"(lo), "=f"(hi) : "r"(h));
}
__device__ __forceinline__ uint32_t prmt(const uint32_t a, const uint32_t b, const uint32_t sel) {
    uint32_t r;
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(sel));
    return r;
}
__device__ __forceinline__ uint32_t pack_h2(const float lo, const float hi) {
    return (uint32_t) __half_as_ushort(__float2half_rn(lo)) | ((uint32_t) __half_as_ushort(__float2half_rn(hi)) << 16);
}

// the weight value of a 4-bit code (what the int8 kernels' decode gives)
template <int FAM>
__device__ __forceinline__ float code_value(const int c) {
    if constexpr (FAM == 3) {
        return (float) (2 * c - 15);
    } else if constexpr (FAM == 2) {
        const float mag = (c & 7) == 7 ? 62.0f : (float) (4 + 8 * (c & 7));
        return (c & 8) ? -mag : mag;
    } else {
        const int m = c & 3;
        const float mag = m == 0 ? 8.0f : m == 1 ? 25.0f : m == 2 ? 43.0f : 0.0f;
        return (c & 4) ? -mag : mag;
    }
}

__device__ __forceinline__ Chunk cta_work(const Chunk* __restrict__ chunks, int& r0) {
    r0 = (int) (blockIdx.x % PIECES) * NT;
    const int4 c = __ldg((const int4*) chunks + blockIdx.x / PIECES);
    return Chunk{c.x, c.y, c.z, c.w};
}

// y[row][1280] (the GU rows) for the chunk's rows: TPP tokens a pass, a weight row a thread (128 rows a CTA).
template <int TPP, int FAM>
__global__ void __launch_bounds__(NT, 3) gu_kernel(const Chunk* __restrict__ chunks, const char* __restrict__ w,
                                                  const int64_t* __restrict__ xoff, const uint4* __restrict__ xh,
                                                  const int32_t* __restrict__ src, float* __restrict__ dst, const int layer) {
    static_assert(TPP == 16 || TPP == 32, "");
    constexpr int NAS = TPP / 8;   // 16-byte activation pieces a thread stages a step (TPP tokens x 16 pieces / 128 threads)
    __shared__ __align__(16) uint4 act_s[TPP * 16];   // [token][16 pieces]; piece 4 s + m = 8 values (group m of item s) as 4 half2 words
    __shared__ __align__(16) uint32_t csm[NT * CS];
    __shared__ uint32_t lut[256];     // code byte (two nibbles) -> the two weight values as half2
    __shared__ uint32_t hs_tab[32];   // scale index -> S 2^-k as half2 (both lanes)
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;

    // the weights' shift for this launch, from the layer's xmax
    const unsigned cur = *(volatile const unsigned*) &g_hf.cur;
    const float xmax = __half2float(__ushort_as_half((unsigned short) cur));   // inf / NaN stay
    const float need = xmax * (wmax(FAM) * (1.0f / 512.0f));                    // 2^k >= 64 Wmax xmax / 2^15
    int k;
    if (!(need <= 1e30f)) {
        k = KMAX;
    } else if (need <= 0.0f) {
        k = KMIN;
    } else {
        int e;
        (void) frexpf(need, &e);
        k = e < KMIN ? KMIN : e > KMAX ? KMAX : e;
    }
    if (blockIdx.x == 0 && tid == 0) {
        atomicAdd(&g_hf.launches, 1u);
        atomicMax(&g_hf.all_max, cur);
        atomicMax(&g_hf.layer_max[layer & 63], cur);
        atomicAdd(&g_hf.khist[k - KMIN], 1u);
        if (cur >= 0x7C00u) atomicAdd(&g_hf.bad_x, 1u);
    }
    const float kdn = ldexpf(1.0f, -k), kup = ldexpf(1.0f, k);
    if (tid < 32) {
        const int sv = (FAM == 1 || FAM == 3) ? tid : 2 * tid + 1;
        const uint32_t h = (uint32_t) __half_as_ushort(__float2half_rn((float) sv * kdn));
        hs_tab[tid] = h | (h << 16);
    }
    for (int e = tid; e < 256; e += NT) lut[e] = pack_h2(code_value<FAM>(e & 15), code_value<FAM>(e >> 4));

    int r0;
    const Chunk c = cta_work(chunks, r0);
    const char* __restrict__ slot = w + xoff[c.j];
    const uint32_t* __restrict__ scw = (const uint32_t*) (slot + nb::kScOff + (size_t) (r0 + tid) * ITEMS);
    const unsigned short* __restrict__ dr = (const unsigned short*) (slot + nb::kDOff) + (size_t) (r0 + tid) * NB;
    const int srow = warp * 8 + (lane >> 2), sq = lane & 3;   // the staging lane's weight row (+ 32 i) and item
    const uint4* __restrict__ cg = (const uint4*) slot + (size_t) (r0 + srow) * ITEMS + sq;
    uint32_t* __restrict__ cs_st = csm + srow * CS + 4 * sq;
    const uint32_t* cw = csm + tid * CS;
    const int piece = tid & 15, trow = tid >> 4;   // the staging lane's activation piece and first token
    int bad = 0;
    for (int p0 = 0; p0 < c.rows; p0 += TPP) {
        const int nt = min(TPP, c.rows - p0);
        const int trow0 = c.row0 + p0;
        const uint4* ap[NAS];
#pragma unroll
        for (int i = 0; i < NAS; ++i) {
            const int t = trow + 8 * i;
            ap[i] = t < nt ? xh + (size_t) __ldg(src + trow0 + t) * XROW16 + piece : nullptr;
        }
        float acc[TPP];
        uint32_t hacc[TPP];
#pragma unroll
        for (int t = 0; t < TPP; ++t) { acc[t] = 0.0f; hacc[t] = 0u; }
#pragma unroll 1
        for (int hc = 0; hc < NSTEP; ++hc) {
            const int kc = hc >> 1;
            __syncthreads();
            uint4 v[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) v[i] = __ldg(cg + (size_t) i * 32 * ITEMS + 4 * hc);
            const uint32_t scv = __ldg(scw + hc);
            const int dv = (int) __ldg(dr + kc);
            uint4 av[NAS];
#pragma unroll
            for (int i = 0; i < NAS; ++i) av[i] = ap[i] != nullptr ? __ldg(ap[i] + hc * 16) : make_uint4(0u, 0u, 0u, 0u);
#pragma unroll
            for (int i = 0; i < 4; ++i) *(uint4*) (cs_st + i * 32 * CS) = v[i];
#pragma unroll
            for (int i = 0; i < NAS; ++i) {
                uint4 a = av[i];
                if constexpr (FAM == 3) {   // IQ3_S's code byte j holds the values j and j + 4 of its group: pair them
                    const uint4 o = make_uint4(prmt(a.x, a.z, 0x5410u), prmt(a.x, a.z, 0x7632u), prmt(a.y, a.w, 0x5410u), prmt(a.y, a.w, 0x7632u));
                    a = o;
                }
                act_s[(trow + 8 * i) * 16 + piece] = a;
            }
            __syncthreads();
            const float fd = __half2float(__ushort_as_half((unsigned short) (dv & 0xFFFF))) * (cfmt(FAM) * kup);
#pragma unroll 1
            for (int s = 0; s < 4; ++s) {
                const uint4 u = *(const uint4*) (cw + 4 * s);
                const uint32_t q[4] = {u.x, u.y, u.z, u.w};
                const int ls = (int) ((scv >> (8 * s)) & 0xFF);
                uint32_t h0, h1;
                if constexpr (FAM == 0) {
                    h0 = hs_tab[ls & 15];
                    h1 = hs_tab[(ls >> 4) & 15];
                } else {
                    h0 = hs_tab[ls & 31];
                    h1 = h0;
                }
                uint32_t wv[16];
#pragma unroll
                for (int m = 0; m < 4; ++m) {
                    const uint32_t hsm = m < 2 ? h0 : h1;
#pragma unroll
                    for (int j = 0; j < 4; ++j) wv[4 * m + j] = hmul2(lut[(q[m] >> (8 * j)) & 0xFF], hsm);
                }
#pragma unroll
                for (int t0 = 0; t0 < TPP; t0 += 4) {
                    if (t0 < nt) {
#pragma unroll
                        for (int t = t0; t < t0 + 4; ++t) {
                            const uint4* a4 = act_s + t * 16 + 4 * s;
                            const uint4 a0 = a4[0], a1 = a4[1], a2 = a4[2], a3 = a4[3];
                            uint32_t hh = hacc[t];
                            hh = hfma2(wv[0], a0.x, hh);  hh = hfma2(wv[1], a0.y, hh);  hh = hfma2(wv[2], a0.z, hh);  hh = hfma2(wv[3], a0.w, hh);
                            hh = hfma2(wv[4], a1.x, hh);  hh = hfma2(wv[5], a1.y, hh);  hh = hfma2(wv[6], a1.z, hh);  hh = hfma2(wv[7], a1.w, hh);
                            hh = hfma2(wv[8], a2.x, hh);  hh = hfma2(wv[9], a2.y, hh);  hh = hfma2(wv[10], a2.z, hh); hh = hfma2(wv[11], a2.w, hh);
                            hh = hfma2(wv[12], a3.x, hh); hh = hfma2(wv[13], a3.y, hh); hh = hfma2(wv[14], a3.z, hh); hh = hfma2(wv[15], a3.w, hh);
                            hacc[t] = hh;
                        }
                    }
                }
            }
            // 64 products a lane: widen into the FP32 sums
#pragma unroll
            for (int t0 = 0; t0 < TPP; t0 += 4) {
                if (t0 < nt) {
#pragma unroll
                    for (int t = t0; t < t0 + 4; ++t) {
                        float lo, hi;
                        h2f(hacc[t], lo, hi);
                        acc[t] = __fmaf_rn(fd, lo + hi, acc[t]);
                        hacc[t] = 0u;
                    }
                }
            }
        }
#pragma unroll
        for (int t = 0; t < TPP; ++t)
            if (t < nt) {
                dst[(size_t) (trow0 + t) * GU_ROWS + r0 + tid] = acc[t];
                bad += !(fabsf(acc[t]) <= 3.0e38f);
            }
    }
    if (bad != 0) atomicAdd(&g_hf.bad_y, (unsigned) bad);
}

// max |x| of the FP16 rows (bit pattern with the sign cleared: for finite values the order of the patterns is the order
// of the magnitudes, inf is 0x7C00 and a NaN is above it); `cur` was zeroed by a memset before
__global__ void __launch_bounds__(256) xmax_kernel(const uint4* __restrict__ x, const int64_t n16) {
    uint32_t m = 0;
    for (int64_t i = (int64_t) blockIdx.x * 256 + threadIdx.x; i < n16; i += (int64_t) gridDim.x * 256) {
        const uint4 v = __ldg(x + i);
        const uint32_t ws[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            m = max(m, ws[j] & 0x7FFFu);
            m = max(m, (ws[j] >> 16) & 0x7FFFu);
        }
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) m = max(m, (uint32_t) __shfl_xor_sync(0xFFFFFFFFu, (int) m, o));
    if ((threadIdx.x & 31) == 0 && m != 0) atomicMax(&g_hf.cur, m);
}

}  // namespace
}  // namespace strata::prefill::ws::hf
