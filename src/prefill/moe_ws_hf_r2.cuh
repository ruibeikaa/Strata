// src/prefill/moe_ws_hf_r2.cuh - STRATA_PF_HFMA_R2 (moe_ws.hpp): the gate/up kernel of moe_ws_hf.cuh with two weight rows a thread.
// Found by the Kernel Forge search (kf-work, node 17 of 19); same operator, same epilogue contract as
// moe_ws_hf.cuh's gu_kernel (it is included after that header, shares its g_hf state, its shift rule and its output layout):
//   y[routed row][r], r = 0 .. 1279, K = 2560, FP16 lane sums of 64 products (4 items x 16 half2 a 128-K step) widened to FP32
//   with fd = d * c_fmt * 2^k each step, the 20 steps summed in order.
// What is different (all in the arithmetic's schedule, none in its grouping):
//  * rows2: a thread owns TWO weight rows (a 128-thread CTA covers 256 rows of the expert, 5 pieces a chunk, 2 CTAs an SM = 8 warps),
//    so every activation word read from shared memory (a broadcast LDS.128 as four LDS.32) feeds 8 HFMA2 instead of 4;
//  * the previous step's FP32 widening (flush) runs in the shadow of the next step's global loads, between their issue and their
//    store to shared memory;
//  * IQ3_S (family 3) decodes a code word by one shift, one LOP3 and one HFMA2 a half2: the nibble u becomes the FP16 number
//    1024 + 2u + 1 (0x6401 | u << 1), and fma(that, S', -1040 S') = (2u - 15) S' exactly (1040 S' is an integer multiple of
//    2^-k below 2^11, the fused product is exact).  The pair (nibble i, nibble i + 4) is one half2, so the activation staging
//    pairs (k0,k2) (k4,k6) (k1,k3) (k5,k7) instead of the compiler kernel's (k0,k4) (k1,k5) (k2,k6) (k3,k7); the sums regroup
//    the same 64 products, and the weights are the very same values (rel. rms error against FP64 1.002x the compiler kernel's).
//    -1040 S' must fit FP16 (S' <= 31 2^-k): the weights' shift is clamped to k >= -1 for the decode (a power of two, exact; fd
//    compensates), the launch's recorded shift is the unclamped one.
//  * families 0, 1, 2 keep the compiler kernel's table decode (bit-identical outputs: rows2 and the flush move change only when
//    the same operations run).
// Everything here has internal linkage; the host side is moe_ws_gu_hf.cu.
#pragma once

#include "moe_ws_hf.cuh"

namespace strata::prefill::ws::hf {
namespace {

constexpr int R2_NT = 128, R2_RPT = 2, R2_ROWS = R2_NT * R2_RPT, R2_PIECES = GU_ROWS / R2_ROWS;   // 5 pieces of 256 rows a chunk

__device__ __forceinline__ uint32_t lds32(const uint32_t addr) {
    uint32_t r;
    asm("ld.shared.b32 %0, [%1];" : "=r"(r) : "r"(addr));
    return r;
}

// (a & m) | c in ONE lop3 (ptxas turns the plain C expression into LOP32I.AND + LOP32I.OR: two instructions)
__device__ __forceinline__ uint32_t and_or(const uint32_t a, const uint32_t m, const uint32_t c) {
    uint32_t d;
    asm("lop3.b32 %0, %1, %2, %3, 0xEA;" : "=r"(d) : "r"(a), "r"(m), "r"(c));
    return d;
}

// IQ3_S: a code word (8 nibbles u, v = 2u - 15) -> 4 half2 of (value x scale): pairs (nibble i, nibble i + 4); hs = S' (both lanes),
// nk = -1040 S' (both lanes)
__device__ __forceinline__ void dec_iq3s_l3(const uint32_t q, const uint32_t hs, const uint32_t nk, uint32_t (&w)[4]) {
    uint32_t m = 0x001E001Eu, c = 0x64016401u;
    asm("" : "+r"(m), "+r"(c));   // keep the two constants in registers (hoisted out of the loops) so that lop3 takes them as register operands
    w[0] = hfma2(and_or(q << 1, m, c), hs, nk);
    w[1] = hfma2(and_or(q >> 3, m, c), hs, nk);
    w[2] = hfma2(and_or(q >> 7, m, c), hs, nk);
    w[3] = hfma2(and_or(q >> 11, m, c), hs, nk);
}

// one 32-value item (4 code words) -> 16 half2 weights
template <int FAM>
__device__ __forceinline__ void decode_item2(const uint4 u, const uint32_t h0, const uint32_t h1, const uint32_t nkv, const uint32_t* lut, uint32_t (&wv)[16]) {
    const uint32_t q[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
    for (int m = 0; m < 4; ++m) {
        uint32_t d4[4];
        if constexpr (FAM == 3) {
            dec_iq3s_l3(q[m], h0, nkv, d4);
        } else {
#pragma unroll
            for (int j = 0; j < 4; ++j) d4[j] = hmul2(lut[(q[m] >> (8 * j)) & 0xFF], m < 2 ? h0 : h1);
        }
#pragma unroll
        for (int j = 0; j < 4; ++j) wv[4 * m + j] = d4[j];
    }
}

// y[row][1280] for the chunk's rows: TPP tokens a pass, two weight rows a thread (256 rows a CTA)
template <int TPP, int FAM>
__global__ void __launch_bounds__(R2_NT, 2) gu2_kernel(const Chunk* __restrict__ chunks, const char* __restrict__ w,
                                                      const int64_t* __restrict__ xoff, const uint4* __restrict__ xh,
                                                      const int32_t* __restrict__ src, float* __restrict__ dst, const int layer) {
    static_assert(TPP == 16 || TPP == 32, "");
    constexpr int NAS = TPP / 8;   // 16-byte activation pieces a thread stages a step
    __shared__ __align__(16) uint4 act_s[TPP * 16];   // [token][16 pieces]; piece 4 s + m = 8 values (group m of item s) as 4 half2 words
    __shared__ __align__(16) uint32_t csm[R2_ROWS * CS];
    __shared__ uint32_t lut[256];     // code byte (two nibbles) -> the two weight values as half2
    __shared__ uint32_t hs_tab[32];   // scale index -> S 2^-k as half2 (both lanes)
    __shared__ uint32_t nk_tab[32];   // FAM 3: -1040 S 2^-k as half2
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;

    // the weights' shift for this launch, from the layer's xmax (moe_ws_hf.cuh's gu_kernel prologue, the same arithmetic)
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
    const int kw = FAM == 3 ? max(k, -1) : k;   // FAM 3 ALU decode: 1040 S 2^-kw <= 65504; fd below compensates the power of two exactly
    const float kdn = ldexpf(1.0f, -kw), kup = ldexpf(1.0f, kw);
    if (tid < 32) {
        const int sv = (FAM == 1 || FAM == 3) ? tid : 2 * tid + 1;
        const uint32_t h = (uint32_t) __half_as_ushort(__float2half_rn((float) sv * kdn));
        hs_tab[tid] = h | (h << 16);
        if constexpr (FAM == 3) {
            const uint32_t g = (uint32_t) __half_as_ushort(__float2half_rn(-1040.0f * (float) sv * kdn));
            nk_tab[tid] = g | (g << 16);
        }
    }
    for (int e = tid; e < 256; e += R2_NT) lut[e] = pack_h2(code_value<FAM>(e & 15), code_value<FAM>(e >> 4));

    const int r0 = (int) (blockIdx.x % R2_PIECES) * R2_ROWS;
    const int4 cc = __ldg((const int4*) chunks + blockIdx.x / R2_PIECES);
    const Chunk c{cc.x, cc.y, cc.z, cc.w};
    const char* __restrict__ slot = w + xoff[c.j];
    const uint32_t* __restrict__ scwA = (const uint32_t*) (slot + nb::kScOff + (size_t) (r0 + tid) * ITEMS);
    const uint32_t* __restrict__ scwB = scwA + (size_t) R2_NT * ITEMS / 4;
    const unsigned short* __restrict__ drA = (const unsigned short*) (slot + nb::kDOff) + (size_t) (r0 + tid) * NB;
    const unsigned short* __restrict__ drB = drA + (size_t) R2_NT * NB;
    const int srow = warp * 8 + (lane >> 2), sq = lane & 3;   // the staging lane's weight row (+ 32 i) and item
    const uint4* __restrict__ cg = (const uint4*) slot + (size_t) (r0 + srow) * ITEMS + sq;
    uint32_t* __restrict__ cs_st = csm + srow * CS + 4 * sq;
    const uint32_t* cwA = csm + tid * CS;
    const uint32_t* cwB = csm + (tid + R2_NT) * CS;
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
        float accA[TPP], accB[TPP];
        uint32_t hA[TPP], hB[TPP];
#pragma unroll
        for (int t = 0; t < TPP; ++t) { accA[t] = 0.0f; accB[t] = 0.0f; hA[t] = 0u; hB[t] = 0u; }
        auto flush = [&](const float fA, const float fB) {   // 64 products a lane: widen into the FP32 sums
#pragma unroll
            for (int t0 = 0; t0 < TPP; t0 += 4) {
                if (t0 < nt) {
#pragma unroll
                    for (int t = t0; t < t0 + 4; ++t) {
                        float lo, hi;
                        h2f(hA[t], lo, hi);
                        accA[t] = __fmaf_rn(fA, lo + hi, accA[t]);
                        hA[t] = 0u;
                        h2f(hB[t], lo, hi);
                        accB[t] = __fmaf_rn(fB, lo + hi, accB[t]);
                        hB[t] = 0u;
                    }
                }
            }
        };
        float fdA = 0.0f, fdB = 0.0f;
#pragma unroll 1
        for (int hc = 0; hc < NSTEP; ++hc) {
            const int kc = hc >> 1;
            __syncthreads();
            uint4 v[8];
#pragma unroll
            for (int i = 0; i < 8; ++i) v[i] = __ldg(cg + (size_t) i * 32 * ITEMS + 4 * hc);
            const uint32_t scvA = __ldg(scwA + hc), scvB = __ldg(scwB + hc);
            const int dvA = (int) __ldg(drA + kc), dvB = (int) __ldg(drB + kc);
            uint4 av[NAS];
#pragma unroll
            for (int i = 0; i < NAS; ++i) av[i] = ap[i] != nullptr ? __ldg(ap[i] + hc * 16) : make_uint4(0u, 0u, 0u, 0u);
            if (hc > 0) flush(fdA, fdB);   // the previous step's FP32 widening runs in the shadow of this step's global loads
#pragma unroll
            for (int i = 0; i < 8; ++i) *(uint4*) (cs_st + i * 32 * CS) = v[i];
#pragma unroll
            for (int i = 0; i < NAS; ++i) {
                uint4 a = av[i];
                if constexpr (FAM == 3) {   // the ALU decode pairs the nibbles (i, i + 4) of a code word as (k0,k2) (k4,k6) (k1,k3) (k5,k7)
                    const uint4 o = make_uint4(prmt(a.x, a.y, 0x5410u), prmt(a.z, a.w, 0x5410u), prmt(a.x, a.y, 0x7632u), prmt(a.z, a.w, 0x7632u));
                    a = o;
                }
                act_s[(trow + 8 * i) * 16 + piece] = a;
            }
            __syncthreads();
            fdA = __half2float(__ushort_as_half((unsigned short) (dvA & 0xFFFF))) * (cfmt(FAM) * kup);
            fdB = __half2float(__ushort_as_half((unsigned short) (dvB & 0xFFFF))) * (cfmt(FAM) * kup);
#pragma unroll 1
            for (int s = 0; s < 4; ++s) {
                uint32_t wa[16], wb[16];
                {
                    const int ls = (int) ((scvA >> (8 * s)) & 0xFF);
                    uint32_t h0, h1;
                    if constexpr (FAM == 0) { h0 = hs_tab[ls & 15]; h1 = hs_tab[(ls >> 4) & 15]; }
                    else { h0 = hs_tab[ls & 31]; h1 = h0; }
                    decode_item2<FAM>(*(const uint4*) (cwA + 4 * s), h0, h1, FAM == 3 ? nk_tab[ls & 31] : 0u, lut, wa);
                }
                {
                    const int ls = (int) ((scvB >> (8 * s)) & 0xFF);
                    uint32_t h0, h1;
                    if constexpr (FAM == 0) { h0 = hs_tab[ls & 15]; h1 = hs_tab[(ls >> 4) & 15]; }
                    else { h0 = hs_tab[ls & 31]; h1 = h0; }
                    decode_item2<FAM>(*(const uint4*) (cwB + 4 * s), h0, h1, FAM == 3 ? nk_tab[ls & 31] : 0u, lut, wb);
                }
                const uint32_t abase = (uint32_t) __cvta_generic_to_shared(act_s + 4 * s);
#pragma unroll
                for (int t0 = 0; t0 < TPP; t0 += 4) {
                    if (t0 < nt) {
#pragma unroll
                        for (int t = t0; t < t0 + 4; ++t) {
                            uint32_t ha = hA[t], hb = hB[t];
#pragma unroll
                            for (int m = 0; m < 4; ++m) {
                                const uint32_t a0 = lds32(abase + (t * 16 + m) * 16 + 0), a1 = lds32(abase + (t * 16 + m) * 16 + 4);
                                const uint32_t a2 = lds32(abase + (t * 16 + m) * 16 + 8), a3 = lds32(abase + (t * 16 + m) * 16 + 12);
                                ha = hfma2(wa[4 * m + 0], a0, ha);  hb = hfma2(wb[4 * m + 0], a0, hb);
                                ha = hfma2(wa[4 * m + 1], a1, ha);  hb = hfma2(wb[4 * m + 1], a1, hb);
                                ha = hfma2(wa[4 * m + 2], a2, ha);  hb = hfma2(wb[4 * m + 2], a2, hb);
                                ha = hfma2(wa[4 * m + 3], a3, ha);  hb = hfma2(wb[4 * m + 3], a3, hb);
                            }
                            hA[t] = ha;
                            hB[t] = hb;
                        }
                    }
                }
            }
        }
        flush(fdA, fdB);
#pragma unroll
        for (int t = 0; t < TPP; ++t)
            if (t < nt) {
                dst[(size_t) (trow0 + t) * GU_ROWS + r0 + tid] = accA[t];
                dst[(size_t) (trow0 + t) * GU_ROWS + r0 + R2_NT + tid] = accB[t];
                bad += !(fabsf(accA[t]) <= 3.0e38f) + !(fabsf(accB[t]) <= 3.0e38f);
            }
    }
    if (bad != 0) atomicAdd(&g_hf.bad_y, (unsigned) bad);
}

}  // namespace
}  // namespace strata::prefill::ws::hf
