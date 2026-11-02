// src/prefill/moe_ws_hf_cls.cuh - STRATA_PF_HFMA_CLASSES (moe_ws.hpp): the gate/up kernels of moe_ws_hf.cuh / moe_ws_hf_r2.cuh with the
// layer's passes split by SIZE into classes, each class on the kernel that is fastest for it.
// Found by the Kernel Forge "wide" search (kf-work\wide, node 26 of 27, cands\n26.cuh); same operator, same epilogue
// contract as moe_ws_hf.cuh's gu_kernel and moe_ws_hf_r2.cuh's gu2_kernel (it is included after both, shares their g_hf state, their shift
// rule and their output layout):
//   y[routed row][r], r = 0 .. 1279, K = 2560, FP16 lane sums of 64 products (4 items x 16 half2 a 128-K step) widened to FP32 with
//   fd = d * c_fmt * 2^k each step, the 20 steps summed in order.
// A PASS is up to 32 routed rows of one expert (a chunk of <= 64 rows is one or two passes).  The host (moe_ws_gu_hf.cu, at plan time) splits
// the layer's chunk list into pass items, in four classes:
//   full   32 rows            gd_kernel<32, FAM, 2, true>   direct code loads, no guards
//   mid    25 .. 31 rows      gd_kernel<32, FAM, 2, false>
//   mid2   17 .. 24 rows      gd_kernel<24, FAM, 2, false>  24 tokens a pass (226 registers, 2 CTAs an SM)
//   small  1 .. 16 rows       gs_kernel<16, FAM>            staged codes, 12 warps an SM (3 CTAs)
// Each kernel is a CTA of 128 threads x 2 weight rows (256 rows, five CTAs a pass item).  What differs from the R2 kernel (all in the schedule,
// none in the arithmetic's grouping):
//  * gd_kernel (the big passes) reads each thread's two rows' item codes straight from global memory, one item ahead (no code staging, ONE
//    barrier a step, the activation tile double buffered); the IQ2 formats and IQ3_XXS decode a code word by PRMT byte tables (no shared LUT)
//    and ONE HFMA2 a half2: the table byte b = 128 +- |v| (IQ3_XXS: 64 +- |v|) lands in the low byte of the FP16 number 1024 + b and
//    fma(x, S', -1152 S') (IQ3_XXS: -1088 S') = (b - 128) S' = +-|v| S' exactly, so the weights are the very same values as the LUT's;
//  * gs_kernel (the small passes) keeps the R2 kernel's staging of the codes through shared memory (16 words a row, XOR swizzled) and runs
//    three CTAs an SM;
//  * IQ3_S decodes as the R2 kernel does (shift + LOP3 + HFMA2, the activations re-paired), so its outputs are the R2 kernel's; the other
//    three families' outputs are bit-identical to the compiler kernel's (and so to the R2 kernel's).
// The kernels of different classes write disjoint rows of dst and run on separate streams.  Everything here has internal linkage; the host
// side is moe_ws_gu_hf.cu.
#pragma once

#include "moe_ws_hf.cuh"
#include "moe_ws_hf_r2.cuh"   // lds32, and_or, dec_iq3s_l3

namespace strata::prefill::ws::hf {
namespace {

constexpr int W_NT = 128, W_RPT = 2, W_ROWS = W_NT * W_RPT, W_PIECES = GU_ROWS / W_ROWS;   // 5 pieces of 256 rows
constexpr int W_CS = 16;   // gs_kernel's staged words a row (XOR swizzle)
// the class bounds (rows in a pass): full = 32, mid > W_MID_MIN, mid2 > W_SMALL_MAX, small <= W_SMALL_MAX
constexpr int W_MID_MIN = 24, W_SMALL_MAX = 16;

// the launch's weights' shift from the layer's xmax (the same arithmetic as gu_kernel's / gu2_kernel's prologue)
template <int FAM>
__device__ __forceinline__ int w_shift(const unsigned cur) {
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
    return k;
}
// from ONE thread of block 0 of ONE launch of the layer (what gu_kernel does with blockIdx.x == 0 && tid == 0)
__device__ __forceinline__ void w_note(const unsigned cur, const int k, const int layer) {
    atomicAdd(&g_hf.launches, 1u);
    atomicMax(&g_hf.all_max, cur);
    atomicMax(&g_hf.layer_max[layer & 63], cur);
    atomicAdd(&g_hf.khist[k - KMIN], 1u);
    if (cur >= 0x7C00u) atomicAdd(&g_hf.bad_x, 1u);
}

// FAM 0/1 (nibble = idx{8,25,43} | sign << 2, bit 3 clear, idx 3 -> 0): the 8 table bytes b(n) = 128 +- mag, indexed by the raw nibbles as PRMT selectors;
// the byte lands in the low byte of an FP16 pair 0x64:b = 1024 + b (exact), and ONE HFMA2 fma(x, S', -1152 S') = (b - 128) S' = +-mag S' exactly.
// 11 instructions for the 4 half2 of a code word (shift, 2 table PRMT, 4 pack PRMT, 4 HFMA2): no shared-memory LUT, no bank conflicts.
__device__ __forceinline__ void dec_iq2_tab(const uint32_t q, const uint32_t hs, const uint32_t nk, uint32_t (&w)[4]) {
    uint32_t t0 = 0x80AB9988u, t1 = 0x80556778u, cc = 0x64646464u;
    asm("" : "+r"(t0), "+r"(t1), "+r"(cc));
    uint32_t r = prmt(t0, t1, q);
    w[0] = hfma2(prmt(r, cc, 0x5140u), hs, nk);
    w[1] = hfma2(prmt(r, cc, 0x7362u), hs, nk);
    r = prmt(t0, t1, q >> 16);
    w[2] = hfma2(prmt(r, cc, 0x5140u), hs, nk);
    w[3] = hfma2(prmt(r, cc, 0x7362u), hs, nk);
}

// FAM 2 (IQ3_XXS, nibble = n | sign << 3, magnitudes {4,12,..,52,62}): b = 64 +- mag (all < 128).  Selector nibbles with bit 3 set make PRMT answer 0x00 (the msb of a table byte
// is 0), so  prmt(T+, q) | prmt(T-, q ^ 0x8888..)  = the byte of the right table for either sign; the byte goes below 0x64 and ONE HFMA2 with (S', -1088 S') gives +-mag S' exactly.
__device__ __forceinline__ void dec_iq3xxs_tab(const uint32_t q, const uint32_t hs, const uint32_t nk, uint32_t (&w)[4]) {
    uint32_t p0 = 0x5C544C44u, p1 = 0x7E746C64u, n0 = 0x242C343Cu, n1 = 0x020C141Cu, cc = 0x64646464u, xm = 0x88888888u;
    asm("" : "+r"(p0), "+r"(p1), "+r"(n0), "+r"(n1), "+r"(cc), "+r"(xm));
    const uint32_t cx = q ^ xm;
    uint32_t r = prmt(p0, p1, q) | prmt(n0, n1, cx);
    w[0] = hfma2(prmt(r, cc, 0x5140u), hs, nk);
    w[1] = hfma2(prmt(r, cc, 0x7362u), hs, nk);
    r = prmt(p0, p1, q >> 16) | prmt(n0, n1, cx >> 16);
    w[2] = hfma2(prmt(r, cc, 0x5140u), hs, nk);
    w[3] = hfma2(prmt(r, cc, 0x7362u), hs, nk);
}

// one 32-value item (4 code words) -> 16 half2 weights; h0 / h1 = S' of the item's first / second 16 values (FAM 0), nk0 / nk1 = -c S' (c = 1152, 1088 or 1040)
template <int FAM>
__device__ __forceinline__ void decode_item_w(const uint4 u, const uint32_t h0, const uint32_t h1, const uint32_t nk0, const uint32_t nk1, uint32_t (&wv)[16]) {
    const uint32_t q[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
    for (int m = 0; m < 4; ++m) {
        uint32_t d4[4];
        if constexpr (FAM == 3) {
            dec_iq3s_l3(q[m], h0, nk0, d4);
        } else if constexpr (FAM <= 1) {
            dec_iq2_tab(q[m], m < 2 ? h0 : h1, m < 2 ? nk0 : nk1, d4);
        } else {
            dec_iq3xxs_tab(q[m], h0, nk0, d4);
        }
#pragma unroll
        for (int j = 0; j < 4; ++j) wv[4 * m + j] = d4[j];
    }
}

// the big passes: direct code loads, one barrier a step
template <int TPP, int FAM, int MINB, bool FULL>
__global__ void __launch_bounds__(W_NT, MINB) gd_kernel(const Chunk* __restrict__ chunks, const char* __restrict__ w,
                                                       const int64_t* __restrict__ xoff, const uint4* __restrict__ xh,
                                                       const int32_t* __restrict__ src, float* __restrict__ dst, const int layer, const int note) {
    static_assert(TPP % 8 == 0, "");
    constexpr int NAS = TPP / 8;
    __shared__ __align__(16) uint4 act_s[2][TPP * 16];
    __shared__ uint32_t lut[FAM == 3 ? 256 : 1];   // FAM 3: the (unused) table changes ptxas' schedule: +9% measured on this kernel
    __shared__ __align__(8) uint2 hn_tab[32];   // (S', -c S') pairs: one LDS.64 a lookup
    const int tid = threadIdx.x;

    const unsigned cur = *(volatile const unsigned*) &g_hf.cur;
    const int k = w_shift<FAM>(cur);
    if (note != 0 && blockIdx.x == 0 && tid == 0) w_note(cur, k, layer);
    const int kw = FAM == 3 ? max(k, -1) : max(k, 0);   // ALU decode: c S 2^-kw <= 65504; fd below compensates the power of two exactly
    const float kdn = ldexpf(1.0f, -kw), kup = ldexpf(1.0f, kw);
    if (tid < 32) {
        const int sv = (FAM == 1 || FAM == 3) ? tid : 2 * tid + 1;
        const uint32_t h = (uint32_t) __half_as_ushort(__float2half_rn((float) sv * kdn));
        const uint32_t hs_ = h | (h << 16);
        const uint32_t g = (uint32_t) __half_as_ushort(__float2half_rn((FAM == 3 ? -1040.0f : FAM == 2 ? -1088.0f : -1152.0f) * (float) sv * kdn));
        hn_tab[tid] = make_uint2(hs_, g | (g << 16));
    }

    if constexpr (FAM == 3) for (int e = tid; e < 256; e += W_NT) lut[e] = pack_h2(code_value<FAM>(e & 15), code_value<FAM>(e >> 4));
    const int r0 = (int) (blockIdx.x % W_PIECES) * W_ROWS;
    const int4 cc = __ldg((const int4*) chunks + blockIdx.x / W_PIECES);
    const Chunk c{cc.x, cc.y, cc.z, cc.w};
    const char* __restrict__ slot = w + xoff[c.j];
    const uint32_t* __restrict__ scwA = (const uint32_t*) (slot + nb::kScOff + (size_t) (r0 + tid) * ITEMS);
    const uint32_t* __restrict__ scwB = scwA + (size_t) W_NT * ITEMS / 4;
    const unsigned short* __restrict__ drA = (const unsigned short*) (slot + nb::kDOff) + (size_t) (r0 + tid) * NB;
    const unsigned short* __restrict__ drB = drA + (size_t) W_NT * NB;
    const uint4* __restrict__ cgA = (const uint4*) slot + (size_t) (r0 + tid) * ITEMS;      // this thread's own rows' codes, item by item
    const uint4* __restrict__ cgB = cgA + (size_t) W_NT * ITEMS;
    const int piece = tid & 15, trow = tid >> 4;
    int bad = 0;
    for (int p0 = 0; p0 < c.rows; p0 += TPP) {
        const int nt = FULL ? TPP : min(TPP, c.rows - p0);
        const int trow0 = c.row0 + p0;
        const uint4* ap[NAS];
#pragma unroll
        for (int i = 0; i < NAS; ++i) {
            const int t = trow + 8 * i;
            ap[i] = (FULL || t < nt) ? xh + (size_t) __ldg(src + trow0 + t) * XROW16 + piece : nullptr;
        }
        float accA[TPP], accB[TPP];
        uint32_t hA[TPP], hB[TPP];
#pragma unroll
        for (int t = 0; t < TPP; ++t) { accA[t] = 0.0f; accB[t] = 0.0f; hA[t] = 0u; hB[t] = 0u; }
        auto flush = [&](const float fA, const float fB) {
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
        uint4 cA = __ldg(cgA), cB = __ldg(cgB);                 // item 0
#pragma unroll 1
        for (int hc = 0; hc < NSTEP; ++hc) {
            const int kc = hc >> 1;
            const uint32_t scvA = __ldg(scwA + hc), scvB = __ldg(scwB + hc);
            const int dvA = (int) __ldg(drA + kc), dvB = (int) __ldg(drB + kc);
            uint4 av[NAS];
#pragma unroll
            for (int i = 0; i < NAS; ++i) av[i] = (FULL || ap[i] != nullptr) ? __ldg(ap[i] + hc * 16) : make_uint4(0u, 0u, 0u, 0u);
            if (hc > 0) flush(fdA, fdB);   // the previous step's FP32 widening runs in the shadow of this step's global loads
            uint4* act = act_s[hc & 1];
#pragma unroll
            for (int i = 0; i < NAS; ++i) {
                uint4 a = av[i];
                if constexpr (FAM == 3) {
                    const uint4 o = make_uint4(prmt(a.x, a.y, 0x5410u), prmt(a.z, a.w, 0x5410u), prmt(a.x, a.y, 0x7632u), prmt(a.z, a.w, 0x7632u));   // pairs (k0,k2) (k4,k6) (k1,k3) (k5,k7)
                    a = o;
                }
                act[(trow + 8 * i) * 16 + piece] = a;
            }
            __syncthreads();   // the only barrier of the step: the tile of this step is visible; everybody has left the other buffer's previous step
            fdA = __half2float(__ushort_as_half((unsigned short) (dvA & 0xFFFF))) * (cfmt(FAM) * kup);
            fdB = __half2float(__ushort_as_half((unsigned short) (dvB & 0xFFFF))) * (cfmt(FAM) * kup);
#pragma unroll 1
            for (int s = 0; s < 4; ++s) {
                const int itn = min(4 * hc + s + 1, ITEMS - 1);
                const uint4 nA = __ldg(cgA + itn), nB = __ldg(cgB + itn);   // next item's codes, in flight during this item's work
                uint32_t wa[16], wb[16];
                {
                    const int ls = (int) ((scvA >> (8 * s)) & 0xFF);
                    uint32_t h0, h1, n0 = 0u, n1 = 0u;
                    if constexpr (FAM == 0) { const uint2 u0 = hn_tab[ls & 15], u1 = hn_tab[(ls >> 4) & 15]; h0 = u0.x; n0 = u0.y; h1 = u1.x; n1 = u1.y; }
                    else { const uint2 u0 = hn_tab[ls & 31]; h0 = u0.x; h1 = h0; n0 = u0.y; n1 = n0; }
                    decode_item_w<FAM>(cA, h0, h1, n0, n1, wa);
                }
                {
                    const int ls = (int) ((scvB >> (8 * s)) & 0xFF);
                    uint32_t h0, h1, n0 = 0u, n1 = 0u;
                    if constexpr (FAM == 0) { const uint2 u0 = hn_tab[ls & 15], u1 = hn_tab[(ls >> 4) & 15]; h0 = u0.x; n0 = u0.y; h1 = u1.x; n1 = u1.y; }
                    else { const uint2 u0 = hn_tab[ls & 31]; h0 = u0.x; h1 = h0; n0 = u0.y; n1 = n0; }
                    decode_item_w<FAM>(cB, h0, h1, n0, n1, wb);
                }
                cA = nA; cB = nB;
                const uint32_t abase = (uint32_t) __cvta_generic_to_shared(act + 4 * s);
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
                dst[(size_t) (trow0 + t) * GU_ROWS + r0 + W_NT + tid] = accB[t];
                bad += !(fabsf(accA[t]) <= 3.0e38f) + !(fabsf(accB[t]) <= 3.0e38f);
            }
    }
    if (bad != 0) atomicAdd(&g_hf.bad_y, (unsigned) bad);
}

// the small passes (<= 16 tokens): staged codes, three CTAs an SM
template <int TPP, int FAM>
__global__ void __launch_bounds__(W_NT, 3) gs_kernel(const Chunk* __restrict__ chunks, const char* __restrict__ w,
                                                    const int64_t* __restrict__ xoff, const uint4* __restrict__ xh,
                                                    const int32_t* __restrict__ src, float* __restrict__ dst, const int layer, const int note) {
    static_assert(TPP == 16 || TPP == 32, "");
    constexpr int NAS = TPP / 8;
    __shared__ __align__(16) uint4 act_s[TPP * 16];
    __shared__ __align__(16) uint32_t csm[W_ROWS * W_CS];
    __shared__ uint32_t hs_tab[32];
    __shared__ uint32_t nk_tab[32];   // -c S 2^-k as half2
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;

    const unsigned cur = *(volatile const unsigned*) &g_hf.cur;
    const int k = w_shift<FAM>(cur);
    if (note != 0 && blockIdx.x == 0 && tid == 0) w_note(cur, k, layer);
    const int kw = FAM == 3 ? max(k, -1) : max(k, 0);   // ALU decode: c S 2^-kw <= 65504; fd below compensates the power of two exactly
    const float kdn = ldexpf(1.0f, -kw), kup = ldexpf(1.0f, kw);
    if (tid < 32) {
        const int sv = (FAM == 1 || FAM == 3) ? tid : 2 * tid + 1;
        const uint32_t h = (uint32_t) __half_as_ushort(__float2half_rn((float) sv * kdn));
        hs_tab[tid] = h | (h << 16);
        const uint32_t g = (uint32_t) __half_as_ushort(__float2half_rn((FAM == 3 ? -1040.0f : FAM == 2 ? -1088.0f : -1152.0f) * (float) sv * kdn));
        nk_tab[tid] = g | (g << 16);
    }

    const int r0 = (int) (blockIdx.x % W_PIECES) * W_ROWS;
    const int4 cc = __ldg((const int4*) chunks + blockIdx.x / W_PIECES);
    const Chunk c{cc.x, cc.y, cc.z, cc.w};
    const char* __restrict__ slot = w + xoff[c.j];
    const uint32_t* __restrict__ scwA = (const uint32_t*) (slot + nb::kScOff + (size_t) (r0 + tid) * ITEMS);
    const uint32_t* __restrict__ scwB = scwA + (size_t) W_NT * ITEMS / 4;
    const unsigned short* __restrict__ drA = (const unsigned short*) (slot + nb::kDOff) + (size_t) (r0 + tid) * NB;
    const unsigned short* __restrict__ drB = drA + (size_t) W_NT * NB;
    const int srow = warp * 8 + (lane >> 2), sq = lane & 3;
    const uint4* __restrict__ cg = (const uint4*) slot + (size_t) (r0 + srow) * ITEMS + sq;
    uint32_t* __restrict__ cs_st = csm + srow * W_CS + 4 * (sq ^ ((srow >> 1) & 3));
    const int gA = (tid >> 1) & 3, gB = ((tid + W_NT) >> 1) & 3;   // swizzle phases of this thread's two rows
    const uint32_t* cwA = csm + tid * W_CS;
    const uint32_t* cwB = csm + (tid + W_NT) * W_CS;
    const int piece = tid & 15, trow = tid >> 4;
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
        auto flush = [&](const float fA, const float fB) {
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
            for (int i = 0; i < 8; ++i) *(uint4*) (cs_st + i * 32 * W_CS) = v[i];
#pragma unroll
            for (int i = 0; i < NAS; ++i) {
                uint4 a = av[i];
                if constexpr (FAM == 3) {
                    const uint4 o = make_uint4(prmt(a.x, a.y, 0x5410u), prmt(a.z, a.w, 0x5410u), prmt(a.x, a.y, 0x7632u), prmt(a.z, a.w, 0x7632u));   // pairs (k0,k2) (k4,k6) (k1,k3) (k5,k7)
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
                    uint32_t h0, h1, n0 = 0u, n1 = 0u;
                    if constexpr (FAM == 0) { h0 = hs_tab[ls & 15]; h1 = hs_tab[(ls >> 4) & 15]; n0 = nk_tab[ls & 15]; n1 = nk_tab[(ls >> 4) & 15]; }
                    else { h0 = hs_tab[ls & 31]; h1 = h0; n0 = nk_tab[ls & 31]; n1 = n0; }
                    decode_item_w<FAM>(*(const uint4*) (cwA + 4 * (s ^ gA)), h0, h1, n0, n1, wa);
                }
                {
                    const int ls = (int) ((scvB >> (8 * s)) & 0xFF);
                    uint32_t h0, h1, n0 = 0u, n1 = 0u;
                    if constexpr (FAM == 0) { h0 = hs_tab[ls & 15]; h1 = hs_tab[(ls >> 4) & 15]; n0 = nk_tab[ls & 15]; n1 = nk_tab[(ls >> 4) & 15]; }
                    else { h0 = hs_tab[ls & 31]; h1 = h0; n0 = nk_tab[ls & 31]; n1 = n0; }
                    decode_item_w<FAM>(*(const uint4*) (cwB + 4 * (s ^ gB)), h0, h1, n0, n1, wb);
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
                dst[(size_t) (trow0 + t) * GU_ROWS + r0 + W_NT + tid] = accB[t];
                bad += !(fabsf(accA[t]) <= 3.0e38f) + !(fabsf(accB[t]) <= 3.0e38f);
            }
    }
    if (bad != 0) atomicAdd(&g_hf.bad_y, (unsigned) bad);
}

}  // namespace
}  // namespace strata::prefill::ws::hf
