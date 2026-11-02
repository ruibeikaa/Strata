// src/prefill/moe_ws_dn_hf_r2.cuh - STRATA_PF_HFMA_R2_DOWN (moe_ws.hpp): the Q2_0 and IQ4_NL down kernels of moe_ws_dn_hf.cuh with two weight rows a thread.
// Found by the Kernel Forge search (kf-work\kfdown, node 9 of 11); the same operator, the same epilogue contract and the same bits as
// moe_ws_dn_hf.cuh's kernels (the outputs are bit for bit equal on every mix tried): per row and step the shift k = step_k(...), the weight word =
// (integer code value) * fl16(d 2^-k) by one FP16 multiply, products accumulated in FP16 lanes over 4 items x 16 half2 = 64 products per 128-K step, widened to
// FP32 with 2^k, the 5 steps summed in order, times the H row's 2^a.  What is different is the schedule only:
//  * rows2: a thread owns TWO weight rows (a 128-thread CTA covers 256 rows of the expert, 10 pieces a chunk, 2 CTAs an SM = 8 warps), so every H word read from
//    shared memory (a broadcast LDS.128) feeds 8 HFMA2 instead of 4 and the decoded weights are used by twice the work;
//  * Q2_0 (r2q::dn2_kernel<16>, 246 registers): 16 tokens a pass; the next step's weights and H pieces are loaded into registers while the step computes and stored
//    to shared at the start of the next (the engine kernel's structure, widened to 256 rows);
//  * IQ4_NL (r2n::dn2_kernel<32>, 239 registers): 32 tokens a pass (the decode and the staging are shared by twice the tokens); the weights and H pieces are loaded
//    and stored to shared in the same step (no register-held prefetch: wpre / offsets would not fit), the offsets recomputed (idx / 9 by multiply-shift), and the
//    previous step's FP32 widening runs in the shadow of those global loads.
// Included by moe_ws_dn_hf.cu after moe_ws_dn_hf.cuh; everything here has internal linkage.
#pragma once

#include "moe_ws_dn_hf.cuh"

namespace strata::prefill::ws::dn {
namespace {

constexpr int RPT = 2, ROWS_CTA = NT * RPT, P2 = DN_ROWS / ROWS_CTA;   // two rows a thread: 10 pieces of 256 rows a chunk

// the sums of the pass's tokens (two rows) into the FP32 ones (64 products a lane), scaled by 2^k
template <int TPP>
__device__ __forceinline__ void flush2(uint32_t (&hA)[TPP], uint32_t (&hB)[TPP], float (&accA)[TPP], float (&accB)[TPP], const float fdA, const float fdB, const int nt) {
#pragma unroll
    for (int t0 = 0; t0 < TPP; t0 += 4) {
        if (t0 < nt) {
#pragma unroll
            for (int t = t0; t < t0 + 4; ++t) {
                float lo, hi;
                h2f(hA[t], lo, hi);
                accA[t] = __fmaf_rn(fdA, lo + hi, accA[t]);
                hA[t] = 0u;
                h2f(hB[t], lo, hi);
                accB[t] = __fmaf_rn(fdB, lo + hi, accB[t]);
                hB[t] = 0u;
            }
        }
    }
}
// one 32-value item of two rows against the pass's tokens: 16 weight words a row, 16 H words a token
template <int TPP>
__device__ __forceinline__ void item_fma2(const uint32_t (&wa)[16], const uint32_t (&wb)[16], const uint4* __restrict__ act_s, const int s, const int nt,
                                          uint32_t (&hA)[TPP], uint32_t (&hB)[TPP]) {
#pragma unroll
    for (int t0 = 0; t0 < TPP; t0 += 4) {
        if (t0 < nt) {
#pragma unroll
            for (int t = t0; t < t0 + 4; ++t) {
                const uint4* a4 = act_s + t * 16 + 4 * s;
                uint32_t ha = hA[t], hb = hB[t];
#pragma unroll
                for (int m = 0; m < 4; ++m) {
                    const uint4 a = a4[m];
                    ha = hfma2(wa[4 * m + 0], a.x, ha);  hb = hfma2(wb[4 * m + 0], a.x, hb);
                    ha = hfma2(wa[4 * m + 1], a.y, ha);  hb = hfma2(wb[4 * m + 1], a.y, hb);
                    ha = hfma2(wa[4 * m + 2], a.z, ha);  hb = hfma2(wb[4 * m + 2], a.z, hb);
                    ha = hfma2(wa[4 * m + 3], a.w, ha);  hb = hfma2(wb[4 * m + 3], a.w, hb);
                }
                hA[t] = ha;
                hB[t] = hb;
            }
        }
    }
}
// the pass's outputs of two rows: acc * 2^a of the token's row, the non-finite ones counted
template <int TPP>
__device__ __forceinline__ int store_out2(const float (&accA)[TPP], const float (&accB)[TPP], const float* __restrict__ ex_s, float* __restrict__ dst, const int trow0, const int nt, const int r0) {
    int bad = 0;
#pragma unroll
    for (int t = 0; t < TPP; ++t)
        if (t < nt) {
            const float vA = accA[t] * ex_s[t], vB = accB[t] * ex_s[t];
            dst[(size_t) (trow0 + t) * DN_ROWS + r0 + threadIdx.x] = vA;
            dst[(size_t) (trow0 + t) * DN_ROWS + r0 + NT + threadIdx.x] = vB;
            bad += !(fabsf(vA) <= 3.0e38f) + !(fabsf(vB) <= 3.0e38f);
        }
    return bad;
}

__device__ __forceinline__ int row_of(const int idx) { return (idx * 7282) >> 16; }   // idx / 9 for 0 <= idx < 2304
constexpr bool row_of_ok() { for (int i = 0; i < 2304; ++i) if (((i * 7282) >> 16) != i / 9) return false; return true; }
static_assert(row_of_ok(), "idx / 9 by multiply-shift");


namespace r2q {
constexpr int WPR = 9, ROWB = 180;

// one 32-value item (block b / 2, half b % 2) of the row whose step words are cw[0..8] -> 16 FP16 pairs (value x dh)
__device__ __forceinline__ void decode_item(const uint32_t* cw, const int b, const uint32_t dh0, const uint32_t dh1, uint32_t (&wv)[16]) {
    const int pq = b >> 1, h = b & 1;
    const int base = pq ? 5 + 2 * h : 2 * h, sh = pq ? 0 : 16;
    const uint32_t c0 = cw[base], c1 = cw[base + 1], c2 = cw[min(base + 2, WPR - 1)];
    const uint32_t dh = pq ? dh1 : dh0;
    uint32_t wa[4], wb[4];
    q20::dec2_h16(__funnelshift_r(c0, c1, sh), wa, wb);
#pragma unroll
    for (int j = 0; j < 4; ++j) { wv[j] = hmul2(wa[j], dh); wv[4 + j] = hmul2(wb[j], dh); }
    q20::dec2_h16(__funnelshift_r(c1, c2, sh), wa, wb);
#pragma unroll
    for (int j = 0; j < 4; ++j) { wv[8 + j] = hmul2(wa[j], dh); wv[12 + j] = hmul2(wb[j], dh); }
}

template <int TPP>
__global__ void __launch_bounds__(NT, 2) dn2_kernel(const Chunk* __restrict__ chunks, const char* __restrict__ w, const int64_t* __restrict__ xoff, const int64_t xadd,
                                                   const uint4* __restrict__ hb, float* __restrict__ dst) {
    static_assert(TPP == 16 || TPP == 24 || TPP == 32, "");
    constexpr int AI = TPP * 16 / NT;
    constexpr int NW = RPT * WPR;   // staged words a thread per step (18)
    __shared__ int wsm[ROWS_CTA * WPR];
    __shared__ __align__(16) uint4 act_s[TPP * 16];
    __shared__ float ex_s[TPP];
    const int tid = threadIdx.x;
    const int r0 = (int) (blockIdx.x % P2) * ROWS_CTA;
    const int4 cc = __ldg((const int4*) chunks + blockIdx.x / P2);
    const Chunk c{cc.x, cc.y, cc.z, cc.w};
    const char* __restrict__ W = w + xoff[c.j] + xadd + (size_t) r0 * ROWB;
    int woff[NW];   // word tid + NT i of the step's 256 x 9 words: row idx / 9, word idx % 9
#pragma unroll
    for (int i = 0; i < NW; ++i) {
        const int idx = tid + i * NT, r = idx / WPR;
        woff[i] = r * ROWB + 4 * (idx - r * WPR);
    }
    const int piece = tid & 15, trow = tid >> 4;
    int wpre[NW];
    uint4 apre[AI];
    auto load_w = [&](const int st) {
        const char* __restrict__ Ws = W + st * 36;
#pragma unroll
        for (int i = 0; i < NW; ++i) wpre[i] = __ldg((const int*) (Ws + woff[i]));
    };
    auto store_w = [&]() {
#pragma unroll
        for (int i = 0; i < NW; ++i) wsm[tid + i * NT] = wpre[i];
    };
    auto load_a = [&](const int st, const int trow0, const int nt) {
#pragma unroll
        for (int i = 0; i < AI; ++i) {
            const int t = trow + 8 * i;
            apre[i] = t < nt ? __ldg(hb + (size_t) (trow0 + t) * HROW_U4 + st * 16 + piece) : make_uint4(0u, 0u, 0u, 0u);
        }
    };
    auto store_a = [&]() {
#pragma unroll
        for (int i = 0; i < AI; ++i) act_s[(trow + 8 * i) * 16 + piece] = pair_i4(apre[i]);
    };
    int bad = 0;
    load_w(0);
    load_a(0, c.row0, min(TPP, c.rows));
    for (int p0 = 0; p0 < c.rows; p0 += TPP) {
        const int nt = min(TPP, c.rows - p0);
        const int trow0 = c.row0 + p0;
        float accA[TPP], accB[TPP];
        uint32_t hA[TPP], hB[TPP];
#pragma unroll
        for (int t = 0; t < TPP; ++t) { accA[t] = 0.0f; accB[t] = 0.0f; hA[t] = 0u; hB[t] = 0u; }
#pragma unroll 1
        for (int st = 0; st < NSTEP; ++st) {
            __syncthreads();
            store_w();
            store_a();
            if (st == 0 && tid < TPP) ex_s[tid] = tid < nt ? pow2i(__ldg((const int*) (hb + (size_t) (trow0 + tid) * HROW_U4 + HEXP_U4))) : 0.0f;
            __syncthreads();
            if (st + 1 < NSTEP) {
                load_w(st + 1);
                load_a(st + 1, trow0, nt);
            } else if (p0 + TPP < c.rows) {
                load_w(0);
                load_a(0, trow0 + TPP, min(TPP, c.rows - p0 - TPP));
            }
            const uint32_t* cwA = (const uint32_t*) (wsm + tid * WPR);
            const uint32_t* cwB = (const uint32_t*) (wsm + (tid + NT) * WPR);
            const float d0A = h16f(cwA[0] & 0xFFFFu), d1A = h16f(cwA[4] >> 16), d0B = h16f(cwB[0] & 0xFFFFu), d1B = h16f(cwB[4] >> 16);
            const int kA = step_k(kWintQ20 * (fabsf(d0A) + fabsf(d1A))), kB = step_k(kWintQ20 * (fabsf(d0B) + fabsf(d1B)));
            const float fdA = pow2i(kA), fdB = pow2i(kB);
            const float kdnA = pow2i(-kA), kdnB = pow2i(-kB);
            const uint32_t dhA0 = dup_h2(d0A * kdnA), dhA1 = dup_h2(d1A * kdnA), dhB0 = dup_h2(d0B * kdnB), dhB1 = dup_h2(d1B * kdnB);
            if (tid == 0 && p0 == 0 && st == 0) atomicAdd(&g_dn.khist[kA - KMIN], 1u);
#pragma unroll 1
            for (int b = 0; b < 4; ++b) {
                uint32_t wa[16], wb[16];
                decode_item(cwA, b, dhA0, dhA1, wa);
                decode_item(cwB, b, dhB0, dhB1, wb);
                item_fma2<TPP>(wa, wb, act_s, b, nt, hA, hB);
            }
            flush2<TPP>(hA, hB, accA, accB, fdA, fdB, nt);
        }
        bad += store_out2<TPP>(accA, accB, ex_s, dst, trow0, nt, r0);
    }
    if (bad != 0) atomicAdd(&g_dn.bad_y, (unsigned) bad);
}
}  // namespace r2q

namespace r2n {
constexpr int WPS = 18, S = 19, ROWB = 360;

template <int TPP>
__global__ void __launch_bounds__(NT, 2) dn2_kernel(const Chunk* __restrict__ chunks, const char* __restrict__ w, const int64_t* __restrict__ xoff, const int64_t xadd,
                                                   const uint4* __restrict__ hb, float* __restrict__ dst) {
    static_assert(TPP == 16 || TPP == 24 || TPP == 32, "");
    constexpr int NTASK = 8 * TPP;
    constexpr int AI = (NTASK + NT - 1) / NT;
    constexpr int NP = 9 * RPT;   // 8-byte pieces a thread per step (18)
    __shared__ int wsm[ROWS_CTA * S];
    __shared__ __align__(16) uint4 act_s[TPP * 16];
    __shared__ uint32_t lut[256];
    __shared__ float ex_s[TPP];
    const int tid = threadIdx.x;
    for (int e = tid; e < 256; e += NT) lut[e] = pack_h2((float) kKv[e & 15], (float) kKv[e >> 4]);
    const int r0 = (int) (blockIdx.x % P2) * ROWS_CTA;
    const int4 cc = __ldg((const int4*) chunks + blockIdx.x / P2);
    const Chunk c{cc.x, cc.y, cc.z, cc.w};
    const char* __restrict__ W = w + xoff[c.j] + xadd + (size_t) r0 * ROWB;
    int bad = 0;
    for (int p0 = 0; p0 < c.rows; p0 += TPP) {
        const int nt = min(TPP, c.rows - p0);
        const int trow0 = c.row0 + p0;
        float accA[TPP], accB[TPP];
        uint32_t hA[TPP], hB[TPP];
#pragma unroll
        for (int t = 0; t < TPP; ++t) { accA[t] = 0.0f; accB[t] = 0.0f; hA[t] = 0u; hB[t] = 0u; }
        float fdpA = 0.0f, fdpB = 0.0f;
#pragma unroll 1
        for (int st = 0; st < NSTEP; ++st) {
            __syncthreads();   // everyone is done with the previous step's shared data
            int tq = tid;
            asm("" : "+r"(tq));   // the offsets below are recomputed each step, not held across the compute
            int2 wv[NP];
            const char* __restrict__ Ws = W + st * 72;
#pragma unroll
            for (int i = 0; i < NP; ++i) {
                const int idx = tq + i * NT, r = row_of(idx), pc = idx - r * 9;
                wv[i] = __ldg((const int2*) (Ws + r * ROWB + 8 * pc));
            }
            uint4 apre[AI][2];
#pragma unroll
            for (int i = 0; i < AI; ++i) {
                const int idx = tid + i * NT, t = idx >> 3, s = (idx >> 1) & 3, hf = idx & 1;
                if (NTASK % NT == 0 || idx < NTASK) {
                    if (t < nt) {
                        const uint4* p = hb + (size_t) (trow0 + t) * HROW_U4 + st * 16 + 4 * s + hf;
                        apre[i][0] = __ldg(p);
                        apre[i][1] = __ldg(p + 2);
                    } else {
                        apre[i][0] = apre[i][1] = make_uint4(0u, 0u, 0u, 0u);
                    }
                }
            }
            if (st > 0) flush2<TPP>(hA, hB, accA, accB, fdpA, fdpB, nt);   // the previous step's FP32 widening runs in the shadow of this step's global loads
#pragma unroll
            for (int i = 0; i < NP; ++i) {
                const int idx = tq + i * NT, r = row_of(idx), pc = idx - r * 9;
                const int si = r * S + 2 * pc;
                wsm[si] = wv[i].x;
                wsm[si + 1] = wv[i].y;
            }
#pragma unroll
            for (int i = 0; i < AI; ++i) {
                const int idx = tid + i * NT, t = idx >> 3, s = (idx >> 1) & 3, hf = idx & 1;
                if (NTASK % NT == 0 || idx < NTASK) {
                    const uint4 x = apre[i][0], y = apre[i][1];
                    uint4* d = act_s + t * 16 + 4 * s + 2 * hf;
                    d[0] = make_uint4(prmt(x.x, y.x, 0x5410u), prmt(x.x, y.x, 0x7632u), prmt(x.y, y.y, 0x5410u), prmt(x.y, y.y, 0x7632u));
                    d[1] = make_uint4(prmt(x.z, y.z, 0x5410u), prmt(x.z, y.z, 0x7632u), prmt(x.w, y.w, 0x5410u), prmt(x.w, y.w, 0x7632u));
                }
            }
            if (st == 0 && tid < TPP) ex_s[tid] = tid < nt ? pow2i(__ldg((const int*) (hb + (size_t) (trow0 + tid) * HROW_U4 + HEXP_U4))) : 0.0f;
            __syncthreads();
            const uint32_t* cwA = (const uint32_t*) (wsm + tid * S);
            const uint32_t* cwB = (const uint32_t*) (wsm + (tid + NT) * S);
            const float dA[4] = {h16f(cwA[0] & 0xFFFFu), h16f(cwA[4] >> 16), h16f(cwA[9] & 0xFFFFu), h16f(cwA[13] >> 16)};
            const float dB[4] = {h16f(cwB[0] & 0xFFFFu), h16f(cwB[4] >> 16), h16f(cwB[9] & 0xFFFFu), h16f(cwB[13] >> 16)};
            const int kA = step_k(kWintNl * (fabsf(dA[0]) + fabsf(dA[1]) + fabsf(dA[2]) + fabsf(dA[3])));
            const int kB = step_k(kWintNl * (fabsf(dB[0]) + fabsf(dB[1]) + fabsf(dB[2]) + fabsf(dB[3])));
            const float fdA = pow2i(kA), fdB = pow2i(kB), kdnA = pow2i(-kA), kdnB = pow2i(-kB);
            uint32_t dhA[4], dhB[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) { dhA[i] = dup_h2(dA[i] * kdnA); dhB[i] = dup_h2(dB[i] * kdnB); }
            if (tid == 0 && p0 == 0 && st == 0) atomicAdd(&g_dn.khist[kA - KMIN], 1u);
#pragma unroll 1
            for (int b = 0; b < 4; ++b) {
                uint32_t wa[16], wb[16];
                {
                    const uint32_t* cb = cwA + 9 * (b >> 1);
                    uint32_t q[4];
                    if (b & 1) {
#pragma unroll
                        for (int j = 0; j < 4; ++j) q[j] = cb[5 + j];
                    } else {
#pragma unroll
                        for (int j = 0; j < 4; ++j) q[j] = __funnelshift_r(cb[j], cb[j + 1], 16);
                    }
                    const uint32_t dhb = dhA[b];
#pragma unroll
                    for (int j = 0; j < 4; ++j)
#pragma unroll
                        for (int i = 0; i < 4; ++i) wa[4 * j + i] = hmul2(lut[(q[j] >> (8 * i)) & 0xFF], dhb);
                }
                {
                    const uint32_t* cb = cwB + 9 * (b >> 1);
                    uint32_t q[4];
                    if (b & 1) {
#pragma unroll
                        for (int j = 0; j < 4; ++j) q[j] = cb[5 + j];
                    } else {
#pragma unroll
                        for (int j = 0; j < 4; ++j) q[j] = __funnelshift_r(cb[j], cb[j + 1], 16);
                    }
                    const uint32_t dhb = dhB[b];
#pragma unroll
                    for (int j = 0; j < 4; ++j)
#pragma unroll
                        for (int i = 0; i < 4; ++i) wb[4 * j + i] = hmul2(lut[(q[j] >> (8 * i)) & 0xFF], dhb);
                }
                item_fma2<TPP>(wa, wb, act_s, b, nt, hA, hB);
            }
            fdpA = fdA;
            fdpB = fdB;
        }
        flush2<TPP>(hA, hB, accA, accB, fdpA, fdpB, nt);
        bad += store_out2<TPP>(accA, accB, ex_s, dst, trow0, nt, r0);
    }
    if (bad != 0) atomicAdd(&g_dn.bad_y, (unsigned) bad);
}
}  // namespace r2n

}  // namespace
}  // namespace strata::prefill::ws::dn
