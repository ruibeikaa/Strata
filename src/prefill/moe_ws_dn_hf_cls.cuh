// src/prefill/moe_ws_dn_hf_cls.cuh - STRATA_PF_HFMA_DOWN_CLASSES (moe_ws.hpp): the Q2_0 and IQ4_NL down kernels of moe_ws_dn_hf_r2.cuh with the layer's passes split by SIZE
// into classes, each class on the kernel that is fastest for it.
// Found by the Kernel Forge "wide-down" search (kf-work\wide-down, node 15 = cands\n15.cuh from the generator source n18_src.cuh; libs dl_q_v13 / dl_n_v13);
// same operator, same epilogue contract and the same bits as the one-row and two-rows kernels (moe_ws_dn_hf.cuh, moe_ws_dn_hf_r2.cuh): per weight row and 128-K step the shift
// k = step_k(...), the weight word = (integer code value) * fl16(d 2^-k) by one FP16 multiply, FP16 lane sums of 64 products (4 items x 16 half2) widened to FP32 with 2^k,
// the 5 steps summed in order, times the H row's 2^a.  The GGUF block layout of the down matrix in the expert slot is read as it is (decode and the int8 path read the same bytes).
// A PASS is up to 32 routed rows of one expert (a chunk of <= 64 rows is one or two passes).  The host (moe_ws_dn_hf.cu, at plan time) splits the layer's chunk list into pass
// items {j, row0 + p0, nt, first-pass-of-its-chunk}, in three classes:
//   full   32 rows            <32, FULL=true,  2 CTAs/SM>   unguarded (nt = 32 for every token, no guards anywhere in the compute and the stores)
//   mid    17 .. 31 rows      <32, FULL=false, 2 CTAs/SM>   guarded by the pass's token count
//   small  1 .. 16 rows       <16, FULL=false, 2 CTAs/SM>   guarded 16-token kernel
// One CTA is one pass item x 256 weight rows (128 threads x 2 rows; ten CTAs a pass item).  What differs from the R2 kernels (the schedule, not the arithmetic):
//  * Q2_0 (cls::q::kernel): the first search's staged form at TPP tokens with LINEAR staging (thread t < 126 loads word t % 9 of row t / 9 + 14 i, i = 0..18: every load and
//    shared store offset is a constant times i), register-held prefetch of the next step's codes and H, two barriers a step, the FIRST-item trick (the item loop unrolled, the first FMA
//    of a (row, token) in a step adds 0, the FP16 sums are not zeroed by the flush) and a FULL variant without guards;
//  * IQ4_NL (cls::n::kernel): the R2 kernel's body (shared staging of the 8-byte pieces, LUT decode, the previous step's flush in the shadow of the next step's loads) with the same
//    linear staging, the FIRST trick and the FULL variant, at 32 and 16 tokens a pass.
// The kernels of different classes write disjoint rows of dst and run on separate streams.  Included by moe_ws_dn_hf.cu after moe_ws_dn_hf_r2.cuh; everything here has internal linkage.
#pragma once

#include "moe_ws_dn_hf_r2.cuh"

namespace strata::prefill::ws::dn {
namespace {
namespace cls {

// the sums of the pass's tokens (two rows) into the FP32 ones; FULL: no token guard.  The FP16 sums are NOT zeroed (item_fmag's FIRST item starts from 0)
template <int TPP, bool FULL>
__device__ __forceinline__ void flushg(uint32_t (&hA)[TPP], uint32_t (&hB)[TPP], float (&accA)[TPP], float (&accB)[TPP], const float fdA, const float fdB, const int nt) {
#pragma unroll
    for (int t0 = 0; t0 < TPP; t0 += 4) {
        if (FULL || t0 < nt) {
#pragma unroll
            for (int t = t0; t < t0 + 4; ++t) {
                float lo, hi;
                h2f(hA[t], lo, hi);
                accA[t] = __fmaf_rn(fdA, lo + hi, accA[t]);
                h2f(hB[t], lo, hi);
                accB[t] = __fmaf_rn(fdB, lo + hi, accB[t]);
            }
        }
    }
}
// one 32-value item of two rows against the pass's tokens; FIRST: the item starts the step's FP16 sums (adds to 0)
template <int TPP, bool FULL, bool FIRST>
__device__ __forceinline__ void item_fmag(const uint32_t (&wa)[16], const uint32_t (&wb)[16], const uint4* __restrict__ act_s, const int s, const int nt, uint32_t (&hA)[TPP],
                                          uint32_t (&hB)[TPP]) {
#pragma unroll
    for (int t0 = 0; t0 < TPP; t0 += 4) {
        if (FULL || t0 < nt) {
#pragma unroll
            for (int t = t0; t < t0 + 4; ++t) {
                const uint4* a4 = act_s + t * 16 + 4 * s;
                uint32_t ha = FIRST ? 0u : hA[t], hb = FIRST ? 0u : hB[t];
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
template <int TPP, bool FULL>
__device__ __forceinline__ int store_outg(const float (&accA)[TPP], const float (&accB)[TPP], const float* __restrict__ ex_s, float* __restrict__ dst, const int trow0, const int nt,
                                          const int r0) {
    int bad = 0;
#pragma unroll
    for (int t = 0; t < TPP; ++t)
        if (FULL || t < nt) {
            const float vA = accA[t] * ex_s[t], vB = accB[t] * ex_s[t];
            dst[(size_t) (trow0 + t) * DN_ROWS + r0 + threadIdx.x] = vA;
            dst[(size_t) (trow0 + t) * DN_ROWS + r0 + NT + threadIdx.x] = vB;
            bad += !(fabsf(vA) <= 3.0e38f) + !(fabsf(vB) <= 3.0e38f);
        }
    return bad;
}

// ---- Q2_0.  `items` = the class's pass items (Chunk{j, row0, rows <= TPP, pad = 1 for the first pass of its chunk}); a CTA = item blockIdx.x / P2, piece blockIdx.x % P2
namespace q {
constexpr int WPR = r2q::WPR, ROWB = r2q::ROWB;   // 9 words (36 bytes) of a row a step; a row 180 bytes

template <int TPP, bool FULL, int MINB>
__global__ void __launch_bounds__(NT, MINB) kernel(const Chunk* __restrict__ items, const char* __restrict__ w, const int64_t* __restrict__ xoff, const int64_t xadd,
                                                   const uint4* __restrict__ hb, float* __restrict__ dst) {
    static_assert(TPP == 16 || TPP == 24 || TPP == 32, "");
    constexpr int AI = TPP * 16 / NT;
    __shared__ int wsm[ROWS_CTA * WPR];
    __shared__ __align__(16) uint4 act_s[TPP * 16];
    __shared__ float ex_s[TPP];
    const int tid = threadIdx.x;
    const int r0 = (int) (blockIdx.x % P2) * ROWS_CTA;
    const int4 cc = __ldg((const int4*) items + blockIdx.x / P2);
    const Chunk c{cc.x, cc.y, cc.z, cc.w};
    const char* __restrict__ W = w + xoff[c.j] + xadd + (size_t) r0 * ROWB;
    const int piece = tid & 15, trow = tid >> 4;
    constexpr int NW2 = 19;   // staged words a thread per step: 18, +1 for t / 9 < 4 (linear staging)
    const int t9 = tid / 9, k9 = tid - t9 * 9;
    const bool wact = tid < 126;
    const char* __restrict__ Wt = W + (size_t) t9 * ROWB + 4 * k9;
    int wpre[NW2];
    uint4 apre[AI];
    auto load_w = [&](const int st) {
        const char* __restrict__ p = Wt + st * 36;
#pragma unroll
        for (int i = 0; i < NW2; ++i)
            if (wact && (i < 18 || t9 < 4)) wpre[i] = __ldg((const int*) (p + 2520 * i));
    };
    auto store_w = [&]() {
#pragma unroll
        for (int i = 0; i < NW2; ++i)
            if (wact && (i < 18 || t9 < 4)) wsm[tid + 126 * i] = wpre[i];
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
        const int nt = FULL ? TPP : min(TPP, c.rows - p0);
        const int trow0 = c.row0 + p0;
        float accA[TPP], accB[TPP];
        uint32_t hA[TPP], hB[TPP];
#pragma unroll
        for (int t = 0; t < TPP; ++t) { accA[t] = 0.0f; accB[t] = 0.0f; }
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
            if (tid == 0 && p0 == 0 && c.pad == 1 && st == 0) atomicAdd(&g_dn.khist[kA - KMIN], 1u);   // one sample a chunk piece, as the R2 kernel's
#pragma unroll
            for (int b = 0; b < 4; ++b) {
                uint32_t wa[16], wb[16];
                r2q::decode_item(cwA, b, dhA0, dhA1, wa);
                r2q::decode_item(cwB, b, dhB0, dhB1, wb);
                if (b == 0) item_fmag<TPP, FULL, true>(wa, wb, act_s, b, nt, hA, hB);
                else item_fmag<TPP, FULL, false>(wa, wb, act_s, b, nt, hA, hB);
            }
            flushg<TPP, FULL>(hA, hB, accA, accB, fdA, fdB, nt);
        }
        bad += store_outg<TPP, FULL>(accA, accB, ex_s, dst, trow0, nt, r0);
    }
    if (bad != 0) atomicAdd(&g_dn.bad_y, (unsigned) bad);
}
}  // namespace q

// ---- IQ4_NL
namespace n {
constexpr int S = r2n::S, ROWB = r2n::ROWB;   // shared words a row (19); a row 360 bytes

template <int TPP, bool FULL, int MINB>
__global__ void __launch_bounds__(NT, MINB) kernel(const Chunk* __restrict__ items, const char* __restrict__ w, const int64_t* __restrict__ xoff, const int64_t xadd,
                                                   const uint4* __restrict__ hb, float* __restrict__ dst) {
    static_assert(TPP % 4 == 0 && TPP >= 4 && TPP <= 32, "");
    constexpr int NTASK = 8 * TPP;
    constexpr int AI = (NTASK + NT - 1) / NT;
    constexpr int NP = 19;   // 8-byte pieces a thread per step: 18, +1 for t / 9 < 4 (linear staging)
    __shared__ int wsm[ROWS_CTA * S];
    __shared__ __align__(16) uint4 act_s[TPP * 16];
    __shared__ uint32_t lut[256];
    __shared__ float ex_s[TPP];
    const int tid = threadIdx.x;
    const int t9 = tid / 9, k9 = tid - t9 * 9;
    const bool wact = tid < 126;
    for (int e = tid; e < 256; e += NT) lut[e] = pack_h2((float) kKv[e & 15], (float) kKv[e >> 4]);
    const int r0 = (int) (blockIdx.x % P2) * ROWS_CTA;
    const int4 cc = __ldg((const int4*) items + blockIdx.x / P2);
    const Chunk c{cc.x, cc.y, cc.z, cc.w};
    const char* __restrict__ W = w + xoff[c.j] + xadd + (size_t) r0 * ROWB;
    int bad = 0;
    for (int p0 = 0; p0 < c.rows; p0 += TPP) {
        const int nt = FULL ? TPP : min(TPP, c.rows - p0);
        const int trow0 = c.row0 + p0;
        float accA[TPP], accB[TPP];
        uint32_t hA[TPP], hB[TPP];
#pragma unroll
        for (int t = 0; t < TPP; ++t) { accA[t] = 0.0f; accB[t] = 0.0f; }
        float fdpA = 0.0f, fdpB = 0.0f;
#pragma unroll 1
        for (int st = 0; st < NSTEP; ++st) {
            __syncthreads();   // everyone is done with the previous step's shared data
            int tq = tid;
            asm("" : "+r"(tq));   // the offsets below are recomputed each step, not held across the compute
            int2 wv[NP];
            const char* __restrict__ Ws = W + (size_t) t9 * ROWB + 8 * k9 + st * 72;
#pragma unroll
            for (int i = 0; i < NP; ++i)
                if (wact && (i < 18 || t9 < 4)) wv[i] = __ldg((const int2*) (Ws + 5040 * i));
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
            if (st > 0) flushg<TPP, FULL>(hA, hB, accA, accB, fdpA, fdpB, nt);   // the previous step's FP32 widening runs in the shadow of this step's global loads
#pragma unroll
            for (int i = 0; i < NP; ++i)
                if (wact && (i < 18 || t9 < 4)) {
                    const int si = (t9 + 14 * i) * S + 2 * k9;
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
            if (tid == 0 && p0 == 0 && c.pad == 1 && st == 0) atomicAdd(&g_dn.khist[kA - KMIN], 1u);   // one sample a chunk piece, as the R2 kernel's
#pragma unroll
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
                if (b == 0) item_fmag<TPP, FULL, true>(wa, wb, act_s, b, nt, hA, hB);
                else item_fmag<TPP, FULL, false>(wa, wb, act_s, b, nt, hA, hB);
            }
            fdpA = fdA;
            fdpB = fdB;
        }
        flushg<TPP, FULL>(hA, hB, accA, accB, fdpA, fdpB, nt);
        bad += store_outg<TPP, FULL>(accA, accB, ex_s, dst, trow0, nt, r0);
    }
    if (bad != 0) atomicAdd(&g_dn.bad_y, (unsigned) bad);
}
}  // namespace n

}  // namespace cls
}  // namespace
}  // namespace strata::prefill::ws::dn
