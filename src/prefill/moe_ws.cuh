// src/prefill/moe_ws.cuh - the weight-stationary expert kernels' shared pieces (see moe_ws.hpp): included after
// ggml-cuda's common.cuh and mmq.cuh by moe_ws.cu and the moe_ws_*.cu that hold the kernels (a few formats per file,
// so the instances compile in parallel).
#pragma once

#include "strata/prefill/moe_ws.hpp"

#include <cuda_runtime.h>

namespace strata::prefill::ws::detail {

// The two ways every kernel multiplies: int8 values sign-extended to int16 pairs (one PRMT per pair: bytes 0,2 and 1,3)
// and a pair of products added to a 32-bit sum by two XMAD (mad.wide.s16) - MMQ's dp4a on sm_60, one product at a time.
// A 32-value block's integer sum is exact (|sum| < 2^22 for every format here), so the order of its MACs is free.
__device__ __forceinline__ void split2(const int v, int& lo02, int& hi13) {
    asm("prmt.b32 %0, %1, 0, 0xA280;" : "=r"(lo02) : "r"(v));
    asm("prmt.b32 %0, %1, 0, 0xB391;" : "=r"(hi13) : "r"(v));
}
__device__ __forceinline__ int mad2(const int w, const int a, int c) {
    asm("{\n\t.reg .s16 al, ah, bl, bh;\n\t"
        "mov.b32 {al, ah}, %1;\n\tmov.b32 {bl, bh}, %2;\n\t"
        "mad.wide.s16 %0, al, bl, %0;\n\tmad.wide.s16 %0, ah, bh, %0;\n\t}"
        : "+r"(c) : "r"(w), "r"(a));
    return c;
}
// float(sum) for |sum| < 2^22 without the quarter-rate I2F: a chain seeded with 1.5 * 2^23 holds the float's bits
constexpr int kMagicI = 0x4B400000;
constexpr float kMagicF = 12582912.0f;

// CTA b's chunk and first weight row: chunk b / pieces, piece b % pieces (`pieces` = weight rows / rows per CTA) - the
// chunks' order, each chunk's pieces next to each other
template <int PIECES, int ROWS>
__device__ __forceinline__ Chunk cta_work(const Chunk* __restrict__ chunks, int& r0) {
    r0 = (int) (blockIdx.x % PIECES) * ROWS;
    const int4 c = __ldg((const int4*) chunks + blockIdx.x / PIECES);
    return Chunk{c.x, c.y, c.z, c.w};
}

// ---- the gate/up products' shared pieces (moe_ws_gu_iq2.cu's IQ2_XS / IQ2_S kernels and moe_ws_gu_nib.cu's 4-bit
// kernels): a CTA of kGuNT threads = kGuNT weight rows, K 128 values a step, the step's q8_1 rows in shared memory
constexpr int kGuNT = 128;
constexpr uint32_t PX = 0x002B1908u, PY = 0x00D5E7F8u;   // codes 0-2: +8 +25 +43 | codes 4-6: -8 -25 -43

// the step's activations of IQ2_XS / IQ2_S and the 4-bit kernels: nt consecutive 144-B block_q8_1_mmq, d4 -> dy_s,
// int8 -> int16 pairs
template <int TPP>
__device__ __forceinline__ void stage_act(const block_q8_1_mmq* __restrict__ xq, const int hc, const int total_rows,
                                          const int trow0, const int nt, int (*act_s)[4][16], float (*dy_s)[4]) {
    constexpr int NA = (TPP * 36 + kGuNT - 1) / kGuNT;
    int av[NA];
    const int* xr = (const int*) (xq + (size_t) hc * total_rows + trow0);
#pragma unroll
    for (int i = 0; i < NA; ++i) {
        const int idx = threadIdx.x + i * kGuNT;
        if (idx < nt * 36) av[i] = __ldg(xr + idx);
    }
#pragma unroll
    for (int i = 0; i < NA; ++i) {
        const int idx = threadIdx.x + i * kGuNT;
        if (idx < nt * 36) {
            const int t = idx / 36, wd = idx - t * 36;
            if (wd < 4) {
                dy_s[t][wd] = __int_as_float(av[i]);
            } else {
                const int m = wd - 4;
                int lo, hi;
                split2(av[i], lo, hi);
                *(int2*) &act_s[t][m / 8][2 * (m % 8)] = make_int2(lo, hi);
            }
        }
    }
}

// IQ2_XS / IQ2_S: sub-block s's 16 int16-pair words wv and scales d0 / d1 against the tokens (vec_dot_q8_0_16_q8_1's
// epilogue).  CVT 0: four XMAD chains and I2F; 1: two chains seeded with kMagicI (exact for the 16-value sums,
// |sum| <= 16 * 43 * 127 < 2^22)
template <int TPP, int CVT, int TG>
__device__ __forceinline__ void mac16(const int (&wv)[16], const float dx0, const float dx1, const int s,
                                      const int (*act_s)[4][16], const float (*dy_s)[4], const int nt, float (&acc)[TPP]) {
#pragma unroll
    for (int t0 = 0; t0 < TPP; t0 += TG) {
        if (t0 < nt) {
#pragma unroll
            for (int t = t0; t < t0 + TG; ++t) {
                const int4* a4 = (const int4*) act_s[t][s];
                const int4 a0 = a4[0], a1 = a4[1], a2 = a4[2], a3 = a4[3];
                const float dy = dy_s[t][s];
                float fa, fb;   // float(sum of values 0-15), float(sum of values 16-31): exact
                if constexpr (CVT == 0) {
                    int s0 = 0, s1 = 0, s2 = 0, s3 = 0;
                    s0 = mad2(wv[0], a0.x, s0); s1 = mad2(wv[1], a0.y, s1); s0 = mad2(wv[2], a0.z, s0); s1 = mad2(wv[3], a0.w, s1);
                    s0 = mad2(wv[4], a1.x, s0); s1 = mad2(wv[5], a1.y, s1); s0 = mad2(wv[6], a1.z, s0); s1 = mad2(wv[7], a1.w, s1);
                    s2 = mad2(wv[8], a2.x, s2); s3 = mad2(wv[9], a2.y, s3); s2 = mad2(wv[10], a2.z, s2); s3 = mad2(wv[11], a2.w, s3);
                    s2 = mad2(wv[12], a3.x, s2); s3 = mad2(wv[13], a3.y, s3); s2 = mad2(wv[14], a3.z, s2); s3 = mad2(wv[15], a3.w, s3);
                    fa = __int2float_rn(s0 + s1);
                    fb = __int2float_rn(s2 + s3);
                } else {
                    int sa = kMagicI, sb = kMagicI;
                    sa = mad2(wv[0], a0.x, sa); sb = mad2(wv[8], a2.x, sb);
                    sa = mad2(wv[1], a0.y, sa); sb = mad2(wv[9], a2.y, sb);
                    sa = mad2(wv[2], a0.z, sa); sb = mad2(wv[10], a2.z, sb);
                    sa = mad2(wv[3], a0.w, sa); sb = mad2(wv[11], a2.w, sb);
                    sa = mad2(wv[4], a1.x, sa); sb = mad2(wv[12], a3.x, sb);
                    sa = mad2(wv[5], a1.y, sa); sb = mad2(wv[13], a3.y, sb);
                    sa = mad2(wv[6], a1.z, sa); sb = mad2(wv[14], a3.z, sb);
                    sa = mad2(wv[7], a1.w, sa); sb = mad2(wv[15], a3.w, sb);
                    fa = __fsub_rn(__int_as_float(sa), kMagicF);
                    fb = __fsub_rn(__int_as_float(sb), kMagicF);
                }
                float tf = __fmaf_rn(fa, dx0, 0.0f);
                tf = __fmaf_rn(fb, dx1, tf);
                acc[t] = __fmaf_rn(tf, dy, acc[t]);
            }
        }
    }
}

// d0 / d1 = ((ls & 15 | ls >> 4) * d + d / 2) / 4 as MMQ's sm_60 SASS computes them
__device__ __forceinline__ void scales16(const int ls, const int dbits, float& d0, float& d1) {
    const float d = __half2float(__ushort_as_half((unsigned short) (dbits & 0xFFFF)));
    const float dh = __fmul_rn(d, 0.5f);
    d0 = __fmul_rn(__fmaf_rn(d, __int2float_rn(ls & 0x0F), dh), 0.25f);
    d1 = __fmul_rn(__fmaf_rn(d, __int2float_rn(ls >> 4), dh), 0.25f);
}

// one launch of a layer's product, the variant picked by u (rows per routed expert); grid = nchunks x pieces
void gu_iq2_xxs(const Layer& L, double u, cudaStream_t s);
void gu_iq2_xs(const Layer& L, double u, cudaStream_t s);
void gu_iq2_s(const Layer& L, double u, cudaStream_t s);
void gu_iq3_xxs(const Layer& L, double u, cudaStream_t s);
void gu_iq3_s(const Layer& L, double u, cudaStream_t s);
// STRATA_EXPERT_NIBBLE: a layer whose gate/up experts are 4-bit codes (Layer::gu_nib; IQ2_S, IQ3_XXS, IQ3_S;
// moe_ws_gu_nib.cu, built where STRATA_EXPERT_NIBBLE_BUILT).  `fast`: ws::fast() (STRATA_WS_FAST=1) - the 32-token
// tier's fast kernel, the same GU rows
// `v2`: ws::v2() (STRATA_WS_V2=1) - moe_ws_v2.cuh's one-row kernels with the tiers moved (gu_nib), the same GU rows
void gu_nib(const Layer& L, double u, cudaStream_t s, bool fast, bool v2);
void dn_q2_0(const Layer& L, double u, cudaStream_t s);
// `fast`: ws::fast() (STRATA_WS_FAST=1) - the 32-token tier's fast kernel, the same Dm rows
void dn_iq4_nl(const Layer& L, double u, cudaStream_t s, bool fast);

}  // namespace strata::prefill::ws::detail
