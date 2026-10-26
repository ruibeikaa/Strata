// src/prefill/moe_ws_gu_nib.cu - the weight-stationary gate/up product (see moe_ws.hpp) of a layer
// whose experts are 4-bit codes in their VRAM slots (STRATA_EXPERT_NIBBLE=1, ws::Layer::gu_nib; the slot layout and the
// item code: strata/core/expert_nibble.hpp).  IQ2_S, IQ3_XXS and IQ3_S, [1280 x 2560], read in place.
//
// One template, gu_kernel<TPP, CVT, TG, FMT>: a CTA = 128 threads = 128 weight rows of one expert (a row per thread,
// the chunk's activation rows TPP at a time, the sums in registers), K 128 values = 4 items a step, as the GGUF WS
// kernels (moe_ws_gu_iq2.cu, moe_ws_gu_iq3.cu).  The slot puts each 32-weight item in one aligned 16-byte code, so the
// staging is the same for the three formats: lane -> (row lane >> 2 of the warp's 8, item lane & 3),
// one LDG.128 per lane and round, 4 rounds for the 128 rows, stored as STS.128 into shared memory rows 20 words apart
// (a quarter warp's LDS.128 of 8 rows hit 8 distinct 16-byte bank groups); the row's scale word (the step's 4 items)
// and its fp16 d are the row's own thread's LDG.  Nothing reads outside [codes, scales, d) of a slot.
//
// FMT changes only the nibble -> int8 words and dx; the int8 values are the GGUF WS kernels', byte for byte:
//   IQ2_S    word m of the code = 8 nibbles = the GGUF kernel's sel = tabA | tabB: one PRMT per 4 values (__byte_perm(PX, PY))
//   IQ3_XXS  magnitudes {4 .. 62} and their negatives by PRMT on the 3 low bits, the sign bytes by PRMT's
//            sign-replicate mode on bit 3, one LOP3 picks per byte (the GGUF kernel's (g ^ s) + (s & 0x01010101) values)
//   IQ3_S    v = 2u - 15 per byte (((2u | 0x80) - 0x0F) ^ 0x80: every byte >= 0x80 > 0x0F, no borrow between bytes)
// and dx from the kept scale byte and d bits is each format's GGUF WS expression (IQ2_S scales16; IQ3_XXS
// (ls*d + d/2)/2; IQ3_S ls * d).  Then the GGUF WS epilogue in K order: IQ2_S mac16 (moe_ws.cuh); IQ3_XXS / IQ3_S
// acc = fma(dx * dy, float(sum), acc) over the 32-value sum, float(sum) exact either way (|sum| < 2^22: the kMagicI
// chains, CH of them, give I2F's value).  So every GU row is the GGUF WS kernels' bit for bit (checked on the
// 0.1.41-based tree: one real 512-expert layer per format, T 1..8192, every tier, 0 rows of 1280 floats differ).
#include "common.cuh"
#include "mmq.cuh"

#include "moe_ws.cuh"
#include "strata/core/expert_nibble.hpp"
#include "moe_ws_v2.cuh"   // STRATA_WS_V2's two-row kernel

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <type_traits>

#if STRATA_EXPERT_NIBBLE_BUILT
namespace strata::prefill::ws::detail {
namespace {

namespace nb = strata::core::nibble;

constexpr int GU_ROWS = 1280, GU_K = 2560, NB = GU_K / QK_K, NSTEP = 2 * NB, NT = kGuNT, PIECES = GU_ROWS / NT;
constexpr int ITEMS = (int) nb::kItems;
static_assert(nb::kRows == GU_ROWS && nb::kNEmbd == GU_K && ITEMS == GU_K / 32 && nb::kSuper == NB && nb::kCodesOff == 0,
              "the 4-bit slot of strata/core/expert_nibble.hpp is the WS kernels' [1280 x 2560]");
static_assert(NSTEP * 4 == ITEMS, "a step = 4 items");
constexpr int CS = 20;   // shared memory row stride in words: 16 code words + 4 (S1W)

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "prefill ws: %s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

__device__ __forceinline__ uint32_t prmt(const uint32_t a, const uint32_t b, const uint32_t sel) {
    uint32_t r;
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(sel));   // default mode: bit 3 = sign replicate
    return r;
}

// one item's 16-byte code -> its 8 int8x4 words (the GGUF kernel's W.g[0..7]) split to the 16 int16-pair words wv; dx
template <int FMT> struct Dec;
template <> struct Dec<nb::kIQ2_S> {
    __device__ static __forceinline__ void wv(const uint32_t (&q)[4], int (&w)[16]) {
#pragma unroll
        for (int m = 0; m < 4; ++m) {
            split2((int) __byte_perm(PX, PY, q[m]), w[4 * m + 0], w[4 * m + 1]);
            split2((int) __byte_perm(PX, PY, q[m] >> 16), w[4 * m + 2], w[4 * m + 3]);
        }
    }
    __device__ static __forceinline__ void dx(const int ls, const int dbits, float& d0, float& d1) { scales16(ls, dbits, d0, d1); }
};
template <> struct Dec<nb::kIQ3_XXS> {
    static constexpr uint32_t T0 = 0x1C140C04u, T1 = 0x3E342C24u;   // {4, 12, 20, 28} {36, 44, 52, 62}
    static constexpr uint32_t N0 = 0xE4ECF4FCu, N1 = 0xC2CCD4DCu;   // their negatives: (g ^ s) + (s & 1) for s = 0xFF
    __device__ static __forceinline__ void wv(const uint32_t (&q)[4], int (&w)[16]) {
#pragma unroll
        for (int m = 0; m < 4; ++m) {
            const uint32_t c = q[m], ms = c & 0x77777777u, a = c << 4;
            // sign bytes: nibble b's bit 3 is the top bit of byte b/2 of (c << 4) (b even) or of c (b odd)
            const uint32_t s0 = prmt(a, c, 0xD9C8u), s1 = prmt(a, c, 0xFBEAu);
            // each byte the magnitude or its negative by its sign byte (one LOP3): the GGUF kernel's int8 value
            const uint32_t g0 = (prmt(T0, T1, ms) & ~s0) | (prmt(N0, N1, ms) & s0);
            const uint32_t g1 = (prmt(T0, T1, ms >> 16) & ~s1) | (prmt(N0, N1, ms >> 16) & s1);
            split2((int) g0, w[4 * m + 0], w[4 * m + 1]);
            split2((int) g1, w[4 * m + 2], w[4 * m + 3]);
        }
    }
    __device__ static __forceinline__ float dx(const int ls, const int dbits) {
        const float d = __ushort_as_half((unsigned short) (dbits & 0xFFFF));
        return (ls*d + d/2)/2;   // moe_ws_gu_iq3.cu xxs3 (MMQ's expression), verbatim
    }
};
template <> struct Dec<nb::kIQ3_S> {
    __device__ static __forceinline__ void wv(const uint32_t (&q)[4], int (&w)[16]) {
#pragma unroll
        for (int m = 0; m < 4; ++m) {
            const uint32_t lo2 = (q[m] << 1) & 0x1E1E1E1Eu, hi2 = (q[m] >> 3) & 0x1E1E1E1Eu;   // 2u per byte
            split2((int) (((lo2 | 0x80808080u) - 0x0F0F0F0Fu) ^ 0x80808080u), w[4 * m + 0], w[4 * m + 1]);
            split2((int) (((hi2 | 0x80808080u) - 0x0F0F0F0Fu) ^ 0x80808080u), w[4 * m + 2], w[4 * m + 3]);
        }
    }
    __device__ static __forceinline__ float dx(const int ls, const int dbits) {
        const float d = __ushort_as_half((unsigned short) (dbits & 0xFFFF));
        return ls * d;   // moe_ws_gu_iq3.cu s3::dec, verbatim
    }
};

// IQ3_XXS / IQ3_S: an item's 16 int16-pair words wv against the tokens, MMQ's epilogue acc = fma(dx * dy, float(sum),
// acc).  CH chains, the first seeded with kMagicI (exact: |sum| <= 32 * 62 * 127 < 2^22)
template <int TPP, int CH, int TG>
__device__ __forceinline__ void mac32(const int (&wv)[16], const float dx, const int s, const int (*act_s)[4][16],
                                      const float (*dy_s)[4], const int nt, float (&acc)[TPP]) {
#pragma unroll
    for (int t0 = 0; t0 < TPP; t0 += TG) {
        if (t0 < nt) {
#pragma unroll
            for (int t = t0; t < t0 + TG; ++t) {
                const int4* a4 = (const int4*) act_s[t][s];
                const int4 a0 = a4[0], a1 = a4[1], a2 = a4[2], a3 = a4[3];
                const int a[16] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w, a2.x, a2.y, a2.z, a2.w, a3.x, a3.y, a3.z, a3.w};
                const float dy = dy_s[t][s];
                int sc[CH];
                sc[0] = kMagicI;
#pragma unroll
                for (int c = 1; c < CH; ++c) sc[c] = 0;
#pragma unroll
                for (int m = 0; m < 16; ++m) sc[m % CH] = mad2(wv[m], a[m], sc[m % CH]);
                int sum = sc[0];
#pragma unroll
                for (int c = 1; c < CH; ++c) sum += sc[c];
                acc[t] = __fmaf_rn(__fmul_rn(dx, dy), __fsub_rn(__int_as_float(sum), kMagicF), acc[t]);
            }
        }
    }
}

// CVT: IQ2_S mac16's (0: four chains and I2F, 1: two kMagicI chains); IQ3_XXS / IQ3_S mac32's chain count
template <int TPP, int CVT, int TG, int FMT>
__global__ void __launch_bounds__(NT, 1) gu_kernel(const Chunk* __restrict__ chunks, const char* __restrict__ w,
                                                  const int64_t* __restrict__ xoff, const block_q8_1_mmq* __restrict__ xq,
                                                  const int total_rows, float* __restrict__ dst) {
    static_assert(TPP % TG == 0, "");
    __shared__ __align__(16) int act_s[TPP][4][16];
    __shared__ float dy_s[TPP][4];
    __shared__ __align__(16) uint32_t csm[NT * CS];
    int r0;
    const Chunk c = cta_work<PIECES, NT>(chunks, r0);
    const char* __restrict__ slot = w + xoff[c.j];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    // the thread's row: its scale word (+ hc: the step's 4 items) and its d (+ kc)
    const uint32_t* __restrict__ scw = (const uint32_t*) (slot + nb::kScOff + (size_t) (r0 + tid) * ITEMS);
    const unsigned short* __restrict__ dr = (const unsigned short*) (slot + nb::kDOff) + (size_t) (r0 + tid) * NB;
    // the staging lane's rows (srow + 32 i, i = 0..3) and item (sq) of the step
    const int srow = warp * 8 + (lane >> 2), sq = lane & 3;
    const uint4* __restrict__ cg = (const uint4*) slot + (size_t) (r0 + srow) * ITEMS + sq;   // + i*32*ITEMS + 4hc
    uint32_t* __restrict__ cs_st = csm + srow * CS + 4 * sq;                                  // + i*32*CS
    const uint32_t* cw = csm + tid * CS;
    for (int p0 = 0; p0 < c.rows; p0 += TPP) {
        const int nt = min(TPP, c.rows - p0);
        const int trow0 = c.row0 + p0;
        float acc[TPP];
#pragma unroll
        for (int t = 0; t < TPP; ++t) acc[t] = 0.0f;
#pragma unroll 1
        for (int hc = 0; hc < NSTEP; ++hc) {
            const int kc = hc >> 1;
            __syncthreads();
            uint4 v[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) v[i] = __ldg(cg + (size_t) i * 32 * ITEMS + 4 * hc);
            const uint32_t scv = __ldg(scw + hc);
            const int dv = (int) __ldg(dr + kc);
            stage_act<TPP>(xq, hc, total_rows, trow0, nt, act_s, dy_s);
#pragma unroll
            for (int i = 0; i < 4; ++i) *(uint4*) (cs_st + i * 32 * CS) = v[i];
            __syncthreads();
#pragma unroll 1
            for (int s = 0; s < 4; ++s) {
                const uint4 u = *(const uint4*) (cw + 4 * s);
                const uint32_t q[4] = {u.x, u.y, u.z, u.w};
                const int ls = (int) ((scv >> (8 * s)) & 0xFF);
                int wv[16];
                Dec<FMT>::wv(q, wv);
                if constexpr (FMT == nb::kIQ2_S) {
                    float dx0, dx1;
                    Dec<FMT>::dx(ls, dv, dx0, dx1);
                    mac16<TPP, CVT, TG>(wv, dx0, dx1, s, act_s, dy_s, nt, acc);
                } else {
                    mac32<TPP, CVT, TG>(wv, Dec<FMT>::dx(ls, dv), s, act_s, dy_s, nt, acc);
                }
            }
        }
#pragma unroll
        for (int t = 0; t < TPP; ++t)
            if (t < nt) dst[(size_t) (trow0 + t) * GU_ROWS + r0 + tid] = acc[t];
    }
}

// ---- STRATA_WS_FAST=1 (ws::fast(), moe_ws.hpp): the 32-token tier with fewer instructions per pass, the same GU rows
// bit for bit.  Every token keeps its own operations in their order - the items in K order, each item's 16 XMAD pairs
// into the same chains, the same epilogue - and only the staging, the control flow and the shared-memory layout change.
// (Prototyped as moe_ws_prof_bench's variants, branch ws-prof: memcmp-equal to gu_kernel above on real layers of each
// format at T 1024 and 1536, lognormal counts sigma 0.5 / 1.0 and a flat-32 control; 2-5% faster at T 1536, 5-9% at
// T 1024 on a GP100 die.  The kernel issues as many instructions as the SM can - 99-101% of the issue slots - so its
// time follows the instruction count.)
//   ST16  (always) the step's activations in 16-byte pieces: a token's 144-B block_q8_1_mmq is 9 LDG.128 - piece 0 its
//         4 item d, pieces 1-8 16 int8 values each, split to 8 int16-pair words and stored as two STS.128 - instead
//         of 36 LDG.32, each with its own divide by 36 and branch (stage_act); the same act_s and dy values
//   TAIL  the unrolled loop runs whole TG-token groups only; the last nt % TG tokens are computed one by one into
//         their own sums (acct) - not a whole group with stale rows past nt
//   DUFF  a switch on the number of groups falls through the unrolled groups (a jump table) instead of a guard per
//         group and item
//   DECF  IQ3_S: ((2u | 0x80) - 0x0F) ^ 0x80 per byte as (2u + 0x71) ^ 0x80 (2u <= 0x1E has bit 7 clear: no carry)
//   DYT   dy as [item][token]: a group's TG dy in one LDS.128 (TG 4) or LDS.64 (TG 2)
// The picks per format (WS-PROFILE.md section 4, the best of the measured combinations): IQ2_S DUFF, IQ3_XXS TAIL + DYT,
// IQ3_S TAIL + DUFF + DECF + DYT, all with ST16.
namespace fast {
enum : int { TAIL = 1, DUFF = 2, DECF = 4, DYT = 8 };

// dy of token t, item s: [t][4] (gu_kernel's dy_s) or [4][TPP] (DYT)
template <int TPP, bool T_> struct DyS {
    float a[T_ ? 4 : TPP][T_ ? TPP : 4];
    __device__ __forceinline__ float at(const int t, const int s) const {
        if constexpr (T_) return a[s][t];
        else return a[t][s];
    }
};

// ST16: stage_act's act_s / dy values from 16-byte pieces (Xq 16-byte aligned: the caller checks)
template <int TPP, bool T_>
__device__ __forceinline__ void stage_act16(const block_q8_1_mmq* __restrict__ xq, const int hc, const int total_rows,
                                            const int trow0, const int nt, int (*act_s)[4][16], DyS<TPP, T_>& dy_s) {
    constexpr int NA = (TPP * 9 + NT - 1) / NT;
    int4 av[NA];
    const int4* xr = (const int4*) (xq + (size_t) hc * total_rows + trow0);
#pragma unroll
    for (int i = 0; i < NA; ++i) {
        const int idx = threadIdx.x + i * NT;
        if (idx < nt * 9) av[i] = __ldg(xr + idx);
    }
#pragma unroll
    for (int i = 0; i < NA; ++i) {
        const int idx = threadIdx.x + i * NT;
        if (idx < nt * 9) {
            const int t = idx / 9, pc = idx - t * 9;
            const int4 v = av[i];
            if (pc == 0) {
                if constexpr (T_) {
                    dy_s.a[0][t] = __int_as_float(v.x); dy_s.a[1][t] = __int_as_float(v.y);
                    dy_s.a[2][t] = __int_as_float(v.z); dy_s.a[3][t] = __int_as_float(v.w);
                } else {
                    *(int4*) &dy_s.a[t][0] = v;
                }
            } else {
                // words m0 .. m0 + 3 of the 32: item m0 / 8, pair slots 2 (m0 % 8) .. 2 (m0 % 8) + 7 (stage_act's places)
                const int m0 = 4 * (pc - 1);
                int a0, a1, b0, b1, c0, c1, d0, d1;
                split2(v.x, a0, a1); split2(v.y, b0, b1); split2(v.z, c0, c1); split2(v.w, d0, d1);
                int4* dst = (int4*) &act_s[t][m0 >> 3][2 * (m0 & 7)];
                dst[0] = make_int4(a0, a1, b0, b1);
                dst[1] = make_int4(c0, c1, d0, d1);
            }
        }
    }
}

// the TG dy of tokens t0 .. t0 + TG - 1, item s
template <int TPP, bool T_, int TG>
__device__ __forceinline__ void dy_group(const DyS<TPP, T_>& dy_s, const int t0, const int s, float (&d)[TG]) {
    if constexpr (T_ && TG == 4) {
        const float4 v = *(const float4*) &dy_s.a[s][t0];
        d[0] = v.x; d[1] = v.y; d[2] = v.z; d[3] = v.w;
    } else if constexpr (T_ && TG == 2) {
        const float2 v = *(const float2*) &dy_s.a[s][t0];
        d[0] = v.x; d[1] = v.y;
    } else {
#pragma unroll
        for (int k = 0; k < TG; ++k) d[k] = dy_s.at(t0 + k, s);
    }
}

// one token's IQ2_S product: mac16's body (CVT as mac16)
template <int CVT>
__device__ __forceinline__ float mac16_one(const int (&wv)[16], const float dx0, const float dx1, const int4* a4, const float dy,
                                           const float acc) {
    const int4 a0 = a4[0], a1 = a4[1], a2 = a4[2], a3 = a4[3];
    float fa, fb;
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
    return __fmaf_rn(tf, dy, acc);
}

// one token's IQ3_XXS / IQ3_S product: mac32's body (CH chains as mac32)
template <int CH>
__device__ __forceinline__ float mac32_one(const int (&wv)[16], const float dx, const int4* a4, const float dy, const float acc) {
    const int4 a0 = a4[0], a1 = a4[1], a2 = a4[2], a3 = a4[3];
    const int a[16] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w, a2.x, a2.y, a2.z, a2.w, a3.x, a3.y, a3.z, a3.w};
    int sc[CH];
    sc[0] = kMagicI;
#pragma unroll
    for (int c = 1; c < CH; ++c) sc[c] = 0;
#pragma unroll
    for (int m = 0; m < 16; ++m) sc[m % CH] = mad2(wv[m], a[m], sc[m % CH]);
    int sum = sc[0];
#pragma unroll
    for (int c = 1; c < CH; ++c) sum += sc[c];
    return __fmaf_rn(__fmul_rn(dx, dy), __fsub_rn(__int_as_float(sum), kMagicF), acc);
}

// item s against the chunk's tokens; one(t, dy, acc) is the format's per-token product.  TAIL: whole TG groups only,
// the last nt % TG tokens into acct[0 ..); DUFF: a switch into the unrolled groups (8 of them: TPP / TG == 8)
template <int TPP, int TG, bool T_, bool TAIL_, bool DUFF_, typename One>
__device__ __forceinline__ void mac_tokens(const One& one, const int s, const DyS<TPP, T_>& dy_s, const int nt,
                                           float (&acc)[TPP], float (&acct)[TG]) {
    if constexpr (TAIL_) {
        const int nf = nt & ~(TG - 1);
#pragma unroll
        for (int k = 0; k < TG - 1; ++k)
            if (nf + k < nt) acct[k] = one(nf + k, dy_s.at(nf + k, s), acct[k]);
    }
    if constexpr (DUFF_) {
        static_assert(TPP / TG == 8, "the switch below has 8 cases");
        auto grp = [&](auto K) {
            constexpr int t0 = decltype(K)::value * TG;
            float dyg[TG];
            dy_group<TPP, T_, TG>(dy_s, t0, s, dyg);
#pragma unroll
            for (int t = t0; t < t0 + TG; ++t) acc[t] = one(t, dyg[t - t0], acc[t]);
        };
        using std::integral_constant;
        switch (TAIL_ ? nt / TG : (nt + TG - 1) / TG) {
            case 8: grp(integral_constant<int, 7>{}); [[fallthrough]];
            case 7: grp(integral_constant<int, 6>{}); [[fallthrough]];
            case 6: grp(integral_constant<int, 5>{}); [[fallthrough]];
            case 5: grp(integral_constant<int, 4>{}); [[fallthrough]];
            case 4: grp(integral_constant<int, 3>{}); [[fallthrough]];
            case 3: grp(integral_constant<int, 2>{}); [[fallthrough]];
            case 2: grp(integral_constant<int, 1>{}); [[fallthrough]];
            case 1: grp(integral_constant<int, 0>{}); [[fallthrough]];
            default: break;
        }
    } else {
#pragma unroll
        for (int t0 = 0; t0 < TPP; t0 += TG) {
            if (TAIL_ ? t0 + TG <= nt : t0 < nt) {
                float dyg[TG];
                dy_group<TPP, T_, TG>(dy_s, t0, s, dyg);
#pragma unroll
                for (int t = t0; t < t0 + TG; ++t) acc[t] = one(t, dyg[t - t0], acc[t]);
            }
        }
    }
}

// DECF: IQ3_S's decode with the fused byte arithmetic; the other formats' as Dec
template <int FMT> struct DecF : Dec<FMT> {};
template <> struct DecF<nb::kIQ3_S> {
    __device__ static __forceinline__ void wv(const uint32_t (&q)[4], int (&w)[16]) {
#pragma unroll
        for (int m = 0; m < 4; ++m) {
            const uint32_t lo2 = (q[m] << 1) & 0x1E1E1E1Eu, hi2 = (q[m] >> 3) & 0x1E1E1E1Eu;   // 2u per byte, bit 7 clear
            split2((int) ((lo2 + 0x71717171u) ^ 0x80808080u), w[4 * m + 0], w[4 * m + 1]);
            split2((int) ((hi2 + 0x71717171u) ^ 0x80808080u), w[4 * m + 2], w[4 * m + 3]);
        }
    }
};

// gu_kernel's arguments and work; V: the TAIL / DUFF / DECF / DYT bits (ST16 always)
template <int TPP, int CVT, int TG, int FMT, int V>
__global__ void __launch_bounds__(NT, 1) gu_kernel(const Chunk* __restrict__ chunks, const char* __restrict__ w,
                                                  const int64_t* __restrict__ xoff, const block_q8_1_mmq* __restrict__ xq,
                                                  const int total_rows, float* __restrict__ dst) {
    static_assert(TPP % TG == 0, "");
    constexpr bool TAIL_ = (V & TAIL) != 0, DUFF_ = (V & DUFF) != 0, DECF_ = (V & DECF) != 0, DYT_ = (V & DYT) != 0;
    __shared__ __align__(16) int act_s[TPP][4][16];
    __shared__ __align__(16) DyS<TPP, DYT_> dy_s;
    __shared__ __align__(16) uint32_t csm[NT * CS];
    int r0;
    const Chunk c = cta_work<PIECES, NT>(chunks, r0);
    const char* __restrict__ slot = w + xoff[c.j];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const uint32_t* __restrict__ scw = (const uint32_t*) (slot + nb::kScOff + (size_t) (r0 + tid) * ITEMS);
    const unsigned short* __restrict__ dr = (const unsigned short*) (slot + nb::kDOff) + (size_t) (r0 + tid) * NB;
    const int srow = warp * 8 + (lane >> 2), sq = lane & 3;
    const uint4* __restrict__ cg = (const uint4*) slot + (size_t) (r0 + srow) * ITEMS + sq;
    uint32_t* __restrict__ cs_st = csm + srow * CS + 4 * sq;
    const uint32_t* cw = csm + tid * CS;
    for (int p0 = 0; p0 < c.rows; p0 += TPP) {
        const int nt = min(TPP, c.rows - p0);
        const int trow0 = c.row0 + p0;
        float acc[TPP], acct[TG];
#pragma unroll
        for (int t = 0; t < TPP; ++t) acc[t] = 0.0f;
#pragma unroll
        for (int t = 0; t < TG; ++t) acct[t] = 0.0f;
#pragma unroll 1
        for (int hc = 0; hc < NSTEP; ++hc) {
            const int kc = hc >> 1;
            __syncthreads();
            uint4 v[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) v[i] = __ldg(cg + (size_t) i * 32 * ITEMS + 4 * hc);
            const uint32_t scv = __ldg(scw + hc);
            const int dv = (int) __ldg(dr + kc);
            stage_act16<TPP, DYT_>(xq, hc, total_rows, trow0, nt, act_s, dy_s);
#pragma unroll
            for (int i = 0; i < 4; ++i) *(uint4*) (cs_st + i * 32 * CS) = v[i];
            __syncthreads();
#pragma unroll 1
            for (int s = 0; s < 4; ++s) {
                const uint4 u = *(const uint4*) (cw + 4 * s);
                const uint32_t q[4] = {u.x, u.y, u.z, u.w};
                const int ls = (int) ((scv >> (8 * s)) & 0xFF);
                int wv[16];
                if constexpr (DECF_) DecF<FMT>::wv(q, wv);
                else Dec<FMT>::wv(q, wv);
                if constexpr (FMT == nb::kIQ2_S) {
                    float dx0, dx1;
                    Dec<FMT>::dx(ls, dv, dx0, dx1);
                    const auto one = [&](const int t, const float dy, const float a) {
                        return mac16_one<CVT>(wv, dx0, dx1, (const int4*) act_s[t][s], dy, a);
                    };
                    mac_tokens<TPP, TG, DYT_, TAIL_, DUFF_>(one, s, dy_s, nt, acc, acct);
                } else {
                    const float dx = Dec<FMT>::dx(ls, dv);
                    const auto one = [&](const int t, const float dy, const float a) {
                        return mac32_one<CVT>(wv, dx, (const int4*) act_s[t][s], dy, a);
                    };
                    mac_tokens<TPP, TG, DYT_, TAIL_, DUFF_>(one, s, dy_s, nt, acc, acct);
                }
            }
        }
        if constexpr (TAIL_) {
            const int nf = nt & ~(TG - 1);
#pragma unroll
            for (int t = 0; t < TPP; ++t)
                if (t < nf) dst[(size_t) (trow0 + t) * GU_ROWS + r0 + tid] = acc[t];
#pragma unroll
            for (int k = 0; k < TG - 1; ++k)
                if (nf + k < nt) dst[(size_t) (trow0 + nf + k) * GU_ROWS + r0 + tid] = acct[k];
        } else {
#pragma unroll
            for (int t = 0; t < TPP; ++t)
                if (t < nt) dst[(size_t) (trow0 + t) * GU_ROWS + r0 + tid] = acc[t];
        }
    }
}
}  // namespace fast

template <typename F>
void go(F kernel, const Layer& L, cudaStream_t s) {
    kernel<<<(unsigned) (L.nchunks * PIECES), NT, 0, s>>>(L.chunks, (const char*) L.w, L.xoff, (const block_q8_1_mmq*) L.xq,
                                                         (int) L.rows, L.gu);
}

// STRATA_WS_V2: v2::gu_kernel<32, CH, 4, FMT, V> (v2::gu_pieces<V>() pieces a chunk)
template <int CH, int FMT, int V>
void go_v2(const Layer& L, cudaStream_t s) {
    v2::gu_kernel<32, CH, 4, FMT, V><<<(unsigned) (L.nchunks * v2::gu_pieces<V>()), v2::NT, 0, s>>>(
        L.chunks, (const char*) L.w, L.xoff, (const block_q8_1_mmq*) L.xq, (int) L.rows, L.gu);
}

}  // namespace

// tokens per pass by u (rows per routed expert), the crossovers measured on the 4-bit kernels (one 512-expert layer of
// the Swift pack per format, lognormal counts sigma 0.5 and 1.0, a GP100 die): IQ2_S 8 below 6, 16 below 25 (four chains and
// I2F: faster here than the two kMagicI chains), then 32 in 4-token groups; IQ3_XXS 8 below 4, 16 below 18 (one chain),
// then 32 (one chain, 4-token groups); IQ3_S 8 below 4, 16 below 18, then 32 (two chains).  `fast` (STRATA_WS_FAST=1):
// the 32-token tier runs fast::gu_kernel instead - the same GU rows; its 16-byte loads need Xq 16-byte aligned (the
// prompt path's buffers are 256-byte aligned; anything else keeps gu_kernel)
//
// `v2_on` (STRATA_WS_V2=1, WS2.md; measured on a GP100 die, one 512-expert layer per format, lognormal sigma 0.5 / 1.0
// and flat-32 counts at T 384 .. 4096, every arm memcmp-equal to these kernels): the 32-token kernels of moe_ws_v2.cuh
// from lower u - the 32-token FAST kernel already beats the 16-token one at u 7.5 (IQ3_S, 3-6%) / 10 (IQ3_XXS) - and
// for IQ2_S (u >= 17) and IQ3_S (u >= 16) the form whose threads read their row's item codes from global memory one
// item ahead (no code staging: 8.5 KB of shared memory, four CTAs an SM for IQ3_S).
void gu_nib(const Layer& L, double u, cudaStream_t s, bool fast_on, bool v2_on, bool v3_on) {
    if (L.nchunks <= 0) return;
    const bool aligned = ((uintptr_t) L.xq & 15) == 0;
    const bool f = fast_on && aligned, f2 = v2_on && aligned, f3 = v3_on && aligned;
    if (f2) {
        switch (L.gu_type) {
            case nb::kIQ2_S:
                if (u < 6.0) go(gu_kernel<8, 1, 2, nb::kIQ2_S>, L, s);
                else if (u < 17.0) go(gu_kernel<16, 0, 2, nb::kIQ2_S>, L, s);
                else go_v2<1, nb::kIQ2_S, v2::DUFF | v2::CD | v2::LB3>(L, s);
                break;
            case nb::kIQ3_XXS:
                if (u < 4.0) go(gu_kernel<8, 2, 2, nb::kIQ3_XXS>, L, s);
                else if (u < 9.0) go(gu_kernel<16, 1, 2, nb::kIQ3_XXS>, L, s);
                else go_v2<1, nb::kIQ3_XXS, v2::TAIL | v2::DYT>(L, s);
                break;
            case nb::kIQ3_S:
                if (u < 4.0) go(gu_kernel<8, 2, 2, nb::kIQ3_S>, L, s);
                else if (u < 7.0) go(gu_kernel<16, 2, 2, nb::kIQ3_S>, L, s);
                else if (u < 16.0) go_v2<2, nb::kIQ3_S, v2::TAIL | v2::DUFF | v2::DECF | v2::DYT>(L, s);
                else go_v2<2, nb::kIQ3_S, v2::TAIL | v2::DUFF | v2::DECF | v2::DYT | v2::CD | v2::LB4>(L, s);
                break;
            default:
                std::fprintf(stderr, "prefill ws: 4-bit gate/up type %d has no kernel\n", L.gu_type);
                std::exit(1);
        }
        ck(cudaGetLastError(), "gate/up 4-bit");
        return;
    }
    if (f3) {
        switch (L.gu_type) {
            case nb::kIQ2_S:
                if (u < 6.0) go(gu_kernel<8, 1, 2, nb::kIQ2_S>, L, s);
                else if (u < 25.0) go(fast::gu_kernel<16, 0, 2, nb::kIQ2_S, fast::TAIL | fast::DUFF | fast::DYT>, L, s);
                else if (f) go(fast::gu_kernel<32, 1, 4, nb::kIQ2_S, fast::DUFF>, L, s);
                else go(gu_kernel<32, 1, 4, nb::kIQ2_S>, L, s);
                break;
            case nb::kIQ3_XXS:
                if (u < 4.0) go(gu_kernel<8, 2, 2, nb::kIQ3_XXS>, L, s);
                else if (u < 18.0)
                    go(fast::gu_kernel<16, 1, 2, nb::kIQ3_XXS, fast::TAIL | fast::DUFF | fast::DYT>, L, s);
                else if (f) go(fast::gu_kernel<32, 1, 4, nb::kIQ3_XXS, fast::TAIL | fast::DYT>, L, s);
                else go(gu_kernel<32, 1, 4, nb::kIQ3_XXS>, L, s);
                break;
            case nb::kIQ3_S:
                if (u < 4.0) go(gu_kernel<8, 2, 2, nb::kIQ3_S>, L, s);
                else if (u < 18.0)
                    go(fast::gu_kernel<16, 2, 2, nb::kIQ3_S, fast::TAIL | fast::DUFF | fast::DECF | fast::DYT>, L, s);
                else if (f) go(fast::gu_kernel<32, 2, 4, nb::kIQ3_S, fast::TAIL | fast::DUFF | fast::DECF | fast::DYT>, L, s);
                else go(gu_kernel<32, 2, 4, nb::kIQ3_S>, L, s);
                break;
            default:
                std::fprintf(stderr, "prefill ws: 4-bit gate/up type %d has no kernel\n", L.gu_type);
                std::exit(1);
        }
        ck(cudaGetLastError(), "gate/up 4-bit");
        return;
    }
    switch (L.gu_type) {
        case nb::kIQ2_S:
            if (u < 6.0) go(gu_kernel<8, 1, 2, nb::kIQ2_S>, L, s);
            else if (u < 25.0) go(gu_kernel<16, 0, 2, nb::kIQ2_S>, L, s);
            else if (f) go(fast::gu_kernel<32, 1, 4, nb::kIQ2_S, fast::DUFF>, L, s);
            else go(gu_kernel<32, 1, 4, nb::kIQ2_S>, L, s);
            break;
        case nb::kIQ3_XXS:
            if (u < 4.0) go(gu_kernel<8, 2, 2, nb::kIQ3_XXS>, L, s);
            else if (u < 18.0) go(gu_kernel<16, 1, 2, nb::kIQ3_XXS>, L, s);
            else if (f) go(fast::gu_kernel<32, 1, 4, nb::kIQ3_XXS, fast::TAIL | fast::DYT>, L, s);
            else go(gu_kernel<32, 1, 4, nb::kIQ3_XXS>, L, s);
            break;
        case nb::kIQ3_S:
            if (u < 4.0) go(gu_kernel<8, 2, 2, nb::kIQ3_S>, L, s);
            else if (u < 18.0) go(gu_kernel<16, 2, 2, nb::kIQ3_S>, L, s);
            else if (f) go(fast::gu_kernel<32, 2, 4, nb::kIQ3_S, fast::TAIL | fast::DUFF | fast::DECF | fast::DYT>, L, s);
            else go(gu_kernel<32, 2, 4, nb::kIQ3_S>, L, s);
            break;
        default:
            std::fprintf(stderr, "prefill ws: 4-bit gate/up type %d has no kernel\n", L.gu_type);
            std::exit(1);
    }
    ck(cudaGetLastError(), "gate/up 4-bit");
}

}  // namespace strata::prefill::ws::detail
#endif  // STRATA_EXPERT_NIBBLE_BUILT
