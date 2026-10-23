// src/prefill/moe_ws_v2.cuh - STRATA_WS_V2 (moe_ws.hpp): the prompt path's weight-stationary 32-token tier with TWO
// weight rows a thread at three CTAs an SM.  Included after common.cuh, mmq.cuh, moe_ws.cuh and
// strata/core/expert_nibble.hpp by moe_ws_gu_nib.cu / moe_ws_down.cu (the picked instances) and by moe_ws_prof_bench.cu
// (every prototype variant); everything here has internal linkage, so each including file has its own copies.
//
// Why: the WS kernels issue as many instructions as the SM can (WS-PROFILE.md: 99-101% of the issue slots), so their
// time follows the instruction count.  With one weight row a thread every 32 XMAD of a token's 32-value item come
// with 4 activation LDS.128, 3 FP32 and (two chains) 1 IADD - the shared activation words are read once per ROW.  Two
// rows a thread read them once per two rows (2 LDS.128 per 32 XMAD) and, with 8 independent chains in a 4-token group,
// run one chain per row and token (no IADD).  STRATA_WS_2ROW (pf-batch2) tried this with the codes staged in shared
// memory (29 KB, 178 registers: 2 CTAs = 8 warps an SM) and lost 1-10% on real routing: the staging and barrier phase
// of a ~20-token pass was not hidden by 8 warps.  Here the codes are NOT staged: each thread reads its own rows' 16-byte
// item codes from global memory one item ahead (an LDG.128 per row and item, no STS / LDS / shared memory for them), so
// shared memory is the step's activations only (8.5 KB) and the register cap of __launch_bounds__(128, 3) keeps three
// CTAs (12 warps) an SM, as the one-row kernels have.
//
// Bit for bit: every output keeps its own operations in their order - the items in K order, each item's exact
// 32-value integer sum (an integer chain: any number of chains and any order give the same int32, no overflow:
// |sum| < 2^22 on a 1.5 * 2^23 seed), the same float(sum) (kMagicI / kMagicF) and the same epilogue
// fma(dx * dy, float(sum), acc) or mac16's two-half form; tokens are independent.  Only which thread computes a row,
// where its codes come from and how many chains hold the integer sum change.
#pragma once

#include <cstdint>
#include <type_traits>

namespace strata::prefill::ws::detail {
namespace {
namespace v2 {

namespace nbv = strata::core::nibble;

// ---- variant bits (the bench's prototypes; the production picks are moe_ws_gu_nib.cu's / moe_ws_down.cu's)
enum : int {
    TAIL = 1,      // whole TG-token groups only, the last nt % TG tokens one by one (FAST's TAIL)
    DUFF = 2,      // a switch into the unrolled groups (FAST's DUFF)
    DECF = 4,      // IQ3_S's fused byte decode (FAST's DECF)
    DYT = 8,       // dy as [item][token] (FAST's DYT)
    R2 = 16,       // two weight rows a thread (256-row pieces)
    CD = 32,       // codes read per thread from global memory one item ahead (no shared-memory staging)
    LB3 = 64,      // __launch_bounds__(128, 3)
    LB4 = 128,     // __launch_bounds__(128, 4)
    LB2 = 256,     // __launch_bounds__(128, 2)
    // bench only (moe_ws_prof_bench.cu defines STRATA_WS_V2_PROF): per-CTA %globaltimer / per-warp clock64 phases, and
    // two ablations whose results are wrong (timing only)
    PTIME = 512,     // record into g_v2rec (wsprof::Rec's layout)
    PNODEC = 1024,   // the item's code words used as the weight words (no decode)
    PNOSTAGE = 2048, // the activations staged at the first step only
    QG = 4096,       // a whole TG = 4 group by K quarters: the 4 tokens' LDS.128 of a quarter together, then every row's
                     // and token's chain (8 independent chains with R2), instead of token after token
};
constexpr int minb(const int V) { return (V & LB4) ? 4 : (V & LB3) ? 3 : (V & LB2) ? 2 : 1; }

constexpr int GU_ROWS = 1280, GU_K = 2560, NB = GU_K / QK_K, NSTEP = 2 * NB, NT = kGuNT;
constexpr int ITEMS = (int) nbv::kItems;
constexpr int CS = 20;   // staged codes' shared-memory row stride in words (the one-row kernels')

#ifdef STRATA_WS_V2_PROF
struct alignas(16) Rec2 {
    unsigned long long t0, t1, cyc, cycL, cycC;
    unsigned smid, rows, passes, pad;
};
__device__ Rec2* g_v2rec = nullptr;
__device__ __forceinline__ unsigned long long v2_gtimer() {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}
#endif

__device__ __forceinline__ uint32_t prmt(const uint32_t a, const uint32_t b, const uint32_t sel) {
    uint32_t r;
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(sel));
    return r;
}

// moe_ws_gu_nib.cu's Dec / DecF, verbatim (the same int8 values and dx)
template <int FMT> struct Dec;
template <> struct Dec<nbv::kIQ2_S> {
    __device__ static __forceinline__ void wv(const uint32_t (&q)[4], int (&w)[16]) {
#pragma unroll
        for (int m = 0; m < 4; ++m) {
            split2((int) __byte_perm(PX, PY, q[m]), w[4 * m + 0], w[4 * m + 1]);
            split2((int) __byte_perm(PX, PY, q[m] >> 16), w[4 * m + 2], w[4 * m + 3]);
        }
    }
    __device__ static __forceinline__ void dx(const int ls, const int dbits, float& d0, float& d1) { scales16(ls, dbits, d0, d1); }
};
template <> struct Dec<nbv::kIQ3_XXS> {
    static constexpr uint32_t T0 = 0x1C140C04u, T1 = 0x3E342C24u;
    static constexpr uint32_t N0 = 0xE4ECF4FCu, N1 = 0xC2CCD4DCu;
    __device__ static __forceinline__ void wv(const uint32_t (&q)[4], int (&w)[16]) {
#pragma unroll
        for (int m = 0; m < 4; ++m) {
            const uint32_t c = q[m], ms = c & 0x77777777u, a = c << 4;
            const uint32_t s0 = prmt(a, c, 0xD9C8u), s1 = prmt(a, c, 0xFBEAu);
            const uint32_t g0 = (prmt(T0, T1, ms) & ~s0) | (prmt(N0, N1, ms) & s0);
            const uint32_t g1 = (prmt(T0, T1, ms >> 16) & ~s1) | (prmt(N0, N1, ms >> 16) & s1);
            split2((int) g0, w[4 * m + 0], w[4 * m + 1]);
            split2((int) g1, w[4 * m + 2], w[4 * m + 3]);
        }
    }
    __device__ static __forceinline__ float dx(const int ls, const int dbits) {
        const float d = __ushort_as_half((unsigned short) (dbits & 0xFFFF));
        return (ls*d + d/2)/2;
    }
};
template <> struct Dec<nbv::kIQ3_S> {
    __device__ static __forceinline__ void wv(const uint32_t (&q)[4], int (&w)[16]) {
#pragma unroll
        for (int m = 0; m < 4; ++m) {
            const uint32_t lo2 = (q[m] << 1) & 0x1E1E1E1Eu, hi2 = (q[m] >> 3) & 0x1E1E1E1Eu;
            split2((int) (((lo2 | 0x80808080u) - 0x0F0F0F0Fu) ^ 0x80808080u), w[4 * m + 0], w[4 * m + 1]);
            split2((int) (((hi2 | 0x80808080u) - 0x0F0F0F0Fu) ^ 0x80808080u), w[4 * m + 2], w[4 * m + 3]);
        }
    }
    __device__ static __forceinline__ float dx(const int ls, const int dbits) {
        const float d = __ushort_as_half((unsigned short) (dbits & 0xFFFF));
        return ls * d;
    }
};
template <int FMT> struct DecF : Dec<FMT> {};
template <> struct DecF<nbv::kIQ3_S> : Dec<nbv::kIQ3_S> {
    __device__ static __forceinline__ void wv(const uint32_t (&q)[4], int (&w)[16]) {
#pragma unroll
        for (int m = 0; m < 4; ++m) {
            const uint32_t lo2 = (q[m] << 1) & 0x1E1E1E1Eu, hi2 = (q[m] >> 3) & 0x1E1E1E1Eu;
            split2((int) ((lo2 + 0x71717171u) ^ 0x80808080u), w[4 * m + 0], w[4 * m + 1]);
            split2((int) ((hi2 + 0x71717171u) ^ 0x80808080u), w[4 * m + 2], w[4 * m + 3]);
        }
    }
};

// dy of token t, item s: [t][4] or [4][TPP] (DYT)
template <int TPP, bool T_> struct DyS {
    float a[T_ ? 4 : TPP][T_ ? TPP : 4];
    __device__ __forceinline__ float at(const int t, const int s) const {
        if constexpr (T_) return a[s][t];
        else return a[t][s];
    }
};

// FAST's ST16 staging (the same act_s / dy values as moe_ws.cuh stage_act)
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

// one token's IQ2_S product from its activation words in registers (mac16's body; CVT 1: two kMagicI chains)
template <int CVT>
__device__ __forceinline__ float mac16_v(const int (&wv)[16], const float dx0, const float dx1, const int4 a0, const int4 a1,
                                         const int4 a2, const int4 a3, const float dy, const float acc) {
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
// one token's IQ3_XXS / IQ3_S product: CH integer chains (the first seeded with kMagicI) - the same int32 sum for any CH
template <int CH>
__device__ __forceinline__ float mac32_v(const int (&wv)[16], const float dx, const int4 a0, const int4 a1, const int4 a2,
                                         const int4 a3, const float dy, const float acc) {
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

// item s against the pass's tokens: one(t, dy, slot) updates every row's sum of token t in accumulator `slot` (t for the
// whole groups, TPP + k for TAIL's k-th last token).  TAIL / DUFF as FAST's mac_tokens (8 groups: TPP / TG == 8)
struct NoGroup {};
template <int TPP, int TG, bool T_, bool TAIL_, bool DUFF_, typename One, typename Grp = NoGroup>
__device__ __forceinline__ void tokens(const One& one, const int s, const DyS<TPP, T_>& dy_s, const int nt,
                                       const Grp& group = Grp{}) {
    constexpr bool G_ = !std::is_same_v<Grp, NoGroup>;
    if constexpr (TAIL_) {
        const int nf = nt & ~(TG - 1);
#pragma unroll
        for (int k = 0; k < TG - 1; ++k)
            if (nf + k < nt) one(nf + k, dy_s.at(nf + k, s), TPP + k);
    }
    if constexpr (DUFF_) {
        static_assert(TPP / TG == 8, "the switch below has 8 cases");
        auto grp = [&](auto K) {
            constexpr int t0 = decltype(K)::value * TG;
            float dyg[TG];
            dy_group<TPP, T_, TG>(dy_s, t0, s, dyg);
            if constexpr (G_) {
                group(t0, dyg);
            } else {
#pragma unroll
                for (int t = t0; t < t0 + TG; ++t) one(t, dyg[t - t0], t);
            }
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
                if constexpr (G_) {
                    group(t0, dyg);
                } else {
#pragma unroll
                    for (int t = t0; t < t0 + TG; ++t) one(t, dyg[t - t0], t);
                }
            }
        }
    }
}

// The gate/up kernel on 4-bit slots: NR (1, or 2 with R2) weight rows a thread, rows r0 + r NT + tid of a NR NT-row
// piece; CH: IQ2_S mac16's CVT, IQ3_XXS / IQ3_S mac32's chain count.  Without CD the codes are staged as gu_kernel's
// (NR x 4 rounds of LDG.128 / STS.128, rows CS words apart); with CD each thread reads its rows' item codes itself.
template <int TPP, int CH, int TG, int FMT, int V>
__global__ void __launch_bounds__(NT, minb(V)) gu_kernel(const Chunk* __restrict__ chunks, const char* __restrict__ w,
                                                        const int64_t* __restrict__ xoff, const block_q8_1_mmq* __restrict__ xq,
                                                        const int total_rows, float* __restrict__ dst) {
    static_assert(TPP % TG == 0, "");
    constexpr bool TAIL_ = (V & TAIL) != 0, DUFF_ = (V & DUFF) != 0, DECF_ = (V & DECF) != 0, DYT_ = (V & DYT) != 0;
    constexpr bool CD_ = (V & CD) != 0;
    constexpr bool PT_ = (V & PTIME) != 0, PND_ = (V & PNODEC) != 0, PNS_ = (V & PNOSTAGE) != 0, QG_ = (V & QG) != 0;
    static_assert(!QG_ || (TG == 4 && CH == 1), "QG: 4-token groups, one chain (IQ3) / mac16's two kMagicI chains (IQ2_S)");
    constexpr int NR = (V & R2) ? 2 : 1, PIECES = GU_ROWS / (NR * NT);
    static_assert(GU_ROWS % (NR * NT) == 0, "");
    unsigned long long pk0 = 0, pkL = 0, pkC = 0, pka = 0, pkb = 0;
    (void) pk0; (void) pkL; (void) pkC; (void) pka; (void) pkb;
#ifdef STRATA_WS_V2_PROF
    if constexpr (PT_) {
        pk0 = clock64();
        if (threadIdx.x == 0) {
            unsigned sm;
            asm volatile("mov.u32 %0, %%smid;" : "=r"(sm));
            g_v2rec[blockIdx.x].t0 = v2_gtimer();
            g_v2rec[blockIdx.x].smid = sm;
        }
    }
#endif
    __shared__ __align__(16) int act_s[TPP][4][16];
    __shared__ __align__(16) DyS<TPP, DYT_> dy_s;
    __shared__ __align__(16) uint32_t csm[CD_ ? 4 : NR * NT * CS];
    int r0;
    const Chunk c = cta_work<PIECES, NR * NT>(chunks, r0);
    const char* __restrict__ slot = w + xoff[c.j];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const uint32_t* __restrict__ scw[NR];
    const unsigned short* __restrict__ dr[NR];
    const uint4* __restrict__ cr[NR];   // CD: the row's item codes
#pragma unroll
    for (int r = 0; r < NR; ++r) {
        const int row = r0 + r * NT + tid;
        scw[r] = (const uint32_t*) (slot + nbv::kScOff + (size_t) row * ITEMS);
        dr[r] = (const unsigned short*) (slot + nbv::kDOff) + (size_t) row * NB;
        cr[r] = (const uint4*) slot + (size_t) row * ITEMS;
    }
    const int srow = warp * 8 + (lane >> 2), sq = lane & 3;
    const uint4* __restrict__ cg = (const uint4*) slot + (size_t) (r0 + srow) * ITEMS + sq;
    uint32_t* __restrict__ cs_st = csm + srow * CS + 4 * sq;
    uint4 qn[NR];   // CD: the next item's codes
    if constexpr (CD_) {
#pragma unroll
        for (int r = 0; r < NR; ++r) qn[r] = __ldg(cr[r]);
    }
    for (int p0 = 0; p0 < c.rows; p0 += TPP) {
        const int nt = min(TPP, c.rows - p0);
        const int trow0 = c.row0 + p0;
        float acc[NR][TPP + TG];
#pragma unroll
        for (int r = 0; r < NR; ++r)
#pragma unroll
            for (int t = 0; t < TPP + TG; ++t) acc[r][t] = 0.0f;
#pragma unroll 1
        for (int hc = 0; hc < NSTEP; ++hc) {
            const int kc = hc >> 1;
            if constexpr (PT_) pka = clock64();
            __syncthreads();
            uint32_t scv[NR];
            int dv[NR];
            if constexpr (PNS_ && CD_) {
#pragma unroll
                for (int r = 0; r < NR; ++r) { scv[r] = __ldg(scw[r] + hc); dv[r] = (int) __ldg(dr[r] + kc); }
                if (hc == 0) stage_act16<TPP, DYT_>(xq, hc, total_rows, trow0, nt, act_s, dy_s);
            } else if constexpr (!CD_) {
                uint4 v[4 * NR];
#pragma unroll
                for (int i = 0; i < 4 * NR; ++i) v[i] = __ldg(cg + (size_t) i * 32 * ITEMS + 4 * hc);
#pragma unroll
                for (int r = 0; r < NR; ++r) { scv[r] = __ldg(scw[r] + hc); dv[r] = (int) __ldg(dr[r] + kc); }
                stage_act16<TPP, DYT_>(xq, hc, total_rows, trow0, nt, act_s, dy_s);
#pragma unroll
                for (int i = 0; i < 4 * NR; ++i) *(uint4*) (cs_st + i * 32 * CS) = v[i];
            } else {
#pragma unroll
                for (int r = 0; r < NR; ++r) { scv[r] = __ldg(scw[r] + hc); dv[r] = (int) __ldg(dr[r] + kc); }
                stage_act16<TPP, DYT_>(xq, hc, total_rows, trow0, nt, act_s, dy_s);
            }
            __syncthreads();
            if constexpr (PT_) { pkb = clock64(); pkL += pkb - pka; }
#pragma unroll 1
            for (int s = 0; s < 4; ++s) {
                uint4 u[NR];
                if constexpr (CD_) {
                    // this item's codes (loaded one item ahead) and the next item's: item 4 hc + s + 1, or item 0 (the
                    // next pass's first) after the last
                    const int it = 4 * hc + s + 1;
#pragma unroll
                    for (int r = 0; r < NR; ++r) {
                        u[r] = qn[r];
                        qn[r] = __ldg(cr[r] + (it < ITEMS ? it : 0));
                    }
                } else {
#pragma unroll
                    for (int r = 0; r < NR; ++r) u[r] = *(const uint4*) (csm + (r * NT + tid) * CS + 4 * s);
                }
                int wv[NR][16];
#pragma unroll
                for (int r = 0; r < NR; ++r) {
                    const uint32_t q[4] = {u[r].x, u[r].y, u[r].z, u[r].w};
                    if constexpr (PND_) {
#pragma unroll
                        for (int m = 0; m < 16; ++m) wv[r][m] = (int) (q[m & 3] >> (m >> 2));
                    } else if constexpr (DECF_) DecF<FMT>::wv(q, wv[r]);
                    else Dec<FMT>::wv(q, wv[r]);
                }
                if constexpr (FMT == nbv::kIQ2_S) {
                    float d0[NR], d1[NR];
#pragma unroll
                    for (int r = 0; r < NR; ++r) Dec<FMT>::dx((int) ((scv[r] >> (8 * s)) & 0xFF), dv[r], d0[r], d1[r]);
                    const auto one = [&](const int t, const float dy, const int sl) {
                        const int4* a4 = (const int4*) act_s[t][s];
                        const int4 a0 = a4[0], a1 = a4[1], a2 = a4[2], a3 = a4[3];
#pragma unroll
                        for (int r = 0; r < NR; ++r) acc[r][sl] = mac16_v<CH>(wv[r], d0[r], d1[r], a0, a1, a2, a3, dy, acc[r][sl]);
                    };
                    if constexpr (QG_) {
                        // mac16_v<1>'s two kMagicI chains (values 0-15: words 0-7, quarters 0-1; values 16-31: words 8-15,
                        // quarters 2-3), each in its word order, then its epilogue - per row and token
                        const auto group = [&](const int t0, const float (&dyg)[4]) {
                            int ca[NR][4], cb[NR][4];
#pragma unroll
                            for (int r = 0; r < NR; ++r)
#pragma unroll
                                for (int k = 0; k < 4; ++k) { ca[r][k] = kMagicI; cb[r][k] = kMagicI; }
#pragma unroll
                            for (int q = 0; q < 4; ++q) {
                                int4 a[4];
#pragma unroll
                                for (int k = 0; k < 4; ++k) a[k] = ((const int4*) act_s[t0 + k][s])[q];
#pragma unroll
                                for (int k = 0; k < 4; ++k)
#pragma unroll
                                    for (int r = 0; r < NR; ++r) {
                                        int& c = q < 2 ? ca[r][k] : cb[r][k];
                                        c = mad2(wv[r][4 * q + 0], a[k].x, c);
                                        c = mad2(wv[r][4 * q + 1], a[k].y, c);
                                        c = mad2(wv[r][4 * q + 2], a[k].z, c);
                                        c = mad2(wv[r][4 * q + 3], a[k].w, c);
                                    }
                            }
#pragma unroll
                            for (int k = 0; k < 4; ++k)
#pragma unroll
                                for (int r = 0; r < NR; ++r) {
                                    const float fa = __fsub_rn(__int_as_float(ca[r][k]), kMagicF);
                                    const float fb = __fsub_rn(__int_as_float(cb[r][k]), kMagicF);
                                    float tf = __fmaf_rn(fa, d0[r], 0.0f);
                                    tf = __fmaf_rn(fb, d1[r], tf);
                                    acc[r][t0 + k] = __fmaf_rn(tf, dyg[k], acc[r][t0 + k]);
                                }
                        };
                        tokens<TPP, TG, DYT_, TAIL_, DUFF_>(one, s, dy_s, nt, group);
                    } else {
                        tokens<TPP, TG, DYT_, TAIL_, DUFF_>(one, s, dy_s, nt);
                    }
                } else {
                    float dx[NR];
#pragma unroll
                    for (int r = 0; r < NR; ++r) dx[r] = Dec<FMT>::dx((int) ((scv[r] >> (8 * s)) & 0xFF), dv[r]);
                    const auto one = [&](const int t, const float dy, const int sl) {
                        const int4* a4 = (const int4*) act_s[t][s];
                        const int4 a0 = a4[0], a1 = a4[1], a2 = a4[2], a3 = a4[3];
#pragma unroll
                        for (int r = 0; r < NR; ++r) acc[r][sl] = mac32_v<CH>(wv[r], dx[r], a0, a1, a2, a3, dy, acc[r][sl]);
                    };
                    if constexpr (QG_) {
                        // mac32_v<1>'s one kMagicI chain per row and token, words 0-15 in order (4 a quarter), then its
                        // epilogue
                        const auto group = [&](const int t0, const float (&dyg)[4]) {
                            int c[NR][4];
#pragma unroll
                            for (int r = 0; r < NR; ++r)
#pragma unroll
                                for (int k = 0; k < 4; ++k) c[r][k] = kMagicI;
#pragma unroll
                            for (int q = 0; q < 4; ++q) {
                                int4 a[4];
#pragma unroll
                                for (int k = 0; k < 4; ++k) a[k] = ((const int4*) act_s[t0 + k][s])[q];
#pragma unroll
                                for (int k = 0; k < 4; ++k)
#pragma unroll
                                    for (int r = 0; r < NR; ++r) {
                                        c[r][k] = mad2(wv[r][4 * q + 0], a[k].x, c[r][k]);
                                        c[r][k] = mad2(wv[r][4 * q + 1], a[k].y, c[r][k]);
                                        c[r][k] = mad2(wv[r][4 * q + 2], a[k].z, c[r][k]);
                                        c[r][k] = mad2(wv[r][4 * q + 3], a[k].w, c[r][k]);
                                    }
                            }
#pragma unroll
                            for (int k = 0; k < 4; ++k)
#pragma unroll
                                for (int r = 0; r < NR; ++r)
                                    acc[r][t0 + k] = __fmaf_rn(__fmul_rn(dx[r], dyg[k]),
                                                               __fsub_rn(__int_as_float(c[r][k]), kMagicF), acc[r][t0 + k]);
                        };
                        tokens<TPP, TG, DYT_, TAIL_, DUFF_>(one, s, dy_s, nt, group);
                    } else {
                        tokens<TPP, TG, DYT_, TAIL_, DUFF_>(one, s, dy_s, nt);
                    }
                }
            }
            if constexpr (PT_) pkC += clock64() - pkb;
        }
#pragma unroll
        for (int r = 0; r < NR; ++r) {
            float* __restrict__ d = dst + r0 + r * NT + tid;
            if constexpr (TAIL_) {
                const int nf = nt & ~(TG - 1);
#pragma unroll
                for (int t = 0; t < TPP; ++t)
                    if (t < nf) d[(size_t) (trow0 + t) * GU_ROWS] = acc[r][t];
#pragma unroll
                for (int k = 0; k < TG - 1; ++k)
                    if (nf + k < nt) d[(size_t) (trow0 + nf + k) * GU_ROWS] = acc[r][TPP + k];
            } else {
#pragma unroll
                for (int t = 0; t < TPP; ++t)
                    if (t < nt) d[(size_t) (trow0 + t) * GU_ROWS] = acc[r][t];
            }
        }
    }
#ifdef STRATA_WS_V2_PROF
    if constexpr (PT_) {
        const unsigned long long tot = clock64() - pk0;
        if ((threadIdx.x & 31) == 0) {
            atomicAdd(&g_v2rec[blockIdx.x].cyc, tot);
            atomicAdd(&g_v2rec[blockIdx.x].cycL, pkL);
            atomicAdd(&g_v2rec[blockIdx.x].cycC, pkC);
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            g_v2rec[blockIdx.x].t1 = v2_gtimer();
            g_v2rec[blockIdx.x].rows = (unsigned) c.rows;
            g_v2rec[blockIdx.x].passes = (unsigned) ((c.rows + TPP - 1) / TPP);
        }
    }
#endif
}
template <int V> constexpr int gu_pieces() { return GU_ROWS / (((V & R2) ? 2 : 1) * NT); }

// ---- IQ4_NL down [2560 x 640], the 32-token tier: moe_ws_down.cu nl4::dn_kernel_fast's per-token operations (one XMAD
// chain per row, token and block, I2F, FMUL dx*dy, FFMA; blocks in K order) with NR (1, or 2 with R2) rows a thread of
// 128 threads.  Without CD the step's weights are staged as dn_kernel's (2-block steps, rows 9 words apart, lanes 0-26
// of a warp a row each 3); with CD each thread reads its rows' 36-byte steps from global memory itself (9 LDG.32 a row,
// 4-byte aligned: rows are 360 B, the matrix 8-byte aligned).  No register prefetch of the next step (dn_kernel's
// pre / apre): the other CTAs of the SM cover the loads.
namespace dn {
constexpr int DN_ROWS = 2560, DN_K = 640, NBK = DN_K / QK4_NL, BB = 18, ROWB = NBK * BB, SB = 2, NSTEPD = NBK / SB;
constexpr int WPS = 9;   // words a row and step (2 blocks of 18 B)
__device__ __forceinline__ void split01(const int v, int& p01, int& p23) {
    asm("prmt.b32 %0, %1, 0, 0x9180;" : "=r"(p01) : "r"(v));
    asm("prmt.b32 %0, %1, 0, 0xB3A2;" : "=r"(p23) : "r"(v));
}
__device__ __forceinline__ void load_tab(uint32_t (&tab)[4]) {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        uint32_t v = 0;
#pragma unroll
        for (int b = 0; b < 4; ++b) v |= (uint32_t) (uint8_t) kvalues_iq4nl[4 * i + b] << (8 * b);
        tab[i] = v;
    }
}
// moe_ws_down.cu nl4::dec, verbatim
__device__ __forceinline__ void dec(const int (&q)[4], const uint32_t (&tab)[4], int (&wv)[16]) {
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        const uint32_t q4 = (uint32_t) q[k];
        uint32_t tmp[2];
        const uint32_t low_high_selection_indices = (0x32103210 | ((q4 & 0x88888888) >> 1));
#pragma unroll
        for (uint32_t i = 0; i < 2; ++i) {
            const uint32_t shift = 16 * i;
            const uint32_t low = __byte_perm(tab[0], tab[1], q4 >> shift);
            const uint32_t high = __byte_perm(tab[2], tab[3], q4 >> shift);
            tmp[i] = __byte_perm(low, high, low_high_selection_indices >> shift);
        }
        split2((int) tmp[0], wv[2 * k], wv[2 * (k + 4)]);
        split2((int) tmp[1], wv[2 * k + 1], wv[2 * (k + 4) + 1]);
    }
}
// block PAR of the step's 9 words c (moe_ws_down.cu load_blk with p = 0)
template <int PAR>
__device__ __forceinline__ float load_blk(const int (&c)[WPS], int (&q)[4]) {
    if constexpr (PAR == 0) {
#pragma unroll
        for (int k = 0; k < 4; ++k) q[k] = (int) __funnelshift_r((uint32_t) c[k], (uint32_t) c[k + 1], 16);
        return __half2float(__ushort_as_half((unsigned short) ((uint32_t) c[0] & 0xFFFFu)));
    } else {
#pragma unroll
        for (int k = 0; k < 4; ++k) q[k] = c[5 + k];
        return __half2float(__ushort_as_half((unsigned short) ((uint32_t) c[4] >> 16)));
    }
}

template <int TPP, int V>
__global__ void __launch_bounds__(NT, minb(V)) dn_kernel(const Chunk* __restrict__ chunks, const char* __restrict__ w,
                                                        const int64_t* __restrict__ xoff, const int64_t xadd,
                                                        const block_q8_1_mmq* __restrict__ xq, const int total_rows,
                                                        float* __restrict__ dst) {
    constexpr bool CD_ = (V & CD) != 0;
    constexpr int NR = (V & R2) ? 2 : 1, PIECES = DN_ROWS / (NR * NT), G = 2, NP = TPP / G;
    static_assert(NP == 16 || NP == 8, "the switch below: 8 or 16 token pairs");
    constexpr int S = WPS;   // staged row stride (9: odd, conflict-free per-thread reads)
    constexpr int AW = 9 * SB, AI = (TPP * AW + NT - 1) / NT;
    __shared__ int wsm[CD_ ? 1 : NR * NT * S];
    __shared__ __align__(16) int act_s[TPP][SB][16];
    __shared__ __align__(8) float dy_s[SB * TPP];
    uint32_t tab[4];
    load_tab(tab);
    int r0;
    const Chunk c = cta_work<PIECES, NR * NT>(chunks, r0);
    const char* __restrict__ base = w + xoff[c.j] + xadd;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    // staged: dn_kernel's pattern for NR NT rows with 4 warps (12 rows a round)
    constexpr int NW = NT / 32, KI = (NR * NT + 3 * NW - 1) / (3 * NW);
    const int sub = lane / 9, pc = lane - 9 * sub;
    const int srow0 = 3 * warp + sub;
    const bool sact = sub < 3;
    const char* __restrict__ Wg = base + (size_t) (r0 + srow0) * ROWB + pc * 4;
    int* __restrict__ ws_st = wsm + srow0 * S + pc;
    const int* __restrict__ Wr[NR];   // CD: the thread's rows
#pragma unroll
    for (int r = 0; r < NR; ++r) Wr[r] = (const int*) (base + (size_t) (r0 + r * NT + threadIdx.x) * ROWB);
    for (int p0 = 0; p0 < c.rows; p0 += TPP) {
        const int nt = min(TPP, c.rows - p0);
        const int trow0 = c.row0 + p0;
        float acc[NR][TPP];
#pragma unroll
        for (int r = 0; r < NR; ++r)
#pragma unroll
            for (int t = 0; t < TPP; ++t) acc[r][t] = 0.0f;
#pragma unroll 1
        for (int st = 0; st < NSTEPD; ++st) {
            __syncthreads();
            int cw[NR][WPS];
            if constexpr (CD_) {
#pragma unroll
                for (int r = 0; r < NR; ++r)
#pragma unroll
                    for (int k = 0; k < WPS; ++k) cw[r][k] = __ldg(Wr[r] + st * WPS + k);
            } else {
                int pre[KI];
                if (sact) {
                    const char* g = Wg + st * (SB * BB);
#pragma unroll
                    for (int k = 0; k < KI; ++k)
                        if (k < KI - 1 || srow0 + 3 * NW * k < NR * NT) pre[k] = __ldg((const int*) (g + (size_t) k * 3 * NW * ROWB));
#pragma unroll
                    for (int k = 0; k < KI; ++k)
                        if (k < KI - 1 || srow0 + 3 * NW * k < NR * NT) ws_st[k * 3 * NW * S] = pre[k];
                }
            }
            {   // the step's activations: dn_kernel_fast's load_a / store_a, dy as [SB][TPP]
                const int kb = (st * SB) >> 2, b0 = (st * SB) & 3;
                int apre[AI];
#pragma unroll
                for (int i = 0; i < AI; ++i) {
                    const int idx = threadIdx.x + i * NT;
                    if (idx < nt * AW) {
                        const int t = idx / AW, wd = idx - t * AW;
                        const int* blk = (const int*) (xq + (size_t) kb * total_rows + trow0 + t);
                        apre[i] = __ldg(blk + (wd < SB ? b0 + wd : 4 + 8 * b0 + (wd - SB)));
                    }
                }
#pragma unroll
                for (int i = 0; i < AI; ++i) {
                    const int idx = threadIdx.x + i * NT;
                    if (idx < nt * AW) {
                        const int t = idx / AW, wd = idx - t * AW;
                        if (wd < SB) {
                            dy_s[wd * TPP + t] = __int_as_float(apre[i]);
                        } else {
                            const int m = wd - SB, bb = m >> 3, mm = m & 7;
                            int lo, hi;
                            split01(apre[i], lo, hi);
                            *(int2*) &act_s[t][bb][2 * mm] = make_int2(lo, hi);
                        }
                    }
                }
            }
            __syncthreads();
            if constexpr (!CD_) {
#pragma unroll
                for (int r = 0; r < NR; ++r)
#pragma unroll
                    for (int k = 0; k < WPS; ++k) cw[r][k] = wsm[(r * NT + threadIdx.x) * S + k];
            }
            auto blk = [&](auto PARc) {
                constexpr int PAR = decltype(PARc)::value;
                int wv[NR][16];
                float dx[NR];
#pragma unroll
                for (int r = 0; r < NR; ++r) {
                    int q[4];
                    dx[r] = load_blk<PAR>(cw[r], q);
                    dec(q, tab, wv[r]);
                }
                auto grp = [&](auto K) {
                    constexpr int t0 = decltype(K)::value * G;
                    const float2 d2 = *(const float2*) &dy_s[PAR * TPP + t0];
#pragma unroll
                    for (int t = t0; t < t0 + G; ++t) {
                        const int4* a4 = (const int4*) act_s[t][PAR];
                        const int4 a0 = a4[0], a1 = a4[1], a2 = a4[2], a3 = a4[3];
                        const float dy = t == t0 ? d2.x : d2.y;
#pragma unroll
                        for (int r = 0; r < NR; ++r) {
                            int s0 = 0;
                            s0 = mad2(wv[r][0], a0.x, s0); s0 = mad2(wv[r][1], a0.y, s0); s0 = mad2(wv[r][2], a0.z, s0); s0 = mad2(wv[r][3], a0.w, s0);
                            s0 = mad2(wv[r][4], a1.x, s0); s0 = mad2(wv[r][5], a1.y, s0); s0 = mad2(wv[r][6], a1.z, s0); s0 = mad2(wv[r][7], a1.w, s0);
                            s0 = mad2(wv[r][8], a2.x, s0); s0 = mad2(wv[r][9], a2.y, s0); s0 = mad2(wv[r][10], a2.z, s0); s0 = mad2(wv[r][11], a2.w, s0);
                            s0 = mad2(wv[r][12], a3.x, s0); s0 = mad2(wv[r][13], a3.y, s0); s0 = mad2(wv[r][14], a3.z, s0); s0 = mad2(wv[r][15], a3.w, s0);
                            acc[r][t] = __fmaf_rn(__fmul_rn(dx[r], dy), __int2float_rn(s0), acc[r][t]);
                        }
                    }
                };
                using std::integral_constant;
                switch ((nt + G - 1) / G) {
                    case 16: if constexpr (NP >= 16) grp(integral_constant<int, 15>{}); [[fallthrough]];
                    case 15: if constexpr (NP >= 15) grp(integral_constant<int, 14>{}); [[fallthrough]];
                    case 14: if constexpr (NP >= 14) grp(integral_constant<int, 13>{}); [[fallthrough]];
                    case 13: if constexpr (NP >= 13) grp(integral_constant<int, 12>{}); [[fallthrough]];
                    case 12: if constexpr (NP >= 12) grp(integral_constant<int, 11>{}); [[fallthrough]];
                    case 11: if constexpr (NP >= 11) grp(integral_constant<int, 10>{}); [[fallthrough]];
                    case 10: if constexpr (NP >= 10) grp(integral_constant<int, 9>{}); [[fallthrough]];
                    case 9: if constexpr (NP >= 9) grp(integral_constant<int, 8>{}); [[fallthrough]];
                    case 8: if constexpr (NP >= 8) grp(integral_constant<int, 7>{}); [[fallthrough]];
                    case 7: if constexpr (NP >= 7) grp(integral_constant<int, 6>{}); [[fallthrough]];
                    case 6: if constexpr (NP >= 6) grp(integral_constant<int, 5>{}); [[fallthrough]];
                    case 5: if constexpr (NP >= 5) grp(integral_constant<int, 4>{}); [[fallthrough]];
                    case 4: if constexpr (NP >= 4) grp(integral_constant<int, 3>{}); [[fallthrough]];
                    case 3: if constexpr (NP >= 3) grp(integral_constant<int, 2>{}); [[fallthrough]];
                    case 2: if constexpr (NP >= 2) grp(integral_constant<int, 1>{}); [[fallthrough]];
                    case 1: if constexpr (NP >= 1) grp(integral_constant<int, 0>{}); [[fallthrough]];
                    default: break;
                }
            };
            blk(std::integral_constant<int, 0>{});
            blk(std::integral_constant<int, 1>{});
        }
#pragma unroll
        for (int r = 0; r < NR; ++r)
#pragma unroll
            for (int t = 0; t < TPP; ++t)
                if (t < nt) dst[(size_t) (trow0 + t) * DN_ROWS + r0 + r * NT + threadIdx.x] = acc[r][t];
    }
}
template <int V> constexpr int dn_pieces() { return DN_ROWS / (((V & R2) ? 2 : 1) * NT); }
}  // namespace dn

}  // namespace v2
}  // namespace
}  // namespace strata::prefill::ws::detail
