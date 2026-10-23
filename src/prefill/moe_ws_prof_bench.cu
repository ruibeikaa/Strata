// src/prefill/moe_ws_prof_bench.cu - bench only (2026-10-23, branch ws-prof; on pf-batch1 with the production
// STRATA_WS_FAST arms): where the prompt path's weight-stationary expert kernels (moe_ws.hpp) lose time on one GP100 die,
// at production shapes, and bit-exact candidate fixes.  Needs STRATA_MMQ_WS=1 (ws::mode() is opt-in on this branch).
//
//     moe_ws_prof_bench <shard.gguf> [<shard.gguf> ...] [--layers 2,0,17,21] [--t 1536] [--models ln0.5,ln1.0,flat32]
//                       [--reps 15] [--arms all|prod] [--csv dir]
//
// Per layer (all 512 experts of the GGUF read into the pack's blob, the 4-bit slots made by the engine's own transcoder,
// as moe_ws_nib_parity.cu does) and per count model (rows = 10 T; lognormal counts over 512 experts with sigma 0.5 / 1.0,
// each at most T - the ws2 model of moe_ws_nib_parity; flat32: rows / 32 experts of exactly 32 rows, the no-tail
// control), on the same q8_1 rows:
//   production  ws::gate_up on the 4-bit slots (STRATA_EXPERT_NIBBLE=1's path) and on the GGUF blobs, one
//               mmq::swiglu_quant, ws::down - each timed alone (cudaEvent, median of --reps, arms interleaved); the
//               references are the default kernels (ws::set_fast(0)); "FAST" arms: the same calls with ws::set_fast(1)
//               (STRATA_WS_FAST=1's kernels, the 32-token tier), checked by memcmp against the references
//   copies      the production gate/up 4-bit kernel and IQ4_NL down kernel copied here with variant bits (V): V = 0
//               must equal production bit for bit and in time; V_TIME records per CTA (SM, %globaltimer start/end,
//               per-warp clock64 in the load/stage phase and the compute phase); the ablations (V_NOLOAD, V_NOSYNC,
//               V_NODEC: wrong results, timing only) bound each phase; V_DYT (dy transposed: one LDS.128 per 4 tokens)
//               and V_PF (next step's codes / scales / d / act words prefetched into registers before this step's
//               compute) are the bit-exact candidates, checked by memcmp against production.
// The chunk table is ws::chunks' (production); the host also counts passes and token slots computed vs used.
// --arms v2 / v2r2 (branch ws2): instead of the copies, moe_ws_v2.cuh's variants (STRATA_WS_V2's kernels and the
// prototypes: two rows a thread, per-thread code loads, launch bounds, phase records and ablations), each memcmp'd
// against production; "PROD ... V2" arms run the production entry points with ws::set_v2(1).
#include "common.cuh"
#include "mmq.cuh"

#include "moe_ws.cuh"
#include "strata/artifact/gguf_reader.hpp"
#include "strata/core/expert_nibble.hpp"
#include "strata/kernels/cpu/native_expert.hpp"
#include "strata/prefill/moe_mmq.hpp"
#include "strata/prefill/moe_ws.hpp"
#define STRATA_WS_V2_PROF 1   // moe_ws_v2.cuh's PTIME / PNODEC / PNOSTAGE bits
#include "moe_ws_v2.cuh"   // STRATA_WS_V2's kernels and the prototypes (--arms v2)

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <map>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <type_traits>
#include <vector>

#define CK(x)                                                                                                       \
    do {                                                                                                            \
        cudaError_t e_ = (x);                                                                                       \
        if (e_ != cudaSuccess) {                                                                                    \
            std::fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_));                 \
            std::exit(2);                                                                                           \
        }                                                                                                           \
    } while (0)

namespace wsprof {

using strata::prefill::ws::Chunk;
using namespace strata::prefill::ws::detail;
namespace nb = strata::core::nibble;

enum : int { V_TIME = 1, V_NOLOAD = 2, V_NOSYNC = 4, V_NODEC = 8, V_DYT = 16, V_PF = 32, V_ST16 = 64, V_TAIL = 128,
             V_DUFF = 256, V_DECF = 512, V_LB2 = 1024 };

struct alignas(16) Rec {
    unsigned long long t0, t1;     // %globaltimer at entry / exit (thread 0)
    unsigned long long cyc;        // clock64 entry -> exit, summed over the CTA's warps
    unsigned long long cycL;       // clock64 in the load / stage phase (barrier to barrier), summed over warps
    unsigned long long cycC;       // clock64 in the compute phase, summed over warps
    unsigned smid, rows, passes, pad;
};

__device__ __forceinline__ unsigned long long gtimer() {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}
__device__ __forceinline__ unsigned smid_reg() {
    unsigned s;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(s));
    return s;
}
__device__ __forceinline__ void rec_warp(Rec* rec, unsigned long long cyc, unsigned long long cl, unsigned long long cc) {
    if ((threadIdx.x & 31) == 0) {
        atomicAdd(&rec[blockIdx.x].cyc, cyc);
        atomicAdd(&rec[blockIdx.x].cycL, cl);
        atomicAdd(&rec[blockIdx.x].cycC, cc);
    }
}

// ================================ gate/up on 4-bit slots: moe_ws_gu_nib.cu's gu_kernel, copied ======================
namespace gu {
constexpr int GU_ROWS = 1280, GU_K = 2560, NB = GU_K / QK_K, NSTEP = 2 * NB, NT = kGuNT, PIECES = GU_ROWS / NT;
constexpr int ITEMS = (int) nb::kItems;
constexpr int CS = 20;

__device__ __forceinline__ uint32_t prmt(const uint32_t a, const uint32_t b, const uint32_t sel) {
    uint32_t r;
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(sel));
    return r;
}
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
template <int FMT> struct DecF { template <typename... A> __device__ static __forceinline__ void wv(A&&... a) { Dec<FMT>::wv(a...); } };
template <> struct DecF<nb::kIQ3_S> {
    __device__ static __forceinline__ void wv(const uint32_t (&q)[4], int (&w)[16]) {
#pragma unroll
        for (int m = 0; m < 4; ++m) {
            const uint32_t lo2 = (q[m] << 1) & 0x1E1E1E1Eu, hi2 = (q[m] >> 3) & 0x1E1E1E1Eu;
            split2((int) ((lo2 + 0x71717171u) ^ 0x80808080u), w[4 * m + 0], w[4 * m + 1]);
            split2((int) ((hi2 + 0x71717171u) ^ 0x80808080u), w[4 * m + 2], w[4 * m + 3]);
        }
    }
};
template <> struct Dec<nb::kIQ3_S> {
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
// V_NODEC: the code words as weights (timing only)
__device__ __forceinline__ void wv_raw(const uint32_t (&q)[4], int (&w)[16]) {
#pragma unroll
    for (int m = 0; m < 4; ++m) {
        split2((int) q[m], w[4 * m + 0], w[4 * m + 1]);
        split2((int) (q[m] >> 4), w[4 * m + 2], w[4 * m + 3]);
    }
}

// dy of token t, item s: [t][4] (production) or [4][TPP] (V_DYT)
template <int TPP, bool DYT>
struct DyS {
    float a[DYT ? 4 : TPP][DYT ? TPP : 4];
    __device__ __forceinline__ float& at(int t, int s) { if constexpr (DYT) return a[s][t]; else return a[t][s]; }
    __device__ __forceinline__ float at(int t, int s) const { if constexpr (DYT) return a[s][t]; else return a[t][s]; }
};

// moe_ws.cuh stage_act, split into the loads (to registers) and the stores, dy placed by DyS
template <int TPP> struct ActPre { int av[(TPP * 36 + kGuNT - 1) / kGuNT]; };
template <int TPP>
__device__ __forceinline__ void act_load(const block_q8_1_mmq* __restrict__ xq, const int hc, const int total_rows,
                                         const int trow0, const int nt, ActPre<TPP>& P) {
    constexpr int NA = (TPP * 36 + kGuNT - 1) / kGuNT;
    const int* xr = (const int*) (xq + (size_t) hc * total_rows + trow0);
#pragma unroll
    for (int i = 0; i < NA; ++i) {
        const int idx = threadIdx.x + i * kGuNT;
        if (idx < nt * 36) P.av[i] = __ldg(xr + idx);
    }
}
template <int TPP, bool DYT>
__device__ __forceinline__ void act_store(const ActPre<TPP>& P, const int nt, int (*act_s)[4][16], DyS<TPP, DYT>& dy_s) {
    constexpr int NA = (TPP * 36 + kGuNT - 1) / kGuNT;
#pragma unroll
    for (int i = 0; i < NA; ++i) {
        const int idx = threadIdx.x + i * kGuNT;
        if (idx < nt * 36) {
            const int t = idx / 36, wd = idx - t * 36;
            if (wd < 4) {
                dy_s.at(t, wd) = __int_as_float(P.av[i]);
            } else {
                const int m = wd - 4;
                int lo, hi;
                split2(P.av[i], lo, hi);
                *(int2*) &act_s[t][m / 8][2 * (m % 8)] = make_int2(lo, hi);
            }
        }
    }
}

// V_ST16: the step's activations as 16-byte pieces: piece 0 of a token = its 4 d (one per 32-value item), pieces 1-8 =
// int8 values 16 (p - 1) .. 16 p - 1, each split to 8 int16-pair words (two STS.128) - the same words stage_act writes
template <int TPP> struct ActPre16 { int4 av[(TPP * 9 + kGuNT - 1) / kGuNT]; };
template <int TPP>
__device__ __forceinline__ void act_load16(const block_q8_1_mmq* __restrict__ xq, const int hc, const int total_rows,
                                           const int trow0, const int nt, ActPre16<TPP>& P) {
    constexpr int NA = (TPP * 9 + kGuNT - 1) / kGuNT;
    const int4* xr = (const int4*) (xq + (size_t) hc * total_rows + trow0);
#pragma unroll
    for (int i = 0; i < NA; ++i) {
        const int idx = threadIdx.x + i * kGuNT;
        if (idx < nt * 9) P.av[i] = __ldg(xr + idx);
    }
}
template <int TPP, bool DYT>
__device__ __forceinline__ void act_store16(const ActPre16<TPP>& P, const int nt, int (*act_s)[4][16], DyS<TPP, DYT>& dy_s) {
    constexpr int NA = (TPP * 9 + kGuNT - 1) / kGuNT;
#pragma unroll
    for (int i = 0; i < NA; ++i) {
        const int idx = threadIdx.x + i * kGuNT;
        if (idx < nt * 9) {
            const int t = idx / 9, pc = idx - t * 9;
            const int4 v = P.av[i];
            if (pc == 0) {
                if constexpr (DYT) {
                    dy_s.a[0][t] = __int_as_float(v.x); dy_s.a[1][t] = __int_as_float(v.y);
                    dy_s.a[2][t] = __int_as_float(v.z); dy_s.a[3][t] = __int_as_float(v.w);
                } else {
                    *(int4*) &dy_s.a[t][0] = v;
                }
            } else {
                const int m0 = 4 * (pc - 1);   // words m0 .. m0 + 3 of the 32: item m0 / 8, pair slots 2 (m0 % 8) ..
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
template <int TPP, bool DYT, int TG>
__device__ __forceinline__ void dy_group(const DyS<TPP, DYT>& dy_s, const int t0, const int s, float (&d)[TG]) {
    if constexpr (DYT && TG == 4) {
        const float4 v = *(const float4*) &dy_s.a[s][t0];
        d[0] = v.x; d[1] = v.y; d[2] = v.z; d[3] = v.w;
    } else if constexpr (DYT && TG == 2) {
        const float2 v = *(const float2*) &dy_s.a[s][t0];
        d[0] = v.x; d[1] = v.y;
    } else {
#pragma unroll
        for (int k = 0; k < TG; ++k) d[k] = dy_s.at(t0 + k, s);
    }
}

// moe_ws.cuh mac16 with dy from dy_group
// one token's IQ2_S product (mac16's body), CVT as mac16
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
// one token's IQ3 product (mac32's body)
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

template <int TPP, int CVT, int TG, bool DYT, bool TAIL = false, bool DUFF = false>
__device__ __forceinline__ void mac16v(const int (&wv)[16], const float dx0, const float dx1, const int s,
                                       const int (*act_s)[4][16], const DyS<TPP, DYT>& dy_s, const int nt, float (&acc)[TPP],
                                       float (&acct)[TG]) {
    if constexpr (TAIL) {
        const int nf = nt & ~(TG - 1);
#pragma unroll
        for (int k = 0; k < TG - 1; ++k)
            if (nf + k < nt) acct[k] = mac16_one<CVT>(wv, dx0, dx1, (const int4*) act_s[nf + k][s], dy_s.at(nf + k, s), acct[k]);
    }
    if constexpr (DUFF) {
        static_assert(TPP / TG == 8, "the switch below has 8 cases");
        auto grp = [&](auto K) {
            constexpr int t0 = decltype(K)::value * TG;
            float dyg[TG];
            dy_group<TPP, DYT, TG>(dy_s, t0, s, dyg);
#pragma unroll
            for (int t = t0; t < t0 + TG; ++t)
                acc[t] = mac16_one<CVT>(wv, dx0, dx1, (const int4*) act_s[t][s], dyg[t - t0], acc[t]);
        };
        switch (TAIL ? nt / TG : (nt + TG - 1) / TG) {
            case 8: grp(std::integral_constant<int, 7>{}); [[fallthrough]];
            case 7: grp(std::integral_constant<int, 6>{}); [[fallthrough]];
            case 6: grp(std::integral_constant<int, 5>{}); [[fallthrough]];
            case 5: grp(std::integral_constant<int, 4>{}); [[fallthrough]];
            case 4: grp(std::integral_constant<int, 3>{}); [[fallthrough]];
            case 3: grp(std::integral_constant<int, 2>{}); [[fallthrough]];
            case 2: grp(std::integral_constant<int, 1>{}); [[fallthrough]];
            case 1: grp(std::integral_constant<int, 0>{}); [[fallthrough]];
            default: break;
        }
        return;
    }
#pragma unroll
    for (int t0 = 0; t0 < TPP; t0 += TG) {
        if (TAIL ? t0 + TG <= nt : t0 < nt) {
            float dyg[TG];
            dy_group<TPP, DYT, TG>(dy_s, t0, s, dyg);
#pragma unroll
            for (int t = t0; t < t0 + TG; ++t) {
                const int4* a4 = (const int4*) act_s[t][s];
                const int4 a0 = a4[0], a1 = a4[1], a2 = a4[2], a3 = a4[3];
                const float dy = dyg[t - t0];
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
                acc[t] = __fmaf_rn(tf, dy, acc[t]);
            }
        }
    }
}

// moe_ws_gu_nib.cu mac32 with dy from dy_group
template <int TPP, int CH, int TG, bool DYT, bool TAIL = false, bool DUFF = false>
__device__ __forceinline__ void mac32v(const int (&wv)[16], const float dx, const int s, const int (*act_s)[4][16],
                                       const DyS<TPP, DYT>& dy_s, const int nt, float (&acc)[TPP], float (&acct)[TG]) {
    if constexpr (TAIL) {
        const int nf = nt & ~(TG - 1);
#pragma unroll
        for (int k = 0; k < TG - 1; ++k)
            if (nf + k < nt) acct[k] = mac32_one<CH>(wv, dx, (const int4*) act_s[nf + k][s], dy_s.at(nf + k, s), acct[k]);
    }
    if constexpr (DUFF) {
        static_assert(TPP / TG == 8, "the switch below has 8 cases");
        auto grp = [&](auto K) {
            constexpr int t0 = decltype(K)::value * TG;
            float dyg[TG];
            dy_group<TPP, DYT, TG>(dy_s, t0, s, dyg);
#pragma unroll
            for (int t = t0; t < t0 + TG; ++t)
                acc[t] = mac32_one<CH>(wv, dx, (const int4*) act_s[t][s], dyg[t - t0], acc[t]);
        };
        switch (TAIL ? nt / TG : (nt + TG - 1) / TG) {
            case 8: grp(std::integral_constant<int, 7>{}); [[fallthrough]];
            case 7: grp(std::integral_constant<int, 6>{}); [[fallthrough]];
            case 6: grp(std::integral_constant<int, 5>{}); [[fallthrough]];
            case 5: grp(std::integral_constant<int, 4>{}); [[fallthrough]];
            case 4: grp(std::integral_constant<int, 3>{}); [[fallthrough]];
            case 3: grp(std::integral_constant<int, 2>{}); [[fallthrough]];
            case 2: grp(std::integral_constant<int, 1>{}); [[fallthrough]];
            case 1: grp(std::integral_constant<int, 0>{}); [[fallthrough]];
            default: break;
        }
        return;
    }
#pragma unroll
    for (int t0 = 0; t0 < TPP; t0 += TG) {
        if (TAIL ? t0 + TG <= nt : t0 < nt) {
            float dyg[TG];
            dy_group<TPP, DYT, TG>(dy_s, t0, s, dyg);
#pragma unroll
            for (int t = t0; t < t0 + TG; ++t) {
                const int4* a4 = (const int4*) act_s[t][s];
                const int4 a0 = a4[0], a1 = a4[1], a2 = a4[2], a3 = a4[3];
                const int a[16] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w, a2.x, a2.y, a2.z, a2.w, a3.x, a3.y, a3.z, a3.w};
                const float dy = dyg[t - t0];
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

template <int TPP, int CVT, int TG, int FMT, int V>
__global__ void __launch_bounds__(NT, 1) gu_kernel(const Chunk* __restrict__ chunks, const char* __restrict__ w,
                                                  const int64_t* __restrict__ xoff, const block_q8_1_mmq* __restrict__ xq,
                                                  const int total_rows, float* __restrict__ dst, Rec* __restrict__ rec) {
    static_assert(TPP % TG == 0, "");
    constexpr bool TIME = V & V_TIME, NOLOAD = V & V_NOLOAD, NOSYNC = V & V_NOSYNC, NODEC = V & V_NODEC, DYT = V & V_DYT,
                   PF = V & V_PF, ST16 = V & V_ST16, TAIL = V & V_TAIL, DUFF = V & V_DUFF, DECF = V & V_DECF;
    static_assert(!(PF && NOLOAD), "");
    using AP = typename std::conditional<ST16, ActPre16<TPP>, ActPre<TPP>>::type;
    __shared__ __align__(16) int act_s[TPP][4][16];
    __shared__ __align__(16) DyS<TPP, DYT> dy_s;
    __shared__ __align__(16) uint32_t csm[NT * CS];
    unsigned long long c_entry = 0, cL = 0, cC = 0, mk = 0;
    if constexpr (TIME) {
        c_entry = clock64();
        if (threadIdx.x == 0) rec[blockIdx.x].t0 = gtimer();
    }
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
    // V_PF: the next step's loads, in registers across this step's compute
    uint4 pv[4];
    uint32_t pscv = 0;
    int pdv = 0;
    AP pa;
    auto load_step = [&](const int hc, const int trow0, const int nt) {
#pragma unroll
        for (int i = 0; i < 4; ++i) pv[i] = __ldg(cg + (size_t) i * 32 * ITEMS + 4 * hc);
        pscv = __ldg(scw + hc);
        pdv = (int) __ldg(dr + (hc >> 1));
        if constexpr (ST16) act_load16<TPP>(xq, hc, total_rows, trow0, nt, pa);
        else act_load<TPP>(xq, hc, total_rows, trow0, nt, pa);
    };
    auto store_act = [&](const int nn) {
        if constexpr (ST16) act_store16<TPP, DYT>(pa, nn, act_s, dy_s);
        else act_store<TPP, DYT>(pa, nn, act_s, dy_s);
    };
    if constexpr (PF) load_step(0, c.row0, min(TPP, c.rows));
    uint32_t scv = 0;
    int dv = 0;
    int passes = 0;
    for (int p0 = 0; p0 < c.rows; p0 += TPP) {
        const int nt = min(TPP, c.rows - p0);
        const int trow0 = c.row0 + p0;
        ++passes;
        float acc[TPP], acct[TG];
#pragma unroll
        for (int t = 0; t < TPP; ++t) acc[t] = 0.0f;
#pragma unroll
        for (int t = 0; t < TG; ++t) acct[t] = 0.0f;
#pragma unroll 1
        for (int hc = 0; hc < NSTEP; ++hc) {
            if constexpr (TIME) mk = clock64();
            if constexpr (!NOSYNC) __syncthreads();
            if constexpr (PF) {
#pragma unroll
                for (int i = 0; i < 4; ++i) *(uint4*) (cs_st + i * 32 * CS) = pv[i];
                store_act(nt);
                scv = pscv;
                dv = pdv;
            } else if (!NOLOAD || (p0 == 0 && hc == 0)) {
                load_step(hc, trow0, nt);
                store_act(nt);
#pragma unroll
                for (int i = 0; i < 4; ++i) *(uint4*) (cs_st + i * 32 * CS) = pv[i];
                scv = pscv;
                dv = pdv;
            }
            if constexpr (!NOSYNC) __syncthreads();
            if constexpr (PF) {
                if (hc + 1 < NSTEP) load_step(hc + 1, trow0, nt);
                else if (p0 + TPP < c.rows) load_step(0, trow0 + TPP, min(TPP, c.rows - p0 - TPP));
            }
            if constexpr (TIME) {
                const unsigned long long m2 = clock64();
                cL += m2 - mk;
                mk = m2;
            }
#pragma unroll 1
            for (int s = 0; s < 4; ++s) {
                const uint4 u = *(const uint4*) (cw + 4 * s);
                const uint32_t q[4] = {u.x, u.y, u.z, u.w};
                const int ls = (int) ((scv >> (8 * s)) & 0xFF);
                int wv[16];
                if constexpr (NODEC) wv_raw(q, wv);
                else if constexpr (DECF) DecF<FMT>::wv(q, wv);
                else Dec<FMT>::wv(q, wv);
                if constexpr (FMT == nb::kIQ2_S) {
                    float dx0, dx1;
                    Dec<FMT>::dx(ls, dv, dx0, dx1);
                    mac16v<TPP, CVT, TG, DYT, TAIL, DUFF>(wv, dx0, dx1, s, act_s, dy_s, nt, acc, acct);
                } else {
                    mac32v<TPP, CVT, TG, DYT, TAIL, DUFF>(wv, Dec<FMT>::dx(ls, dv), s, act_s, dy_s, nt, acc, acct);
                }
            }
            if constexpr (TIME) cC += clock64() - mk;
        }
        if constexpr (TAIL) {
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
    if constexpr (TIME) {
        rec_warp(rec, clock64() - c_entry, cL, cC);
        if (threadIdx.x == 0) {
            rec[blockIdx.x].t1 = gtimer();
            rec[blockIdx.x].smid = smid_reg();
            rec[blockIdx.x].rows = (unsigned) c.rows;
            rec[blockIdx.x].passes = (unsigned) passes;
        }
    }
}
}  // namespace gu

// ================================ down IQ4_NL: moe_ws_down.cu's nl4::dn_kernel, copied (+ chains CHN) =============
namespace nl4 {
constexpr int DN_ROWS = 2560, DN_K = 640;
constexpr int NB = DN_K / QK4_NL, BB = (int) sizeof(block_iq4_nl), ROWB = NB * BB;

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
template <int PAR>
__device__ __forceinline__ float load_blk(const int* __restrict__ cw, const int p, int (&q)[4]) {
    const int* c = cw + 9 * p;
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

template <int NT, int TPP, int SB, int G, int PAR, int CHN, bool DUFF = false, bool DYT = false>
__device__ __forceinline__ void blk_compute(const int* __restrict__ cw, const int p, const uint32_t (&tab)[4],
                                            const int (*act_s)[SB][16], const float* __restrict__ dyb, const int nt,
                                            float (&acc)[TPP]) {
    const int b = 2 * p + PAR;
    // dy of token t, block b: production's dy_s[TPP][SB], or [SB][TPP] (DYT)
    auto dyat = [&](const int t) { return DYT ? dyb[b * TPP + t] : dyb[t * SB + b]; };
    int wv[16];
    float dx;
    {
        int q[4];
        dx = load_blk<PAR>(cw, p, q);
        dec(q, tab, wv);
    }
    if constexpr (DUFF) {
        static_assert(TPP / G == 16 && G == 2 && CHN == 1, "the switch below: 16 token pairs, one chain");
        auto grp = [&](auto K) {
            constexpr int t0 = decltype(K)::value * G;
            float d0, d1;
            if constexpr (DYT) {
                const float2 v = *(const float2*) &dyb[b * TPP + t0];
                d0 = v.x; d1 = v.y;
            } else {
                d0 = dyb[t0 * SB + b]; d1 = dyb[(t0 + 1) * SB + b];
            }
#pragma unroll
            for (int t = t0; t < t0 + G; ++t) {
                const int4* a4 = (const int4*) act_s[t][b];
                const int4 a0 = a4[0], a1 = a4[1], a2 = a4[2], a3 = a4[3];
                int s0 = 0;
                s0 = mad2(wv[0], a0.x, s0); s0 = mad2(wv[1], a0.y, s0); s0 = mad2(wv[2], a0.z, s0); s0 = mad2(wv[3], a0.w, s0);
                s0 = mad2(wv[4], a1.x, s0); s0 = mad2(wv[5], a1.y, s0); s0 = mad2(wv[6], a1.z, s0); s0 = mad2(wv[7], a1.w, s0);
                s0 = mad2(wv[8], a2.x, s0); s0 = mad2(wv[9], a2.y, s0); s0 = mad2(wv[10], a2.z, s0); s0 = mad2(wv[11], a2.w, s0);
                s0 = mad2(wv[12], a3.x, s0); s0 = mad2(wv[13], a3.y, s0); s0 = mad2(wv[14], a3.z, s0); s0 = mad2(wv[15], a3.w, s0);
                acc[t] = __fmaf_rn(__fmul_rn(dx, t == t0 ? d0 : d1), __int2float_rn(s0), acc[t]);
            }
        };
        using std::integral_constant;
        switch ((nt + G - 1) / G) {
            case 16: grp(integral_constant<int, 15>{}); [[fallthrough]];
            case 15: grp(integral_constant<int, 14>{}); [[fallthrough]];
            case 14: grp(integral_constant<int, 13>{}); [[fallthrough]];
            case 13: grp(integral_constant<int, 12>{}); [[fallthrough]];
            case 12: grp(integral_constant<int, 11>{}); [[fallthrough]];
            case 11: grp(integral_constant<int, 10>{}); [[fallthrough]];
            case 10: grp(integral_constant<int, 9>{}); [[fallthrough]];
            case 9: grp(integral_constant<int, 8>{}); [[fallthrough]];
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
        return;
    }
#pragma unroll
    for (int t0 = 0; t0 < TPP; t0 += G) {
        if (t0 < nt) {
#pragma unroll
            for (int t = t0; t < t0 + G; ++t) {
                const int4* a4 = (const int4*) act_s[t][b];
                const int4 a0 = a4[0], a1 = a4[1], a2 = a4[2], a3 = a4[3];
                const float dy = dyat(t);
                int s0;
                if constexpr (CHN == 1) {
                    s0 = 0;
                    s0 = mad2(wv[0], a0.x, s0); s0 = mad2(wv[1], a0.y, s0); s0 = mad2(wv[2], a0.z, s0); s0 = mad2(wv[3], a0.w, s0);
                    s0 = mad2(wv[4], a1.x, s0); s0 = mad2(wv[5], a1.y, s0); s0 = mad2(wv[6], a1.z, s0); s0 = mad2(wv[7], a1.w, s0);
                    s0 = mad2(wv[8], a2.x, s0); s0 = mad2(wv[9], a2.y, s0); s0 = mad2(wv[10], a2.z, s0); s0 = mad2(wv[11], a2.w, s0);
                    s0 = mad2(wv[12], a3.x, s0); s0 = mad2(wv[13], a3.y, s0); s0 = mad2(wv[14], a3.z, s0); s0 = mad2(wv[15], a3.w, s0);
                } else {   // two chains: the 32-value sum is exact, so the same int
                    int sa = 0, sb = 0;
                    sa = mad2(wv[0], a0.x, sa); sb = mad2(wv[8], a2.x, sb); sa = mad2(wv[1], a0.y, sa); sb = mad2(wv[9], a2.y, sb);
                    sa = mad2(wv[2], a0.z, sa); sb = mad2(wv[10], a2.z, sb); sa = mad2(wv[3], a0.w, sa); sb = mad2(wv[11], a2.w, sb);
                    sa = mad2(wv[4], a1.x, sa); sb = mad2(wv[12], a3.x, sb); sa = mad2(wv[5], a1.y, sa); sb = mad2(wv[13], a3.y, sb);
                    sa = mad2(wv[6], a1.z, sa); sb = mad2(wv[14], a3.z, sb); sa = mad2(wv[7], a1.w, sa); sb = mad2(wv[15], a3.w, sb);
                    s0 = sa + sb;
                }
                acc[t] = __fmaf_rn(__fmul_rn(dx, dy), __int2float_rn(s0), acc[t]);
            }
        }
    }
}

template <int NT, int TPP, int SB, int G, int CHN, int V>
__global__ void __launch_bounds__(NT, (V & V_LB2) ? 2 : 1) dn_kernel(const Chunk* __restrict__ chunks, const char* __restrict__ w,
                                                   const int64_t* __restrict__ xoff, const int64_t xadd,
                                                   const block_q8_1_mmq* __restrict__ xq, const int total_rows,
                                                   float* __restrict__ dst, Rec* __restrict__ rec) {
    static_assert((SB == 2 || SB == 4) && NT % 32 == 0 && TPP % G == 0, "");
    constexpr bool TIME = V & V_TIME, NOSYNC = V & V_NOSYNC, DUFF = V & V_DUFF, DYT = V & V_DYT;
    constexpr int NW = NT / 32, WPS = 9 * SB / 2, S = WPS | 1, NSTEP = NB / SB, PIECES = DN_ROWS / NT;
    constexpr int KI = (NT + 3 * NW - 1) / (3 * NW);
    constexpr int AW = 9 * SB;
    constexpr int AI = (TPP * AW + NT - 1) / NT;
    using PT = typename std::conditional<SB == 4, int2, int>::type;
    __shared__ int wsm[NT * S];
    __shared__ __align__(16) int act_s[TPP][SB][16];
    __shared__ __align__(8) float dy_s[TPP * SB];   // [TPP][SB], or [SB][TPP] (DYT)
    unsigned long long c_entry = 0, cL = 0, cC = 0, mk = 0;
    if constexpr (TIME) {
        c_entry = clock64();
        if (threadIdx.x == 0) rec[blockIdx.x].t0 = gtimer();
    }
    uint32_t tab[4];
    load_tab(tab);
    int r0;
    const Chunk c = cta_work<PIECES, NT>(chunks, r0);
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int sub = lane / 9, pc = lane - 9 * sub;
    const int srow0 = 3 * warp + sub;
    const bool sact = sub < 3;
    const char* __restrict__ Wg = w + xoff[c.j] + xadd + (size_t) (r0 + srow0) * ROWB + pc * (int) sizeof(PT);
    int* __restrict__ ws_st = wsm + srow0 * S + pc * (int) (sizeof(PT) / 4);
    PT pre[KI];
    int apre[AI];
    auto load_w = [&](const int st) {
        if (sact) {
            const char* g = Wg + st * (SB * BB);
#pragma unroll
            for (int k = 0; k < KI; ++k)
                if (k < KI - 1 || srow0 + 3 * NW * k < NT) pre[k] = __ldg((const PT*) (g + (size_t) k * 3 * NW * ROWB));
        }
    };
    auto store_w = [&]() {
        if (sact) {
#pragma unroll
            for (int k = 0; k < KI; ++k)
                if (k < KI - 1 || srow0 + 3 * NW * k < NT) {
                    if constexpr (SB == 4) {
                        ws_st[k * 3 * NW * S] = pre[k].x;
                        ws_st[k * 3 * NW * S + 1] = pre[k].y;
                    } else {
                        ws_st[k * 3 * NW * S] = pre[k];
                    }
                }
        }
    };
    auto load_a = [&](const int st, const int trow0, const int nt) {
        const int kb = (st * SB) >> 2, b0 = (st * SB) & 3;
#pragma unroll
        for (int i = 0; i < AI; ++i) {
            const int idx = threadIdx.x + i * NT;
            if (idx < nt * AW) {
                const int t = idx / AW, wd = idx - t * AW;
                const int* blk = (const int*) (xq + (size_t) kb * total_rows + trow0 + t);
                apre[i] = __ldg(blk + (wd < SB ? b0 + wd : 4 + 8 * b0 + (wd - SB)));
            }
        }
    };
    auto store_a = [&](const int nt) {
#pragma unroll
        for (int i = 0; i < AI; ++i) {
            const int idx = threadIdx.x + i * NT;
            if (idx < nt * AW) {
                const int t = idx / AW, wd = idx - t * AW;
                if (wd < SB) {
                    dy_s[DYT ? wd * TPP + t : t * SB + wd] = __int_as_float(apre[i]);
                } else {
                    const int m = wd - SB, bb = m >> 3, mm = m & 7;
                    int lo, hi;
                    split01(apre[i], lo, hi);
                    *(int2*) &act_s[t][bb][2 * mm] = make_int2(lo, hi);
                }
            }
        }
    };
    load_w(0);
    load_a(0, c.row0, min(TPP, c.rows));
    const int* cw = wsm + threadIdx.x * S;
    int passes = 0;
    for (int p0 = 0; p0 < c.rows; p0 += TPP) {
        const int nt = min(TPP, c.rows - p0);
        const int trow0 = c.row0 + p0;
        ++passes;
        float acc[TPP];
#pragma unroll
        for (int t = 0; t < TPP; ++t) acc[t] = 0.0f;
#pragma unroll 1
        for (int st = 0; st < NSTEP; ++st) {
            if constexpr (TIME) mk = clock64();
            if constexpr (!NOSYNC) __syncthreads();
            store_w();
            store_a(nt);
            if constexpr (!NOSYNC) __syncthreads();
            if (st + 1 < NSTEP) {
                load_w(st + 1);
                load_a(st + 1, trow0, nt);
            } else if (p0 + TPP < c.rows) {
                load_w(0);
                load_a(0, trow0 + TPP, min(TPP, c.rows - p0 - TPP));
            }
            if constexpr (TIME) {
                const unsigned long long m2 = clock64();
                cL += m2 - mk;
                mk = m2;
            }
#pragma unroll 1
            for (int p = 0; p < SB / 2; ++p) {
                blk_compute<NT, TPP, SB, G, 0, CHN, DUFF, DYT>(cw, p, tab, act_s, dy_s, nt, acc);
                blk_compute<NT, TPP, SB, G, 1, CHN, DUFF, DYT>(cw, p, tab, act_s, dy_s, nt, acc);
            }
            if constexpr (TIME) cC += clock64() - mk;
        }
#pragma unroll
        for (int t = 0; t < TPP; ++t)
            if (t < nt) dst[(size_t) (trow0 + t) * DN_ROWS + r0 + threadIdx.x] = acc[t];
    }
    if constexpr (TIME) {
        rec_warp(rec, clock64() - c_entry, cL, cC);
        if (threadIdx.x == 0) {
            rec[blockIdx.x].t1 = gtimer();
            rec[blockIdx.x].smid = smid_reg();
            rec[blockIdx.x].rows = (unsigned) c.rows;
            rec[blockIdx.x].passes = (unsigned) passes;
        }
    }
}
}  // namespace nl4

// ---- launch helpers -------------------------------------------------------------------------------------------------
struct Args {
    const Chunk* chunks;
    const char* w;
    const int64_t* xoff;
    int64_t down_off;
    const block_q8_1_mmq* xq;   // gate/up's q8_1 rows
    const block_q8_1_mmq* hq;   // down's
    int rows;
    float* gu;
    float* dm;
    Rec* rec;
    int64_t nchunks;
};
using Launch = std::function<void(const Args&)>;

template <int TPP, int CVT, int TG, int FMT, int V>
Launch gu_launch() {
    return [](const Args& a) {
        gu::gu_kernel<TPP, CVT, TG, FMT, V><<<(unsigned) (a.nchunks * gu::PIECES), gu::NT>>>(a.chunks, a.w, a.xoff, a.xq, a.rows,
                                                                                         a.gu, a.rec);
    };
}
template <int TPP, int CVT, int TG, int FMT, int V>
void gu_attr(int& regs, int& smem, int& ctas) {
    cudaFuncAttributes fa{};
    CK(cudaFuncGetAttributes(&fa, gu::gu_kernel<TPP, CVT, TG, FMT, V>));
    regs = fa.numRegs;
    smem = (int) fa.sharedSizeBytes;
    CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&ctas, gu::gu_kernel<TPP, CVT, TG, FMT, V>, gu::NT, 0));
}
template <int NT, int TPP, int SB, int G, int CHN, int V>
Launch dn_launch() {
    return [](const Args& a) {
        nl4::dn_kernel<NT, TPP, SB, G, CHN, V><<<(unsigned) (a.nchunks * (nl4::DN_ROWS / NT)), NT>>>(
            a.chunks, a.w, a.xoff, a.down_off, a.hq, a.rows, a.dm, a.rec);
    };
}
template <int NT, int TPP, int SB, int G, int CHN, int V>
void dn_attr(int& regs, int& smem, int& ctas) {
    cudaFuncAttributes fa{};
    CK(cudaFuncGetAttributes(&fa, nl4::dn_kernel<NT, TPP, SB, G, CHN, V>));
    regs = fa.numRegs;
    smem = (int) fa.sharedSizeBytes;
    CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&ctas, nl4::dn_kernel<NT, TPP, SB, G, CHN, V>, NT, 0));
}

// ---- STRATA_WS_V2 prototypes (moe_ws_v2.cuh), --arms v2
namespace v2n = strata::prefill::ws::detail::v2;
template <int TPP, int CH, int TG, int FMT, int V>
Launch v2gu_launch() {
    return [](const Args& a) {
        v2n::gu_kernel<TPP, CH, TG, FMT, V><<<(unsigned) (a.nchunks * v2n::gu_pieces<V>()), v2n::NT>>>(a.chunks, a.w, a.xoff, a.xq,
                                                                                                 a.rows, a.gu);
    };
}
template <int TPP, int CH, int TG, int FMT, int V>
void v2gu_attr(int& regs, int& smem, int& ctas) {
    cudaFuncAttributes fa{};
    CK(cudaFuncGetAttributes(&fa, v2n::gu_kernel<TPP, CH, TG, FMT, V>));
    regs = fa.numRegs;
    smem = (int) fa.sharedSizeBytes;
    CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&ctas, v2n::gu_kernel<TPP, CH, TG, FMT, V>, v2n::NT, 0));
}
template <int TPP, int V>
Launch v2dn_launch() {
    return [](const Args& a) {
        v2n::dn::dn_kernel<TPP, V><<<(unsigned) (a.nchunks * v2n::dn::dn_pieces<V>()), v2n::NT>>>(a.chunks, a.w, a.xoff, a.down_off,
                                                                                               a.hq, a.rows, a.dm);
    };
}
template <int TPP, int V>
void v2dn_attr(int& regs, int& smem, int& ctas) {
    cudaFuncAttributes fa{};
    CK(cudaFuncGetAttributes(&fa, v2n::dn::dn_kernel<TPP, V>));
    regs = fa.numRegs;
    smem = (int) fa.sharedSizeBytes;
    CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&ctas, v2n::dn::dn_kernel<TPP, V>, v2n::NT, 0));
}

}  // namespace wsprof

namespace {

namespace nib = strata::core::nibble;
namespace ws = strata::prefill::ws;
namespace mmq = strata::prefill::mmq;
namespace cpu = strata::kernels::cpu;
using wsprof::Args;
using wsprof::Launch;
using wsprof::Rec;

constexpr int64_t kEmbd = nib::kNEmbd, kFf = nib::kNFf, kK = 10, kNE = 512;

__global__ void fill_x(float* x, long long n, unsigned long long seed) {
    const long long i = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    unsigned long long h = seed ^ (unsigned long long) i * 0x9E3779B97F4A7C15ull;
    h ^= h >> 33; h *= 0xFF51AFD7ED558CCDull; h ^= h >> 33; h *= 0xC4CEB9FE1A85EC53ull; h ^= h >> 33;
    const float u = (float) (h & 0xFFFFFF) / 16777216.0f - 0.5f;
    x[i] = ((h >> 40) % 97 == 0) ? 24.0f * u : 2.0f * u;
}

struct Layer {
    int layer = -1;
    cpu::NativeFmt f;
    nib::LayerVram V;
    uint8_t* raw = nullptr;
    uint8_t* slots = nullptr;
};

// moe_ws_nib_parity.cu load_layer (without the negative control)
bool load_layer(const strata::GgufModel& m, int l, Layer& Y) {
    const char* roles[3] = { "gate", "up", "down" };
    const strata::TensorInfo* t[3] = {};
    size_t sh[3] = {};
    for (int r = 0; r < 3; ++r) t[r] = m.find("blk." + std::to_string(l) + ".ffn_" + roles[r] + "_exps.weight", &sh[r]);
    if (!t[0] || !t[1] || !t[2]) return false;
    std::string err;
    if (!cpu::native_fmt((int) t[0]->type, (int) t[2]->type, kEmbd, kFf, Y.f, err)) {
        std::printf("layer %d: %s\n", l, err.c_str());
        return false;
    }
    Y.layer = l;
    const size_t gu_role = Y.f.up_off, dn_role = Y.f.bytes - Y.f.down_off;
    std::vector<uint8_t> host((size_t) kNE * Y.f.bytes);
    const uint8_t* G = m.shard(sh[0]).tensor_data(*t[0]);
    const uint8_t* U = m.shard(sh[1]).tensor_data(*t[1]);
    const uint8_t* D = m.shard(sh[2]).tensor_data(*t[2]);
    for (int64_t e = 0; e < kNE; ++e) {
        uint8_t* b = host.data() + (size_t) e * Y.f.bytes;
        std::memcpy(b, G + (size_t) e * gu_role, gu_role);
        std::memcpy(b + Y.f.up_off, U + (size_t) e * gu_role, gu_role);
        std::memcpy(b + Y.f.down_off, D + (size_t) e * dn_role, dn_role);
    }
    nib::Config c;
    c.on = true;
    Y.V = nib::plan_layer(l, Y.f, c);
    CK(cudaMalloc(&Y.raw, host.size()));
    CK(cudaMemcpy(Y.raw, host.data(), host.size(), cudaMemcpyHostToDevice));
    if (!Y.V.on) {
        std::printf("layer %d: plan_layer does not convert it (no 4-bit slots)\n", l);
        return true;
    }
    const size_t sb = (size_t) (kNE - 1) * Y.V.slot_bytes + Y.V.hashed;
    CK(cudaMalloc(&Y.slots, sb));
    CK(cudaMemset(Y.slots, 0xA5, sb));
    std::vector<unsigned long long> hp(kNE);
    for (int64_t e = 0; e < kNE; ++e) hp[e] = (unsigned long long) (Y.slots + (size_t) e * Y.V.slot_bytes);
    unsigned long long *dp = nullptr, *dbad = nullptr;
    CK(cudaMalloc(&dp, kNE * 8));
    CK(cudaMalloc(&dbad, 8));
    CK(cudaMemcpy(dp, hp.data(), kNE * 8, cudaMemcpyHostToDevice));
    CK(cudaMemset(dbad, 0, 8));
    for (int64_t e0 = 0; e0 < kNE; e0 += 64)
        if (nib::transcode(Y.V, Y.raw + (size_t) e0 * Y.V.raw_bytes, 64, dp + e0, dbad, nullptr) != 0) std::exit(2);
    for (int64_t e = 0; e < kNE; ++e)
        CK(cudaMemcpyAsync(Y.slots + (size_t) e * Y.V.slot_bytes + nib::kDownOff,
                           Y.raw + (size_t) e * Y.V.raw_bytes + Y.V.raw_down_off, Y.V.down_bytes, cudaMemcpyDeviceToDevice));
    CK(cudaDeviceSynchronize());
    unsigned long long bad = 0;
    CK(cudaMemcpy(&bad, dbad, 8, cudaMemcpyDeviceToHost));
    if (bad) {
        std::printf("layer %d: transcode: %llu items do not round-trip\n", l, bad);
        std::exit(2);
    }
    CK(cudaFree(dp));
    CK(cudaFree(dbad));
    return true;
}

std::vector<std::string> split_list(const char* s) {
    std::vector<std::string> v;
    std::string t(s);
    size_t p = 0;
    while (p < t.size()) {
        size_t q = t.find(',', p);
        if (q == std::string::npos) q = t.size();
        v.push_back(t.substr(p, q - p));
        p = q + 1;
    }
    return v;
}

// counts over the 512 experts summing to rows: "lnS" lognormal sigma S, each <= T (moe_ws_nib_parity's model);
// "flatN": rows / N experts of exactly N rows
std::vector<int32_t> make_counts(const std::string& model, int T, int layer) {
    const int rows = T * (int) kK;
    std::vector<int32_t> cnt(kNE, 0);
    if (model.rfind("flat", 0) == 0) {
        const int n = std::atoi(model.c_str() + 4);
        int left = rows;
        for (int e = 0; e < kNE && left > 0; ++e) {
            cnt[e] = std::min(n, left);
            left -= cnt[e];
        }
        return cnt;
    }
    const double sigma = std::atof(model.c_str() + 2);
    std::mt19937_64 rng(1000003ull * (uint64_t) T + (uint64_t) (sigma * 1000) + 7 + (uint64_t) layer);
    std::normal_distribution<double> nd(0.0, 1.0);
    std::vector<double> wgt(kNE);
    for (auto& x : wgt) x = std::exp(sigma * nd(rng));
    const double S = std::accumulate(wgt.begin(), wgt.end(), 0.0);
    std::vector<std::pair<double, int>> fr;
    int sum = 0;
    for (int e = 0; e < kNE; ++e) {
        const double cc = rows * wgt[e] / S;
        cnt[e] = std::min((int) std::floor(cc), T);
        sum += cnt[e];
        fr.push_back({ cc - std::floor(cc), e });
    }
    std::sort(fr.begin(), fr.end(), [](auto& a, auto& b) { return a.first > b.first; });
    for (size_t i = 0; sum < rows && i < 4 * (size_t) kNE; ++i) {
        const int e = fr[i % kNE].second;
        if (cnt[e] < T) { ++cnt[e]; ++sum; }
    }
    return cnt;
}

double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v.empty() ? 0.0 : v[v.size() / 2];
}

// the per-CTA records of one V_TIME launch: span, SM idle / partly filled time, the tail, the phases
void analyze(const char* tag, const std::vector<Rec>& r, int slots_per_sm, int nsm, double kernel_ms, FILE* csv) {
    if (r.empty()) return;
    unsigned long long t0 = ~0ull, t1 = 0, last_start = 0;
    double cyc = 0, cl = 0, cc = 0;
    for (const Rec& x : r) {
        t0 = std::min(t0, x.t0);
        t1 = std::max(t1, x.t1);
        last_start = std::max(last_start, x.t0);
        cyc += (double) x.cyc;
        cl += (double) x.cycL;
        cc += (double) x.cycC;
    }
    const double span = (double) (t1 - t0);
    // per SM: time with 0 CTAs, with fewer than the full slots, over [t0, t1]
    std::vector<std::vector<std::pair<unsigned long long, int>>> ev(nsm);
    double ctadur = 0;
    for (const Rec& x : r) {
        if ((int) x.smid < nsm) {
            ev[x.smid].push_back({ x.t0, +1 });
            ev[x.smid].push_back({ x.t1, -1 });
        }
        ctadur += (double) (x.t1 - x.t0);
    }
    double idle = 0, part = 0, occ_int = 0, tail_idle = 0;
    for (int s = 0; s < nsm; ++s) {
        auto& e = ev[s];
        std::sort(e.begin(), e.end(), [](auto& a, auto& b) { return a.first != b.first ? a.first < b.first : a.second < b.second; });
        int act = 0;
        unsigned long long prev = t0;
        for (auto& [t, d] : e) {
            const double dt = (double) (t - prev);
            if (act == 0) idle += dt;
            else if (act < slots_per_sm) part += dt * (double) (slots_per_sm - act) / slots_per_sm;
            if (act == 0 && prev >= last_start) tail_idle += dt;
            occ_int += dt * act;
            act += d;
            prev = t;
        }
        idle += (double) (t1 - prev);
        if (prev >= last_start) tail_idle += (double) (t1 - prev);
        else tail_idle += (double) (t1 - std::max(prev, last_start));
    }
    const double smspan = span * nsm;
    std::printf("CTAS  %-34s %zu CTAs, span %.3f ms (event %.3f ms); SM idle %.1f%% (after the last CTA start %.1f%%), "
                "slots unfilled while busy %.1f%%; mean CTAs/SM %.2f of %d; last CTA starts at %.1f%% of the span; "
                "warp-time phases: load/stage+barriers %.1f%%, compute %.1f%%, other %.1f%%\n",
                tag, r.size(), span * 1e-6, kernel_ms, 100.0 * idle / smspan, 100.0 * tail_idle / smspan,
                100.0 * part / smspan, occ_int / smspan, slots_per_sm, 100.0 * (double) (last_start - t0) / span,
                100.0 * cl / cyc, 100.0 * cc / cyc, 100.0 * (cyc - cl - cc) / cyc);
    if (csv) {
        for (const Rec& x : r)
            std::fprintf(csv, "%s,%u,%llu,%llu,%u,%u,%llu,%llu,%llu\n", tag, x.smid, x.t0 - t0, x.t1 - t0, x.rows, x.passes,
                         x.cyc, x.cycL, x.cycC);
    }
}

}  // namespace

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    std::vector<std::string> shards, models = { "ln0.5", "ln1.0", "flat32" };
    std::vector<int> layers = { 2, 0, 17, 21 }, ts = { 1536 };
    int reps = 15;
    bool prod_only = false, v2_only = false, v2_round2 = false;
    std::string csvdir;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> const char* {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "%s needs a value\n", a.c_str());
                std::exit(2);
            }
            return argv[++i];
        };
        if (a == "--layers") { layers.clear(); for (auto& s : split_list(next())) layers.push_back(std::atoi(s.c_str())); }
        else if (a == "--t") { ts.clear(); for (auto& s : split_list(next())) ts.push_back(std::atoi(s.c_str())); }
        else if (a == "--models") models = split_list(next());
        else if (a == "--reps") reps = std::atoi(next());
        else if (a == "--arms") {
            const std::string v = next();
            prod_only = v == "prod";
            v2_only = v == "v2" || v == "v2r2";
            v2_round2 = v == "v2r2";
        }
        else if (a == "--csv") csvdir = next();
        else if (a.size() > 5 && a.substr(a.size() - 5) == ".gguf") shards.push_back(a);
        else {
            std::fprintf(stderr, "usage: moe_ws_prof_bench <shard.gguf> [...] [--layers L,..] [--t T,..] "
                                 "[--models ln0.5,ln1.0,flat32] [--reps N] [--arms all|prod] [--csv dir]\n");
            return 2;
        }
    }
    if (shards.empty()) {
        std::fprintf(stderr, "moe_ws_prof_bench: give a model shard (.gguf)\n");
        return 2;
    }
    cudaDeviceProp p{};
    CK(cudaGetDeviceProperties(&p, 0));
    char bus[64] = {};
    CK(cudaDeviceGetPCIBusId(bus, sizeof bus, 0));
    std::printf("device 0: %s, bus %s, cc %d.%d, %d SMs, clock %d kHz; ws::mode() %d; mmq::built() %d\n", p.name, bus, p.major,
                p.minor, p.multiProcessorCount, p.clockRate, ws::mode(), (int) mmq::built());
    if (ws::mode() != 1 || !mmq::built()) {
        std::fprintf(stderr, "needs the WS kernels (sm_60, STRATA_MMQ_WS=1) and MMQ\n");
        return 2;
    }
    const int nsm = p.multiProcessorCount;
    std::unique_ptr<strata::GgufModel> model;
    try {
        model = std::make_unique<strata::GgufModel>(strata::GgufModel::open(shards[0]));
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "%s\n", ex.what());
        return 2;
    }
    // the variants (V bits) of the copies, per format: name, launch, attributes; [0] = V 0 (production's code)
    struct Arm {
        std::string name;
        Launch go;
        bool exact;   // must equal production bit for bit
        int regs = 0, smem = 0, ctas = 0;
    };
    using namespace wsprof;
    auto gu_arms = [&](int fmt) {
        std::vector<Arm> v;
#define GU_ARMS(TPP, CVT, TG, FMT)                                                                                           \
        v.push_back({ "copy V0", gu_launch<TPP, CVT, TG, FMT, 0>(), true }); gu_attr<TPP, CVT, TG, FMT, 0>(v.back().regs, v.back().smem, v.back().ctas); \
        v.push_back({ "copy TIME", gu_launch<TPP, CVT, TG, FMT, V_TIME>(), true }); gu_attr<TPP, CVT, TG, FMT, V_TIME>(v.back().regs, v.back().smem, v.back().ctas); \
        v.push_back({ "abl NOLOAD", gu_launch<TPP, CVT, TG, FMT, V_NOLOAD>(), false }); gu_attr<TPP, CVT, TG, FMT, V_NOLOAD>(v.back().regs, v.back().smem, v.back().ctas); \
        v.push_back({ "abl NOSYNC", gu_launch<TPP, CVT, TG, FMT, V_NOSYNC>(), false }); gu_attr<TPP, CVT, TG, FMT, V_NOSYNC>(v.back().regs, v.back().smem, v.back().ctas); \
        v.push_back({ "abl NOLOAD+NOSYNC", gu_launch<TPP, CVT, TG, FMT, V_NOLOAD | V_NOSYNC>(), false }); gu_attr<TPP, CVT, TG, FMT, V_NOLOAD | V_NOSYNC>(v.back().regs, v.back().smem, v.back().ctas); \
        v.push_back({ "abl NODEC", gu_launch<TPP, CVT, TG, FMT, V_NODEC>(), false }); gu_attr<TPP, CVT, TG, FMT, V_NODEC>(v.back().regs, v.back().smem, v.back().ctas); \
        v.push_back({ "abl MACONLY", gu_launch<TPP, CVT, TG, FMT, V_NOLOAD | V_NOSYNC | V_NODEC>(), false }); gu_attr<TPP, CVT, TG, FMT, V_NOLOAD | V_NOSYNC | V_NODEC>(v.back().regs, v.back().smem, v.back().ctas); \
        v.push_back({ "fix DYT", gu_launch<TPP, CVT, TG, FMT, V_DYT>(), true }); gu_attr<TPP, CVT, TG, FMT, V_DYT>(v.back().regs, v.back().smem, v.back().ctas); \
        v.push_back({ "fix PF", gu_launch<TPP, CVT, TG, FMT, V_PF>(), true }); gu_attr<TPP, CVT, TG, FMT, V_PF>(v.back().regs, v.back().smem, v.back().ctas); \
        v.push_back({ "fix PF+DYT", gu_launch<TPP, CVT, TG, FMT, V_PF | V_DYT>(), true }); gu_attr<TPP, CVT, TG, FMT, V_PF | V_DYT>(v.back().regs, v.back().smem, v.back().ctas); \
        v.push_back({ "fix ST16", gu_launch<TPP, CVT, TG, FMT, V_ST16>(), true }); gu_attr<TPP, CVT, TG, FMT, V_ST16>(v.back().regs, v.back().smem, v.back().ctas); \
        v.push_back({ "fix TAIL", gu_launch<TPP, CVT, TG, FMT, V_TAIL>(), true }); gu_attr<TPP, CVT, TG, FMT, V_TAIL>(v.back().regs, v.back().smem, v.back().ctas); \
        v.push_back({ "fix ST16+TAIL", gu_launch<TPP, CVT, TG, FMT, V_ST16 | V_TAIL>(), true }); gu_attr<TPP, CVT, TG, FMT, V_ST16 | V_TAIL>(v.back().regs, v.back().smem, v.back().ctas); \
        v.push_back({ "fix ST16+TAIL+DYT", gu_launch<TPP, CVT, TG, FMT, V_ST16 | V_TAIL | V_DYT>(), true }); gu_attr<TPP, CVT, TG, FMT, V_ST16 | V_TAIL | V_DYT>(v.back().regs, v.back().smem, v.back().ctas); \
        v.push_back({ "fix DUFF", gu_launch<TPP, CVT, TG, FMT, V_DUFF>(), true }); gu_attr<TPP, CVT, TG, FMT, V_DUFF>(v.back().regs, v.back().smem, v.back().ctas); \
        v.push_back({ "fix DECF", gu_launch<TPP, CVT, TG, FMT, V_DECF>(), true }); gu_attr<TPP, CVT, TG, FMT, V_DECF>(v.back().regs, v.back().smem, v.back().ctas); \
        v.push_back({ "fix ST16+DUFF", gu_launch<TPP, CVT, TG, FMT, V_ST16 | V_DUFF>(), true }); gu_attr<TPP, CVT, TG, FMT, V_ST16 | V_DUFF>(v.back().regs, v.back().smem, v.back().ctas); \
        v.push_back({ "fix ST16+DUFF+DECF", gu_launch<TPP, CVT, TG, FMT, V_ST16 | V_DUFF | V_DECF>(), true }); gu_attr<TPP, CVT, TG, FMT, V_ST16 | V_DUFF | V_DECF>(v.back().regs, v.back().smem, v.back().ctas); \
        v.push_back({ "fix ST16+DUFF+DECF+DYT", gu_launch<TPP, CVT, TG, FMT, V_ST16 | V_DUFF | V_DECF | V_DYT>(), true }); gu_attr<TPP, CVT, TG, FMT, V_ST16 | V_DUFF | V_DECF | V_DYT>(v.back().regs, v.back().smem, v.back().ctas); \
        v.push_back({ "fix ST16+TAIL+DUFF+DECF", gu_launch<TPP, CVT, TG, FMT, V_ST16 | V_TAIL | V_DUFF | V_DECF>(), true }); gu_attr<TPP, CVT, TG, FMT, V_ST16 | V_TAIL | V_DUFF | V_DECF>(v.back().regs, v.back().smem, v.back().ctas); \
        v.push_back({ "fix ALL (ST16 TAIL DUFF DECF DYT)", gu_launch<TPP, CVT, TG, FMT, V_ST16 | V_TAIL | V_DUFF | V_DECF | V_DYT>(), true }); gu_attr<TPP, CVT, TG, FMT, V_ST16 | V_TAIL | V_DUFF | V_DECF | V_DYT>(v.back().regs, v.back().smem, v.back().ctas);
        // production's tier at u >= 25 / 18 / 18 (moe_ws_gu_nib.cu gu_nib)
        if (fmt == nib::kIQ2_S) { GU_ARMS(32, 1, 4, nib::kIQ2_S) }
        else if (fmt == nib::kIQ3_XXS) { GU_ARMS(32, 1, 4, nib::kIQ3_XXS) }
        else if (fmt == nib::kIQ3_S) { GU_ARMS(32, 2, 4, nib::kIQ3_S) }
#undef GU_ARMS
        return v;
    };
    auto dn_arms = [&]() {
        std::vector<Arm> v;
        // production's IQ4_NL tier at u >= 20: 256 x 32, 2-block steps, G 2, one chain
        v.push_back({ "copy V0", dn_launch<256, 32, 2, 2, 1, 0>(), true }); dn_attr<256, 32, 2, 2, 1, 0>(v.back().regs, v.back().smem, v.back().ctas);
        v.push_back({ "copy TIME", dn_launch<256, 32, 2, 2, 1, V_TIME>(), true }); dn_attr<256, 32, 2, 2, 1, V_TIME>(v.back().regs, v.back().smem, v.back().ctas);
        v.push_back({ "abl NOSYNC", dn_launch<256, 32, 2, 2, 1, V_NOSYNC>(), false }); dn_attr<256, 32, 2, 2, 1, V_NOSYNC>(v.back().regs, v.back().smem, v.back().ctas);
        v.push_back({ "fix CH2", dn_launch<256, 32, 2, 2, 2, 0>(), true }); dn_attr<256, 32, 2, 2, 2, 0>(v.back().regs, v.back().smem, v.back().ctas);
        v.push_back({ "fix G4", dn_launch<256, 32, 2, 4, 1, 0>(), true }); dn_attr<256, 32, 2, 4, 1, 0>(v.back().regs, v.back().smem, v.back().ctas);
        v.push_back({ "fix G4 CH2", dn_launch<256, 32, 2, 4, 2, 0>(), true }); dn_attr<256, 32, 2, 4, 2, 0>(v.back().regs, v.back().smem, v.back().ctas);
        v.push_back({ "alt 128x32 SB2", dn_launch<128, 32, 2, 2, 1, 0>(), true }); dn_attr<128, 32, 2, 2, 1, 0>(v.back().regs, v.back().smem, v.back().ctas);
        v.push_back({ "alt 256x32 SB4", dn_launch<256, 32, 4, 2, 1, 0>(), true }); dn_attr<256, 32, 4, 2, 1, 0>(v.back().regs, v.back().smem, v.back().ctas);
        v.push_back({ "fix DUFF", dn_launch<256, 32, 2, 2, 1, V_DUFF>(), true }); dn_attr<256, 32, 2, 2, 1, V_DUFF>(v.back().regs, v.back().smem, v.back().ctas);
        v.push_back({ "fix DYT", dn_launch<256, 32, 2, 2, 1, V_DYT>(), true }); dn_attr<256, 32, 2, 2, 1, V_DYT>(v.back().regs, v.back().smem, v.back().ctas);
        v.push_back({ "fix DUFF+DYT", dn_launch<256, 32, 2, 2, 1, V_DUFF | V_DYT>(), true }); dn_attr<256, 32, 2, 2, 1, V_DUFF | V_DYT>(v.back().regs, v.back().smem, v.back().ctas);
        v.push_back({ "fix DUFF+DYT LB2", dn_launch<256, 32, 2, 2, 1, V_DUFF | V_DYT | V_LB2>(), true }); dn_attr<256, 32, 2, 2, 1, V_DUFF | V_DYT | V_LB2>(v.back().regs, v.back().smem, v.back().ctas);
        v.push_back({ "fix G4 LB2", dn_launch<256, 32, 2, 4, 1, V_LB2>(), true }); dn_attr<256, 32, 2, 4, 1, V_LB2>(v.back().regs, v.back().smem, v.back().ctas);
        return v;
    };
    // STRATA_WS_V2 prototypes (moe_ws_v2.cuh): the 32-token tier, every arm must memcmp equal to production
    auto v2_gu_arms = [&](int fmt) {
        std::vector<Arm> v;
        using namespace v2n;
#define V2GU(NAME, CH, FMT, VV) \
        v.push_back({ NAME, v2gu_launch<32, CH, 4, FMT, VV>(), true }); v2gu_attr<32, CH, 4, FMT, VV>(v.back().regs, v.back().smem, v.back().ctas);
#define V2GUX(NAME, CH, FMT, VV) \
        v.push_back({ NAME, v2gu_launch<32, CH, 4, FMT, VV>(), ((VV) & PTIME) != 0 }); v2gu_attr<32, CH, 4, FMT, VV>(v.back().regs, v.back().smem, v.back().ctas);
#define V2GU16(NAME, CH, FMT, VV) \
        v.push_back({ NAME, v2gu_launch<16, CH, 2, FMT, VV>(), true }); v2gu_attr<16, CH, 2, FMT, VV>(v.back().regs, v.back().smem, v.back().ctas);
        if (fmt == nib::kIQ2_S) {
            constexpr int F = DUFF;   // FAST's pick
            V2GU("v2 R1 staged (=FAST)", 1, nib::kIQ2_S, F)
            V2GU("v2 R1 CD LB4", 1, nib::kIQ2_S, F | CD | LB4)
            V2GU("v2 R1 CD LB3", 1, nib::kIQ2_S, F | CD | LB3)
            V2GU("v2 R2 CD LB3", 1, nib::kIQ2_S, F | R2 | CD | LB3)
            V2GU("v2 R2 CD LB3 TAIL DYT", 1, nib::kIQ2_S, F | TAIL | DYT | R2 | CD | LB3)
            V2GU("v2 R2 CD LB2", 1, nib::kIQ2_S, F | R2 | CD | LB2)
            V2GUX("v2 R2 TIME", 1, nib::kIQ2_S, F | R2 | CD | LB3 | PTIME)
            V2GU("v2 R2 QG CD LB3", 1, nib::kIQ2_S, F | QG | R2 | CD | LB3)
            V2GU("v2 R2 QG CD LB3 TAIL DYT", 1, nib::kIQ2_S, F | TAIL | DYT | QG | R2 | CD | LB3)
            V2GU("v2 R1 QG CD LB4", 1, nib::kIQ2_S, F | QG | CD | LB4)
            V2GUX("v2 R2 QG TIME", 1, nib::kIQ2_S, F | QG | R2 | CD | LB3 | PTIME)
            V2GUX("v2 R2 abl NODEC", 1, nib::kIQ2_S, F | R2 | CD | LB3 | PNODEC)
            V2GUX("v2 R2 abl NOSTAGE", 1, nib::kIQ2_S, F | R2 | CD | LB3 | PNOSTAGE)
            V2GU16("v2 t16 R2 CD LB3", 1, nib::kIQ2_S, TAIL | DUFF | DYT | R2 | CD | LB3)
            V2GU16("v2 t16 R2 CD LB4", 1, nib::kIQ2_S, TAIL | DUFF | DYT | R2 | CD | LB4)
        } else if (fmt == nib::kIQ3_XXS) {
            constexpr int F = TAIL | DYT;
            V2GU("v2 R1 staged (=FAST)", 1, nib::kIQ3_XXS, F)
            V2GU("v2 R1 CD LB4", 1, nib::kIQ3_XXS, F | CD | LB4)
            V2GU("v2 R1 CD LB3", 1, nib::kIQ3_XXS, F | CD | LB3)
            V2GU("v2 R2 CD LB3", 1, nib::kIQ3_XXS, F | R2 | CD | LB3)
            V2GU("v2 R2 CD LB3 DUFF", 1, nib::kIQ3_XXS, F | DUFF | R2 | CD | LB3)
            V2GU("v2 R2 CD LB2", 1, nib::kIQ3_XXS, F | R2 | CD | LB2)
            V2GUX("v2 R2 TIME", 1, nib::kIQ3_XXS, F | R2 | CD | LB3 | PTIME)
            V2GU("v2 R2 QG CD LB3", 1, nib::kIQ3_XXS, F | QG | R2 | CD | LB3)
            V2GU("v2 R2 QG CD LB3 DUFF", 1, nib::kIQ3_XXS, F | DUFF | QG | R2 | CD | LB3)
            V2GU("v2 R1 QG CD LB4", 1, nib::kIQ3_XXS, F | QG | CD | LB4)
            V2GUX("v2 R2 QG TIME", 1, nib::kIQ3_XXS, F | QG | R2 | CD | LB3 | PTIME)
            V2GUX("v2 R2 abl NODEC", 1, nib::kIQ3_XXS, F | R2 | CD | LB3 | PNODEC)
            V2GUX("v2 R2 abl NOSTAGE", 1, nib::kIQ3_XXS, F | R2 | CD | LB3 | PNOSTAGE)
            V2GU16("v2 t16 R2 CD LB3", 1, nib::kIQ3_XXS, TAIL | DUFF | DYT | R2 | CD | LB3)
            V2GU16("v2 t16 R2 CD LB4", 1, nib::kIQ3_XXS, TAIL | DUFF | DYT | R2 | CD | LB4)
        } else if (fmt == nib::kIQ3_S) {
            constexpr int F = TAIL | DUFF | DECF | DYT;
            V2GU("v2 R1 staged (=FAST)", 2, nib::kIQ3_S, F)
            V2GU("v2 R1 CD LB4", 2, nib::kIQ3_S, F | CD | LB4)
            V2GU("v2 R1 CD LB3", 2, nib::kIQ3_S, F | CD | LB3)
            V2GU("v2 R2 CD LB3 CH1", 1, nib::kIQ3_S, F | R2 | CD | LB3)
            V2GU("v2 R2 CD LB3 CH2", 2, nib::kIQ3_S, F | R2 | CD | LB3)
            V2GU("v2 R2 CD LB2 CH1", 1, nib::kIQ3_S, F | R2 | CD | LB2)
            V2GUX("v2 R2 TIME", 1, nib::kIQ3_S, F | R2 | CD | LB3 | PTIME)
            V2GU("v2 R2 QG CD LB3", 1, nib::kIQ3_S, F | QG | R2 | CD | LB3)
            V2GU("v2 R1 QG CD LB4", 1, nib::kIQ3_S, F | QG | CD | LB4)
            V2GU("v2 R1 QG CD LB3", 1, nib::kIQ3_S, F | QG | CD | LB3)
            V2GUX("v2 R2 QG TIME", 1, nib::kIQ3_S, F | QG | R2 | CD | LB3 | PTIME)
            V2GUX("v2 R2 abl NODEC", 1, nib::kIQ3_S, F | R2 | CD | LB3 | PNODEC)
            V2GUX("v2 R2 abl NOSTAGE", 1, nib::kIQ3_S, F | R2 | CD | LB3 | PNOSTAGE)
            V2GU16("v2 t16 R2 CD LB3", 1, nib::kIQ3_S, TAIL | DUFF | DECF | DYT | R2 | CD | LB3)
            V2GU16("v2 t16 R2 CD LB4", 1, nib::kIQ3_S, TAIL | DUFF | DECF | DYT | R2 | CD | LB4)
        }
#undef V2GU
#undef V2GU16
#undef V2GUX
        return v;
    };
    auto v2_dn_arms = [&]() {
        std::vector<Arm> v;
        using namespace v2n;
#define V2DN(NAME, VV) \
        v.push_back({ NAME, v2dn_launch<32, VV>(), true }); v2dn_attr<32, VV>(v.back().regs, v.back().smem, v.back().ctas);
        V2DN("v2 dn R1 staged LB4", LB4)
        V2DN("v2 dn R1 CD LB4", CD | LB4)
        V2DN("v2 dn R2 staged LB3", R2 | LB3)
        V2DN("v2 dn R2 CD LB3", R2 | CD | LB3)
        V2DN("v2 dn R2 CD LB2", R2 | CD | LB2)
        v.push_back({ "v2 dn t16 R2 CD LB3", v2dn_launch<16, R2 | CD | LB3>(), true }); v2dn_attr<16, R2 | CD | LB3>(v.back().regs, v.back().smem, v.back().ctas);
        v.push_back({ "v2 dn t16 R2 CD LB4", v2dn_launch<16, R2 | CD | LB4>(), true }); v2dn_attr<16, R2 | CD | LB4>(v.back().regs, v.back().smem, v.back().ctas);
#undef V2DN
        return v;
    };

    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0));
    CK(cudaEventCreate(&e1));
    auto time_once = [&](const std::function<void()>& f) {
        CK(cudaEventRecord(e0));
        f();
        CK(cudaEventRecord(e1));
        CK(cudaEventSynchronize(e1));
        CK(cudaGetLastError());
        float ms = 0;
        CK(cudaEventElapsedTime(&ms, e0, e1));
        return (double) ms;
    };

    long long fails = 0;
    for (int l : layers) {
        Layer Y;
        if (!load_layer(*model, l, Y)) {
            std::printf("layer %d: not found / no native format\n", l);
            ++fails;
            continue;
        }
        const int gt = Y.f.gu_type, dt = Y.f.d_type;
        std::printf("\n##### layer %d: gate/up %s (%d), down %d; blob %zu B, 4-bit slot %llu B (%s)\n", l, nib::fmt_name(gt), gt,
                    dt, Y.f.bytes, (unsigned long long) Y.V.slot_bytes, Y.V.on ? "converted" : "NOT converted");
        int64_t* dxp = nullptr;
        int64_t* dxn = nullptr;
        CK(cudaMalloc(&dxp, kNE * 8));
        CK(cudaMalloc(&dxn, kNE * 8));
        std::vector<Arm> GA = (Y.V.on && !prod_only) ? (v2_only ? v2_gu_arms(gt) : gu_arms(gt)) : std::vector<Arm>{};
        std::vector<Arm> DA = (dt == GGML_TYPE_IQ4_NL && !prod_only) ? (v2_only ? v2_dn_arms() : dn_arms()) : std::vector<Arm>{};
        if (v2_round2) {   // round 2: the one-row kernels, the tier crossover, two-row references and phases only
            const char* keep[] = { "v2 R1 staged (=FAST)", "v2 R1 CD LB4", "v2 R1 CD LB3", "v2 R2 CD LB3", "v2 R2 CD LB3 CH1",
                                   "v2 R2 QG CD LB3", "v2 R2 TIME", "v2 R2 QG TIME", "v2 R2 abl NODEC", "v2 R2 abl NOSTAGE",
                                   "v2 t16 R2 CD LB4", "v2 R2 CD LB2", "v2 R2 CD LB2 CH1" };
            std::vector<Arm> g2;
            for (auto& a : GA)
                for (const char* k : keep)
                    if (a.name == k) g2.push_back(a);
            GA.swap(g2);
            DA.clear();
        }
        for (auto& a : GA)
            std::printf("ATTR  gate/up %-18s regs %3d  smem %5d B  %d CTAs/SM\n", a.name.c_str(), a.regs, a.smem, a.ctas);
        for (auto& a : DA)
            std::printf("ATTR  down    %-18s regs %3d  smem %5d B  %d CTAs/SM\n", a.name.c_str(), a.regs, a.smem, a.ctas);
        for (int T : ts)
            for (const std::string& mdl : models) {
                const std::vector<int32_t> cnt = make_counts(mdl, T, l);
                std::vector<int32_t> order;
                int R = 0;
                for (int e = 0; e < kNE; ++e)
                    if (cnt[e] > 0) { order.push_back(e); R += cnt[e]; }
                std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return cnt[a] != cnt[b] ? cnt[a] < cnt[b] : a < b; });
                const int n = (int) order.size();
                std::vector<int32_t> bounds(n + 1, 0);
                for (int j = 0; j < n; ++j) bounds[j + 1] = bounds[j] + cnt[order[j]];
                std::vector<ws::Chunk> ch(ws::max_chunks(R, n));
                const size_t nch = ws::chunks(order.data(), (size_t) n, cnt.data(), bounds.data(), ch.data());
                const double u = (double) R / n;
                // host accounting of the production tiers: gate/up TPP 32 / TG 4 (x10 pieces), down TPP 32 / G 2 (x10)
                auto acct = [&](int TPP, int TG, long long& passes, long long& slots) {
                    passes = slots = 0;
                    for (size_t i = 0; i < nch; ++i)
                        for (int p0 = 0; p0 < ch[i].rows; p0 += TPP) {
                            const int nt = std::min(TPP, ch[i].rows - p0);
                            ++passes;
                            slots += (nt + TG - 1) / TG * TG;
                        }
                };
                long long gp, gs, dp, ds;
                acct(32, 4, gp, gs);
                acct(32, 2, dp, ds);
                int cmax = 0, c1 = 0, c32 = 0;
                for (int e = 0; e < kNE; ++e) { cmax = std::max(cmax, cnt[e]); c1 += cnt[e] == 1; c32 += cnt[e] > 0 && cnt[e] <= 32; }
                std::printf("\n=== L%d %s %s T %d: rows %d, routed %d (u = %.2f, max %d, %d with 1 row, %d with <= 32), chunks %zu; "
                            "gate/up tier 32/4: %lld passes (mean nt %.1f), token slots %lld = x%.3f of rows; down 32/2: x%.3f\n",
                            l, nib::fmt_name(gt), mdl.c_str(), T, R, n, u, cmax, c1, c32, nch, gp, (double) R / gp, gs,
                            (double) gs / R, (double) ds / R);
                // device tables
                std::vector<int64_t> hxp(n), hxn(n);
                for (int j = 0; j < n; ++j) {
                    hxp[j] = (int64_t) order[j] * (int64_t) Y.f.bytes;
                    hxn[j] = (int64_t) order[j] * (int64_t) Y.V.slot_bytes;
                }
                CK(cudaMemcpy(dxp, hxp.data(), (size_t) n * 8, cudaMemcpyHostToDevice));
                CK(cudaMemcpy(dxn, hxn.data(), (size_t) n * 8, cudaMemcpyHostToDevice));
                ws::Chunk* dch = nullptr;
                CK(cudaMalloc(&dch, nch * sizeof(ws::Chunk)));
                CK(cudaMemcpy(dch, ch.data(), nch * sizeof(ws::Chunk), cudaMemcpyHostToDevice));
                float* X = nullptr;
                void* xq = nullptr;
                CK(cudaMalloc(&X, (size_t) R * kEmbd * 4));
                CK(cudaMalloc(&xq, mmq::q8_bytes(R, kEmbd)));
                fill_x<<<(unsigned) (((long long) R * kEmbd + 255) / 256), 256>>>(X, (long long) R * kEmbd, 0xA11CEull + (uint64_t) R + (uint64_t) gt);
                CK(cudaGetLastError());
                mmq::quantize(X, nullptr, xq, gt, kEmbd, kEmbd, R, nullptr);
                CK(cudaDeviceSynchronize());
                CK(cudaFree(X));
                const size_t gub = (size_t) R * 1280 * 4, dmb = (size_t) R * kEmbd * 4, hqb = mmq::q8_bytes(R, kFf);
                float *GUp = nullptr, *GUg = nullptr, *GUx = nullptr, *DMp = nullptr, *DMx = nullptr;
                void* HQ = nullptr;
                CK(cudaMalloc(&GUp, gub));
                CK(cudaMalloc(&GUg, gub));
                CK(cudaMalloc(&GUx, gub));
                CK(cudaMalloc(&DMp, dmb));
                CK(cudaMalloc(&DMx, dmb));
                CK(cudaMalloc(&HQ, hqb));
                CK(cudaMemset(HQ, 0xFF, hqb));
                Rec* drec = nullptr;
                const size_t maxcta = nch * 20;
                CK(cudaMalloc(&drec, maxcta * sizeof(Rec)));
                {   // the v2 TIME arm's records (moe_ws_v2.cuh g_v2rec)
                    void* p = drec;
                    CK(cudaMemcpyToSymbol(wsprof::v2n::g_v2rec, &p, sizeof p));
                }
                ws::Layer Ln, Lg;   // production: 4-bit slots / GGUF blobs
                Lg.gu_type = gt; Lg.d_type = dt; Lg.w = Y.raw; Lg.xoff = dxp; Lg.down_off = (int64_t) Y.f.down_off;
                Lg.chunks = dch; Lg.nchunks = (int64_t) nch; Lg.rows = R; Lg.n = n; Lg.xq = xq; Lg.gu = GUg; Lg.hq = HQ; Lg.dm = DMp;
                Ln = Lg;
                if (Y.V.on) {
                    Ln.w = Y.slots; Ln.xoff = dxn; Ln.down_off = (int64_t) nib::kDownOff; Ln.gu = GUp; Ln.gu_nib = true;
                }
                // reference outputs: production 4-bit gate/up (or GGUF where not converted), swiglu_quant, down - the
                // default kernels (STRATA_WS_FAST off)
                ws::set_fast(0);
                CK(cudaMemset(GUp, 0xFF, gub));
                CK(cudaMemset(GUg, 0xFF, gub));
                CK(cudaMemset(DMp, 0xFF, dmb));
                const bool gu_ok_n = Y.V.on ? ws::gate_up(Ln, nullptr) : false;
                const bool gu_ok_g = ws::gate_up(Lg, nullptr);
                mmq::swiglu_quant(Y.V.on ? GUp : GUg, HQ, R, false, nullptr);
                ws::down(Lg, nullptr);
                CK(cudaDeviceSynchronize());
                CK(cudaGetLastError());
                if (Y.V.on && !gu_ok_n) { std::printf("FAIL ws::gate_up declined the 4-bit layer\n"); ++fails; }
                if (!gu_ok_g) std::printf("note: ws::gate_up declined the GGUF layer (not timed)\n");
                std::vector<uint8_t> refgu(gub), refdm(dmb), tmp(std::max(gub, dmb));
                CK(cudaMemcpy(refgu.data(), Y.V.on ? GUp : GUg, gub, cudaMemcpyDeviceToHost));
                CK(cudaMemcpy(refdm.data(), DMp, dmb, cudaMemcpyDeviceToHost));
                if (Y.V.on && gu_ok_g) {
                    CK(cudaMemcpy(tmp.data(), GUg, gub, cudaMemcpyDeviceToHost));
                    const bool same = std::memcmp(tmp.data(), refgu.data(), gub) == 0;
                    std::printf("CHECK production gate/up 4-bit vs GGUF: %s\n", same ? "memcmp equal" : "DIFFER");
                    if (!same) ++fails;
                }
                // STRATA_WS_FAST=1 (ws::set_fast(1)): the 32-token tier's fast kernels - gate/up on the 4-bit slots at
                // u >= 25 (IQ2_S) / 18 (IQ3_XXS, IQ3_S), IQ4_NL down at u >= 20 - against the default ones, byte for byte
                // (below those u the same default kernels run either way)
                {
                    const bool t32_gu = gt == nib::kIQ2_S ? u >= 25.0 : u >= 18.0, t32_dn = dt == GGML_TYPE_IQ4_NL && u >= 20.0;
                    if (Y.V.on && gu_ok_n) {
                        ws::Layer Lf = Ln;
                        Lf.gu = GUx;
                        CK(cudaMemset(GUx, 0xFF, gub));
                        ws::set_fast(1);
                        const bool ok = ws::gate_up(Lf, nullptr);
                        ws::set_fast(0);
                        CK(cudaDeviceSynchronize());
                        CK(cudaGetLastError());
                        CK(cudaMemcpy(tmp.data(), GUx, gub, cudaMemcpyDeviceToHost));
                        const bool same = ok && std::memcmp(tmp.data(), refgu.data(), gub) == 0;
                        std::printf("CHECK production gate/up 4-bit FAST (%s) vs default: %s\n",
                                    t32_gu ? "fast kernel" : "below the 32-token tier: default kernel",
                                    same ? "memcmp equal" : "DIFFER");
                        if (!same) ++fails;
                    }
                    ws::Layer Ldf = Lg;
                    Ldf.dm = DMx;
                    CK(cudaMemset(DMx, 0xFF, dmb));
                    ws::set_fast(1);
                    ws::down(Ldf, nullptr);
                    ws::set_fast(0);
                    CK(cudaDeviceSynchronize());
                    CK(cudaGetLastError());
                    CK(cudaMemcpy(tmp.data(), DMx, dmb, cudaMemcpyDeviceToHost));
                    const bool same = std::memcmp(tmp.data(), refdm.data(), dmb) == 0;
                    std::printf("CHECK production down FAST (%s) vs default: %s\n",
                                t32_dn ? "fast kernel" : "not the IQ4_NL 32-token tier: default kernel",
                                same ? "memcmp equal" : "DIFFER");
                    if (!same) ++fails;
                }
                // STRATA_WS_V2=1 (ws::set_v2(1)): moe_ws_gu_nib.cu's V2 picks (4-bit gate/up only; the down is fast()'s
                // either way, here the default), against the default kernels
                {
                    const bool t32_gu = gt == nib::kIQ2_S ? u >= 17.0 : gt == nib::kIQ3_XXS ? u >= 9.0 : u >= 7.0, t32_dn = false;
                    if (Y.V.on && gu_ok_n) {
                        ws::Layer Lf = Ln;
                        Lf.gu = GUx;
                        CK(cudaMemset(GUx, 0xFF, gub));
                        ws::set_v2(1);
                        const bool ok = ws::gate_up(Lf, nullptr);
                        ws::set_v2(0);
                        CK(cudaDeviceSynchronize());
                        CK(cudaGetLastError());
                        CK(cudaMemcpy(tmp.data(), GUx, gub, cudaMemcpyDeviceToHost));
                        const bool same = ok && std::memcmp(tmp.data(), refgu.data(), gub) == 0;
                        std::printf("CHECK production gate/up 4-bit V2 (%s) vs default: %s\n",
                                    t32_gu ? "V2 32-token kernel" : "below the V2 tier: default kernel",
                                    same ? "memcmp equal" : "DIFFER");
                        if (!same) ++fails;
                    }
                    ws::Layer Ldf = Lg;
                    Ldf.dm = DMx;
                    CK(cudaMemset(DMx, 0xFF, dmb));
                    ws::set_v2(1);
                    ws::down(Ldf, nullptr);
                    ws::set_v2(0);
                    CK(cudaDeviceSynchronize());
                    CK(cudaGetLastError());
                    CK(cudaMemcpy(tmp.data(), DMx, dmb, cudaMemcpyDeviceToHost));
                    const bool same = std::memcmp(tmp.data(), refdm.data(), dmb) == 0;
                    std::printf("CHECK production down V2 (%s) vs default: %s\n",
                                t32_dn ? "-" : "V2 leaves the down alone: default kernel",
                                same ? "memcmp equal" : "DIFFER");
                    if (!same) ++fails;
                }
                Args A{ dch, (const char*) Ln.w, Ln.xoff, Lg.down_off, (const block_q8_1_mmq*) xq, (const block_q8_1_mmq*) HQ, R,
                        GUx, DMx, drec, (int64_t) nch };
                Args AD = A;
                AD.w = (const char*) Y.raw;
                AD.xoff = dxp;
                // parity of the copies (exact ones must memcmp equal)
                for (auto& a : GA) {
                    CK(cudaMemset(GUx, 0xFF, gub));
                    CK(cudaMemset(drec, 0, maxcta * sizeof(Rec)));
                    a.go(A);
                    CK(cudaDeviceSynchronize());
                    CK(cudaGetLastError());
                    if (!a.exact) continue;
                    CK(cudaMemcpy(tmp.data(), GUx, gub, cudaMemcpyDeviceToHost));
                    const bool same = std::memcmp(tmp.data(), refgu.data(), gub) == 0;
                    std::printf("CHECK gate/up %-18s vs production ws::gate_up (4-bit): %s\n", a.name.c_str(), same ? "memcmp equal" : "DIFFER");
                    if (!same) ++fails;
                }
                for (auto& a : DA) {
                    CK(cudaMemset(DMx, 0xFF, dmb));
                    CK(cudaMemset(drec, 0, maxcta * sizeof(Rec)));
                    a.go(AD);
                    CK(cudaDeviceSynchronize());
                    CK(cudaGetLastError());
                    if (!a.exact) continue;
                    CK(cudaMemcpy(tmp.data(), DMx, dmb, cudaMemcpyDeviceToHost));
                    const bool same = std::memcmp(tmp.data(), refdm.data(), dmb) == 0;
                    std::printf("CHECK down    %-18s vs production ws::down: %s\n", a.name.c_str(), same ? "memcmp equal" : "DIFFER");
                    if (!same) ++fails;
                }
                // timing: every arm once per round, rounds interleaved, a warm-up round first
                struct T_ { std::string name; std::function<void()> f; double macs; std::vector<double> ms; };
                std::vector<T_> arms;
                const double mac_gu = (double) R * 1280 * 2560, mac_dn = (double) R * 2560 * 640;
                if (Y.V.on) arms.push_back({ "PROD gate/up 4-bit", [&] { ws::gate_up(Ln, nullptr); }, mac_gu, {} });
                if (gu_ok_g) arms.push_back({ "PROD gate/up GGUF", [&] { ws::gate_up(Lg, nullptr); }, mac_gu, {} });
                arms.push_back({ "PROD swiglu_quant", [&] { mmq::swiglu_quant(Y.V.on ? GUp : GUg, HQ, R, false, nullptr); }, 0, {} });
                arms.push_back({ "PROD down", [&] { ws::down(Lg, nullptr); }, mac_dn, {} });
                if (Y.V.on)
                    arms.push_back({ "PROD gate/up 4-bit FAST", [&] { ws::set_fast(1); ws::gate_up(Ln, nullptr); ws::set_fast(0); },
                                     mac_gu, {} });
                arms.push_back({ "PROD down FAST", [&] { ws::set_fast(1); ws::down(Lg, nullptr); ws::set_fast(0); }, mac_dn, {} });
                if (Y.V.on)
                    arms.push_back({ "PROD gate/up 4-bit V2", [&] { ws::set_v2(1); ws::gate_up(Ln, nullptr); ws::set_v2(0); },
                                     mac_gu, {} });
                arms.push_back({ "PROD down V2", [&] { ws::set_v2(1); ws::down(Lg, nullptr); ws::set_v2(0); }, mac_dn, {} });
                for (auto& a : GA) arms.push_back({ "gu " + a.name, [&a, &A] { a.go(A); }, mac_gu, {} });
                for (auto& a : DA) arms.push_back({ "dn " + a.name, [&a, &AD] { a.go(AD); }, mac_dn, {} });
                // production again at the end of the round: position / clock drift within a round
                if (Y.V.on) arms.push_back({ "PROD gate/up 4-bit #2", [&] { ws::gate_up(Ln, nullptr); }, mac_gu, {} });
                arms.push_back({ "PROD down #2", [&] { ws::down(Lg, nullptr); }, mac_dn, {} });
                for (int rep = -1; rep < reps; ++rep)
                    for (auto& a : arms) {
                        const double ms = time_once(a.f);
                        if (rep >= 0) a.ms.push_back(ms);
                    }
                double prod_gu = 0, prod_dn = 0;
                for (auto& a : arms) {
                    if (a.name == "PROD gate/up 4-bit" || (!Y.V.on && a.name == "PROD gate/up GGUF")) prod_gu = median(a.ms);
                    if (a.name == "PROD down") prod_dn = median(a.ms);
                }
                for (auto& a : arms) {
                    const double md = median(a.ms);
                    const double ref = a.name.rfind("dn ", 0) == 0 || a.name.rfind("PROD down", 0) == 0 ? prod_dn : prod_gu;
                    std::printf("TIME  L%d %-7s %-6s T %d  %-26s med %8.3f ms  min %8.3f  %s  x%.3f of PROD\n", l, nib::fmt_name(gt),
                                mdl.c_str(), T, a.name.c_str(), md, *std::min_element(a.ms.begin(), a.ms.end()),
                                a.macs > 0 ? (std::to_string(a.macs / md * 1e-9).substr(0, 5) + " T MAC/s").c_str() : "             ",
                                ref > 0 && a.macs > 0 ? md / ref : 0.0);
                }
                // per-CTA records of the TIME copies
                FILE* csv = nullptr;
                if (!csvdir.empty()) {
                    const std::string fn = csvdir + "/cta_L" + std::to_string(l) + "_" + mdl + "_T" + std::to_string(T) + ".csv";
                    csv = std::fopen(fn.c_str(), "w");
                    if (csv) std::fprintf(csv, "kernel,smid,t0_ns,t1_ns,rows,passes,cyc,cycL,cycC\n");
                }
                for (int which = 0; which < 2; ++which) {
                    auto& V = which == 0 ? GA : DA;
                    for (auto& a : V) {
                        if (a.name != "copy TIME" && a.name != "v2 R2 TIME" && a.name != "v2 R2 QG TIME") continue;
                        const bool v2t = a.name != "copy TIME";
                        const size_t ncta = (size_t) nch * (v2t ? 5 : which == 0 ? wsprof::gu::PIECES : 10);
                        if (v2t) {
                            static_assert(sizeof(wsprof::v2n::Rec2) == sizeof(Rec), "the same record layout");
                            void* p = drec;
                            CK(cudaMemcpyToSymbol(wsprof::v2n::g_v2rec, &p, sizeof p));
                        }
                        CK(cudaMemset(drec, 0, maxcta * sizeof(Rec)));
                        const double ms = time_once([&] { a.go(which == 0 ? A : AD); });
                        std::vector<Rec> r(ncta);
                        CK(cudaMemcpy(r.data(), drec, ncta * sizeof(Rec), cudaMemcpyDeviceToHost));
                        const std::string tag = std::string(which == 0 ? "gu" : "dn") + " L" + std::to_string(l) + " " + mdl;
                        analyze(tag.c_str(), r, a.ctas, nsm, ms, csv);
                    }
                }
                if (csv) std::fclose(csv);
                CK(cudaFree(GUp)); CK(cudaFree(GUg)); CK(cudaFree(GUx)); CK(cudaFree(DMp)); CK(cudaFree(DMx));
                CK(cudaFree(HQ)); CK(cudaFree(xq)); CK(cudaFree(dch)); CK(cudaFree(drec));
            }
        CK(cudaFree(dxp));
        CK(cudaFree(dxn));
        CK(cudaFree(Y.raw));
        if (Y.slots) CK(cudaFree(Y.slots));
    }
    std::printf("\n%s: %lld failures\n", fails ? "FAIL" : "ALL CHECKS EQUAL", fails);
    return fails ? 1 : 0;
}
