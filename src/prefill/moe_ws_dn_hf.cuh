// src/prefill/moe_ws_dn_hf.cuh - STRATA_PF_HFMA_DOWN (moe_ws.hpp): the weight-stationary down product of a layer (Q2_0 or
// IQ4_NL experts, [2560 x 640] read in place at the blob's down offset) on FP16 HFMA2 pairs, with H - the swiglu of the
// layer's gate/up rows - kept as FP16 rows instead of q8_1 blocks.  Depends on CUDA and moe_ws.hpp's Chunk only (the test
// program includes it alone); included by moe_ws_dn_hf.cu.  Everything here has internal linkage.
//
// H.  swiglu_h16_kernel: one block a row, in place in the layer's GU rows (a GU row is 1280 floats, H takes its first 1280
// bytes as 640 halves, the row's exponent its next 4 bytes; nothing else reads GU once the down is on this route).  The
// value is swiglu_kernel's expression (g / (1 + exp(-g)) * u, FP32).  A row is scaled by an exact power of two 2^-a so its
// largest |H| lands in [64, 128): the FP16 image can neither overflow nor lose its small values to the subnormal range, and
// the down kernel puts 2^a back into the FP32 sums.  A non-finite H is counted (g_dn.bad_h) and kept (the output is then
// non-finite and counted too); such a row is not scaled.
//
// WEIGHTS.  A weight is the format's small integer w (Q2_0: {-1, 0, 1, 2}; IQ4_NL: kvalues_iq4nl, |w| <= 127) times its
// block's FP16 scale d.  The kernel multiplies w by dh = fl16(d 2^-k) (a half2 of both lanes; exact for Q2_0, one FP16
// rounding for IQ4_NL) - one multiply a weight pair per step, shared by the pass's 16 or 32 tokens - and accumulates 64
// products a lane (128 K values: a step) in FP16 before widening them to the FP32 sum with 2^k.  The shift k is chosen
// per weight row and step from the step's own scales: (sum of |d| over its blocks) * wint_max * 2^-k in [2, 4), so a
// weight |dh w| < 4; with |H| < 128 a product is < 512 and a lane's 64 are < 2^15: no lane can overflow whatever the data.
// A non-finite scale makes the row's output non-finite, which is counted.
//
// WORK.  The tiles of moe_ws_gu_hf.cuh's kernels: a CTA is 128 weight rows of one expert (a row a thread) against the
// chunk's rows TPP (16 or 32) tokens a pass, the step's weights and H rows staged through shared memory, the next step's
// loads issued before this step's compute.
#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

#include "strata/prefill/moe_ws.hpp"

namespace strata::prefill::ws::dn {
namespace {

constexpr int DN_ROWS = 2560, DN_K = 640, NFF = 640, GU_LD = 1280;
constexpr int NT = 128, PIECES = DN_ROWS / NT, NSTEP = DN_K / 128;
constexpr int HROW_U4 = GU_LD * 4 / 16;      // a GU row in uint4 (320)
constexpr int HEXP_U4 = NFF * 2 / 16;        // the row's exponent int: byte 1280, uint4 80
constexpr int KMIN = -32, KMAX = 32;         // the weights' shift: 2^-k
constexpr int AMIN = -100, AMAX = 120;       // H's row exponent a: H = fl16(h 2^-a) 2^a
constexpr float kWintQ20 = 2.0f, kWintNl = 127.0f;   // the largest |w| of Q2_0 ({-1, 0, 1, 2}) and IQ4_NL (kvalues_iq4nl: -127 .. 113)

struct DnState {
    unsigned bad_h;                 // H values that were not finite
    unsigned bad_y;                 // (token, weight row) outputs that were not finite
    unsigned hmax_all;              // max |H| (FP32 bit pattern)
    unsigned layer_max[64];         // per layer
    unsigned khist[KMAX - KMIN + 1];   // the weights' shift, one sample a CTA (its first pass, first step)
};
__device__ DnState g_dn;

__constant__ signed char kKv[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};   // ggml's kvalues_iq4nl

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
__device__ __forceinline__ uint32_t dup_h2(const float v) {
    const uint32_t h = (uint32_t) __half_as_ushort(__float2half_rn(v));
    return h | (h << 16);
}
__device__ __forceinline__ float h16f(const uint32_t bits16) { return __half2float(__ushort_as_half((unsigned short) bits16)); }
__device__ __forceinline__ float pow2i(const int e) { return __int_as_float((e + 127) << 23); }   // e in [-126, 127]

// the shift k with wb 2^-k in [2, 4): wb = the step's bound on |dh w| before the shift.  A non-finite wb gives KMAX (the
// outputs are non-finite then and are counted); wb = 0 (all-zero scales): 0.
__device__ __forceinline__ int step_k(const float wb) {
    if (!(wb <= 1e30f)) return KMAX;
    if (wb <= 1e-30f) return 0;
    int e;
    (void) frexpf(wb * 0.25f, &e);
    return e < KMIN ? KMIN : e > KMAX ? KMAX : e;
}

// ---- H: swiglu in place, scaled per row.  A block is one GU row (160 threads, 4 values each).
__global__ void __launch_bounds__(160) swiglu_h16_kernel(float* __restrict__ gu, const int layer) {
    static_assert(NFF == 160 * 4, "");
    __shared__ float wm_s[5];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    float* row = gu + (size_t) blockIdx.x * GU_LD;
    const float4 g = *(const float4*) (row + 4 * tid);
    const float4 u = *(const float4*) (row + NFF + 4 * tid);
    float h[4];
    h[0] = g.x / (1.0f + __expf(-g.x)) * u.x;
    h[1] = g.y / (1.0f + __expf(-g.y)) * u.y;
    h[2] = g.z / (1.0f + __expf(-g.z)) * u.z;
    h[3] = g.w / (1.0f + __expf(-g.w)) * u.w;
    float am = 0.0f;
    int bad = 0;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float a = fabsf(h[i]);
        bad += !(a <= 3.0e38f);
        am = fmaxf(am, a);   // (a NaN is dropped here and counted above)
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        am = fmaxf(am, __shfl_xor_sync(0xFFFFFFFFu, am, o));
        bad += __shfl_xor_sync(0xFFFFFFFFu, bad, o);
    }
    if (lane == 0) {
        wm_s[warp] = am;
        if (bad != 0) atomicAdd(&g_dn.bad_h, (unsigned) bad);   // (the warp's sum: every lane holds it)
    }
    __syncthreads();   // every thread has read its g / u: the stores below overwrite this row's first 1280 bytes
    float amax = wm_s[0];
#pragma unroll
    for (int i = 1; i < 5; ++i) amax = fmaxf(amax, wm_s[i]);
    int a = 0;
    if (amax > 0.0f && amax <= 3.0e38f) {
        int e;
        (void) frexpf(amax, &e);   // amax = m 2^e, m in [0.5, 1): amax 2^-(e - 7) in [64, 128)
        a = e - 7;
        a = a < AMIN ? AMIN : a > AMAX ? AMAX : a;
    }
    const float sc = pow2i(-a);
    uint2 o;
    o.x = (uint32_t) __half_as_ushort(__float2half_rn(h[0] * sc)) | ((uint32_t) __half_as_ushort(__float2half_rn(h[1] * sc)) << 16);
    o.y = (uint32_t) __half_as_ushort(__float2half_rn(h[2] * sc)) | ((uint32_t) __half_as_ushort(__float2half_rn(h[3] * sc)) << 16);
    *(uint2*) ((char*) row + 8 * tid) = o;
    if (tid == 0) {
        ((int*) row)[NFF / 2] = a;   // byte 1280
        const unsigned bits = __float_as_uint(amax);
        if (bits > *(volatile const unsigned*) &g_dn.layer_max[layer & 63]) atomicMax(&g_dn.layer_max[layer & 63], bits);
        if (bits > *(volatile const unsigned*) &g_dn.hmax_all) atomicMax(&g_dn.hmax_all, bits);
    }
}

__device__ __forceinline__ Chunk cta_work(const Chunk* __restrict__ chunks, int& r0) {
    r0 = (int) (blockIdx.x % PIECES) * NT;
    const int4 c = __ldg((const int4*) chunks + blockIdx.x / PIECES);
    return Chunk{c.x, c.y, c.z, c.w};
}

// the H piece (8 consecutive values as 4 words) as the pairs (v_i, v_{i+4}) of the Q2_0 weight words
__device__ __forceinline__ uint4 pair_i4(const uint4 a) {
    return make_uint4(prmt(a.x, a.z, 0x5410u), prmt(a.x, a.z, 0x7632u), prmt(a.y, a.w, 0x5410u), prmt(a.y, a.w, 0x7632u));
}

// the sums of the pass's tokens into the FP32 ones (64 products a lane), scaled by 2^k
template <int TPP>
__device__ __forceinline__ void flush(uint32_t (&hacc)[TPP], float (&acc)[TPP], const float fd, const int nt) {
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
// one 32-value item against the pass's tokens: 16 weight words, 16 H words a token
template <int TPP>
__device__ __forceinline__ void item_fma(const uint32_t (&wv)[16], const uint4* __restrict__ act_s, const int s, const int nt,
                                         uint32_t (&hacc)[TPP]) {
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

// the pass's outputs: acc * 2^a of the token's row, the non-finite ones counted
template <int TPP>
__device__ __forceinline__ int store_out(const float (&acc)[TPP], const float* __restrict__ ex_s, float* __restrict__ dst,
                                         const int trow0, const int nt, const int r0) {
    int bad = 0;
#pragma unroll
    for (int t = 0; t < TPP; ++t)
        if (t < nt) {
            const float v = acc[t] * ex_s[t];
            dst[(size_t) (trow0 + t) * DN_ROWS + r0 + threadIdx.x] = v;
            bad += !(fabsf(v) <= 3.0e38f);
        }
    return bad;
}

// ---- Q2_0: 18-byte blocks {f16 d; 16 bytes of 2-bit codes (value j at bit 2 (j % 4) of byte j / 4; the code - 1 times d)}
// of 64 values, 180-byte rows; a step is 128 values = 2 blocks = 36 bytes of a row = 9 words (the codes 2 bytes into a
// word: a funnel shift).  The code -> FP16 table is moe_ws_down.cu's: prmt on the high bytes of {-1, 0, 1, 2}.
namespace q20 {
constexpr int WPR = 9, ROWB = 180;

// the 16 values of a 32-bit word of codes (value i at bits 2i) as FP16 pairs (v0,v4) (v1,v5) (v2,v6) (v3,v7), the first
// eight (the low half) to wa and the next eight to wb
__device__ __forceinline__ void dec2_h16(const uint32_t x2, uint32_t (&wa)[4], uint32_t (&wb)[4]) {
    constexpr uint32_t H = 0x3C40BC00u;   // bytes 00 BC 40 3C = fp16 high bytes of {0, -1, +2, +1}[c ^ 1] = {-1,0,1,2}[c]
    const uint32_t x = x2 ^ 0x55555555u;
    const uint32_t m0 = (x << 4) & 0x30303030u, m1 = (x << 2) & 0x30303030u, m2 = x & 0x30303030u, m3 = (x >> 2) & 0x30303030u;
    wa[0] = prmt(H, 0u, m0);
    wa[1] = prmt(H, 0u, m1);
    wa[2] = prmt(H, 0u, m2);
    wa[3] = prmt(H, 0u, m3);
    wb[0] = prmt(H, 0u, m0 >> 16);
    wb[1] = prmt(H, 0u, m1 >> 16);
    wb[2] = prmt(H, 0u, m2 >> 16);
    wb[3] = prmt(H, 0u, m3 >> 16);
}

template <int TPP>
__global__ void __launch_bounds__(NT, 3) dn_kernel(const Chunk* __restrict__ chunks, const char* __restrict__ w,
                                                   const int64_t* __restrict__ xoff, const int64_t xadd,
                                                   const uint4* __restrict__ hb, float* __restrict__ dst) {
    static_assert(TPP == 16 || TPP == 32, "");
    constexpr int AI = TPP * 16 / NT;   // 16-byte H pieces a thread stages a step
    __shared__ int wsm[NT * WPR];
    __shared__ __align__(16) uint4 act_s[TPP * 16];   // [token][16 pieces]: piece 4 s + m = 8 values (group m of item s) as the pairs (v_i, v_{i+4})
    __shared__ float ex_s[TPP];                        // 2^a of the pass's tokens (0 past the chunk)
    const int tid = threadIdx.x;
    int r0;
    const Chunk c = cta_work(chunks, r0);
    const char* __restrict__ W = w + xoff[c.j] + xadd + (size_t) r0 * ROWB;
    int woff[WPR];   // word tid + NT i of the step's NT x 9 words: row idx / 9, word idx % 9
#pragma unroll
    for (int i = 0; i < WPR; ++i) {
        const int idx = tid + i * NT, r = idx / WPR;
        woff[i] = r * ROWB + 4 * (idx - r * WPR);
    }
    const int piece = tid & 15, trow = tid >> 4;
    int wpre[WPR];
    uint4 apre[AI];
    auto load_w = [&](const int st) {
        const char* __restrict__ Ws = W + st * 36;
#pragma unroll
        for (int i = 0; i < WPR; ++i) wpre[i] = __ldg((const int*) (Ws + woff[i]));
    };
    auto store_w = [&]() {
#pragma unroll
        for (int i = 0; i < WPR; ++i) wsm[tid + i * NT] = wpre[i];
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
        float acc[TPP];
        uint32_t hacc[TPP];
#pragma unroll
        for (int t = 0; t < TPP; ++t) { acc[t] = 0.0f; hacc[t] = 0u; }
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
            // the row's 9 step words: block 0 = d0 | qs0 at bytes 0..17, block 1 = d1 in the top half of word 4, qs1 = words 5..8
            const uint32_t* cw = (const uint32_t*) (wsm + tid * WPR);
            const float d0 = h16f(cw[0] & 0xFFFFu), d1 = h16f(cw[4] >> 16);
            const int k = step_k(kWintQ20 * (fabsf(d0) + fabsf(d1)));
            const float kdn = pow2i(-k), fd = pow2i(k);
            const uint32_t dh0 = dup_h2(d0 * kdn), dh1 = dup_h2(d1 * kdn);
            if (tid == 0 && p0 == 0 && st == 0) atomicAdd(&g_dn.khist[k - KMIN], 1u);
#pragma unroll 1
            for (int b = 0; b < 4; ++b) {   // 32-value item: Q2_0 block b / 2, half b % 2
                const int pq = b >> 1, h = b & 1;
                const int base = pq ? 5 + 2 * h : 2 * h, sh = pq ? 0 : 16;
                const uint32_t c0 = cw[base], c1 = cw[base + 1], c2 = cw[min(base + 2, WPR - 1)];
                const uint32_t dh = pq ? dh1 : dh0;
                uint32_t wv[16], wa[4], wb[4];
                dec2_h16(__funnelshift_r(c0, c1, sh), wa, wb);   // codes 0..15 of the half
#pragma unroll
                for (int j = 0; j < 4; ++j) { wv[j] = hmul2(wa[j], dh); wv[4 + j] = hmul2(wb[j], dh); }
                dec2_h16(__funnelshift_r(c1, c2, sh), wa, wb);   // codes 16..31
#pragma unroll
                for (int j = 0; j < 4; ++j) { wv[8 + j] = hmul2(wa[j], dh); wv[12 + j] = hmul2(wb[j], dh); }
                item_fma<TPP>(wv, act_s, b, nt, hacc);
            }
            flush<TPP>(hacc, acc, fd, nt);
        }
        bad += store_out<TPP>(acc, ex_s, dst, trow0, nt, r0);
    }
    if (bad != 0) atomicAdd(&g_dn.bad_y, (unsigned) bad);
}
}  // namespace q20

// ---- IQ4_NL: 18-byte blocks {f16 d; 16 bytes: byte j's low nibble is value j, its high nibble value j + 16, through
// kvalues_iq4nl} of 32 values, 360-byte rows; a step is 128 values = 4 blocks = 72 bytes = 18 words (a pair of blocks is
// 9 words: the even block's d the low half of word 0 and its codes 2 bytes in, the odd block's d the top half of word 4 and
// its codes words 5..8).  A code byte is looked up as the FP16 pair (value j, value j + 16), so a weight word pairs the
// H values j and j + 16 of its block.
namespace nl4 {
constexpr int WPS = 18, S = 19, ROWB = 360;

template <int TPP>
__global__ void __launch_bounds__(NT, 3) dn_kernel(const Chunk* __restrict__ chunks, const char* __restrict__ w,
                                                   const int64_t* __restrict__ xoff, const int64_t xadd,
                                                   const uint4* __restrict__ hb, float* __restrict__ dst) {
    static_assert(TPP == 16 || TPP == 32, "");
    constexpr int NTASK = 8 * TPP;          // (token, item, half of the block's 16 pairs): two pieces each
    constexpr int AI = (NTASK + NT - 1) / NT;
    __shared__ int wsm[NT * S];
    __shared__ __align__(16) uint4 act_s[TPP * 16];   // [token][16 pieces]: piece 4 s + g = H words (v_j, v_{j+16}) for j = 4 g .. 4 g + 3 of item s
    __shared__ uint32_t lut[256];                      // code byte -> (kvalues[low nibble], kvalues[high nibble]) as half2
    __shared__ float ex_s[TPP];
    const int tid = threadIdx.x;
    for (int e = tid; e < 256; e += NT) lut[e] = pack_h2((float) kKv[e & 15], (float) kKv[e >> 4]);
    int r0;
    const Chunk c = cta_work(chunks, r0);
    const char* __restrict__ W = w + xoff[c.j] + xadd + (size_t) r0 * ROWB;
    int woff[9];   // 8-byte piece tid + NT i of the step's NT x 9 pieces: row idx / 9, piece idx % 9
    int wsi[9];
#pragma unroll
    for (int i = 0; i < 9; ++i) {
        const int idx = tid + i * NT, r = idx / 9, pc = idx - r * 9;
        woff[i] = r * ROWB + 8 * pc;
        wsi[i] = r * S + 2 * pc;
    }
    int2 wpre[9];
    uint4 apre[AI][2];
    auto load_w = [&](const int st) {
        const char* __restrict__ Ws = W + st * 72;
#pragma unroll
        for (int i = 0; i < 9; ++i) wpre[i] = __ldg((const int2*) (Ws + woff[i]));
    };
    auto store_w = [&]() {
#pragma unroll
        for (int i = 0; i < 9; ++i) { wsm[wsi[i]] = wpre[i].x; wsm[wsi[i] + 1] = wpre[i].y; }
    };
    // task idx: token idx >> 3, item (idx >> 1) & 3, half idx & 1: pieces 4 item + half and 4 item + half + 2 of the H row
    auto load_a = [&](const int st, const int trow0, const int nt) {
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
    };
    auto store_a = [&]() {
#pragma unroll
        for (int i = 0; i < AI; ++i) {
            const int idx = tid + i * NT, t = idx >> 3, s = (idx >> 1) & 3, hf = idx & 1;
            if (NTASK % NT == 0 || idx < NTASK) {
                const uint4 x = apre[i][0], y = apre[i][1];   // values 8 hf .. 8 hf + 7 and 16 + 8 hf .. of item s
                uint4* d = act_s + t * 16 + 4 * s + 2 * hf;
                d[0] = make_uint4(prmt(x.x, y.x, 0x5410u), prmt(x.x, y.x, 0x7632u), prmt(x.y, y.y, 0x5410u), prmt(x.y, y.y, 0x7632u));
                d[1] = make_uint4(prmt(x.z, y.z, 0x5410u), prmt(x.z, y.z, 0x7632u), prmt(x.w, y.w, 0x5410u), prmt(x.w, y.w, 0x7632u));
            }
        }
    };
    int bad = 0;
    load_w(0);
    load_a(0, c.row0, min(TPP, c.rows));
    for (int p0 = 0; p0 < c.rows; p0 += TPP) {
        const int nt = min(TPP, c.rows - p0);
        const int trow0 = c.row0 + p0;
        float acc[TPP];
        uint32_t hacc[TPP];
#pragma unroll
        for (int t = 0; t < TPP; ++t) { acc[t] = 0.0f; hacc[t] = 0u; }
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
            const uint32_t* cw = (const uint32_t*) (wsm + tid * S);
            const float d0 = h16f(cw[0] & 0xFFFFu), d1 = h16f(cw[4] >> 16), d2 = h16f(cw[9] & 0xFFFFu), d3 = h16f(cw[13] >> 16);
            const int k = step_k(kWintNl * (fabsf(d0) + fabsf(d1) + fabsf(d2) + fabsf(d3)));
            const float kdn = pow2i(-k), fd = pow2i(k);
            const uint32_t dh[4] = {dup_h2(d0 * kdn), dup_h2(d1 * kdn), dup_h2(d2 * kdn), dup_h2(d3 * kdn)};
            if (tid == 0 && p0 == 0 && st == 0) atomicAdd(&g_dn.khist[k - KMIN], 1u);
#pragma unroll 1
            for (int b = 0; b < 4; ++b) {   // a block (item) of the step
                const uint32_t* cb = cw + 9 * (b >> 1);
                uint32_t q[4];
                if (b & 1) {
#pragma unroll
                    for (int j = 0; j < 4; ++j) q[j] = cb[5 + j];
                } else {
#pragma unroll
                    for (int j = 0; j < 4; ++j) q[j] = __funnelshift_r(cb[j], cb[j + 1], 16);
                }
                const uint32_t dhb = dh[b];
                uint32_t wv[16];
#pragma unroll
                for (int j = 0; j < 4; ++j)
#pragma unroll
                    for (int i = 0; i < 4; ++i) wv[4 * j + i] = hmul2(lut[(q[j] >> (8 * i)) & 0xFF], dhb);
                item_fma<TPP>(wv, act_s, b, nt, hacc);
            }
            flush<TPP>(hacc, acc, fd, nt);
        }
        bad += store_out<TPP>(acc, ex_s, dst, trow0, nt, r0);
    }
    if (bad != 0) atomicAdd(&g_dn.bad_y, (unsigned) bad);
}
}  // namespace nl4

// tiny device check of the constant table against the host's copy of ggml's (the host file runs it once)
__global__ void kv_check_kernel(const signed char* __restrict__ ref, int* __restrict__ bad) {
    if (threadIdx.x < 16 && kKv[threadIdx.x] != ref[threadIdx.x]) atomicAdd(bad, 1);
}

}  // namespace
}  // namespace strata::prefill::ws::dn
