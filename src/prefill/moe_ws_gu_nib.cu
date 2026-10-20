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

#include <cstdio>
#include <cstdlib>

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

template <typename F>
void go(F kernel, const Layer& L, cudaStream_t s) {
    kernel<<<(unsigned) (L.nchunks * PIECES), NT, 0, s>>>(L.chunks, (const char*) L.w, L.xoff, (const block_q8_1_mmq*) L.xq,
                                                         (int) L.rows, L.gu);
}

}  // namespace

// tokens per pass by u (rows per routed expert), the crossovers measured on the 4-bit kernels (one 512-expert layer of
// the Swift pack per format, lognormal counts sigma 0.5 and 1.0, a GP100 die): IQ2_S 8 below 6, 16 below 25 (four chains and
// I2F: faster here than the two kMagicI chains), then 32 in 4-token groups; IQ3_XXS 8 below 4, 16 below 18 (one chain),
// then 32 (one chain, 4-token groups); IQ3_S 8 below 4, 16 below 18, then 32 (two chains)
void gu_nib(const Layer& L, double u, cudaStream_t s) {
    if (L.nchunks <= 0) return;
    switch (L.gu_type) {
        case nb::kIQ2_S:
            if (u < 6.0) go(gu_kernel<8, 1, 2, nb::kIQ2_S>, L, s);
            else if (u < 25.0) go(gu_kernel<16, 0, 2, nb::kIQ2_S>, L, s);
            else go(gu_kernel<32, 1, 4, nb::kIQ2_S>, L, s);
            break;
        case nb::kIQ3_XXS:
            if (u < 4.0) go(gu_kernel<8, 2, 2, nb::kIQ3_XXS>, L, s);
            else if (u < 18.0) go(gu_kernel<16, 1, 2, nb::kIQ3_XXS>, L, s);
            else go(gu_kernel<32, 1, 4, nb::kIQ3_XXS>, L, s);
            break;
        case nb::kIQ3_S:
            if (u < 4.0) go(gu_kernel<8, 2, 2, nb::kIQ3_S>, L, s);
            else if (u < 18.0) go(gu_kernel<16, 2, 2, nb::kIQ3_S>, L, s);
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
