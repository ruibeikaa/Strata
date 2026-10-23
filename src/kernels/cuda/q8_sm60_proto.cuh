// src/kernels/cuda/q8_sm60_proto.cuh - STRATA_DENSE_PROTO=1 (opt-in, Pascal GP100 decode): q8sm60's Q8_0 GEMV
// (q8_sm60.cuh) with the same per-row arithmetic and three changes to WHEN and WHERE things happen:
//
//   * loads first: a lane issues all of its row's 16-byte weight chunks (CPL of them) and their fp16 scales before it
//     uses the first one (q8sm60 loads one chunk and uses it at once, so a warp has 1-2 requests in flight);
//   * the q8_1 quantize fused into the prologue: the workgroup reads the window's fp32 activations and quantizes them
//     itself into shared memory - the same int8 values and the same scale (fp32 of the fp16 d) that
//     native_quantize_q8_1_kernel writes and q8sm60 stages, so the dots are the same integers and the fp32 chain is
//     unchanged.  The block's sum (ds.y) is never read by a Q8_0 dot, so it is not computed.  This must be compiled
//     with the quantize kernel's flags (native_mmvq.cu: --use_fast_math) so amax / 127 and x / d are the same ops;
//   * grouped: one launch runs up to three matrices that read the same input (GDN qkv + z, QSA q + k + v), each
//     segment writing its own output with its own column stride.
//
// Per row (bit-exactness): lanes c = lane, lane + TPR, ... in the same order, the same int32 dot of 16 bytes
// (STRATA_DP4A x4, or with X16 the same products as 16-bit multiplies - integer sums are exact in any order), the same
// acc += dw * sd * float(s), the same xor tree over TPR lanes.  Which block runs which row, how many rows a block
// runs and when the loads are issued do not enter the chain.
#pragma once

namespace q8sm60p {

constexpr int MAXSEG = 3;
struct Seg {
    const int8_t* qs;   // packed planes: [n_out][n_in] int8, then [n_out][n_in / 32] fp16
    const half* d;
    float* y;           // [ncols][n_out]
    int n_out;
    int g_end;          // cumulative row groups (RPB rows each) up to and including this segment
};
struct Segs {
    Seg s[MAXSEG];
    int n;
    int groups;
};

inline std::size_t smem_bytes(int ncols, int n_in, bool x16) {
    return std::size_t(ncols) * std::size_t(n_in) * (x16 ? 2u : 1u) + std::size_t(ncols) * std::size_t(n_in / 32) * 4u;
}

__device__ __forceinline__ uint32_t prmt(uint32_t a, uint32_t sel) {
    uint32_t r;
    asm("prmt.b32 %0, %1, 0, %2;" : "=r"(r) : "r"(a), "r"(sel));   // selector msb: the byte's sign replicated
    return r;
}

// s + the dot of 4 int8 weights (one int) with 4 int16 activations (two int16 pairs): four 16x16 multiply-adds
__device__ __forceinline__ int mad16x4(uint32_t w, int xa, int xb, int s) {
    const int wl = int(prmt(w, 0x9180u));   // [w0, w1] as int16
    const int wh = int(prmt(w, 0xB3A2u));   // [w2, w3]
    s += int(short(wl)) * int(short(xa));
    s += (wl >> 16) * (xa >> 16);
    s += int(short(wh)) * int(short(xb));
    s += (wh >> 16) * (xb >> 16);
    return s;
}

__device__ __forceinline__ int pack8(int a, int b, int c, int e) {
    return (a & 0xff) | ((b & 0xff) << 8) | ((c & 0xff) << 16) | (int(uint32_t(e) << 24));
}
__device__ __forceinline__ int pack16(int a, int b) { return (a & 0xffff) | int(uint32_t(b) << 16); }

// The fused quantize: x = [NCOLS][NIN] fp32 (columns contiguous, as native_quantize_q8_1 reads them) -> sx (int8,
// or with X16 int16 in two planes: chunk c's first 8 values in plane 0, its last 8 in plane 1) and sd (per 32).
// One unit = 8 consecutive values = one thread; a q8_1 block = 4 lanes of one warp (THR is a multiple of 32, so a
// unit's lane % 4 is its place in the block).
template<int NCOLS, int THR, int NIN, bool X16>
__device__ __forceinline__ void stage_f32(const float* __restrict__ x, int8_t* sx, float* sd) {
    constexpr int UNITS = NCOLS * NIN / 8;
    constexpr int IT = (UNITS + THR - 1) / THR;
    constexpr int BATCH = IT < 4 ? IT : 4;
    constexpr int NC = NCOLS * NIN / 16;
    const float4* x4 = reinterpret_cast<const float4*>(x);
    const int tid = int(threadIdx.x);
#pragma unroll
    for (int i0 = 0; i0 < IT; i0 += BATCH) {
        float4 a[BATCH], b[BATCH];
#pragma unroll
        for (int k = 0; k < BATCH; ++k) {   // every load of the batch before the first use
            const int u = (i0 + k) * THR + tid;
            if (i0 + k < IT && u < UNITS) {
                a[k] = __ldg(x4 + 2 * u);
                b[k] = __ldg(x4 + 2 * u + 1);
            } else {
                a[k] = make_float4(0.f, 0.f, 0.f, 0.f);
                b[k] = a[k];
            }
        }
#pragma unroll
        for (int k = 0; k < BATCH; ++k) {
            if (i0 + k >= IT) break;   // uniform
            const int u = (i0 + k) * THR + tid;
            const float v[8] = {a[k].x, a[k].y, a[k].z, a[k].w, b[k].x, b[k].y, b[k].z, b[k].w};
            float amax = fabsf(v[0]);
#pragma unroll
            for (int e = 1; e < 8; ++e) amax = fmaxf(amax, fabsf(v[e]));
            amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1));
            amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 2));
            const float d = strata::kernels::q8_1_finite(amax / 127.0f);
            int q[8];
#pragma unroll
            for (int e = 0; e < 8; ++e) q[e] = strata::kernels::q8_1_quant(v[e], d, amax);
            if (u < UNITS) {
                if (X16) {
                    reinterpret_cast<int4*>(sx)[(u & 1) * NC + (u >> 1)] =
                        make_int4(pack16(q[0], q[1]), pack16(q[2], q[3]), pack16(q[4], q[5]), pack16(q[6], q[7]));
                } else {
                    reinterpret_cast<int2*>(sx)[u] = make_int2(pack8(q[0], q[1], q[2], q[3]), pack8(q[4], q[5], q[6], q[7]));
                }
                if ((u & 3) == 0) sd[u >> 2] = __half2float(__float2half(strata::kernels::q8_1_finite(d)));
            }
        }
    }
}

// q8sm60's staging (q8_1 blocks in, int8 + fp32 scale out), any THR
template<int NCOLS, int THR, int NIN>
__device__ __forceinline__ void stage_q81(const Q81Block* __restrict__ x, int8_t* sx, float* sd) {
    constexpr int NB = NIN / 32;
    for (int b = int(threadIdx.x); b < NCOLS * NB; b += THR) {
        const Q81Block* xb = x + b;
        const int* src = reinterpret_cast<const int*>(xb->qs);
        int* dst = reinterpret_cast<int*>(sx + std::size_t(b) * 32);
#pragma unroll
        for (int i = 0; i < 8; ++i) dst[i] = src[i];
        sd[b] = __low2float(xb->ds);
    }
}

// AB (timing only, outputs wrong): 1 loads but no dot, 2 no prologue, 3 no weight loads, 4 prologue only
template<int NCOLS, int TPR, int CPL, int THR, bool QF, bool X16, bool PRE, int AB>   // the order the callers use
__launch_bounds__(THR)
__global__ void kernel(const Segs S, const void* __restrict__ xin) {
    static_assert(!X16 || QF, "X16 stages from fp32 only");
    constexpr int NIN = CPL * TPR * 16;
    constexpr int NB = NIN / 32;
    constexpr int NCH = NIN / 16;
    constexpr int NC = NCOLS * NCH;
    constexpr int RPB = THR / TPR;
    extern __shared__ int4 q8p_smem[];
    int8_t* sx = reinterpret_cast<int8_t*>(q8p_smem);
    float* sd = reinterpret_cast<float*>(sx + std::size_t(NCOLS) * NIN * (X16 ? 2 : 1));
    const int lane = int(threadIdx.x) % TPR;
    const int rloc = int(threadIdx.x) / TPR;

    int4 wv[CPL];
    half dv[CPL];
    const int8_t* qs = nullptr;
    const half* dp = nullptr;
    float* y = nullptr;
    int n_out = 0, row = 0;
    auto locate = [&](int g) {
        int gs = 0;
        Seg sg = S.s[0];
        if (MAXSEG > 1 && g >= S.s[0].g_end) {
            gs = S.s[0].g_end;
            sg = S.s[1];
            if (g >= S.s[1].g_end) {
                gs = S.s[1].g_end;
                sg = S.s[2];
            }
        }
        qs = sg.qs;
        dp = sg.d;
        y = sg.y;
        n_out = sg.n_out;
        row = (g - gs) * RPB + rloc;
    };
    auto load_row = [&]() {
        const int4* w = reinterpret_cast<const int4*>(qs + std::size_t(row) * NIN);
        const half* d = dp + std::size_t(row) * NB;
#pragma unroll
        for (int i = 0; i < CPL; ++i) {
            const int c = lane + i * TPR;
            if (AB == 3 || AB == 4) {
                wv[i] = make_int4(c, c + 1, c + 2, c + 3);
                dv[i] = __float2half(1.0f);
            } else {
                wv[i] = __ldg(w + c);
                dv[i] = __ldg(d + (c >> 1));
            }
        }
    };

    int g = int(blockIdx.x);
    bool have = false;
    if (PRE && g < S.groups) {
        locate(g);
        if (row < n_out) load_row();
        have = true;
    }
    if (AB != 2) {
        if (QF) stage_f32<NCOLS, THR, NIN, X16>(static_cast<const float*>(xin), sx, sd);
        else stage_q81<NCOLS, THR, NIN>(static_cast<const Q81Block*>(xin), sx, sd);
    }
    __syncthreads();
    if (AB == 4) return;

    for (; g < S.groups; g += int(gridDim.x)) {
        if (!have) {
            locate(g);
            if (row < n_out) load_row();
        }
        have = false;
        float acc[NCOLS];
#pragma unroll
        for (int j = 0; j < NCOLS; ++j) acc[j] = 0.0f;
        if (row < n_out) {
#pragma unroll
            for (int i = 0; i < CPL; ++i) {
                const int c = lane + i * TPR;
                const float dw = __half2float(dv[i]);
                if (AB == 1) {
#pragma unroll
                    for (int j = 0; j < NCOLS; ++j) acc[j] += dw + float(wv[i].x ^ wv[i].y ^ wv[i].z ^ wv[i].w);
                } else if (X16) {
#pragma unroll
                    for (int j = 0; j < NCOLS; ++j) {
                        const int4 pa = reinterpret_cast<const int4*>(sx)[j * NCH + c];
                        const int4 pb = reinterpret_cast<const int4*>(sx)[NC + j * NCH + c];
                        int s = mad16x4(uint32_t(wv[i].x), pa.x, pa.y, 0);
                        s = mad16x4(uint32_t(wv[i].y), pa.z, pa.w, s);
                        s = mad16x4(uint32_t(wv[i].z), pb.x, pb.y, s);
                        s = mad16x4(uint32_t(wv[i].w), pb.z, pb.w, s);
                        acc[j] += dw * sd[j * NB + (c >> 1)] * float(s);
                    }
                } else {
#pragma unroll
                    for (int j = 0; j < NCOLS; ++j) {
                        const int4 xv = reinterpret_cast<const int4*>(sx + std::size_t(j) * NIN)[c];
                        int s = STRATA_DP4A(wv[i].x, xv.x, 0);
                        s = STRATA_DP4A(wv[i].y, xv.y, s);
                        s = STRATA_DP4A(wv[i].z, xv.z, s);
                        s = STRATA_DP4A(wv[i].w, xv.w, s);
                        acc[j] += dw * sd[j * NB + (c >> 1)] * float(s);
                    }
                }
            }
        }
#pragma unroll
        for (int j = 0; j < NCOLS; ++j) {   // every lane takes part, row or not
            float v = acc[j];
#pragma unroll
            for (int o = TPR / 2; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o, TPR);
            if (lane == 0 && row < n_out) y[std::size_t(j) * n_out + row] = v;
        }
    }
}

// rows per group for TPR
template<int TPR, int THR>
constexpr int rpb() { return THR / TPR; }

// Fill S's cumulative group ends for RPB-row groups; the unused segments never match.
inline void finish_segs(Segs& S, int rpb_rows) {
    int g = 0;
    for (int i = 0; i < MAXSEG; ++i) {
        if (i < S.n) {
            g += (S.s[i].n_out + rpb_rows - 1) / rpb_rows;
            S.s[i].g_end = g;
        } else {
            S.s[i] = S.s[S.n - 1];
            S.s[i].g_end = 0x7fffffff;
        }
    }
    S.groups = g;
}

} // namespace q8sm60p
