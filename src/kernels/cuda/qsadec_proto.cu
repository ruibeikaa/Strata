// src/kernels/cuda/qsadec_proto.cu - see include/strata/kernels/qsadec_proto.hpp.
//
// Campaign batch 2, item C.  Everything here is bit for bit the production decode path it replaces:
//   - the scores keep block_scores_dx_kernel's per-lane products (k.x*q.x + k.y*q.y + k.z*q.z + k.w*q.w), its 32-leaf
//     tree (leaves in 5-bit-reversed order, depth first: the xor butterfly's association) and its relu sum over the
//     heads (0 + r0 + r1 + r2 + r3);
//   - the ids come from the multi-CTA top-k's integer histograms (order-free sums) and its digit rule and emit;
//   - the attention keeps attn_chunk_kernel_pre75<1, 2>'s products, reduce12 tree, softmax and ascending value fold, and
//     attn_merge_kernel's chunk order (attn_merge_v2_kernel, bitwise the same).
// Only the loads' placement, the staging and the thread-to-work mapping change.  qsadec_bench memcmps every output.
#include "strata/kernels/qsadec_proto.hpp"
#include "strata/kernels/kv_q8.hpp"
#include "strata/core/geometry.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cfloat>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace strata::kernels {

int qsadec_proto_mode() {
    static const int m = [] {
        const char* v = std::getenv("STRATA_QSADEC_PROTO");
        if (v == nullptr || *v == '\0') return 0;
        const int x = std::atoi(v);
        const int m = x >= 0 && x <= 3 ? x : 0;
        if (m != 0)
            std::fprintf(stderr, "strata qsadec: STRATA_QSADEC_PROTO=%d: decode window QSA %s on the prototype kernels "
                                 "(bit for bit the production path; select only with STRATA_SELECT_LONG=1 past its "
                                 "threshold, 1-4 rows; attention int8 K/V)\n",
                         m, m == 1 ? "select + attention" : m == 2 ? "select" : "attention");
        return m;
    }();
    return m;
}

#if !defined(__HIPCC__) && defined(STRATA_EXPERIMENTAL_SM60)
namespace {

// ---------------------------------------------------------------- select: shapes and the multi top-k's constants
constexpr int IDX_DIM = 128, IDX_HEADS = 4, R = 4;
constexpr int MT_T = 256, MT_W = MT_T / 32, MT_U = 4, MT_NMAX = 32, MT_MAXQ = 16;
constexpr int MT_ACC = MT_NMAX * 256, MT_STATE = MT_ACC + 3 * 256, MT_CNT = MT_STATE + 16;
constexpr int MT_Q_INTS = MT_CNT + 2 * MT_NMAX * MT_W + 64;   // = qsa_select.cu's (static_assert in the host code)
constexpr int TK_T = 1024, TK_PER = 33;                         // qsa_select.cu: the multi top-k past TK_T * TK_PER blocks

__device__ __forceinline__ uint32_t order_key(float s) {
    const float v = s + 0.0f;
    if (!(v == v)) return 0u;
    const uint32_t b = __float_as_uint(v);
    return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}

// ---- copy of block_scores_dx_kernel (qsa_select.cu) with ablations: ABL 1 no key loads (the tile is staged from
// registers holding constants), 2 no tree (one product per lane and head)
__device__ __forceinline__ int dx_lane(int j) { return (int) (__brev((unsigned) j) >> 27); }
constexpr int DX_NB = 32, DX_T = 32 * IDX_HEADS, DX_KS = IDX_DIM + 4;

template <int NQ, int ABL>
__global__ void __launch_bounds__(DX_T) p_dx_kernel(const float* __restrict__ pooled, const float* __restrict__ q_idx,
                                                    const int32_t* __restrict__ steps, int64_t max_blocks,
                                                    float* __restrict__ out) {
    extern __shared__ __align__(16) float dx_smem[];
    float* Ks = dx_smem;
    float* Qs = Ks + DX_NB * DX_KS;
    float* Ds = Qs + NQ * IDX_HEADS * IDX_DIM;
    __shared__ int64_t s_nbid[NQ];
    const int t = threadIdx.x, lane = t & 31, h = t >> 5;
    if (t < NQ) s_nbid[t] = steps[t * kStepCount + kStepNBid];
    __syncthreads();
    int64_t top = 0;
#pragma unroll
    for (int q = 0; q < NQ; ++q) top = s_nbid[q] > top ? s_nbid[q] : top;
    if (top > max_blocks) top = max_blocks;
    if ((int64_t) blockIdx.x * DX_NB >= top) return;
    for (int i = t; i < NQ * IDX_HEADS * IDX_DIM / 4; i += DX_T)
        reinterpret_cast<float4*>(Qs)[i] = reinterpret_cast<const float4*>(q_idx)[i];
    const float* krow = Ks + lane * DX_KS;
    constexpr int PF = DX_NB * (IDX_DIM / 4) / DX_T;
    float4 pre[PF];
    const int64_t stride = (int64_t) gridDim.x * DX_NB;
    auto fetch = [&](int64_t tb) {
#pragma unroll
        for (int i = 0; i < PF; ++i) {
            const int idx = t + i * DX_T, row = idx / (IDX_DIM / 4), c4 = idx % (IDX_DIM / 4);
            if (ABL == 1) pre[i] = make_float4((float) (row + 1) * 1e-3f, (float) c4 * 1e-3f, 0.5f, (float) (tb & 7));
            else
                pre[i] = tb + row < top ? __ldg(reinterpret_cast<const float4*>(pooled + (tb + row) * IDX_DIM) + c4)
                                        : make_float4(0.f, 0.f, 0.f, 0.f);
        }
    };
    fetch((int64_t) blockIdx.x * DX_NB);
    for (int64_t b0 = (int64_t) blockIdx.x * DX_NB; b0 < top; b0 += stride) {
#pragma unroll
        for (int i = 0; i < PF; ++i) {
            const int idx = t + i * DX_T, row = idx / (IDX_DIM / 4), c4 = idx % (IDX_DIM / 4);
            *reinterpret_cast<float4*>(Ks + row * DX_KS + c4 * 4) = pre[i];
        }
        __syncthreads();
        if (b0 + stride < top) fetch(b0 + stride);
        float half0[NQ], half1[NQ];
        if (ABL == 2) {
            const float4 k4 = *reinterpret_cast<const float4*>(krow + lane * 4 % IDX_DIM);
#pragma unroll
            for (int q = 0; q < NQ; ++q) {
                const float4 q4 = *reinterpret_cast<const float4*>(Qs + (q * IDX_HEADS + h) * IDX_DIM);
                half0[q] = k4.x * q4.x + k4.y * q4.y;
                half1[q] = k4.z * q4.z + k4.w * q4.w;
            }
        } else {
#pragma unroll 1
        for (int c = 0; c < 4; ++c) {
            float l0[NQ], l1[NQ], l2[NQ];
#pragma unroll
            for (int jj = 0; jj < 8; ++jj) {
                const int L = dx_lane(8 * c + jj);
                const float4 k4 = *reinterpret_cast<const float4*>(krow + L * 4);
#pragma unroll
                for (int q = 0; q < NQ; ++q) {
                    const float4 q4 = *reinterpret_cast<const float4*>(Qs + (q * IDX_HEADS + h) * IDX_DIM + L * 4);
                    const float d = k4.x * q4.x + k4.y * q4.y + k4.z * q4.z + k4.w * q4.w;
                    if (jj == 0 || jj == 2 || jj == 4 || jj == 6) l0[q] = d;
                    else if (jj == 1 || jj == 5) l1[q] = l0[q] + d;
                    else if (jj == 3) l2[q] = l1[q] + (l0[q] + d);
                    else {
                        const float oct = l2[q] + (l1[q] + (l0[q] + d));
                        if (c == 0) half0[q] = oct;
                        else if (c == 1) half0[q] = half0[q] + oct;
                        else if (c == 2) half1[q] = oct;
                        else half1[q] = half1[q] + oct;
                    }
                }
            }
        }
        }
#pragma unroll
        for (int q = 0; q < NQ; ++q) Ds[(q * IDX_HEADS + h) * DX_NB + lane] = half0[q] + half1[q];
        __syncthreads();
        for (int q = h; q < NQ; q += IDX_HEADS) {
            const int64_t b = b0 + lane;
            if (b >= s_nbid[q]) continue;
            float score = 0.0f;
#pragma unroll
            for (int hh = 0; hh < IDX_HEADS; ++hh) {
                const float d = Ds[(q * IDX_HEADS + hh) * DX_NB + lane];
                score += d > 0.0f ? d : 0.0f;
            }
            out[q * max_blocks + b] = score;
        }
    }
}

// copy of block_scores_tail_kernel
__global__ void __launch_bounds__(32) p_tail_kernel(const float* __restrict__ dead, const float* __restrict__ q_idx,
                                                    const int32_t* __restrict__ steps, int64_t max_blocks,
                                                    float* __restrict__ out) {
    const int64_t qi = blockIdx.x;
    const int32_t* st = steps + qi * kStepCount;
    const int64_t n_kv = st[kStepNKv], n_bid = st[kStepNBid];
    if (n_bid >= max_blocks) return;
    const int lane = threadIdx.x & 31;
    const float4 k4 = *reinterpret_cast<const float4*>(dead + lane * 4);
    const float* q = q_idx + qi * IDX_HEADS * IDX_DIM + lane * 4;
    float score = 0.0f;
#pragma unroll
    for (int h = 0; h < IDX_HEADS; ++h) {
        const float4 q4 = *reinterpret_cast<const float4*>(q + h * IDX_DIM);
        float d = k4.x * q4.x + k4.y * q4.y + k4.z * q4.z + k4.w * q4.w;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) d += __shfl_xor_sync(0xffffffffu, d, o);
        score += d > 0.0f ? d : 0.0f;
    }
    if (lane == 0) {
        if (n_kv % R != 0) score += 1e9f;
        out[qi * max_blocks + n_bid] = score;
    }
}

// ---------------------------------------------------------------- the proto scorer
// Warp w of CTA x takes the tiles of 32 blocks (x * WPC + w) + k * gridDim.x * WPC; lane l scores block b0 + l for all
// four heads and all NQ queries, from its key row read straight into registers in two halves: the leaves of octets 0
// and 2 (float4 L = 4m, 4m + 1: full 32-byte sectors), then octets 1 and 3 (L = 4m + 2, 4m + 3).  Octet c's leaf jj is
// L = 4 brev3(jj) + {0, 2, 1, 3}[c] (= brev5(8c + jj)) and its sum ((d0+d1)+(d2+d3))+((d4+d5)+(d6+d7)), the same
// association block_scores_dx_kernel builds; the head sum is (o0 + o1) + (o2 + o3) and the score 0 + relu over the heads.
// FUSE: CTA 0 also scores every query's tail block n_bid with the dead key (the same tree; block_scores_tail_kernel's
// butterfly is that association), zeroes the top-k's pass 1-3 rows, and every CTA adds its scores to the top-k's pass-0
// histogram (row 0 of the query's scratch, by digit: weight R below n_bid, the tail's cell count at n_bid).
__device__ __forceinline__ int brev3(int j) { return ((j & 1) << 2) | (j & 2) | ((j >> 2) & 1); }

template <int NQ>
__device__ __forceinline__ void r_half(const float4 (&k)[16], const float* __restrict__ Qs, int sel0, int sel1,
                                       float (&oa)[IDX_HEADS][NQ], float (&ob)[IDX_HEADS][NQ]) {
    // octet with leaf offset sel0 (k[2m]) into oa, octet with sel1 (k[2m + 1]) into ob
#pragma unroll
    for (int h = 0; h < IDX_HEADS; ++h) {
#pragma unroll
        for (int q = 0; q < NQ; ++q) {
            const float* qh = Qs + (q * IDX_HEADS + h) * IDX_DIM;
            float da[8], db[8];
#pragma unroll
            for (int jj = 0; jj < 8; ++jj) {
                const int m = brev3(jj);
                const float4 qa = *reinterpret_cast<const float4*>(qh + (4 * m + sel0) * 4);
                const float4 qb = *reinterpret_cast<const float4*>(qh + (4 * m + sel1) * 4);
                const float4 ka = k[2 * m], kb = k[2 * m + 1];
                da[jj] = ka.x * qa.x + ka.y * qa.y + ka.z * qa.z + ka.w * qa.w;
                db[jj] = kb.x * qb.x + kb.y * qb.y + kb.z * qb.z + kb.w * qb.w;
            }
            oa[h][q] = ((da[0] + da[1]) + (da[2] + da[3])) + ((da[4] + da[5]) + (da[6] + da[7]));
            ob[h][q] = ((db[0] + db[1]) + (db[2] + db[3])) + ((db[4] + db[5]) + (db[6] + db[7]));
        }
    }
}

// the scores of one key row (registers) for all queries: out[q] = 0 + relu(r_h) over h
template <int NQ>
__device__ __forceinline__ void r_row(const float4* __restrict__ kr, const float* __restrict__ Qs, float (&sc)[NQ]) {
    float4 k[16];
    float o0[IDX_HEADS][NQ], o2[IDX_HEADS][NQ], o1[IDX_HEADS][NQ], o3[IDX_HEADS][NQ];
#pragma unroll
    for (int m = 0; m < 8; ++m) { k[2 * m] = __ldg(kr + 4 * m); k[2 * m + 1] = __ldg(kr + 4 * m + 1); }
    r_half<NQ>(k, Qs, 0, 1, o0, o2);
#pragma unroll
    for (int m = 0; m < 8; ++m) { k[2 * m] = __ldg(kr + 4 * m + 2); k[2 * m + 1] = __ldg(kr + 4 * m + 3); }
    r_half<NQ>(k, Qs, 2, 3, o1, o3);
#pragma unroll
    for (int q = 0; q < NQ; ++q) {
        float score = 0.0f;
#pragma unroll
        for (int h = 0; h < IDX_HEADS; ++h) {
            const float d = (o0[h][q] + o1[h][q]) + (o2[h][q] + o3[h][q]);
            score += d > 0.0f ? d : 0.0f;
        }
        sc[q] = score;
    }
}

template <int NQ, int WPC, bool FUSE>
__global__ void __launch_bounds__(WPC * 32) r_scores_kernel(const float* __restrict__ pooled,
                                                            const float* __restrict__ dead,
                                                            const float* __restrict__ q_idx,
                                                            const int32_t* __restrict__ steps, int64_t max_blocks,
                                                            float* __restrict__ out, int32_t* __restrict__ scratch) {
    __shared__ __align__(16) float Qs[NQ * IDX_HEADS * IDX_DIM];
    __shared__ int64_t s_nbid[NQ], s_nkv[NQ];
    __shared__ int sh[FUSE ? NQ : 1][256];
    const int t = threadIdx.x, lane = t & 31, w = t >> 5;
    if (t < NQ) {
        s_nbid[t] = steps[t * kStepCount + kStepNBid];
        s_nkv[t] = steps[t * kStepCount + kStepNKv];
    }
    for (int i = t; i < NQ * IDX_HEADS * IDX_DIM / 4; i += WPC * 32)
        reinterpret_cast<float4*>(Qs)[i] = __ldg(reinterpret_cast<const float4*>(q_idx) + i);
    if (FUSE) {
        for (int i = t; i < NQ * 256; i += WPC * 32) (&sh[0][0])[i] = 0;
        if (blockIdx.x == 0)                              // the passes 1-3 rows (production: pass 0's CTA 0)
            for (int i = t; i < NQ * 3 * 256; i += WPC * 32)
                scratch[(int64_t) (i / 768) * MT_Q_INTS + MT_ACC + i % 768] = 0;
    }
    __syncthreads();
    int64_t top = 0;
#pragma unroll
    for (int q = 0; q < NQ; ++q) top = s_nbid[q] > top ? s_nbid[q] : top;
    if (top > max_blocks) top = max_blocks;
    const int64_t stride = (int64_t) gridDim.x * WPC * 32;
    for (int64_t b0 = ((int64_t) blockIdx.x * WPC + w) * 32; b0 < top; b0 += stride) {
        const int64_t b = b0 + lane;
        const int64_t bl = b < top ? b : top - 1;
        float sc[NQ];
        r_row<NQ>(reinterpret_cast<const float4*>(pooled + bl * IDX_DIM), Qs, sc);
#pragma unroll
        for (int q = 0; q < NQ; ++q) {
            const bool valid = b < s_nbid[q];
            if (valid) out[q * max_blocks + b] = sc[q];
            if (FUSE) {
                const int bin = valid ? (int) (order_key(sc[q]) >> 24) : -1;
                const unsigned pend = __ballot_sync(0xffffffffu, bin >= 0);
                if (pend) {
                    const int leader = __ffs(pend) - 1;
                    const int lb = __shfl_sync(0xffffffffu, bin, leader);
                    const unsigned same = __ballot_sync(0xffffffffu, bin == lb);
                    if (lane == leader) atomicAdd(&sh[q][lb], __popc(same) * R);
                    else if (bin >= 0 && bin != lb) atomicAdd(&sh[q][bin], R);
                }
            }
        }
    }
    if (FUSE) {
        if (blockIdx.x == 0 && w == 0 && lane < NQ) {    // the tails: lane q, query q
            float sc[NQ];
            r_row<NQ>(reinterpret_cast<const float4*>(dead), Qs, sc);
            float mine = sc[0];
#pragma unroll
            for (int q = 1; q < NQ; ++q) if (lane == q) mine = sc[q];
            const int64_t n_kv = s_nkv[lane], n_bid = s_nbid[lane];
            if (n_bid < max_blocks) {
                if (n_kv % R != 0) mine += 1e9f;
                out[lane * max_blocks + n_bid] = mine;
                const int wt = (int) (n_kv - n_bid * R);
                if (wt > 0) atomicAdd(&sh[lane][order_key(mine) >> 24], wt);
            }
        }
        __syncthreads();
        for (int i = t; i < NQ * 256; i += WPC * 32) {
            const int v = (&sh[0][0])[i];
            if (v != 0) atomicAdd(&scratch[(int64_t) (i / 256) * MT_Q_INTS + i % 256], v);
        }
    }
}

// ---------------------------------------------------------------- copies of the multi-CTA top-k (qsa_select.cu)
struct MtQuery {
    int64_t n_kv, n_bid, width, lo, hi, row0;
    int rpw, wtail;
    bool all;
};

__device__ __forceinline__ MtQuery mt_query(const int32_t* __restrict__ steps) {
    const int32_t* st = steps + (int64_t) blockIdx.y * kStepCount;
    MtQuery m;
    m.n_kv = st[kStepNKv];
    m.n_bid = st[kStepNBid];
    m.width = st[kStepWidth];
    m.all = m.n_kv <= m.width;
    m.wtail = (int) (m.n_kv - m.n_bid * R);
    const int64_t nbe = m.n_bid + (m.wtail > 0 ? 1 : 0);
    const int64_t per = (nbe + gridDim.x - 1) / gridDim.x;
    m.lo = (int64_t) blockIdx.x * per < nbe ? (int64_t) blockIdx.x * per : nbe;
    m.hi = m.lo + per < nbe ? m.lo + per : nbe;
    m.rpw = (int) (((m.hi - m.lo + 31) / 32 + MT_W - 1) / MT_W);
    m.row0 = m.lo + (int64_t) (threadIdx.x >> 5) * m.rpw * 32;
    return m;
}

__device__ __forceinline__ void mt_load(const MtQuery& m, const float* __restrict__ sc, int j0, uint32_t (&key)[MT_U]) {
    const int lane = threadIdx.x & 31;
#pragma unroll
    for (int u = 0; u < MT_U; ++u) {
        const int64_t b = m.row0 + 32 * (j0 + u) + lane;
        key[u] = (j0 + u < m.rpw && b < m.hi) ? order_key(__ldg(sc + b)) : 0u;
    }
}

template <typename V> __device__ __forceinline__ V mt_incl_scan(V v, V* s_w) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    V x = v;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        const V y = __shfl_up_sync(0xffffffffu, x, o);
        if (lane >= o) x += y;
    }
    if (lane == 31) s_w[warp] = x;
    __syncthreads();
    V off = 0;
    for (int w = 0; w < warp; ++w) off += s_w[w];
    __syncthreads();
    return x + off;
}

__device__ __forceinline__ void mt_choose(int v, int64_t width, int shift, uint32_t& prefix, int& above, int* s_w,
                                          int* s_ex, int* s_i) {
    const int t = threadIdx.x;
    if (t == 0) *s_i = 255;
    const int incl = mt_incl_scan(v, s_w);
    s_ex[t] = incl - v;
    const int64_t need = width - above;
    if (t < 255 && incl >= need && incl - v < need) *s_i = t;
    __syncthreads();
    const int i = *s_i;
    prefix |= (uint32_t) (255 - i) << shift;
    above += s_ex[i];
    __syncthreads();
}

// PASS 0..3 as production.  FUSED: pass 0 ran inside the scorer into row 0 - pass 1 reads that row alone, pass 2's CTA 0
// zeroes it for the next call.
template <int PASS, bool FUSED>
__global__ void __launch_bounds__(MT_T) p_hist_kernel(const float* __restrict__ scores, const int32_t* __restrict__ steps,
                                                      int64_t max_blocks, int32_t* __restrict__ scratch) {
    __shared__ int h[MT_W][256];
    __shared__ int s_w[MT_W], s_ex[256], s_i;
    const MtQuery m = mt_query(steps);
    int32_t* qs = scratch + (int64_t) blockIdx.y * MT_Q_INTS;
    if (FUSED && PASS == 2 && blockIdx.x == 0) qs[threadIdx.x] = 0;   // pass 1 (the only reader) is done
    if (m.all) return;
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const float* sc = scores + (int64_t) blockIdx.y * max_blocks;
    uint32_t key[MT_U];
    mt_load(m, sc, 0, key);
    for (int i = t; i < MT_W * 256; i += MT_T) (&h[0][0])[i] = 0;
    uint32_t prefix = 0;
    int above = 0;
    if (PASS == 0) {
        if (blockIdx.x == 0)
            for (int i = t; i < 3 * 256; i += MT_T) qs[MT_ACC + i] = 0;
    } else {
        int v = 0;
        if (PASS == 1) {
            if (FUSED) v = qs[255 - t];
            else {
#pragma unroll 8
                for (unsigned r = 0; r < gridDim.x; ++r) v += qs[r * 256 + 255 - t];
            }
        } else {
            prefix = (uint32_t) qs[MT_STATE + 2 * (PASS - 1)];
            above = qs[MT_STATE + 2 * (PASS - 1) + 1];
            v = qs[MT_ACC + (PASS - 2) * 256 + 255 - t];
        }
        mt_choose(v, m.width, 32 - 8 * PASS, prefix, above, s_w, s_ex, &s_i);
        if (blockIdx.x == 0 && t == 0) {
            qs[MT_STATE + 2 * PASS] = (int) prefix;
            qs[MT_STATE + 2 * PASS + 1] = above;
        }
    }
    __syncthreads();
    const int shift = 24 - 8 * PASS;
    const uint32_t hmask = PASS == 0 ? 0u : (0xffffffffu << (shift + 8));
#pragma unroll 1
    for (int j0 = 0; j0 < m.rpw; j0 += MT_U) {
      if (j0 > 0) mt_load(m, sc, j0, key);
#pragma unroll
      for (int u = 0; u < MT_U; ++u) {
        if (j0 + u >= m.rpw) break;
        const int64_t b = m.row0 + 32 * (j0 + u) + lane;
        const int bin = (b < m.hi && (key[u] & hmask) == prefix) ? (int) ((key[u] >> shift) & 255) : -1;
        const unsigned pend = __ballot_sync(0xffffffffu, bin >= 0);
        if (pend) {
            const int leader = __ffs(pend) - 1;
            const int lb = __shfl_sync(0xffffffffu, bin, leader);
            const unsigned same = __ballot_sync(0xffffffffu, bin == lb);
            if (lane == leader) atomicAdd(&h[warp][lb], __popc(same) * R);
            else if (bin >= 0 && bin != lb) atomicAdd(&h[warp][bin], R);
        }
        if (bin >= 0 && b == m.n_bid) atomicAdd(&h[warp][bin], m.wtail - R);
      }
    }
    __syncthreads();
    int sum = 0;
#pragma unroll
    for (int w = 0; w < MT_W; ++w) sum += h[w][t];
    if (PASS == 0) qs[blockIdx.x * 256 + t] = sum;
    else if (sum != 0) atomicAdd(&qs[MT_ACC + (PASS - 1) * 256 + t], sum);
}

__global__ void __launch_bounds__(MT_T) p_count_kernel(const float* __restrict__ scores, const int32_t* __restrict__ steps,
                                                       int64_t max_blocks, int32_t* __restrict__ scratch) {
    __shared__ int s_w[MT_W], s_ex[256], s_i;
    const MtQuery m = mt_query(steps);
    if (m.all) return;
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const float* sc = scores + (int64_t) blockIdx.y * max_blocks;
    uint32_t key[MT_U];
    mt_load(m, sc, 0, key);
    int32_t* qs = scratch + (int64_t) blockIdx.y * MT_Q_INTS;
    uint32_t prefix = (uint32_t) qs[MT_STATE + 6];
    int above = qs[MT_STATE + 7];
    mt_choose(qs[MT_ACC + 2 * 256 + 255 - t], m.width, 0, prefix, above, s_w, s_ex, &s_i);
    if (blockIdx.x == 0 && t == 0) {
        qs[MT_STATE + 8] = (int) prefix;
        qs[MT_STATE + 9] = above;
    }
    const uint32_t thr = prefix;
    int gt = 0, eq = 0;
#pragma unroll 1
    for (int j0 = 0; j0 < m.rpw; j0 += MT_U) {
        if (j0 > 0) mt_load(m, sc, j0, key);
#pragma unroll
        for (int u = 0; u < MT_U; ++u) {
            const int64_t b = m.row0 + 32 * (j0 + u) + lane;
            if (j0 + u >= m.rpw || b >= m.hi) continue;
            const int w = b == m.n_bid ? m.wtail : R;
            if (key[u] > thr) gt += w;
            else if (key[u] == thr) eq += w;
        }
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        gt += __shfl_xor_sync(0xffffffffu, gt, o);
        eq += __shfl_xor_sync(0xffffffffu, eq, o);
    }
    if (lane == 0) {
        qs[MT_CNT + 2 * (blockIdx.x * MT_W + warp)] = gt;
        qs[MT_CNT + 2 * (blockIdx.x * MT_W + warp) + 1] = eq;
    }
}

__global__ void __launch_bounds__(MT_T) p_emit_kernel(const float* __restrict__ scores, const int32_t* __restrict__ steps,
                                                      int64_t max_blocks, int64_t cap, int32_t* __restrict__ ids,
                                                      const int32_t* __restrict__ scratch) {
    __shared__ unsigned long long s_w[MT_W];
    const MtQuery m = mt_query(steps);
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    int32_t* out = ids + (int64_t) blockIdx.y * cap;
    if (m.all) {
        for (int64_t j = (int64_t) blockIdx.x * MT_T + t; j < m.n_kv; j += (int64_t) gridDim.x * MT_T) out[j] = (int32_t) j;
        return;
    }
    const float* sc = scores + (int64_t) blockIdx.y * max_blocks;
    uint32_t key[MT_U];
    mt_load(m, sc, 0, key);
    const int32_t* qs = scratch + (int64_t) blockIdx.y * MT_Q_INTS;
    const uint32_t thr = (uint32_t) qs[MT_STATE + 8];
    const int eq_budget = (int) (m.width - qs[MT_STATE + 9]);
    unsigned long long c = 0;
    if (t < (int) blockIdx.x * MT_W)
        c = (unsigned long long) (unsigned) qs[MT_CNT + 2 * t] | ((unsigned long long) (unsigned) qs[MT_CNT + 2 * t + 1] << 32);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) c += __shfl_xor_sync(0xffffffffu, c, o);
    if (lane == 0) s_w[warp] = c;
    __syncthreads();
    unsigned long long before = 0;
    for (int w = 0; w < MT_W; ++w) before += s_w[w];
    for (int w = 0; w < warp; ++w) {
        const int i = (int) blockIdx.x * MT_W + w;
        before += (unsigned long long) (unsigned) qs[MT_CNT + 2 * i] |
                  ((unsigned long long) (unsigned) qs[MT_CNT + 2 * i + 1] << 32);
    }
    const int gt_before = (int) (before & 0xffffffffu), eq_before = (int) (before >> 32);
    int run_sel = gt_before + (eq_before < eq_budget ? eq_before : eq_budget), run_eq = eq_before;
#pragma unroll 1
    for (int j0 = 0; j0 < m.rpw; j0 += MT_U) {
      if (j0 > 0) mt_load(m, sc, j0, key);
#pragma unroll
      for (int u = 0; u < MT_U; ++u) {
        if (j0 + u >= m.rpw) break;
        const int64_t b = m.row0 + 32 * (j0 + u) + lane;
        const int w = b < m.hi ? (b == m.n_bid ? m.wtail : R) : 0;
        const uint32_t k = key[u];
        const int my_gt = (w && k > thr) ? w : 0, my_eq = (w && k == thr) ? w : 0;
        if (__ballot_sync(0xffffffffu, (my_gt | my_eq) != 0) == 0u) continue;
        int pe = my_eq;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            const int y = __shfl_up_sync(0xffffffffu, pe, o);
            if (lane >= o) pe += y;
        }
        const int left = eq_budget - (run_eq + pe - my_eq);
        const int take = left < 0 ? 0 : (left > my_eq ? my_eq : left);
        const int my_sel = my_gt + take;
        int ps = my_sel;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            const int y = __shfl_up_sync(0xffffffffu, ps, o);
            if (lane >= o) ps += y;
        }
        int32_t* dst = out + run_sel + ps - my_sel;
        for (int cc = 0; cc < my_sel; ++cc) dst[cc] = (int32_t) (b * R + cc);
        run_eq += __shfl_sync(0xffffffffu, pe, 31);
        run_sel += __shfl_sync(0xffffffffu, ps, 31);
      }
    }
}

int sm_count() {
    static int sms[64] = {};
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 64) { cudaGetLastError(); return 0; }
    if (sms[dev] == 0 && cudaDeviceGetAttribute(&sms[dev], cudaDevAttrMultiProcessorCount, dev) != cudaSuccess) {
        cudaGetLastError();
        sms[dev] = 0;
    }
    return sms[dev];
}

int topk_ctas(int64_t max_blocks) {
    int64_t n = (max_blocks + 2047) / 2048;
    return (int) (n < 1 ? 1 : n > MT_NMAX ? MT_NMAX : n);
}

template <int NQ, int WPC, bool FUSE>
void r_launch(const float* pooled, const float* dead, const float* q_idx, const int32_t* steps, int64_t max_blocks,
              float* scores, int32_t* scratch, int ctas_per_sm, cudaStream_t st) {
    static int occ[64] = {};
    int dev = 0;
    cudaGetDevice(&dev);
    int& o = occ[dev & 63];
    if (o == 0 && (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&o, r_scores_kernel<NQ, WPC, FUSE>, WPC * 32, 0) !=
                       cudaSuccess || o < 1)) {
        cudaGetLastError();
        o = 2;
    }
    const int per = ctas_per_sm > 0 && ctas_per_sm < o ? ctas_per_sm : o;
    const int64_t tiles = (max_blocks + 31) / 32, want = (int64_t) per * sm_count();
    int64_t g = (tiles + WPC - 1) / WPC;
    g = want < g ? want : g;
    r_scores_kernel<NQ, WPC, FUSE><<<(unsigned) (g < 1 ? 1 : g), WPC * 32, 0, st>>>(pooled, dead, q_idx, steps,
                                                                                    max_blocks, scores, scratch);
}

template <int WPC, bool FUSE>
bool r_dispatch(int64_t nq, const float* pooled, const float* dead, const float* q_idx, const int32_t* steps,
                int64_t max_blocks, float* scores, int32_t* scratch, int ctas_per_sm, cudaStream_t st) {
    switch (nq) {
        case 1: r_launch<1, WPC, FUSE>(pooled, dead, q_idx, steps, max_blocks, scores, scratch, ctas_per_sm, st); return true;
        case 2: r_launch<2, WPC, FUSE>(pooled, dead, q_idx, steps, max_blocks, scores, scratch, ctas_per_sm, st); return true;
        case 3: r_launch<3, WPC, FUSE>(pooled, dead, q_idx, steps, max_blocks, scores, scratch, ctas_per_sm, st); return true;
        case 4: r_launch<4, WPC, FUSE>(pooled, dead, q_idx, steps, max_blocks, scores, scratch, ctas_per_sm, st); return true;
        default: return false;   // 5+ queries: the production path
    }
}

template <int NQ, int ABL>
void dx_launch(const float* pooled, const float* q_idx, const int32_t* steps, int64_t max_blocks, float* scores,
               cudaStream_t st) {
    constexpr int DX_NB_ = DX_NB;
    const size_t smem = (size_t) (DX_NB_ * DX_KS + NQ * IDX_HEADS * IDX_DIM + NQ * IDX_HEADS * DX_NB_) * sizeof(float);
    static int occ[64] = {};
    int dev = 0;
    cudaGetDevice(&dev);
    int& o = occ[dev & 63];
    if (o == 0 && (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&o, p_dx_kernel<NQ, ABL>, DX_T, smem) != cudaSuccess ||
                   o < 1)) {
        cudaGetLastError();
        o = 2;
    }
    const int64_t tiles = (max_blocks + DX_NB_ - 1) / DX_NB_, want = (int64_t) o * sm_count();
    p_dx_kernel<NQ, ABL><<<(unsigned) (want < tiles ? want : tiles), DX_T, smem, st>>>(pooled, q_idx, steps, max_blocks,
                                                                                      scores);
}

template <int ABL>
void dx_dispatch(int64_t nq, const float* pooled, const float* q_idx, const int32_t* steps, int64_t max_blocks,
                 float* scores, cudaStream_t st) {
    switch (nq) {
        case 1: dx_launch<1, ABL>(pooled, q_idx, steps, max_blocks, scores, st); break;
        case 2: dx_launch<2, ABL>(pooled, q_idx, steps, max_blocks, scores, st); break;
        case 3: dx_launch<3, ABL>(pooled, q_idx, steps, max_blocks, scores, st); break;
        default: dx_launch<4, ABL>(pooled, q_idx, steps, max_blocks, scores, st); break;
    }
}

// ---------------------------------------------------------------- the proto scorer, second form (dxs)
// block_scores_dx_kernel with the 32 leaves of a (block, head) split over SPLIT warps: warp w takes head w & 3 and the
// octets c of part w >> 2 (SPLIT 2: octets 0-1 = half0 or 2-3 = half1; SPLIT 4: octet w >> 2).  Each warp builds its
// part exactly as dx builds it (the same leaves, the same depth-first association) and stores it; the relu pass adds
// the parts in dx's order: half0 + half1, with half0 = oct0 + oct1 and half1 = oct2 + oct3.  The key tile is staged
// once for 4 x SPLIT warps, so a 32-block tile runs on SPLIT times the warps (dx: 4 warps a CTA, 2 CTAs per SM at T=2,
// each warp one dependent 32-leaf chain per query).
// FUSE: as r_scores_kernel - CTA 0 scores the tails (block_scores_tail_kernel's own code, one warp per query) and
// zeroes the top-k's pass 1-3 rows; every CTA adds its scores to the pass-0 histogram (row 0 of the query's scratch).
template <int NQ, int SPLIT, bool FUSE>
__global__ void __launch_bounds__(DX_T * SPLIT) dxs_kernel(const float* __restrict__ pooled,
                                                           const float* __restrict__ dead,
                                                           const float* __restrict__ q_idx,
                                                           const int32_t* __restrict__ steps, int64_t max_blocks,
                                                           float* __restrict__ out, int32_t* __restrict__ scratch) {
    constexpr int NT = DX_T * SPLIT, NW = NT / 32;
    extern __shared__ __align__(16) float dx_smem[];
    float* Ks = dx_smem;                                     // [DX_NB][DX_KS]
    float* Qs = Ks + DX_NB * DX_KS;                          // [NQ][IDX_HEADS][IDX_DIM]
    float* Ds = Qs + NQ * IDX_HEADS * IDX_DIM;               // [NQ][IDX_HEADS][SPLIT][DX_NB]
    // FUSE: the CTA's pass-0 counts [NQ][128] ints, bin 2i in the low half, 2i + 1 in the high half (the host keeps
    // a CTA's cells below 65536: dxs_launch)
    int* sh = reinterpret_cast<int*>(Ds + NQ * IDX_HEADS * SPLIT * DX_NB);
    __shared__ int64_t s_nbid[NQ], s_nkv[NQ];
    const int t = threadIdx.x, lane = t & 31, w = t >> 5, h = w & 3, part = w >> 2;
    if (t < NQ) {
        s_nbid[t] = steps[t * kStepCount + kStepNBid];
        s_nkv[t] = steps[t * kStepCount + kStepNKv];
    }
    if (FUSE)
        for (int i = t; i < NQ * 128; i += NT) sh[i] = 0;
    __syncthreads();
    int64_t top = 0;
#pragma unroll
    for (int q = 0; q < NQ; ++q) top = s_nbid[q] > top ? s_nbid[q] : top;
    if (top > max_blocks) top = max_blocks;
    if (blockIdx.x > 0 && (int64_t) blockIdx.x * DX_NB >= top) return;   // CTA-uniform; CTA 0 has the fused extras
    for (int i = t; i < NQ * IDX_HEADS * IDX_DIM / 4; i += NT)
        reinterpret_cast<float4*>(Qs)[i] = reinterpret_cast<const float4*>(q_idx)[i];
    if (FUSE && blockIdx.x == 0)
        for (int i = t; i < NQ * 3 * 256; i += NT) scratch[(int64_t) (i / 768) * MT_Q_INTS + MT_ACC + i % 768] = 0;
    const float* krow = Ks + lane * DX_KS;
    constexpr int PF = DX_NB * (IDX_DIM / 4) / NT;
    float4 pre[PF];
    const int64_t stride = (int64_t) gridDim.x * DX_NB;
    auto fetch = [&](int64_t tb) {
#pragma unroll
        for (int i = 0; i < PF; ++i) {
            const int idx = t + i * NT, row = idx / (IDX_DIM / 4), c4 = idx % (IDX_DIM / 4);
            pre[i] = tb + row < top ? __ldg(reinterpret_cast<const float4*>(pooled + (tb + row) * IDX_DIM) + c4)
                                    : make_float4(0.f, 0.f, 0.f, 0.f);
        }
    };
    fetch((int64_t) blockIdx.x * DX_NB);
    for (int64_t b0 = (int64_t) blockIdx.x * DX_NB; b0 < top; b0 += stride) {
#pragma unroll
        for (int i = 0; i < PF; ++i) {
            const int idx = t + i * NT, row = idx / (IDX_DIM / 4), c4 = idx % (IDX_DIM / 4);
            *reinterpret_cast<float4*>(Ks + row * DX_KS + c4 * 4) = pre[i];
        }
        __syncthreads();
        if (b0 + stride < top) fetch(b0 + stride);
        constexpr int NC = 4 / SPLIT;                         // octets per warp
        float acc[NQ];
#pragma unroll 1
        for (int ci = 0; ci < NC; ++ci) {
            const int c = part * NC + ci;
            float l0[NQ], l1[NQ], l2[NQ];
#pragma unroll
            for (int jj = 0; jj < 8; ++jj) {
                const int L = dx_lane(8 * c + jj);
                const float4 k4 = *reinterpret_cast<const float4*>(krow + L * 4);
#pragma unroll
                for (int q = 0; q < NQ; ++q) {
                    const float4 q4 = *reinterpret_cast<const float4*>(Qs + (q * IDX_HEADS + h) * IDX_DIM + L * 4);
                    const float d = k4.x * q4.x + k4.y * q4.y + k4.z * q4.z + k4.w * q4.w;
                    if (jj == 0 || jj == 2 || jj == 4 || jj == 6) l0[q] = d;
                    else if (jj == 1 || jj == 5) l1[q] = l0[q] + d;
                    else if (jj == 3) l2[q] = l1[q] + (l0[q] + d);
                    else {
                        const float oct = l2[q] + (l1[q] + (l0[q] + d));
                        // within the warp's part: dx's half accumulation (oct0 then + oct1; oct2 then + oct3)
                        if (NC == 1 || (c & 1) == 0) acc[q] = oct;
                        else acc[q] = acc[q] + oct;
                    }
                }
            }
            if (NC == 4 && ci == 1) {
#pragma unroll
                for (int q = 0; q < NQ; ++q) Ds[((q * IDX_HEADS + h) * SPLIT + part) * DX_NB + lane] = acc[q];   // SPLIT 1: half0 parked
            }
        }
        if (NC == 4) {
            // SPLIT 1: half0 was parked in Ds above; d = half0 + half1 in the relu pass below reads it back
#pragma unroll
            for (int q = 0; q < NQ; ++q) {
                const float h0 = Ds[((q * IDX_HEADS + h) * SPLIT + part) * DX_NB + lane];
                Ds[((q * IDX_HEADS + h) * SPLIT + part) * DX_NB + lane] = h0 + acc[q];
            }
        } else {
#pragma unroll
            for (int q = 0; q < NQ; ++q) Ds[((q * IDX_HEADS + h) * SPLIT + part) * DX_NB + lane] = acc[q];
        }
        __syncthreads();
        for (int q = w; q < NQ; q += NW) {
            const int64_t b = b0 + lane;
            const bool valid = b < s_nbid[q];
            float score = 0.0f;
#pragma unroll
            for (int hh = 0; hh < IDX_HEADS; ++hh) {
                const float* dp = Ds + (q * IDX_HEADS + hh) * SPLIT * DX_NB + lane;
                float d;
                if (SPLIT == 1) d = dp[0];
                else if (SPLIT == 2) d = dp[0] + dp[DX_NB];
                else d = (dp[0] + dp[DX_NB]) + (dp[2 * DX_NB] + dp[3 * DX_NB]);
                score += d > 0.0f ? d : 0.0f;
            }
            if (valid) out[q * max_blocks + b] = score;
            if (FUSE) {
                const int bin = valid ? (int) (order_key(score) >> 24) : -1;
                const unsigned pend = __ballot_sync(0xffffffffu, bin >= 0);
                if (pend) {
                    const int leader = __ffs(pend) - 1;
                    const int lb = __shfl_sync(0xffffffffu, bin, leader);
                    const unsigned same = __ballot_sync(0xffffffffu, bin == lb);
                    if (lane == leader) atomicAdd(&sh[q * 128 + (lb >> 1)], (__popc(same) * R) << (16 * (lb & 1)));
                    else if (bin >= 0 && bin != lb) atomicAdd(&sh[q * 128 + (bin >> 1)], R << (16 * (bin & 1)));
                }
            }
        }
    }
    if (FUSE) {
        if (blockIdx.x == 0 && w < NQ) {   // query w's tail: block_scores_tail_kernel's arithmetic
            const int64_t n_kv = s_nkv[w], n_bid = s_nbid[w];
            const float4 k4 = *reinterpret_cast<const float4*>(dead + lane * 4);
            const float* qq = q_idx + w * IDX_HEADS * IDX_DIM + lane * 4;
            float score = 0.0f;
#pragma unroll
            for (int hh = 0; hh < IDX_HEADS; ++hh) {
                const float4 q4 = *reinterpret_cast<const float4*>(qq + hh * IDX_DIM);
                float d = k4.x * q4.x + k4.y * q4.y + k4.z * q4.z + k4.w * q4.w;
#pragma unroll
                for (int o = 16; o > 0; o >>= 1) d += __shfl_xor_sync(0xffffffffu, d, o);
                score += d > 0.0f ? d : 0.0f;
            }
            if (lane == 0 && n_bid < max_blocks) {
                if (n_kv % R != 0) score += 1e9f;
                out[w * max_blocks + n_bid] = score;
                const int wt = (int) (n_kv - n_bid * R);
                const int bin = (int) (order_key(score) >> 24);
                if (wt > 0) atomicAdd(&sh[w * 128 + (bin >> 1)], wt << (16 * (bin & 1)));
            }
        }
        __syncthreads();
        for (int i = t; i < NQ * 128; i += NT) {
            const int v = sh[i], lo = v & 0xffff, hi = (int) ((unsigned) v >> 16);
            int32_t* row = scratch + (int64_t) (i / 128) * MT_Q_INTS + 2 * (i % 128);
            if (lo != 0) atomicAdd(row, lo);
            if (hi != 0) atomicAdd(row + 1, hi);
        }
    }
}

template <int NQ, int SPLIT, bool FUSE>
void dxs_launch(const float* pooled, const float* dead, const float* q_idx, const int32_t* steps, int64_t max_blocks,
                float* scores, int32_t* scratch, cudaStream_t st) {
    const size_t smem = (size_t) (DX_NB * DX_KS + NQ * IDX_HEADS * IDX_DIM + NQ * IDX_HEADS * SPLIT * DX_NB) * sizeof(float) +
                        (FUSE ? (size_t) NQ * 128 * sizeof(int) : 0);
    static int occ[64] = {};
    int dev = 0;
    cudaGetDevice(&dev);
    int& o = occ[dev & 63];
    if (o == 0 && (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&o, dxs_kernel<NQ, SPLIT, FUSE>, DX_T * SPLIT, smem) !=
                       cudaSuccess || o < 1)) {
        cudaGetLastError();
        o = 1;
    }
    const int64_t tiles = (max_blocks + DX_NB - 1) / DX_NB, want = (int64_t) o * sm_count();
    const int64_t grid = want < tiles ? want : tiles;
    dxs_kernel<NQ, SPLIT, FUSE><<<(unsigned) grid, DX_T * SPLIT, smem, st>>>(pooled, dead, q_idx, steps, max_blocks,
                                                                           scores, scratch);
}
// FUSE's 16-bit counts: a CTA's blocks (whole tiles, + the tail's 3 cells) times R stay below 65536
inline bool dxs_counts_fit(int64_t max_blocks) {
    const int64_t grid = sm_count() > 0 ? sm_count() : 1;   // at least one CTA per SM
    const int64_t tiles = (max_blocks + DX_NB - 1) / DX_NB;
    return ((tiles + grid - 1) / grid) * DX_NB * R + R < 65536;
}

template <int SPLIT, bool FUSE>
bool dxs_dispatch(int64_t nq, const float* pooled, const float* dead, const float* q_idx, const int32_t* steps,
                  int64_t max_blocks, float* scores, int32_t* scratch, cudaStream_t st) {
    switch (nq) {
        case 1: dxs_launch<1, SPLIT, FUSE>(pooled, dead, q_idx, steps, max_blocks, scores, scratch, st); return true;
        case 2: dxs_launch<2, SPLIT, FUSE>(pooled, dead, q_idx, steps, max_blocks, scores, scratch, st); return true;
        case 3: dxs_launch<3, SPLIT, FUSE>(pooled, dead, q_idx, steps, max_blocks, scores, scratch, st); return true;
        case 4: dxs_launch<4, SPLIT, FUSE>(pooled, dead, q_idx, steps, max_blocks, scores, scratch, st); return true;
        default: return false;
    }
}

// ---------------------------------------------------------------- attention: shapes and helpers (qsa_decode_attn.cu)
constexpr int HD = 256, G = 12, CHUNK = 64, THREADS = 256, WARPS = THREADS / 32;
static_assert(HD == strata::core::kCompiledGeometry.head_dim &&
              G == strata::core::kCompiledGeometry.n_head / strata::core::kCompiledGeometry.n_head_kv,
              "qsadec_proto is compiled for core/geometry.hpp");

__device__ __forceinline__ float warp_sum_x(float v) {
    v += __shfl_xor_sync(0xffffffffu, v, 16); v += __shfl_xor_sync(0xffffffffu, v, 8);
    v += __shfl_xor_sync(0xffffffffu, v, 4);  v += __shfl_xor_sync(0xffffffffu, v, 2);
    v += __shfl_xor_sync(0xffffffffu, v, 1);
    return v;
}
__device__ __forceinline__ float warp_max_x(float v) {
    v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 16)); v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 8));
    v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 4));  v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 2));
    v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 1));
    return v;
}
__device__ __forceinline__ float reduce12(const float (&part)[16], int lane) {
    float r8[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const bool hi = (lane & 16) != 0;
        r8[i] = (hi ? part[i + 8] : part[i]) + __shfl_xor_sync(0xffffffffu, hi ? part[i] : part[i + 8], 16);
    }
    float r4[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const bool hi = (lane & 8) != 0;
        r4[i] = (hi ? r8[i + 4] : r8[i]) + __shfl_xor_sync(0xffffffffu, hi ? r8[i] : r8[i + 4], 8);
    }
    float r2[2];
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const bool hi = (lane & 4) != 0;
        r2[i] = (hi ? r4[i + 2] : r4[i]) + __shfl_xor_sync(0xffffffffu, hi ? r4[i] : r4[i + 2], 4);
    }
    const bool hi2 = (lane & 2) != 0;
    const float r1 = (hi2 ? r2[1] : r2[0]) + __shfl_xor_sync(0xffffffffu, hi2 ? r2[0] : r2[1], 2);
    return r1 + __shfl_xor_sync(0xffffffffu, r1, 1);
}
__device__ __forceinline__ int head_of_lane(int lane) {
    return ((lane >> 4) & 1) * 8 + ((lane >> 3) & 1) * 4 + ((lane >> 2) & 1) * 2 + ((lane >> 1) & 1);
}
__device__ __forceinline__ void load8_q8(const QsaAttnPools& p, long long row, int d0, float* out) {
    const int8_t* codes = p.k_q + row * HD + d0;
    const uint16_t sbits = p.k_scale[row * (HD / KV_Q8_GROUP) + d0 / KV_Q8_GROUP];
    const float sc = __half2float(__ushort_as_half(sbits));
    const uint2 raw = *reinterpret_cast<const uint2*>(codes);
    const int8_t* c = reinterpret_cast<const int8_t*>(&raw);
#pragma unroll
    for (int j = 0; j < 8; ++j) out[j] = (float) c[j] * sc;
}
__device__ __forceinline__ float load_v1_q8(const QsaAttnPools& p, long long row, int d) {
    const float sc = __half2float(__ushort_as_half(p.v_scale[row * (HD / KV_Q8_GROUP) + d / KV_Q8_GROUP]));
    return (float) p.v_q[row * HD + d] * sc;
}

// ---- copy of attn_chunk_kernel_pre75<1, 2, false> with ablation bits:
//   1 q not read (sq from constants)    2 K not read (codes from the lane)   4 V not read (from the thread)
//   8 no value phase                    16 no score phase (sp constant)      32 rows without ids / page table
template <int ABL>
__global__ void __launch_bounds__(THREADS) p_attn_kernel(const float* __restrict__ q, QsaAttnPools p,
                                                         const int32_t* __restrict__ ids,
                                                         const int32_t* __restrict__ step, int n_kv_heads,
                                                         int page_size, float scale, float* __restrict__ part_acc,
                                                         float* __restrict__ part_m, float* __restrict__ part_l,
                                                         int n_chunks, int cap, long long scratch_stride) {
    q += (size_t) blockIdx.z * (size_t) (n_kv_heads * G) * HD;
    ids += (size_t) blockIdx.z * (size_t) cap;
    step += (size_t) blockIdx.z * kStepCount;
    part_acc += (size_t) blockIdx.z * (size_t) scratch_stride;
    part_m += (size_t) blockIdx.z * (size_t) scratch_stride;
    part_l += (size_t) blockIdx.z * (size_t) scratch_stride;
    __shared__ __align__(16) float sq[G][HD];
    __shared__ float sp[G][CHUNK];
    __shared__ long long srow[CHUNK];
    float (*const spt)[G] = reinterpret_cast<float (*)[G]>(&sq[0][0]);
    const int n_ids = __ldg(step + kStepWidth);
    const int chunk = blockIdx.x, kvh = blockIdx.y;
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int c0 = chunk * CHUNK;
    const int n_here = min(CHUNK, n_ids - c0);
    const int slot = kvh * n_chunks + chunk;
    if (n_here <= 0) {
        if (t < G) { part_m[slot * G + t] = -FLT_MAX; part_l[slot * G + t] = 0.0f; }
        return;
    }
    for (int i = t; i < G * HD; i += THREADS)
        sq[i / HD][i % HD] = (ABL & 1) ? 1e-3f * (float) (i & 63) : q[(size_t) (kvh * G) * HD + i];
    if (t < CHUNK) {
        long long r = -1;
        if (t < n_here) {
            if (ABL & 32) r = ((long long) ((c0 + t) * 37 % 4096) * n_kv_heads + kvh) * page_size;
            else {
                const int cell = ids[c0 + t];
                const long long page = (long long) p.page_table[cell / page_size];
                if (page >= 0) r = (page * n_kv_heads + kvh) * page_size + (cell % page_size);
            }
        }
        srow[t] = r;
    }
    __syncthreads();
    const int hd = head_of_lane(lane);
    if (ABL & 16) {
        for (int i = t; i < G * CHUNK; i += THREADS) sp[i / CHUNK][i % CHUNK] = 1e-2f * (float) (i % 7);
    } else
    for (int i = 0; i < CHUNK / WARPS; i += 2) {
        int cc[2];
        bool all_ok = true;
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            cc[j] = warp + WARPS * (i + j);
            all_ok = all_ok && cc[j] < n_here && srow[cc[j]] >= 0;
        }
        if (all_ok) {
            float kk[2][8];
#pragma unroll
            for (int j = 0; j < 2; ++j) {
                if (ABL & 2) {
#pragma unroll
                    for (int e = 0; e < 8; ++e) kk[j][e] = 1e-3f * (float) ((lane + e + cc[j]) & 15);
                } else load8_q8(p, srow[cc[j]], lane * 8, kk[j]);
            }
            float pp[2][16];
#pragma unroll
            for (int h = 0; h < G; ++h) {
                const float4 qa = *reinterpret_cast<const float4*>(&sq[h][lane * 8]);
                const float4 qb = *reinterpret_cast<const float4*>(&sq[h][lane * 8 + 4]);
#pragma unroll
                for (int j = 0; j < 2; ++j)
                    pp[j][h] = kk[j][0] * qa.x + kk[j][1] * qa.y + kk[j][2] * qa.z + kk[j][3] * qa.w +
                               kk[j][4] * qb.x + kk[j][5] * qb.y + kk[j][6] * qb.z + kk[j][7] * qb.w;
            }
#pragma unroll
            for (int j = 0; j < 2; ++j) {
#pragma unroll
                for (int h = G; h < 16; ++h) pp[j][h] = 0.0f;
                const float sj = reduce12(pp[j], lane);
                if ((lane & 1) == 0 && hd < G) sp[hd][cc[j]] = sj * scale;
            }
        } else {
            for (int k = 0; k < 2; ++k) {
                const int c = cc[k];
                if (c >= n_here || srow[c] < 0) {
                    if (lane < G) sp[lane][c] = -FLT_MAX;
                    continue;
                }
                float k8[8];
                load8_q8(p, srow[c], lane * 8, k8);
                float part[16];
#pragma unroll
                for (int h = 0; h < G; ++h) {
                    const float4 qa = *reinterpret_cast<const float4*>(&sq[h][lane * 8]);
                    const float4 qb = *reinterpret_cast<const float4*>(&sq[h][lane * 8 + 4]);
                    part[h] = k8[0] * qa.x + k8[1] * qa.y + k8[2] * qa.z + k8[3] * qa.w +
                              k8[4] * qb.x + k8[5] * qb.y + k8[6] * qb.z + k8[7] * qb.w;
                }
#pragma unroll
                for (int h = G; h < 16; ++h) part[h] = 0.0f;
                const float s = reduce12(part, lane);
                if ((lane & 1) == 0 && hd < G) sp[hd][c] = s * scale;
            }
        }
    }
    __syncthreads();
    for (int h = warp; h < G; h += WARPS) {
        const float a = sp[h][lane], b = sp[h][lane + 32];
        const float m = warp_max_x(fmaxf(a, b));
        const float ea = (lane < n_here && srow[lane] >= 0) ? __expf(a - m) : 0.0f;
        const float eb = (lane + 32 < n_here && srow[lane + 32] >= 0) ? __expf(b - m) : 0.0f;
        spt[lane][h] = ea;
        spt[lane + 32][h] = eb;
        const float l = warp_sum_x(ea + eb);
        if (lane == 0) { part_m[slot * G + h] = m; part_l[slot * G + h] = l; }
    }
    __syncthreads();
    float acc[G];
#pragma unroll
    for (int h = 0; h < G; ++h) acc[h] = 0.0f;
    const auto fold = [&](int c, float v) {
        const float4* w4 = reinterpret_cast<const float4*>(spt[c]);
        const float4 w0 = w4[0], w1 = w4[1], w2 = w4[2];
        acc[0] = fmaf(w0.x, v, acc[0]);  acc[1] = fmaf(w0.y, v, acc[1]);
        acc[2] = fmaf(w0.z, v, acc[2]);  acc[3] = fmaf(w0.w, v, acc[3]);
        acc[4] = fmaf(w1.x, v, acc[4]);  acc[5] = fmaf(w1.y, v, acc[5]);
        acc[6] = fmaf(w1.z, v, acc[6]);  acc[7] = fmaf(w1.w, v, acc[7]);
        acc[8] = fmaf(w2.x, v, acc[8]);  acc[9] = fmaf(w2.y, v, acc[9]);
        acc[10] = fmaf(w2.z, v, acc[10]); acc[11] = fmaf(w2.w, v, acc[11]);
    };
    auto vload = [&](long long r, int c) -> float {
        if (ABL & 4) return 1e-3f * (float) ((t + c) & 31);
        return load_v1_q8(p, r, t);
    };
    if (!(ABL & 8)) {
        int c = 0;
        for (; c + 4 <= n_here; c += 4) {
            const long long r0 = srow[c], r1 = srow[c + 1], r2 = srow[c + 2], r3 = srow[c + 3];
            if ((r0 | r1 | r2 | r3) < 0) break;
            const float v0 = vload(r0, c), v1 = vload(r1, c + 1);
            const float v2 = vload(r2, c + 2), v3 = vload(r3, c + 3);
            fold(c, v0);
            fold(c + 1, v1);
            fold(c + 2, v2);
            fold(c + 3, v3);
        }
        for (; c < n_here; ++c) {
            if (srow[c] < 0) continue;
            fold(c, vload(srow[c], c));
        }
    } else {
#pragma unroll
        for (int h = 0; h < G; ++h) acc[h] = spt[t & 63][h];
    }
#pragma unroll
    for (int h = 0; h < G; ++h) part_acc[((size_t) slot * G + h) * HD + t] = acc[h];
}

// ---- the proto chunk kernel.  Loads are issued as soon as their addresses are known and before anything waits:
//   - the ids (t < 64) and the 12 query heads (3 float4 per thread) before the first store, the page table right after;
//   - after the one barrier that publishes the rows, every K slice of the warp's 8 cells (8 x (8 codes + scale)) and,
//     VST, the chunk's 64 V rows as 16-byte loads (4 per thread) plus their scales, all before the score phase; the V
//     bytes go to shared memory after the scores (the loads land meanwhile) and the value loop reads them there.
// Arithmetic per cell, per head and per dimension as attn_chunk_kernel_pre75<1, 2>: the same scores (load8's
// (float) code * scale, the 8-term expression, reduce12, * scale), the same softmax, the cells folded ascending
// (masked ones skipped) into the same 12 fmaf chains.
template <int VST>
__global__ void __launch_bounds__(THREADS) a_chunk_kernel(const float* __restrict__ q, QsaAttnPools p,
                                                          const int32_t* __restrict__ ids,
                                                          const int32_t* __restrict__ step, int n_kv_heads,
                                                          int page_size, float scale, float* __restrict__ part_acc,
                                                          float* __restrict__ part_m, float* __restrict__ part_l,
                                                          int n_chunks, int cap, long long scratch_stride) {
    q += (size_t) blockIdx.z * (size_t) (n_kv_heads * G) * HD;
    ids += (size_t) blockIdx.z * (size_t) cap;
    step += (size_t) blockIdx.z * kStepCount;
    part_acc += (size_t) blockIdx.z * (size_t) scratch_stride;
    part_m += (size_t) blockIdx.z * (size_t) scratch_stride;
    part_l += (size_t) blockIdx.z * (size_t) scratch_stride;
    __shared__ __align__(16) float sq[G][HD];
    __shared__ float sp[G][CHUNK];
    __shared__ long long srow[CHUNK];
    __shared__ __align__(16) int8_t sv[VST ? CHUNK : 1][HD];
    __shared__ float svs[VST ? CHUNK : 1][HD / KV_Q8_GROUP];
    float (*const spt)[G] = reinterpret_cast<float (*)[G]>(&sq[0][0]);
    const int n_ids = __ldg(step + kStepWidth);
    const int chunk = blockIdx.x, kvh = blockIdx.y;
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int c0 = chunk * CHUNK;
    const int n_here = min(CHUNK, n_ids - c0);
    const int slot = kvh * n_chunks + chunk;
    if (n_here <= 0) {
        if (t < G) { part_m[slot * G + t] = -FLT_MAX; part_l[slot * G + t] = 0.0f; }
        return;
    }
    const bool mine = t < n_here;   // t < CHUNK implied (n_here <= 64)
    const int cell = mine ? __ldg(ids + c0 + t) : 0;
    const float4* q4 = reinterpret_cast<const float4*>(q + (size_t) (kvh * G) * HD);
    const float4 qv0 = __ldg(q4 + t), qv1 = __ldg(q4 + t + THREADS), qv2 = __ldg(q4 + t + 2 * THREADS);
    long long r = -1;
    if (mine) {
        const long long page = (long long) __ldg(p.page_table + cell / page_size);
        if (page >= 0) r = (page * n_kv_heads + kvh) * page_size + (cell % page_size);
    }
    float4* sq4 = reinterpret_cast<float4*>(&sq[0][0]);
    sq4[t] = qv0;
    sq4[t + THREADS] = qv1;
    sq4[t + 2 * THREADS] = qv2;
    if (t < CHUNK) srow[t] = r;
    __syncthreads();
    // the warp's cells warp + 8 i: K slices
    uint2 kc[8];
    uint16_t ks[8];
    bool kok[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int c = warp + WARPS * i;
        const long long row = srow[c];
        kok[i] = c < n_here && row >= 0;
        const long long rr = kok[i] ? row : 0;
        kc[i] = __ldg(reinterpret_cast<const uint2*>(p.k_q + rr * HD + lane * 8));
        ks[i] = __ldg(p.k_scale + rr * (HD / KV_Q8_GROUP) + (lane * 8) / KV_Q8_GROUP);
    }
    uint4 vv[4];
    uint16_t vsb = 0;
    if (VST) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int idx = t + THREADS * j, c = idx >> 4, part = idx & 15;
            const long long row = srow[c];
            const long long rr = (c < n_here && row >= 0) ? row : 0;
            vv[j] = __ldg(reinterpret_cast<const uint4*>(p.v_q + rr * HD) + part);
        }
        const int c = t >> 2, g = t & 3;
        const long long row = srow[c];
        const long long rr = (c < n_here && row >= 0) ? row : 0;
        vsb = __ldg(p.v_scale + rr * (HD / KV_Q8_GROUP) + g);
    }
    const int hd = head_of_lane(lane);
#pragma unroll
    for (int i = 0; i < 8; i += 2) {
        if (kok[i] && kok[i + 1]) {
            float kk[2][8];
#pragma unroll
            for (int j = 0; j < 2; ++j) {
                const float sc = __half2float(__ushort_as_half(ks[i + j]));
                const int8_t* cb = reinterpret_cast<const int8_t*>(&kc[i + j]);
#pragma unroll
                for (int e = 0; e < 8; ++e) kk[j][e] = (float) cb[e] * sc;
            }
            float pp[2][16];
#pragma unroll
            for (int h = 0; h < G; ++h) {
                const float4 qa = *reinterpret_cast<const float4*>(&sq[h][lane * 8]);
                const float4 qb = *reinterpret_cast<const float4*>(&sq[h][lane * 8 + 4]);
#pragma unroll
                for (int j = 0; j < 2; ++j)
                    pp[j][h] = kk[j][0] * qa.x + kk[j][1] * qa.y + kk[j][2] * qa.z + kk[j][3] * qa.w +
                               kk[j][4] * qb.x + kk[j][5] * qb.y + kk[j][6] * qb.z + kk[j][7] * qb.w;
            }
#pragma unroll
            for (int j = 0; j < 2; ++j) {
#pragma unroll
                for (int h = G; h < 16; ++h) pp[j][h] = 0.0f;
                const float sj = reduce12(pp[j], lane);
                if ((lane & 1) == 0 && hd < G) sp[hd][warp + WARPS * (i + j)] = sj * scale;
            }
        } else {
#pragma unroll
            for (int j = 0; j < 2; ++j) {
                const int c = warp + WARPS * (i + j);
                if (!kok[i + j]) {
                    if (lane < G) sp[lane][c] = -FLT_MAX;
                    continue;
                }
                float k8[8];
                const float sc = __half2float(__ushort_as_half(ks[i + j]));
                const int8_t* cb = reinterpret_cast<const int8_t*>(&kc[i + j]);
#pragma unroll
                for (int e = 0; e < 8; ++e) k8[e] = (float) cb[e] * sc;
                float part[16];
#pragma unroll
                for (int h = 0; h < G; ++h) {
                    const float4 qa = *reinterpret_cast<const float4*>(&sq[h][lane * 8]);
                    const float4 qb = *reinterpret_cast<const float4*>(&sq[h][lane * 8 + 4]);
                    part[h] = k8[0] * qa.x + k8[1] * qa.y + k8[2] * qa.z + k8[3] * qa.w +
                              k8[4] * qb.x + k8[5] * qb.y + k8[6] * qb.z + k8[7] * qb.w;
                }
#pragma unroll
                for (int h = G; h < 16; ++h) part[h] = 0.0f;
                const float s = reduce12(part, lane);
                if ((lane & 1) == 0 && hd < G) sp[hd][c] = s * scale;
            }
        }
    }
    if (VST) {
        uint4* sv4 = reinterpret_cast<uint4*>(&sv[0][0]);
#pragma unroll
        for (int j = 0; j < 4; ++j) sv4[t + THREADS * j] = vv[j];
        svs[t >> 2][t & 3] = __half2float(__ushort_as_half(vsb));
    }
    __syncthreads();
    for (int h = warp; h < G; h += WARPS) {
        const float a = sp[h][lane], b = sp[h][lane + 32];
        const float m = warp_max_x(fmaxf(a, b));
        const float ea = (lane < n_here && srow[lane] >= 0) ? __expf(a - m) : 0.0f;
        const float eb = (lane + 32 < n_here && srow[lane + 32] >= 0) ? __expf(b - m) : 0.0f;
        spt[lane][h] = ea;
        spt[lane + 32][h] = eb;
        const float l = warp_sum_x(ea + eb);
        if (lane == 0) { part_m[slot * G + h] = m; part_l[slot * G + h] = l; }
    }
    __syncthreads();
    float acc[G];
#pragma unroll
    for (int h = 0; h < G; ++h) acc[h] = 0.0f;
    const auto fold = [&](int c, float v) {
        const float4* w4 = reinterpret_cast<const float4*>(spt[c]);
        const float4 w0 = w4[0], w1 = w4[1], w2 = w4[2];
        acc[0] = fmaf(w0.x, v, acc[0]);  acc[1] = fmaf(w0.y, v, acc[1]);
        acc[2] = fmaf(w0.z, v, acc[2]);  acc[3] = fmaf(w0.w, v, acc[3]);
        acc[4] = fmaf(w1.x, v, acc[4]);  acc[5] = fmaf(w1.y, v, acc[5]);
        acc[6] = fmaf(w1.z, v, acc[6]);  acc[7] = fmaf(w1.w, v, acc[7]);
        acc[8] = fmaf(w2.x, v, acc[8]);  acc[9] = fmaf(w2.y, v, acc[9]);
        acc[10] = fmaf(w2.z, v, acc[10]); acc[11] = fmaf(w2.w, v, acc[11]);
    };
    if (VST) {
        const int g = t / KV_Q8_GROUP;
#pragma unroll 4
        for (int c = 0; c < n_here; ++c) {
            if (srow[c] < 0) continue;
            fold(c, (float) sv[c][t] * svs[c][g]);
        }
    } else {
        int c = 0;
        for (; c + 4 <= n_here; c += 4) {
            const long long r0 = srow[c], r1 = srow[c + 1], r2 = srow[c + 2], r3 = srow[c + 3];
            if ((r0 | r1 | r2 | r3) < 0) break;
            const float v0 = load_v1_q8(p, r0, t), v1 = load_v1_q8(p, r1, t);
            const float v2 = load_v1_q8(p, r2, t), v3 = load_v1_q8(p, r3, t);
            fold(c, v0);
            fold(c + 1, v1);
            fold(c + 2, v2);
            fold(c + 3, v3);
        }
        for (; c < n_here; ++c) {
            if (srow[c] < 0) continue;
            fold(c, load_v1_q8(p, srow[c], t));
        }
    }
#pragma unroll
    for (int h = 0; h < G; ++h) part_acc[((size_t) slot * G + h) * HD + t] = acc[h];
}

// ---- a_chunk_kernel<1> in 20.5 KB of shared memory (it takes 33 KB: one CTA per SM): the score phase's sq / sp and
// the value phase's V bytes share one 16 KB region (the V loads wait in registers through the scores and the softmax,
// which writes the probabilities to their own 3 KB; one more barrier before the V bytes are stored).  MINB: the CTAs
// per SM the registers are bounded for.  The arithmetic is a_chunk_kernel's (= attn_chunk_kernel_pre75<1, 2>'s).
template <int MINB>
__global__ void __launch_bounds__(THREADS, MINB) a2_chunk_kernel(const float* __restrict__ q, QsaAttnPools p,
                                                                 const int32_t* __restrict__ ids,
                                                                 const int32_t* __restrict__ step, int n_kv_heads,
                                                                 int page_size, float scale, float* __restrict__ part_acc,
                                                                 float* __restrict__ part_m, float* __restrict__ part_l,
                                                                 int n_chunks, int cap, long long scratch_stride) {
    q += (size_t) blockIdx.z * (size_t) (n_kv_heads * G) * HD;
    ids += (size_t) blockIdx.z * (size_t) cap;
    step += (size_t) blockIdx.z * kStepCount;
    part_acc += (size_t) blockIdx.z * (size_t) scratch_stride;
    part_m += (size_t) blockIdx.z * (size_t) scratch_stride;
    part_l += (size_t) blockIdx.z * (size_t) scratch_stride;
    __shared__ __align__(16) float sA[CHUNK * HD / 4];   // 16 KB: sq [G][HD] + sp [G][CHUNK], then sv [CHUNK][HD] bytes
    __shared__ __align__(16) float spt[CHUNK][G];
    __shared__ long long srow[CHUNK];
    __shared__ float svs[CHUNK][HD / KV_Q8_GROUP];
    static_assert(G * HD + G * CHUNK <= CHUNK * HD / 4, "sq + sp must fit in the V region");
    float (*const sq)[HD] = reinterpret_cast<float (*)[HD]>(sA);
    float (*const sp)[CHUNK] = reinterpret_cast<float (*)[CHUNK]>(sA + G * HD);
    const int8_t (*const sv)[HD] = reinterpret_cast<const int8_t (*)[HD]>(sA);
    const int n_ids = __ldg(step + kStepWidth);
    const int chunk = blockIdx.x, kvh = blockIdx.y;
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int c0 = chunk * CHUNK;
    const int n_here = min(CHUNK, n_ids - c0);
    const int slot = kvh * n_chunks + chunk;
    if (n_here <= 0) {
        if (t < G) { part_m[slot * G + t] = -FLT_MAX; part_l[slot * G + t] = 0.0f; }
        return;
    }
    const bool mine = t < n_here;
    const int cell = mine ? __ldg(ids + c0 + t) : 0;
    const float4* q4 = reinterpret_cast<const float4*>(q + (size_t) (kvh * G) * HD);
    const float4 qv0 = __ldg(q4 + t), qv1 = __ldg(q4 + t + THREADS), qv2 = __ldg(q4 + t + 2 * THREADS);
    long long r = -1;
    if (mine) {
        const long long page = (long long) __ldg(p.page_table + cell / page_size);
        if (page >= 0) r = (page * n_kv_heads + kvh) * page_size + (cell % page_size);
    }
    float4* sq4 = reinterpret_cast<float4*>(sA);
    sq4[t] = qv0;
    sq4[t + THREADS] = qv1;
    sq4[t + 2 * THREADS] = qv2;
    if (t < CHUNK) srow[t] = r;
    __syncthreads();
    uint2 kc[8];
    uint16_t ks[8];
    bool kok[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int c = warp + WARPS * i;
        const long long row = srow[c];
        kok[i] = c < n_here && row >= 0;
        const long long rr = kok[i] ? row : 0;
        kc[i] = __ldg(reinterpret_cast<const uint2*>(p.k_q + rr * HD + lane * 8));
        ks[i] = __ldg(p.k_scale + rr * (HD / KV_Q8_GROUP) + (lane * 8) / KV_Q8_GROUP);
    }
    uint4 vv[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int idx = t + THREADS * j, c = idx >> 4, part = idx & 15;
        const long long row = srow[c];
        const long long rr = (c < n_here && row >= 0) ? row : 0;
        vv[j] = __ldg(reinterpret_cast<const uint4*>(p.v_q + rr * HD) + part);
    }
    uint16_t vsb;
    {
        const int c = t >> 2, g = t & 3;
        const long long row = srow[c];
        const long long rr = (c < n_here && row >= 0) ? row : 0;
        vsb = __ldg(p.v_scale + rr * (HD / KV_Q8_GROUP) + g);
    }
    const int hd = head_of_lane(lane);
#pragma unroll
    for (int i = 0; i < 8; i += 2) {
        if (kok[i] && kok[i + 1]) {
            float kk[2][8];
#pragma unroll
            for (int j = 0; j < 2; ++j) {
                const float sc = __half2float(__ushort_as_half(ks[i + j]));
                const int8_t* cb = reinterpret_cast<const int8_t*>(&kc[i + j]);
#pragma unroll
                for (int e = 0; e < 8; ++e) kk[j][e] = (float) cb[e] * sc;
            }
            float pp[2][16];
#pragma unroll
            for (int h = 0; h < G; ++h) {
                const float4 qa = *reinterpret_cast<const float4*>(&sq[h][lane * 8]);
                const float4 qb = *reinterpret_cast<const float4*>(&sq[h][lane * 8 + 4]);
#pragma unroll
                for (int j = 0; j < 2; ++j)
                    pp[j][h] = kk[j][0] * qa.x + kk[j][1] * qa.y + kk[j][2] * qa.z + kk[j][3] * qa.w +
                               kk[j][4] * qb.x + kk[j][5] * qb.y + kk[j][6] * qb.z + kk[j][7] * qb.w;
            }
#pragma unroll
            for (int j = 0; j < 2; ++j) {
#pragma unroll
                for (int h = G; h < 16; ++h) pp[j][h] = 0.0f;
                const float sj = reduce12(pp[j], lane);
                if ((lane & 1) == 0 && hd < G) sp[hd][warp + WARPS * (i + j)] = sj * scale;
            }
        } else {
#pragma unroll
            for (int j = 0; j < 2; ++j) {
                const int c = warp + WARPS * (i + j);
                if (!kok[i + j]) {
                    if (lane < G) sp[lane][c] = -FLT_MAX;
                    continue;
                }
                float k8[8];
                const float sc = __half2float(__ushort_as_half(ks[i + j]));
                const int8_t* cb = reinterpret_cast<const int8_t*>(&kc[i + j]);
#pragma unroll
                for (int e = 0; e < 8; ++e) k8[e] = (float) cb[e] * sc;
                float part[16];
#pragma unroll
                for (int h = 0; h < G; ++h) {
                    const float4 qa = *reinterpret_cast<const float4*>(&sq[h][lane * 8]);
                    const float4 qb = *reinterpret_cast<const float4*>(&sq[h][lane * 8 + 4]);
                    part[h] = k8[0] * qa.x + k8[1] * qa.y + k8[2] * qa.z + k8[3] * qa.w +
                              k8[4] * qb.x + k8[5] * qb.y + k8[6] * qb.z + k8[7] * qb.w;
                }
#pragma unroll
                for (int h = G; h < 16; ++h) part[h] = 0.0f;
                const float s = reduce12(part, lane);
                if ((lane & 1) == 0 && hd < G) sp[hd][c] = s * scale;
            }
        }
    }
    __syncthreads();
    for (int h = warp; h < G; h += WARPS) {
        const float a = sp[h][lane], b = sp[h][lane + 32];
        const float m = warp_max_x(fmaxf(a, b));
        const float ea = (lane < n_here && srow[lane] >= 0) ? __expf(a - m) : 0.0f;
        const float eb = (lane + 32 < n_here && srow[lane + 32] >= 0) ? __expf(b - m) : 0.0f;
        spt[lane][h] = ea;
        spt[lane + 32][h] = eb;
        const float l = warp_sum_x(ea + eb);
        if (lane == 0) { part_m[slot * G + h] = m; part_l[slot * G + h] = l; }
    }
    __syncthreads();   // sq and sp are dead: the V bytes take their place
    {
        uint4* sv4 = reinterpret_cast<uint4*>(sA);
#pragma unroll
        for (int j = 0; j < 4; ++j) sv4[t + THREADS * j] = vv[j];
        svs[t >> 2][t & 3] = __half2float(__ushort_as_half(vsb));
    }
    __syncthreads();
    float acc[G];
#pragma unroll
    for (int h = 0; h < G; ++h) acc[h] = 0.0f;
    const int g = t / KV_Q8_GROUP;
#pragma unroll 4
    for (int c = 0; c < n_here; ++c) {
        if (srow[c] < 0) continue;
        const float v = (float) sv[c][t] * svs[c][g];
        const float4* w4 = reinterpret_cast<const float4*>(spt[c]);
        const float4 w0 = w4[0], w1 = w4[1], w2 = w4[2];
        acc[0] = fmaf(w0.x, v, acc[0]);  acc[1] = fmaf(w0.y, v, acc[1]);
        acc[2] = fmaf(w0.z, v, acc[2]);  acc[3] = fmaf(w0.w, v, acc[3]);
        acc[4] = fmaf(w1.x, v, acc[4]);  acc[5] = fmaf(w1.y, v, acc[5]);
        acc[6] = fmaf(w1.z, v, acc[6]);  acc[7] = fmaf(w1.w, v, acc[7]);
        acc[8] = fmaf(w2.x, v, acc[8]);  acc[9] = fmaf(w2.y, v, acc[9]);
        acc[10] = fmaf(w2.z, v, acc[10]); acc[11] = fmaf(w2.w, v, acc[11]);
    }
#pragma unroll
    for (int h = 0; h < G; ++h) part_acc[((size_t) slot * G + h) * HD + t] = acc[h];
}

// ---- copies of the merges
__global__ void __launch_bounds__(HD) p_merge_kernel(const float* __restrict__ part_acc, const float* __restrict__ part_m,
                                                     const float* __restrict__ part_l, int n_chunks,
                                                     float* __restrict__ attn, long long scratch_stride) {
    part_acc += (size_t) blockIdx.y * (size_t) scratch_stride;
    part_m += (size_t) blockIdx.y * (size_t) scratch_stride;
    part_l += (size_t) blockIdx.y * (size_t) scratch_stride;
    attn += (size_t) blockIdx.y * (size_t) gridDim.x * HD;
    const int h = blockIdx.x;
    const int kvh = h / G, hl = h % G;
    const int d = threadIdx.x;
    float M = -FLT_MAX;
    for (int c = 0; c < n_chunks; ++c) M = fmaxf(M, part_m[(kvh * n_chunks + c) * G + hl]);
    float L = 0.0f, acc = 0.0f;
    for (int c = 0; c < n_chunks; ++c) {
        const int slot = kvh * n_chunks + c;
        const float m = part_m[slot * G + hl];
        if (m == -FLT_MAX) continue;
        const float w = __expf(m - M);
        L = fmaf(part_l[slot * G + hl], w, L);
        acc = fmaf(part_acc[((size_t) slot * G + hl) * HD + d], w, acc);
    }
    attn[(size_t) h * HD + d] = L > 0.0f ? acc / L : 0.0f;
}

template <int U, bool TAIL>
__device__ __forceinline__ void merge_max_block(const float* __restrict__ pm, int c0, int n_act, float& M) {
    float mm[U];
#pragma unroll
    for (int u = 0; u < U; ++u) mm[u] = (!TAIL || c0 + u < n_act) ? pm[(size_t) (c0 + u) * G] : -FLT_MAX;
#pragma unroll
    for (int u = 0; u < U; ++u)
        if (!TAIL || c0 + u < n_act) M = fmaxf(M, mm[u]);
}
template <int U, bool TAIL>
__device__ __forceinline__ void merge_sum_block(const float* __restrict__ pm, const float* __restrict__ pl,
                                                const float* __restrict__ pa, int c0, int n_act, float M, float& L,
                                                float& acc) {
    float mm[U], ll[U], aa[U];
#pragma unroll
    for (int u = 0; u < U; ++u) {
        const bool in = !TAIL || c0 + u < n_act;
        mm[u] = in ? pm[(size_t) (c0 + u) * G] : -FLT_MAX;
        ll[u] = in ? pl[(size_t) (c0 + u) * G] : 0.0f;
        aa[u] = in ? pa[(size_t) (c0 + u) * G * HD] : 0.0f;
    }
#pragma unroll
    for (int u = 0; u < U; ++u) {
        const float w = __expf(mm[u] - M);
        const float L_next = fmaf(ll[u], w, L);
        const float acc_next = fmaf(aa[u], w, acc);
        const bool live = !(mm[u] == -FLT_MAX);
        L = live ? L_next : L;
        acc = live ? acc_next : acc;
    }
}
template <int U>
__global__ void __launch_bounds__(HD) p_merge_v2_kernel(const float* __restrict__ part_acc,
                                                        const float* __restrict__ part_m,
                                                        const float* __restrict__ part_l, int n_chunks,
                                                        const int32_t* __restrict__ steps, float* __restrict__ attn,
                                                        long long scratch_stride) {
    part_acc += (size_t) blockIdx.y * (size_t) scratch_stride;
    part_m += (size_t) blockIdx.y * (size_t) scratch_stride;
    part_l += (size_t) blockIdx.y * (size_t) scratch_stride;
    attn += (size_t) blockIdx.y * (size_t) gridDim.x * HD;
    const int width = __ldg(steps + (size_t) blockIdx.y * kStepCount + kStepWidth);
    const int n_act = width > 0 ? min(n_chunks, (width - 1) / CHUNK + 1) : 0;
    const int h = blockIdx.x;
    const int kvh = h / G, hl = h % G;
    const int d = threadIdx.x;
    const size_t base = (size_t) kvh * (size_t) n_chunks * G + (size_t) hl;
    const float* pm = part_m + base;
    const float* pl = part_l + base;
    const float* pa = part_acc + base * HD + (size_t) d;
    const int n_full = n_act - n_act % U;
    float M = -FLT_MAX;
    for (int c0 = 0; c0 < n_full; c0 += U) merge_max_block<U, false>(pm, c0, n_act, M);
    if (n_full < n_act) merge_max_block<U, true>(pm, n_full, n_act, M);
    float L = 0.0f, acc = 0.0f;
    for (int c0 = 0; c0 < n_full; c0 += U) merge_sum_block<U, false>(pm, pl, pa, c0, n_act, M, L, acc);
    if (n_full < n_act) merge_sum_block<U, true>(pm, pl, pa, n_full, n_act, M, L, acc);
    attn[(size_t) h * HD + d] = L > 0.0f ? acc / L : 0.0f;
}

struct AttnArgs {
    int n_chunks;
    long long stride;
    float *part_acc, *part_m, *part_l;
    float scale;
    dim3 grid;
};
AttnArgs attn_args(int64_t cap, const QsaShapes& s, float* scratch, int64_t nq) {
    AttnArgs a;
    a.n_chunks = (int) ((cap + CHUNK - 1) / CHUNK);
    a.stride = (long long) qsa_decode_attn_scratch_floats(cap, s);
    a.part_acc = scratch;
    a.part_m = scratch + (size_t) a.n_chunks * s.n_head * HD;
    a.part_l = a.part_m + (size_t) a.n_chunks * s.n_head;
    a.scale = 1.0f / sqrtf((float) HD);
    a.grid = dim3((unsigned) a.n_chunks, (unsigned) s.n_head_kv, (unsigned) nq);
    return a;
}
void merge_launch(int merge, const AttnArgs& a, const int32_t* steps, const QsaShapes& s, float* attn, int64_t nq,
                  cudaStream_t st) {
    if (merge == 1)
        p_merge_v2_kernel<16><<<dim3((unsigned) s.n_head, (unsigned) nq), HD, 0, st>>>(a.part_acc, a.part_m, a.part_l,
                                                                                         a.n_chunks, steps, attn, a.stride);
    else if (merge == 0)
        p_merge_kernel<<<dim3((unsigned) s.n_head, (unsigned) nq), HD, 0, st>>>(a.part_acc, a.part_m, a.part_l,
                                                                              a.n_chunks, attn, a.stride);
}

}  // namespace

int64_t qsadec_select_scratch_ints(int64_t nq) { return (nq < MT_MAXQ ? nq : MT_MAXQ) * (int64_t) MT_Q_INTS; }

// variant: 0 = proto default: dxs fused (the tail and pass 0 inside the scorer; SPLIT 4 for 1-2 queries, SPLIT 2 for
//              3-4 - die 3 bench, campaign\QSADEC-PROTO.md), then passes 1-3, count, emit
//          5 = R scorer 4 warps/CTA fused (registers, no staging: slower, kept for the record)
//          1 = R scorer unfused (+ production tail copy + 6 top-k copies)  2 = 8 warps/CTA fused
//          3 = fused, at most 2 CTAs per SM   4 = fused, 3 per SM   10 = copies of the production path (dx + tail + 6)
//          20 / 21 = dxs SPLIT 2 / 4 unfused (+ tail + 6)   22 / 23 / 24 = dxs SPLIT 2 / 4 / 1 fused (+ 5)
bool qsadec_select(const float* pooled, const float* dead, const float* q_idx, const int32_t* steps, int64_t nq,
                   int64_t max_blocks, int64_t cap, const QsaShapes& s, float* scores, int32_t* ids, int32_t* scratch,
                   void* stream, int variant) {
    if (nq < 1 || nq > 4 || scratch == nullptr || s.idx_dim != IDX_DIM || s.idx_n_head != IDX_HEADS ||
        s.idx_block != R || cap < qsa_selection_width(kTopkMaxCells, s) || max_blocks <= (int64_t) TK_T * TK_PER)
        return false;
    static_assert(MT_Q_INTS == (MT_NMAX * 256 + 3 * 256 + 16) + 2 * MT_NMAX * MT_W + 64, "MT layout");
    const cudaStream_t st = (cudaStream_t) stream;
    const dim3 tg((unsigned) topk_ctas(max_blocks), (unsigned) nq);
    bool fused = true;
    switch (variant) {
        case 0:
            if (!dxs_counts_fit(max_blocks)) return false;
            if (nq <= 2) dxs_dispatch<4, true>(nq, pooled, dead, q_idx, steps, max_blocks, scores, scratch, st);
            else dxs_dispatch<2, true>(nq, pooled, dead, q_idx, steps, max_blocks, scores, scratch, st);
            break;
        case 5: r_dispatch<4, true>(nq, pooled, dead, q_idx, steps, max_blocks, scores, scratch, 0, st); break;
        case 1:
            r_dispatch<4, false>(nq, pooled, dead, q_idx, steps, max_blocks, scores, scratch, 0, st);
            p_tail_kernel<<<(unsigned) nq, 32, 0, st>>>(dead, q_idx, steps, max_blocks, scores);
            fused = false;
            break;
        case 2: r_dispatch<8, true>(nq, pooled, dead, q_idx, steps, max_blocks, scores, scratch, 0, st); break;
        case 3: r_dispatch<4, true>(nq, pooled, dead, q_idx, steps, max_blocks, scores, scratch, 2, st); break;
        case 4: r_dispatch<4, true>(nq, pooled, dead, q_idx, steps, max_blocks, scores, scratch, 3, st); break;
        case 20: case 21:
            if (variant == 20) dxs_dispatch<2, false>(nq, pooled, dead, q_idx, steps, max_blocks, scores, scratch, st);
            else dxs_dispatch<4, false>(nq, pooled, dead, q_idx, steps, max_blocks, scores, scratch, st);
            p_tail_kernel<<<(unsigned) nq, 32, 0, st>>>(dead, q_idx, steps, max_blocks, scores);
            fused = false;
            break;
        case 22: case 23: case 24:
            if (!dxs_counts_fit(max_blocks)) return false;
            if (variant == 22) dxs_dispatch<2, true>(nq, pooled, dead, q_idx, steps, max_blocks, scores, scratch, st);
            else if (variant == 23) dxs_dispatch<4, true>(nq, pooled, dead, q_idx, steps, max_blocks, scores, scratch, st);
            else dxs_dispatch<1, true>(nq, pooled, dead, q_idx, steps, max_blocks, scores, scratch, st);
            break;
        case 10:
            dx_dispatch<0>(nq, pooled, q_idx, steps, max_blocks, scores, st);
            p_tail_kernel<<<(unsigned) nq, 32, 0, st>>>(dead, q_idx, steps, max_blocks, scores);
            fused = false;
            break;
        default: return false;
    }
    if (fused) {
        p_hist_kernel<1, true><<<tg, MT_T, 0, st>>>(scores, steps, max_blocks, scratch);
        p_hist_kernel<2, true><<<tg, MT_T, 0, st>>>(scores, steps, max_blocks, scratch);
    } else {
        p_hist_kernel<0, false><<<tg, MT_T, 0, st>>>(scores, steps, max_blocks, scratch);
        p_hist_kernel<1, false><<<tg, MT_T, 0, st>>>(scores, steps, max_blocks, scratch);
        p_hist_kernel<2, false><<<tg, MT_T, 0, st>>>(scores, steps, max_blocks, scratch);
    }
    p_hist_kernel<3, false><<<tg, MT_T, 0, st>>>(scores, steps, max_blocks, scratch);
    p_count_kernel<<<tg, MT_T, 0, st>>>(scores, steps, max_blocks, scratch);
    p_emit_kernel<<<tg, MT_T, 0, st>>>(scores, steps, max_blocks, cap, ids, scratch);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { std::fprintf(stderr, "qsadec_select: %s\n", cudaGetErrorString(e)); std::exit(1); }
    return true;
}

// variant: 0 = proto default: a2 (V region shared) bounded for 3 CTAs/SM at 2 queries, 2 CTAs/SM otherwise, + merge
// v2 (die 3 bench); 8 = a_chunk<1> (V staged, 33 KB) + merge v2, 1 = VST chunk + merge v1, 2 = early loads without V staging +
// merge v2, 3 = production chunk copy + merge v2, 4 = early loads, no V staging + merge v1, 5 = a2 (the V region
// shared, 3 CTAs/SM) + merge v2, 6 = a2 bounded for 2 CTAs/SM + merge v2, 7 = a2 + merge v1
bool qsadec_attn(const float* q, const QsaAttnPools& pools, const int32_t* ids, const int32_t* steps, int64_t cap,
                 const QsaShapes& s, float* scratch, float* attn, int64_t nq, void* stream, int variant) {
    if (nq <= 0 || nq > 65535 || s.head_dim != HD || s.n_head != (int64_t) G * s.n_head_kv || cap <= 0 || !scratch ||
        !ids || !steps || !pools.page_table || pools.k_q == nullptr || pools.v_q == nullptr || pools.k_q4 != nullptr ||
        pools.v_q4 != nullptr || !pools.k_scale || !pools.v_scale)
        return false;
    const cudaStream_t st = (cudaStream_t) stream;
    const AttnArgs a = attn_args(cap, s, scratch, nq);
    int merge = 1;
    switch (variant) {
        case 0:
            if (nq == 2)
                a2_chunk_kernel<3><<<a.grid, THREADS, 0, st>>>(q, pools, ids, steps, (int) s.n_head_kv, (int) s.page_size,
                                                               a.scale, a.part_acc, a.part_m, a.part_l, a.n_chunks,
                                                               (int) cap, a.stride);
            else
                a2_chunk_kernel<2><<<a.grid, THREADS, 0, st>>>(q, pools, ids, steps, (int) s.n_head_kv, (int) s.page_size,
                                                               a.scale, a.part_acc, a.part_m, a.part_l, a.n_chunks,
                                                               (int) cap, a.stride);
            break;
        case 8: case 1:
            a_chunk_kernel<1><<<a.grid, THREADS, 0, st>>>(q, pools, ids, steps, (int) s.n_head_kv, (int) s.page_size,
                                                          a.scale, a.part_acc, a.part_m, a.part_l, a.n_chunks, (int) cap,
                                                          a.stride);
            merge = variant == 8 ? 1 : 0;
            break;
        case 2: case 4:
            a_chunk_kernel<0><<<a.grid, THREADS, 0, st>>>(q, pools, ids, steps, (int) s.n_head_kv, (int) s.page_size,
                                                          a.scale, a.part_acc, a.part_m, a.part_l, a.n_chunks, (int) cap,
                                                          a.stride);
            merge = variant == 2 ? 1 : 0;
            break;
        case 5: case 7:
            a2_chunk_kernel<3><<<a.grid, THREADS, 0, st>>>(q, pools, ids, steps, (int) s.n_head_kv, (int) s.page_size,
                                                           a.scale, a.part_acc, a.part_m, a.part_l, a.n_chunks, (int) cap,
                                                           a.stride);
            merge = variant == 5 ? 1 : 0;
            break;
        case 6:
            a2_chunk_kernel<2><<<a.grid, THREADS, 0, st>>>(q, pools, ids, steps, (int) s.n_head_kv, (int) s.page_size,
                                                           a.scale, a.part_acc, a.part_m, a.part_l, a.n_chunks, (int) cap,
                                                           a.stride);
            break;
        case 3:
            p_attn_kernel<0><<<a.grid, THREADS, 0, st>>>(q, pools, ids, steps, (int) s.n_head_kv, (int) s.page_size,
                                                         a.scale, a.part_acc, a.part_m, a.part_l, a.n_chunks, (int) cap,
                                                         a.stride);
            break;
        default: return false;
    }
    merge_launch(merge, a, steps, s, attn, nq, st);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { std::fprintf(stderr, "qsadec_attn: %s\n", cudaGetErrorString(e)); std::exit(1); }
    return true;
}

void qsadec_bench_select_stage(int stage, int abl, const float* pooled, const float* dead, const float* q_idx,
                               const int32_t* steps, int64_t nq, int64_t max_blocks, int64_t cap, float* scores,
                               int32_t* ids, int32_t* scratch, void* stream) {
    const cudaStream_t st = (cudaStream_t) stream;
    const dim3 tg((unsigned) topk_ctas(max_blocks), (unsigned) nq);
    switch (stage) {
        case 0:
            if (abl == 1) dx_dispatch<1>(nq, pooled, q_idx, steps, max_blocks, scores, st);
            else if (abl == 2) dx_dispatch<2>(nq, pooled, q_idx, steps, max_blocks, scores, st);
            else dx_dispatch<0>(nq, pooled, q_idx, steps, max_blocks, scores, st);
            break;
        case 7: p_tail_kernel<<<(unsigned) nq, 32, 0, st>>>(dead, q_idx, steps, max_blocks, scores); break;
        case 1: p_hist_kernel<0, false><<<tg, MT_T, 0, st>>>(scores, steps, max_blocks, scratch); break;
        case 2: p_hist_kernel<1, false><<<tg, MT_T, 0, st>>>(scores, steps, max_blocks, scratch); break;
        case 3: p_hist_kernel<2, false><<<tg, MT_T, 0, st>>>(scores, steps, max_blocks, scratch); break;
        case 4: p_hist_kernel<3, false><<<tg, MT_T, 0, st>>>(scores, steps, max_blocks, scratch); break;
        case 5: p_count_kernel<<<tg, MT_T, 0, st>>>(scores, steps, max_blocks, scratch); break;
        case 6: p_emit_kernel<<<tg, MT_T, 0, st>>>(scores, steps, max_blocks, cap, ids, scratch); break;
        case 8: r_dispatch<4, false>(nq, pooled, dead, q_idx, steps, max_blocks, scores, scratch, 0, st); break;
        case 9: r_dispatch<4, true>(nq, pooled, dead, q_idx, steps, max_blocks, scores, scratch, 0, st); break;
        default: break;
    }
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { std::fprintf(stderr, "qsadec_bench_select_stage: %s\n", cudaGetErrorString(e)); std::exit(1); }
}

void qsadec_bench_attn_abl(int abl, int merge, const float* q, const QsaAttnPools& pools, const int32_t* ids,
                           const int32_t* steps, int64_t cap, const QsaShapes& s, float* scratch, float* attn,
                           int64_t nq, void* stream) {
    const cudaStream_t st = (cudaStream_t) stream;
    const AttnArgs a = attn_args(cap, s, scratch, nq);
#define QD_ABL(N) case N: p_attn_kernel<N><<<a.grid, THREADS, 0, st>>>(q, pools, ids, steps, (int) s.n_head_kv, (int) s.page_size, a.scale, a.part_acc, a.part_m, a.part_l, a.n_chunks, (int) cap, a.stride); break;
    switch (abl) {
        QD_ABL(0) QD_ABL(1) QD_ABL(2) QD_ABL(4) QD_ABL(6) QD_ABL(7) QD_ABL(8) QD_ABL(16) QD_ABL(32) QD_ABL(39) QD_ABL(63)
        case -1: break;   // the merge alone
        default: std::fprintf(stderr, "qsadec_bench_attn_abl: no variant %d\n", abl); std::exit(1);
    }
#undef QD_ABL
    merge_launch(merge, a, steps, s, attn, nq, st);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { std::fprintf(stderr, "qsadec_bench_attn_abl: %s\n", cudaGetErrorString(e)); std::exit(1); }
}

void qsadec_bench_attrs(char* buf, int len) {
    int n = 0;
    auto add = [&](const char* name, const void* f) {
        cudaFuncAttributes at{};
        if (cudaFuncGetAttributes(&at, f) != cudaSuccess) { cudaGetLastError(); return; }
        if (n < len) n += std::snprintf(buf + n, (size_t) (len - n), "%s regs %d smem %zu local %zu\n", name, at.numRegs,
                                        at.sharedSizeBytes, at.localSizeBytes);
    };
    add("p_dx<2>", (const void*) p_dx_kernel<2, 0>);
    add("r_scores<1,4,F>", (const void*) r_scores_kernel<1, 4, true>);
    add("r_scores<2,4,F>", (const void*) r_scores_kernel<2, 4, true>);
    add("r_scores<4,4,F>", (const void*) r_scores_kernel<4, 4, true>);
    add("p_attn<0>", (const void*) p_attn_kernel<0>);
    add("a_chunk<1>", (const void*) a_chunk_kernel<1>);
    add("a_chunk<0>", (const void*) a_chunk_kernel<0>);
    add("a2_chunk<3>", (const void*) a2_chunk_kernel<3>);
    add("a2_chunk<2>", (const void*) a2_chunk_kernel<2>);
    add("p_hist<1>", (const void*) p_hist_kernel<1, false>);
    add("dxs<2,2,F>", (const void*) dxs_kernel<2, 2, true>);
    add("dxs<2,4,F>", (const void*) dxs_kernel<2, 4, true>);
    add("dxs<4,4,F>", (const void*) dxs_kernel<4, 4, true>);
}

#else  // not the experimental sm_60 CUDA build: nothing applies
int64_t qsadec_select_scratch_ints(int64_t) { return 0; }
bool qsadec_select(const float*, const float*, const float*, const int32_t*, int64_t, int64_t, int64_t, const QsaShapes&,
                   float*, int32_t*, int32_t*, void*, int) { return false; }
bool qsadec_attn(const float*, const QsaAttnPools&, const int32_t*, const int32_t*, int64_t, const QsaShapes&, float*,
                 float*, int64_t, void*, int) { return false; }
void qsadec_bench_select_stage(int, int, const float*, const float*, const float*, const int32_t*, int64_t, int64_t,
                               int64_t, float*, int32_t*, int32_t*, void*) {}
void qsadec_bench_attn_abl(int, int, const float*, const QsaAttnPools&, const int32_t*, const int32_t*, int64_t,
                           const QsaShapes&, float*, float*, int64_t, void*) {}
void qsadec_bench_attrs(char* buf, int len) { if (len > 0) buf[0] = 0; }
#endif

}  // namespace strata::kernels
