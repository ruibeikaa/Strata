// src/kernels/cuda/qsa_pattn_h16p.cuh - STRATA_PF_ATTN_PERSIST: ps_score_h_kernel as a persistent CTA with a fused epilogue.  Included by
// qsa_pattn.cu after qsa_pattn_h16.cuh, inside its anonymous namespace (it uses that header's helpers and constants).
//
// ps_score_h_kernel runs one CTA per (chunk, KV head, query).  Ablations on a PH402 die (qsa_pattn_h16 bench, 100 queries, 1536-query
// equivalents): 13.4 ms; without the HFMA2 loop 8.0 ms; without the combine + softmax 8.6 ms; without the K loads from global memory
// 12.7 ms.  So the HFMA2 loop (4.4 ms) and the per-chunk epilogue (3.8 ms: the four groups' combine, a one-head-a-warp softmax in two
// rounds, 12 heads on 8 warps) are what there is to cut, the K latency less (0.7 ms).  Here:
//   - a CTA owns SP_NPC chunks of one (KV head, query) (chunks x, x + S, x + 2 S, ...): q' (the FP16 image and its power of two 2^-e) is
//     built once, the pool rows of all its chunks in one pass, and the next chunk's K codes and scales are in flight in registers while
//     this chunk computes (SP_NPC 1 is slower than the original: the K loads then start after the barrier);
//   - the epilogue is one round on 192 threads, 16 a head: thread (h, j) holds the cells j, j + 16, j + 32, j + 48, reads the four groups'
//     partial scores straight from shared memory (no sp round trip, one barrier less), and the max / l trees are ps_score_h_kernel's
//     (its warp_sum_x adds xor 16 first, i.e. (e_j + e_{j+32}) + (e_{j+16} + e_{j+48}), then xor 8, 4, 2, 1).
// The arithmetic of a (cell, head) is ps_score_h_kernel's operation for operation (same q', same K conversion, same order of FP16 pair
// sums, the FP32 flush, the four groups added 0..3, the same softmax): the scratch it writes is bitwise the original's
// (src/kernels/qsa_pattn_h16p_bench.cu).  Measured: 13.4 -> 11.2 ms per 1536 queries.
#pragma once

#ifndef SP_NPC_DEFAULT
#define SP_NPC_DEFAULT 6
#endif
constexpr int SP_NPC = SP_NPC_DEFAULT;   // chunks a CTA runs (33 chunks: 6 -> 6 CTAs a (KV head, query)); 1 is slower than ps_score_h_kernel
static_assert(SP_NPC >= 1 && SP_NPC <= 16, "SP_NPC");

template <bool WH>
__global__ void __launch_bounds__(SH_T, 2) ps_score_hp_kernel(const float* __restrict__ q, QsaAttnPools p,
                                                              const int32_t* __restrict__ ids,
                                                              const int32_t* __restrict__ step, int n_kv_heads,
                                                              int page_size, float scale, float* __restrict__ blocks,
                                                              int n_chunks, int cap, int stride) {
    q += (size_t) blockIdx.z * (size_t) (n_kv_heads * G) * HD;
    ids += (size_t) blockIdx.z * (size_t) cap;
    step += (size_t) blockIdx.z * kStepCount;
    const int kvh = blockIdx.y;
    float* const blk_base = blocks + (((size_t) blockIdx.z * n_kv_heads + kvh) * n_chunks) * PS_BLK;
    __shared__ __align__(16) uint8_t sk[CHUNK * SH_KSTR];   // the chunk's K codes, later the four groups' partial scores
    __shared__ __align__(16) uint32_t sq[4 * 4 * G * 8];     // q' as half2: [group][16-dimension step][head][8 words]
    __shared__ float sks[CHUNK][NSC];
    __shared__ int srow[SP_NPC * CHUNK];
    __shared__ float smx[8];
    const int n_ids = __ldg(step + kStepWidth);
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    // this CTA's chunks: x, x + stride, ... below n_chunks, those with cells first
    int my_n = 0;
    for (int j = 0; j < SP_NPC; ++j) {
        const int ch = (int) blockIdx.x + j * stride;
        if (ch < n_chunks && ch * CHUNK < n_ids) my_n = j + 1;
        else if (ch < n_chunks && t < G) {   // a chunk past the selection: m = -FLT_MAX, l = 0
            float* const b = blk_base + (size_t) ch * PS_BLK;
            b[OFF_M + t] = -FLT_MAX;
            b[OFF_M + G + t] = 0.0f;
        }
    }
    if (my_n == 0) return;
    // the pool rows of every chunk (cell_row32's), also into the scratch
    for (int i = t; i < my_n * CHUNK; i += SH_T) {
        const int j = i >> 6, c = i & 63;
        const int ch = (int) blockIdx.x + j * stride;
        const int c0 = ch * CHUNK;
        const int n_here = min(CHUNK, n_ids - c0);
        const int r = cell_row32(p, ids, c0, c, n_here, n_kv_heads, kvh, page_size);
        srow[i] = r;
        reinterpret_cast<long long*>(blk_base + (size_t) ch * PS_BLK + G * CHUNK)[c] = (long long) r;
    }
    // q: 12 heads x 256 floats, three float4 a thread; the CTA's largest |q| sets the power of two
    float4 qv[3];
    float mx = 0.0f;
#pragma unroll
    for (int k = 0; k < 3; ++k) {
        const int i = t + SH_T * k, h = i >> 6, dd = (i & 63) * 4;
        qv[k] = __ldg(reinterpret_cast<const float4*>(q + (size_t) (kvh * G + h) * HD + dd));
        mx = fmaxf(mx, fmaxf(fmaxf(fabsf(qv[k].x), fabsf(qv[k].y)), fmaxf(fabsf(qv[k].z), fabsf(qv[k].w))));
    }
    mx = warp_max_x(mx);
    if (lane == 0) smx[warp] = mx;
    __syncthreads();   // srow, smx
    uint4 kv[4];
    float ksc;
    const auto load_k = [&](const int j) {   // chunk j's K codes (16 B x 4 a thread) and K scale (one a thread), into registers
        const int n_here = min(CHUNK, n_ids - ((int) blockIdx.x + j * stride) * CHUNK);
#pragma unroll
        for (int jj = 0; jj < 4; ++jj) {
            const int idx = t + SH_T * jj, c = idx >> 4, part = idx & 15;
            const int row = srow[j * CHUNK + c];
            const size_t rr = (c < n_here && row >= 0) ? (size_t) row : 0;
            kv[jj] = __ldg(reinterpret_cast<const uint4*>(p.k_q + (rr << 8)) + part);
        }
        const int c = t >> 2, g = t & 3;
        const int row = srow[j * CHUNK + c];
        const size_t rr = (c < n_here && row >= 0) ? (size_t) row : 0;
        ksc = __half2float(__ushort_as_half(__ldg(p.k_scale + (rr << 2) + g)));
    };
    load_k(0);
    float gm = smx[0];
#pragma unroll
    for (int w = 1; w < 8; ++w) gm = fmaxf(gm, smx[w]);
    // gm = f 2^ex with f in [0.5, 1): e = ex - 3 puts max |q'| below 8; a q that is zero, below 8, or not finite keeps e = 0
    int e = 0;
    if (gm >= 8.0f && gm <= FLT_MAX) e = (int) ((__float_as_uint(gm) >> 23) & 255u) - 126 - 3;
    const float qs = __uint_as_float((uint32_t) (127 - e) << 23);       // 2^-e
    const float osc = scale * __uint_as_float((uint32_t) (127 + e) << 23);   // 1/16 * 2^e
#pragma unroll
    for (int k = 0; k < 3; ++k) {
        const int i = t + SH_T * k, h = i >> 6, dd = (i & 63) * 4;
        const int g = dd >> 6, j = (dd & 63) >> 4, w0 = (dd & 15) >> 1;
        uint2 pk;
        pk.x = hf_pack(qv[k].x * qs, qv[k].y * qs);
        pk.y = hf_pack(qv[k].z * qs, qv[k].w * qs);
        *reinterpret_cast<uint2*>(&sq[((g * 4 + j) * G + h) * 8 + w0]) = pk;
    }
    const int g = warp & 3, cell = (warp >> 2) * 32 + lane;
    for (int jc = 0; jc < my_n; ++jc) {
        const int chunk = (int) blockIdx.x + jc * stride;
        const int n_here = min(CHUNK, n_ids - chunk * CHUNK);
        float* const blk = blk_base + (size_t) chunk * PS_BLK;
        const int* const srw = srow + jc * CHUNK;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int idx = t + SH_T * j, c = idx >> 4, part = idx & 15;
            *reinterpret_cast<uint4*>(sk + c * SH_KSTR + part * 16) = kv[j];
        }
        sks[t >> 2][t & 3] = ksc;
        __syncthreads();   // sq, sk, sks
        if (jc + 1 < my_n) load_k(jc + 1);   // the next chunk's K in flight while this one computes
        // thread (cell, group): all 12 heads over the group's 64 dimensions
        uint32_t acc[G];
        float facc[G];
#pragma unroll
        for (int h = 0; h < G; ++h) { acc[h] = 0u; facc[h] = 0.0f; }
        const uint4* krow = reinterpret_cast<const uint4*>(sk + cell * SH_KSTR + g * 64);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const uint4 kk = krow[j];
            uint32_t kh[8];
            i8x4_to_h2(kk.x, kh[0], kh[1]);
            i8x4_to_h2(kk.y, kh[2], kh[3]);
            i8x4_to_h2(kk.z, kh[4], kh[5]);
            i8x4_to_h2(kk.w, kh[6], kh[7]);
            const uint4* qp = reinterpret_cast<const uint4*>(&sq[(g * 4 + j) * G * 8]);
#pragma unroll
            for (int h = 0; h < G; ++h) {
                const uint4 qa = qp[h * 2], qb = qp[h * 2 + 1];   // the same 32 bytes for the whole warp
                uint32_t a = acc[h];
                a = hf_fma2(qa.x, kh[0], a);
                a = hf_fma2(qa.y, kh[1], a);
                a = hf_fma2(qa.z, kh[2], a);
                a = hf_fma2(qa.w, kh[3], a);
                a = hf_fma2(qb.x, kh[4], a);
                a = hf_fma2(qb.y, kh[5], a);
                a = hf_fma2(qb.z, kh[6], a);
                a = hf_fma2(qb.w, kh[7], a);
                acc[h] = a;
            }
            if ((j + 1) % SH_FLUSH == 0) {   // the FP16 pair sums into the FP32 ones
#pragma unroll
                for (int h = 0; h < G; ++h) {
                    const float2 f = hf_to_f2(acc[h]);
                    facc[h] += f.x + f.y;
                    acc[h] = 0u;
                }
            }
        }
        const float scg = sks[cell][g];
        __syncthreads();   // every thread has read its K row: the buffer now takes the partial scores [group][cell][12]
        float* const part = reinterpret_cast<float*>(sk);
        {
            float4* dst = reinterpret_cast<float4*>(part + (size_t) (g * CHUNK + cell) * G);
#pragma unroll
            for (int h4 = 0; h4 < G / 4; ++h4) {
                float v[4];
#pragma unroll
                for (int u = 0; u < 4; ++u) v[u] = facc[h4 * 4 + u] * scg;
                dst[h4] = make_float4(v[0], v[1], v[2], v[3]);
            }
        }
        __syncthreads();
        // combine + chunk softmax in one pass, 16 threads a head (192 threads, one round): thread (h, j) holds cells j, j + 16, j + 32,
        // j + 48; the sums ((p0 + p1) + p2) + p3, the max, __expf and the l tree are ps_score_h_kernel's: its lane j holds cells j and
        // j + 32 and warp_sum_x adds xor 16 first, i.e. ((e_j + e_{j+32}) + (e_{j+16} + e_{j+48})), then xor 8, 4, 2, 1 (the same tree here)
        if (t < G * 16) {
            const int h = t >> 4, j = t & 15;
            float v[4];
            bool live[4];
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                const int c = j + 16 * k;
                const float sc = ((part[(0 * CHUNK + c) * G + h] + part[(1 * CHUNK + c) * G + h]) + part[(2 * CHUNK + c) * G + h]) +
                                 part[(3 * CHUNK + c) * G + h];
                live[k] = c < n_here && srw[c] >= 0;
                v[k] = live[k] ? sc * osc : -FLT_MAX;
            }
            float m = fmaxf(fmaxf(v[0], v[1]), fmaxf(v[2], v[3]));
            m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 8));
            m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 4));
            m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 2));
            m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 1));
            float ev[4];
#pragma unroll
            for (int k = 0; k < 4; ++k) ev[k] = live[k] ? __expf(v[k] - m) : 0.0f;
            float l = (ev[0] + ev[2]) + (ev[1] + ev[3]);
            l += __shfl_xor_sync(0xffffffffu, l, 8);
            l += __shfl_xor_sync(0xffffffffu, l, 4);
            l += __shfl_xor_sync(0xffffffffu, l, 2);
            l += __shfl_xor_sync(0xffffffffu, l, 1);
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                const int c = j + 16 * k;
                if (WH) reinterpret_cast<uint32_t*>(blk)[c * G + h] = hf_pack(ev[k], ev[k]);
                else blk[h * CHUNK + c] = ev[k];
            }
            if (j == 0) { blk[OFF_M + h] = m; blk[OFF_M + G + h] = l; }
        }
        __syncthreads();   // the next chunk's K codes restage sk (part)
    }
}
