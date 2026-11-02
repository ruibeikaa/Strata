// src/kernels/cuda/qsa_pattn_h16.cuh - STRATA_PF_HFMA_ATTN: the score kernel of the prompt's QSA attention on FP16 HFMA2 (GP100).
// Included by qsa_pattn.cu inside its anonymous namespace (it uses CHUNK, G, HD, NSC, PS_BLK, cell_row, warp_max_x, warp_sum_x).
//
// ps_score_x_kernel (the bit-exact route) computes a chunk's 12 x 64 scores with one lane per 8 of a cell's 256 dimensions: 96 FFMA,
// 16 conversion/scale instructions and a 5-level shuffle reduction (25 shuffles, 25 adds per two cells) for 96 products, ~150 issue
// slots a cell.  Here a thread owns one (cell, 64-dimension K scale group) and walks all 12 query heads over it in FP16 pairs:
//   - K arrives as int8 codes; four codes become two exact half2 (code + 128 as the low byte of 0x6400 is 1024 + u, minus 1152 is
//     the code: 1 LOP + 2 PRMT + 2 HSUB2 for four codes), so the K scale never touches an element;
//   - q is the FP16 image of the query rows, scaled by an exact power of two 2^-e per CTA so that max |q'| < 8: a lane's
//     32 products (each half of a pair sums every second dimension) are at most 32 * 127 * 8 = 32512 < 65504 whatever the data,
//     so an FP16 sum can never overflow (SH_FLUSH = 2 widens after 16 products a half);
//   - one thread's sums are finished in FP32: every SH_FLUSH (2) steps of 16 dimensions the pair sums are widened (lo + hi) and
//     added to an FP32 sum, which is multiplied by the K scale of the group (fp16 -> float), the four groups of a cell added in the
//     order 0..3 and the total by 2^e / sqrt(256); no cross-lane reduction at all (the 25 shuffles a cell are gone);
//   - 12 heads x 8 HFMA2 per 16 dimensions need 24 broadcast LDS.128 of q' (every lane of a warp reads the same address) and one
//     LDS.128 of K; the 384 HFMA2 a thread are 4.35 of the kernel's 12.4 ms per 1536 queries (an ablation without them: 8.1 ms), the
//     rest is the chunk's setup, the combine of the four groups and the softmax (qsa_pattn_h16_bench).
// The softmax, the probabilities, rows, m and l written to the scratch are ps_score_x_kernel's (WH false: the same float layout,
// so ps_value_d4_kernel reads them; WH true: the layout of ps_value_h_kernel, see there).  The scores differ from the FP32 route's
// by the FP16 rounding of q' (2^-11 relative) and of the products' sums: qsa_pattn_h16_bench measures it (relative RMS of the
// attention output against an FP64 reference 1.2e-3, the bit-exact route 3e-7).
#pragma once

// ---- FP16 pair helpers (PTX: the compiler's __hfma2 may contract or reorder; these are the instructions)
__device__ __forceinline__ uint32_t hf_fma2(const uint32_t a, const uint32_t b, const uint32_t c) {
    uint32_t d;
    asm("fma.rn.f16x2 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
    return d;
}
__device__ __forceinline__ uint32_t hf_sub2(const uint32_t a, const uint32_t b) {
    uint32_t d;
    asm("sub.rn.f16x2 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b));
    return d;
}
__device__ __forceinline__ uint32_t hf_mul2(const uint32_t a, const uint32_t b) {
    uint32_t d;
    asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b));
    return d;
}
__device__ __forceinline__ float2 hf_to_f2(const uint32_t h) {
    float lo, hi;
    asm("{\n\t.reg .f16 l, h;\n\tmov.b32 {l, h}, %2;\n\tcvt.f32.f16 %0, l;\n\tcvt.f32.f16 %1, h;\n\t}" : "=f"(lo), "=f"(hi) : "r"(h));
    return make_float2(lo, hi);
}
// four int8 codes (the bytes of w, byte 0 first) -> half2 (c0, c1) and (c2, c3), exact
__device__ __forceinline__ void i8x4_to_h2(const uint32_t w, uint32_t& v01, uint32_t& v23) {
    const uint32_t u = w ^ 0x80808080u;                                       // code + 128 as an unsigned byte
    v01 = hf_sub2(__byte_perm(u, 0x64646464u, 0x4140), 0x64806480u);          // bytes (u0, 0x64, u1, 0x64) = half2 (1024 + u0, 1024 + u1)
    v23 = hf_sub2(__byte_perm(u, 0x64646464u, 0x4342), 0x64806480u);          // minus 1152 (0x6480): the codes
}
__device__ __forceinline__ uint32_t hf_pack(const float lo, const float hi) {
    uint32_t r;
    asm("{\n\t.reg .f16 l, h;\n\tcvt.rn.f16.f32 l, %1;\n\tcvt.rn.f16.f32 h, %2;\n\tmov.b32 %0, {l, h};\n\t}" : "=r"(r) : "f"(lo), "f"(hi));
    return r;
}

// the pool row of cell c0 + t of this query's selection (cell_row's, in 32 bits and without the runtime division for 4-cell pages), -1 masked
__device__ __forceinline__ int cell_row32(const QsaAttnPools& p, const int32_t* __restrict__ ids, int c0, int t, int n_here,
                                          int n_kv_heads, int kvh, int page_size) {
    int r = -1;
    if (t < n_here) {
        const int cell = __ldg(ids + c0 + t);
        const int pg = page_size == 4 ? cell >> 2 : cell / page_size;
        const int off = cell - pg * page_size;
        const int page = __ldg(p.page_table + pg);
        if (page >= 0) r = (page * n_kv_heads + kvh) * page_size + off;
    }
    return r;
}

static_assert(HD == 256 && NSC == 4 && CHUNK == 64 && G == 12, "qsa_pattn_h16 is written for 256-wide heads, 4 K scales, 64-cell chunks, 12 heads a KV head");
#ifndef SH_FLUSH_DEFAULT
#define SH_FLUSH_DEFAULT 2
#endif
constexpr int SH_FLUSH = SH_FLUSH_DEFAULT;   // 16-dimension steps a thread sums in FP16 pairs before widening (1, 2 or 4)
constexpr int SH_T = 256;          // threads: 8 warps = 4 K scale groups x 2 halves of the chunk's 64 cells
constexpr int SH_KSTR = 272;       // bytes of a staged K row (256 + 16): eight consecutive rows' 16-byte reads cover all 32 banks
constexpr int OFF_M = G * CHUNK + 2 * CHUNK;   // floats: m[12] then l[12] after the probabilities and the rows

// WH false: probabilities as floats [head][cell] (ps_score_x_kernel's layout); true: half2 (p, p) per (cell, head), [cell][12] words
template <bool WH>
__global__ void __launch_bounds__(SH_T, 2) ps_score_h_kernel(const float* __restrict__ q, QsaAttnPools p,
                                                             const int32_t* __restrict__ ids,
                                                             const int32_t* __restrict__ step, int n_kv_heads,
                                                             int page_size, float scale, float* __restrict__ blocks,
                                                             int n_chunks, int cap) {
    q += (size_t) blockIdx.z * (size_t) (n_kv_heads * G) * HD;
    ids += (size_t) blockIdx.z * (size_t) cap;
    step += (size_t) blockIdx.z * kStepCount;
    const int chunk = blockIdx.x, kvh = blockIdx.y;
    float* const blk = blocks + (((size_t) blockIdx.z * n_kv_heads + kvh) * n_chunks + chunk) * PS_BLK;
    __shared__ __align__(16) uint8_t sk[CHUNK * SH_KSTR];   // the chunk's K codes, later the four groups' partial scores
    __shared__ __align__(16) uint32_t sq[4 * 4 * G * 8];     // q' as half2: [group][16-dimension step][head][8 words]
    __shared__ float sks[CHUNK][NSC];
    __shared__ float sp[G][CHUNK];
    __shared__ int srow[CHUNK];
    __shared__ float smx[8];
    const int n_ids = __ldg(step + kStepWidth);
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int c0 = chunk * CHUNK;
    const int n_here = min(CHUNK, n_ids - c0);
    if (n_here <= 0) {
        if (t < G) { blk[OFF_M + t] = -FLT_MAX; blk[OFF_M + G + t] = 0.0f; }
        return;
    }
    if (t < CHUNK) {
        const int r = cell_row32(p, ids, c0, t, n_here, n_kv_heads, kvh, page_size);
        srow[t] = r;
        reinterpret_cast<long long*>(blk + G * CHUNK)[t] = (long long) r;
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
    // the K chunk: 16-byte global loads in flight while q is converted
    uint4 kv[4];
    float ksc;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int idx = t + SH_T * j, c = idx >> 4, part = idx & 15;
        const int row = srow[c];
        const size_t rr = (c < n_here && row >= 0) ? (size_t) row : 0;
        kv[j] = __ldg(reinterpret_cast<const uint4*>(p.k_q + (rr << 8)) + part);
    }
    {
        const int c = t >> 2, g = t & 3;
        const int row = srow[c];
        const size_t rr = (c < n_here && row >= 0) ? (size_t) row : 0;
        ksc = __half2float(__ushort_as_half(__ldg(p.k_scale + (rr << 2) + g)));
    }
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
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int idx = t + SH_T * j, c = idx >> 4, part = idx & 15;
        *reinterpret_cast<uint4*>(sk + c * SH_KSTR + part * 16) = kv[j];
    }
    sks[t >> 2][t & 3] = ksc;
    __syncthreads();   // sq, sk, sks
    // thread (cell, group): all 12 heads over the group's 64 dimensions
    const int g = warp & 3, cell = (warp >> 2) * 32 + lane;
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
        if ((j + 1) % SH_FLUSH == 0) {   // the FP16 pair sums into the FP32 ones: an FP16 sum holds 16 * SH_FLUSH products at most
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
            for (int u = 0; u < 4; ++u) {
                v[u] = facc[h4 * 4 + u] * scg;
            }
            dst[h4] = make_float4(v[0], v[1], v[2], v[3]);
        }
    }
    __syncthreads();
    for (int i = t; i < G * CHUNK; i += SH_T) {
        const int h = i >> 6, c = i & 63;
        const float s = ((part[(0 * CHUNK + c) * G + h] + part[(1 * CHUNK + c) * G + h]) + part[(2 * CHUNK + c) * G + h]) +
                        part[(3 * CHUNK + c) * G + h];
        sp[h][c] = (c < n_here && srow[c] >= 0) ? s * osc : -FLT_MAX;
    }
    __syncthreads();
    // the chunk softmax: ps_score_x_kernel's (warp_max over the two cells a lane holds, __expf, warp_sum)
    for (int h = warp; h < G; h += SH_T / 32) {
        const float a = sp[h][lane], b = sp[h][lane + 32];
        const float m = warp_max_x(fmaxf(a, b));
        const float ea = (lane < n_here && srow[lane] >= 0) ? __expf(a - m) : 0.0f;
        const float eb = (lane + 32 < n_here && srow[lane + 32] >= 0) ? __expf(b - m) : 0.0f;
        if (WH) {
            uint32_t* const wh = reinterpret_cast<uint32_t*>(blk);
            const uint32_t pa = hf_pack(ea, ea), pb = hf_pack(eb, eb);
            wh[lane * G + h] = pa;
            wh[(lane + 32) * G + h] = pb;
        } else {
            blk[h * CHUNK + lane] = ea;
            blk[h * CHUNK + lane + 32] = eb;
        }
        const float l = warp_sum_x(ea + eb);
        if (lane == 0) { blk[OFF_M + h] = m; blk[OFF_M + G + h] = l; }
    }
}
