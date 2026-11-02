// src/kernels/cuda/qsa_pattn_h16v.cuh - STRATA_PF_HFMA_ATTN=2: the value kernel on FP16 pairs.  Included by qsa_pattn.cu after
// qsa_pattn_h16.cuh, inside its anonymous namespace.
//
// ps_value_h_kernel is ps_value_d4_kernel on HFMA2, reading ps_score_h_kernel<true>'s scratch.  One CTA per (KV head, query) with
// VH_CG chunk groups of two warps each; the two warps of a group split the 256 dimensions (a lane owns 4 of them: two FP16 pairs) and the
// group walks the chunks c = group, group + VH_CG, ... in order.  Per chunk the group's 64 threads stage the chunk's 12 x 64 probabilities
// (each already a half2 (p, p): no conversion) and its 64 pool rows into shared memory with coalesced loads - the next chunk's are in
// flight in registers while this one computes - and a lane then runs its cells with VH_D V loads in flight (one 32-bit load of its four
// codes and the fp16 scale of its group): v = code * scale as two half2, 24 HFMA2 into the chunk's FP16 sums, which are widened to FP32 and
// merged with the chunk weight exp(m - M) when the chunk ends (an FP16 sum holds at most 64 products).  The chunk groups' FP32 sums are
// added in a fixed order: reassociated against ps_value_d4_kernel's single ascending fold, not bitwise.  A masked cell has probability 0
// and zero codes and scale.  An FP16 sum that overflows (or a V that is not finite) shows as a non-finite FP32 accumulator: the CTA then
// redoes its whole output with FP32 products from the same probabilities and counts itself in g_ph16_value_fb (qsa_pattn_h16_report).
#pragma once

#ifndef VH_CG_DEFAULT
#define VH_CG_DEFAULT 1
#endif
constexpr int VH_DW = 2;                    // warps across the 256 dimensions
constexpr int VH_CG = VH_CG_DEFAULT;        // chunk groups a CTA runs
constexpr int VH_T = 32 * VH_DW * VH_CG;
#ifndef VH_D_DEFAULT
#define VH_D_DEFAULT 8
#endif
constexpr int VH_D = VH_D_DEFAULT;           // V loads in flight a warp

__device__ unsigned long long g_ph16_value_fb = 0;

__global__ void __launch_bounds__(VH_T) ps_value_h_kernel(QsaAttnPools p, const int32_t* __restrict__ steps,
                                                          const float* __restrict__ blocks, int n_kv_heads, int n_chunks,
                                                          float* __restrict__ attn) {
    const int kvh = blockIdx.x, z = blockIdx.y;
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int dh = warp & 1, cg = warp >> 1, tg = t & 63;   // dimension half, chunk group, thread of the group
    const float* const blk0 = blocks + ((size_t) z * n_kv_heads + kvh) * n_chunks * PS_BLK;
    __shared__ float sm_m[NCM][G], sm_l[NCM][G], sm_w[NCM][G];
    __shared__ float sL[G];
    __shared__ __align__(16) uint32_t swt[VH_CG][CHUNK * G];
    __shared__ int srw[VH_CG][CHUNK];
    __shared__ __align__(16) float red[VH_CG > 1 ? G : 1][VH_CG > 1 ? HD : 4];
    __shared__ int s_bad;
    const int width = __ldg(steps + (size_t) z * kStepCount + kStepWidth);
    const int n_act = width > 0 ? min(n_chunks, (width - 1) / CHUNK + 1) : 0;
    if (t == 0) s_bad = 0;
    for (int i = t; i < n_act * G; i += VH_T) {
        const int c = i / G, h = i % G;
        sm_m[c][h] = __ldg(blk0 + (size_t) c * PS_BLK + OFF_M + h);
        sm_l[c][h] = __ldg(blk0 + (size_t) c * PS_BLK + OFF_M + G + h);
    }
    __syncthreads();
    if (t < G) {   // attn_merge_kernel's M, w and L
        float M = -FLT_MAX;
        for (int c = 0; c < n_act; ++c) M = fmaxf(M, sm_m[c][t]);
        float L = 0.0f;
        for (int c = 0; c < n_act; ++c) {
            const float m = sm_m[c][t];
            const float w = __expf(m - M);
            sm_w[c][t] = w;
            const float L_next = fmaf(sm_l[c][t], w, L);
            L = (m == -FLT_MAX) ? L : L_next;
        }
        sL[t] = L;
    }
    __syncthreads();
    const int d0 = dh * 128 + lane * 4, gsc = d0 / KV_Q8_GROUP;
    float acc[4][G];
#pragma unroll
    for (int h = 0; h < G; ++h)
#pragma unroll
        for (int d = 0; d < 4; ++d) acc[d][h] = 0.0f;
    const auto group_sync = [&]() {
        if constexpr (VH_CG == 1) __syncthreads();
        else asm volatile("bar.sync %0, 64;" ::"r"(1 + cg));
    };
    uint4 nw0 = make_uint4(0, 0, 0, 0), nw1 = nw0, nw2 = nw0;
    long long nr = -1;
    const auto stage_load = [&](const int c) {   // chunk c's probabilities and rows, three uint4 and one row a thread
        const float* b = blk0 + (size_t) c * PS_BLK;
        const uint4* wv = reinterpret_cast<const uint4*>(b);
        nw0 = __ldg(wv + tg);
        nw1 = __ldg(wv + tg + 64);
        nw2 = __ldg(wv + tg + 128);
        nr = __ldg(reinterpret_cast<const long long*>(b + G * CHUNK) + tg);
    };
    if (cg < n_act) stage_load(cg);
    for (int c = cg; c < n_act; c += VH_CG) {
        uint4* const sw4 = reinterpret_cast<uint4*>(swt[cg]);
        sw4[tg] = nw0;
        sw4[tg + 64] = nw1;
        sw4[tg + 128] = nw2;
        srw[cg][tg] = (int) nr;
        group_sync();
        if (c + VH_CG < n_act) stage_load(c + VH_CG);   // in flight while this chunk computes
        const int n_here = min(CHUNK, width - c * CHUNK);
        const uint32_t* const sw = swt[cg];
        const int* const sr = srw[cg];
        const unsigned live_lo = __ballot_sync(0xffffffffu, sr[lane] >= 0), live_hi = __ballot_sync(0xffffffffu, sr[lane + 32] >= 0);
        uint32_t a[2][G];
#pragma unroll
        for (int h = 0; h < G; ++h) { a[0][h] = 0u; a[1][h] = 0u; }
        uint32_t cw[VH_D], sc[VH_D];
        const auto vload = [&](const int ci, uint32_t& cwv, uint32_t& scv) {
            const int r = sr[ci];
            const size_t rr = r >= 0 ? (size_t) r : 0;
            cwv = __ldg(reinterpret_cast<const uint32_t*>(p.v_q + (rr << 8) + d0));
            scv = (uint32_t) __ldg(p.v_scale + (rr << 2) + gsc);
        };
        const auto fold = [&](const int ci, const uint32_t cwv, const uint32_t scv) {
            const bool live = ci < 32 ? ((live_lo >> ci) & 1u) != 0 : ((live_hi >> (ci - 32)) & 1u) != 0;
            uint32_t v01, v23;
            i8x4_to_h2(live ? cwv : 0u, v01, v23);
            const uint32_t s2 = live ? scv * 0x00010001u : 0u;
            v01 = hf_mul2(v01, s2);
            v23 = hf_mul2(v23, s2);
            const uint4* const wp = reinterpret_cast<const uint4*>(sw + ci * G);
            const uint4 w0 = wp[0], w1 = wp[1], w2 = wp[2];
            const uint32_t ww[G] = {w0.x, w0.y, w0.z, w0.w, w1.x, w1.y, w1.z, w1.w, w2.x, w2.y, w2.z, w2.w};
#pragma unroll
            for (int h = 0; h < G; ++h) {
                a[0][h] = hf_fma2(ww[h], v01, a[0][h]);
                a[1][h] = hf_fma2(ww[h], v23, a[1][h]);
            }
        };
#pragma unroll
        for (int u = 0; u < VH_D; ++u)
            if (u < n_here) vload(u, cw[u], sc[u]);
        int cl = 0;
        for (; cl + VH_D <= n_here; cl += VH_D) {
#pragma unroll
            for (int u = 0; u < VH_D; ++u) {
                fold(cl + u, cw[u], sc[u]);
                if (cl + u + VH_D < n_here) vload(cl + u + VH_D, cw[u], sc[u]);
            }
        }
#pragma unroll
        for (int u = 0; u < VH_D; ++u)
            if (cl + u < n_here) fold(cl + u, cw[u], sc[u]);
#pragma unroll
        for (int h = 0; h < G; ++h) {   // the chunk's FP16 sums into the FP32 ones, weighted as attn_merge_kernel does
            const float w = sm_w[c][h];
            const bool live = !(sm_m[c][h] == -FLT_MAX);
            const float2 f0 = hf_to_f2(a[0][h]), f1 = hf_to_f2(a[1][h]);
            acc[0][h] = live ? fmaf(f0.x, w, acc[0][h]) : acc[0][h];
            acc[1][h] = live ? fmaf(f0.y, w, acc[1][h]) : acc[1][h];
            acc[2][h] = live ? fmaf(f1.x, w, acc[2][h]) : acc[2][h];
            acc[3][h] = live ? fmaf(f1.y, w, acc[3][h]) : acc[3][h];
        }
        group_sync();   // every thread of the group is done with swt / srw: the next chunk restages them
    }
    {
        bool bad = false;
#pragma unroll
        for (int h = 0; h < G; ++h)
#pragma unroll
            for (int d = 0; d < 4; ++d) bad |= !(fabsf(acc[d][h]) <= FLT_MAX);
        if (bad) s_bad = 1;
    }
    __syncthreads();
    if (s_bad) {   // the same sums with FP32 products (the probabilities are the half2 ones the kernel was given)
        if (t == 0) atomicAdd(&g_ph16_value_fb, 1ull);
#pragma unroll
        for (int h = 0; h < G; ++h)
#pragma unroll
            for (int d = 0; d < 4; ++d) acc[d][h] = 0.0f;
        for (int c = cg; c < n_act; c += VH_CG) {
            const float* blk = blk0 + (size_t) c * PS_BLK;
            const uint32_t* const wh = reinterpret_cast<const uint32_t*>(blk);
            const long long* const rows = reinterpret_cast<const long long*>(blk + G * CHUNK);
            const int n_here = min(CHUNK, width - c * CHUNK);
            float a[4][G];
#pragma unroll
            for (int h = 0; h < G; ++h)
#pragma unroll
                for (int d = 0; d < 4; ++d) a[d][h] = 0.0f;
            for (int cl = 0; cl < n_here; ++cl) {
                const long long r = __ldg(rows + cl);
                if (r < 0) continue;
                const uint32_t codes = __ldg(reinterpret_cast<const uint32_t*>(p.v_q + (size_t) r * HD + d0));
                const float sc = __half2float(__ushort_as_half(__ldg(p.v_scale + (size_t) r * NSC + gsc)));
                const float v0 = (float) (int8_t) (codes & 0xFF) * sc, v1 = (float) (int8_t) ((codes >> 8) & 0xFF) * sc;
                const float v2 = (float) (int8_t) ((codes >> 16) & 0xFF) * sc, v3 = (float) (int8_t) (codes >> 24) * sc;
#pragma unroll
                for (int h = 0; h < G; ++h) {
                    const float w = __half2float(__ushort_as_half((unsigned short) (__ldg(wh + (size_t) cl * G + h) & 0xFFFFu)));
                    a[0][h] = fmaf(w, v0, a[0][h]);
                    a[1][h] = fmaf(w, v1, a[1][h]);
                    a[2][h] = fmaf(w, v2, a[2][h]);
                    a[3][h] = fmaf(w, v3, a[3][h]);
                }
            }
#pragma unroll
            for (int h = 0; h < G; ++h) {
                const float w = sm_w[c][h];
                const bool live = !(sm_m[c][h] == -FLT_MAX);
#pragma unroll
                for (int d = 0; d < 4; ++d) acc[d][h] = live ? fmaf(a[d][h], w, acc[d][h]) : acc[d][h];
            }
        }
    }
    if constexpr (VH_CG > 1) {
        if (cg == 1) {
#pragma unroll
            for (int h = 0; h < G; ++h)
                *reinterpret_cast<float4*>(&red[h][d0]) = make_float4(acc[0][h], acc[1][h], acc[2][h], acc[3][h]);
        }
        __syncthreads();
    }
    if (cg == 0) {
        float* out = attn + ((size_t) z * n_kv_heads + kvh) * G * HD;
#pragma unroll
        for (int h = 0; h < G; ++h) {
            float4 r = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            if constexpr (VH_CG > 1) r = *reinterpret_cast<const float4*>(&red[h][d0]);
            const float L = sL[h];
            *reinterpret_cast<float4*>(out + (size_t) h * HD + d0) =
                make_float4(L > 0.0f ? (acc[0][h] + r.x) / L : 0.0f, L > 0.0f ? (acc[1][h] + r.y) / L : 0.0f,
                            L > 0.0f ? (acc[2][h] + r.z) / L : 0.0f, L > 0.0f ? (acc[3][h] + r.w) / L : 0.0f);
        }
    }
}
