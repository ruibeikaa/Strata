// src/kernels/cuda/nib_gu_proto.cuh - gu-proto: latency-hiding variants of nib_gu_kernel (STRATA_EXPERT_NIBBLE's
// 4-bit gate/up), included by iq_kernels.cu inside its anonymous namespace after nib_gu_kernel.
//
// Every variant keeps the S skeleton's per-row arithmetic: 16 lanes a row, lane t's items k = t, t+32, t+64 into s and
// t+16, t+48 into s16, each item NibLoad<TG>::load then Split<TG>::apply on the staged block_q8_1 (the production
// dp4a chain, integer scale step and float tail), __fadd_rn(s, s16), xor 8, 4, 2, 1.  Only the order of the loads,
// the staging and the block -> (group, rows) mapping change:
//   RB  row blocks (16 rows each) per block: grid (80 / RB, gy); a block stages a group's activations once and runs
//       its RB row blocks one after another
//   PF  1: the first row block's codes / scales / d are loaded before the staging (in flight during it), and row block
//       j + 1's before row block j is computed (software pipeline, one ahead); 0: loaded after the staging (production)
//   ST  0: the production word loop; 1: the entries' rows as 16-byte loads, all issued before the stores
//   AB  ablations for timing only (outputs are not the production's): 1 loads only (no unpack, no math), 2 no unpack
//       (raw code words into the dp4a chain), 3 no math (unpack kept, the dp4a chain replaced by an xor fold), 4 no
//       weight loads (codes made from the row and lane), 5 no staging (no barriers, shared memory not written),
//       6 no pointer chain (entry 0, token 0: T=1 only), 7 empty (returns after reading n_groups), 8 = 1 + 5
//   IL  1: each row block runs nib_row_dot_s itself (the production function: each item loaded where it is used, 64
//       registers, 4 blocks / SM); 0: the lane's five items loaded into registers first (nibx_load, 80-144 registers)
// Ids (native_expert_nib_set_variant / STRATA_NIB_GU_PROTO): nibx_launch_tg.  The pick is 31 = RB2 ST1 IL1 (one staging
// per two row blocks, 16-byte staging loads, the production row dot): T=1 x1.25-1.27 against the S / P skeleton on
// PH402 die 5, every gate / up row bitwise equal.
#pragma once

template<int TG, int AB>
struct NibxW {
    using W = typename NibLoad<TG>::W;
    __device__ static W get(const uint4& q, uint32_t sc, __half d) {
        if constexpr (AB == 2) {   // no unpack: the code words as the grid words
            W r;
            const uint32_t qw[4] = { q.x, q.y, q.z, q.w };
#pragma unroll
            for (int m = 0; m < 4; ++m) {
                r.g[2 * m + 0] = (int) qw[m];
                r.g[2 * m + 1] = (int) (qw[m] >> 4);
            }
            if constexpr (TG == 22) { r.ls0 = sc & 0x0F; r.ls1 = sc >> 4; }
            else r.ls = (int) sc;
            r.dw = __half2float(d);
            return r;
        } else {
            return NibLoad<TG>::load(q, sc, d);
        }
    }
};

template<int TG, int AB>
__device__ __forceinline__ float nibx_apply(const typename NibLoad<TG>::W& w, const block_q8_1* xb, int iqs) {
    if constexpr (AB == 3) {   // no math: an xor fold of the unpacked words instead of the dp4a chain
        int a = w.g[0] ^ w.g[1] ^ w.g[2] ^ w.g[3] ^ w.g[4] ^ w.g[5] ^ w.g[6] ^ w.g[7];
        if constexpr (TG == 22) a ^= w.ls0 ^ w.ls1;
        else a ^= w.ls;
        return w.dw * (float) a;
    } else {
        return Split<TG>::apply(w, xb, iqs);
    }
}

// one row against NC staged entries, the codes of the lane's five items in registers (nib_row_dot_s's order)
template<int TG, int NC, int AB>
__device__ __forceinline__ void nibx_row_dot(const uint4 (&q)[5], const uint32_t (&sc)[5], const __half (&dh)[5],
                                             const block_q8_1* x, const int (&off)[NC], int t, float (&s)[NC]) {
    const int iqs = 2 * (t & 7), kb = t >> 3;
    if constexpr (AB == 1 || AB == 8) {   // loads only
        uint32_t a = 0;
#pragma unroll
        for (int i = 0; i < 5; ++i) a ^= q[i].x ^ q[i].y ^ q[i].z ^ q[i].w ^ sc[i] ^ (uint32_t) __half_as_ushort(dh[i]);
#pragma unroll
        for (int c = 0; c < NC; ++c) s[c] = __uint_as_float(a + c);
    } else {
    const block_q8_1* xb[NC];
#pragma unroll
    for (int c = 0; c < NC; ++c) { s[c] = 0.0f; xb[c] = x + off[c] + kb * 8; }
#pragma unroll
    for (int i = 0; i < 3; ++i) {
        const typename NibLoad<TG>::W w = NibxW<TG, AB>::get(q[i], sc[i], dh[i]);
#pragma unroll
        for (int c = 0; c < NC; ++c) s[c] += nibx_apply<TG, AB>(w, xb[c] + 32 * i, iqs);
    }
    float s16[NC];
#pragma unroll
    for (int c = 0; c < NC; ++c) s16[c] = 0.0f;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const typename NibLoad<TG>::W w = NibxW<TG, AB>::get(q[3 + i], sc[3 + i], dh[3 + i]);
#pragma unroll
        for (int c = 0; c < NC; ++c) s16[c] += nibx_apply<TG, AB>(w, xb[c] + 16 + 32 * i, iqs);
    }
#pragma unroll
    for (int c = 0; c < NC; ++c) {
        s[c] = __fadd_rn(s[c], s16[c]);
#pragma unroll
        for (int o = 8; o > 0; o >>= 1) s[c] += __shfl_xor_sync(0xffffffffu, s[c], o);
    }
    }
}

// lane t's five items of `row`: k = t, t+32, t+64, t+16, t+48 (codes, scale bytes, superblock d)
template<int AB>
__device__ __forceinline__ void nibx_load(const uint8_t* slot, int row, int t, uint4 (&q)[5], uint32_t (&sc)[5],
                                          __half (&dh)[5]) {
    const int kb = t >> 3;
    if constexpr (AB == 4) {   // no weight loads
        const uint32_t h = (uint32_t) row * 0x9E3779B9u + (uint32_t) t * 0x85EBCA6Bu + (uint32_t) (uintptr_t) slot;
#pragma unroll
        for (int i = 0; i < 5; ++i) {
            const uint32_t u = h + 0x632BE5ABu * (uint32_t) i;
            q[i] = make_uint4(u & 0x77777777u, (u >> 1) & 0x77777777u, (u ^ 0x5A5A5A5Au) & 0x77777777u, (u * 3u) & 0x77777777u);
            sc[i] = (u >> 24) & 0x0F;
            dh[i] = __ushort_as_half((unsigned short) (0x2000u | (u & 0x3FFu)));
        }
        return;
    }
    const uint4* crow = (const uint4*) (slot + nibk::kCodesOff) + (size_t) row * nibk::kItems;
    const uint8_t* srow = slot + nibk::kScOff + (size_t) row * nibk::kItems;
    const __half* drow = (const __half*) (slot + nibk::kDOff) + (size_t) row * nibk::kSuper;
    q[0] = crow[t]; q[1] = crow[t + 32]; q[2] = crow[t + 64]; q[3] = crow[t + 16]; q[4] = crow[t + 48];
    sc[0] = srow[t]; sc[1] = srow[t + 32]; sc[2] = srow[t + 64]; sc[3] = srow[t + 16]; sc[4] = srow[t + 48];
    dh[0] = drow[kb]; dh[1] = drow[kb + 4]; dh[2] = drow[kb + 8]; dh[3] = drow[kb + 2]; dh[4] = drow[kb + 6];
}

// The n (1..GRP_NC) entries from e into s_x (block_q8_1 rows of 80).  Block-uniform (e, n): every thread reaches both
// barriers.  ST 0: nib_pass_s's word loop; ST 1: 16-byte loads (an entry's row is 2,880 B = 180 x 16, 16-aligned when
// xq is), all in flight before the first store.
template<int ST, int AB>
__device__ __forceinline__ void nibx_stage(block_q8_1* s_x, const int32_t* __restrict__ ent_tok,
                                           const block_q8_1* __restrict__ xq, int e, int n) {
    if constexpr (AB == 5 || AB == 8) return;
    if constexpr (ST == 0) {
        int tok[GRP_NC];
#pragma unroll
        for (int c = 0; c < GRP_NC; ++c) tok[c] = AB == 6 ? 0 : ent_tok[e + min(c, n - 1)];
        __syncthreads();
        for (int i = threadIdx.x; i < n * 80 * 9; i += 256) {
            const int c = i / 720, rem = i % 720;
            ((int*) s_x)[c * 720 + rem] = ((const int*) (xq + (size_t) tok[c] * 80))[rem];
        }
        __syncthreads();
    } else {
        constexpr int kV = (GRP_NC * 180 + 255) / 256;   // 3
        const int nv = n * 180;
        uint4 v[kV];
#pragma unroll
        for (int j = 0; j < kV; ++j) {
            const int i = threadIdx.x + 256 * j;
            if (i < nv) {
                const int c = i >= 540 ? 3 : i >= 360 ? 2 : i >= 180 ? 1 : 0;
                const int tok = AB == 6 ? 0 : ent_tok[e + c];
                v[j] = ((const uint4*) (xq + (size_t) tok * 80))[i - 180 * c];
            }
        }
        __syncthreads();
#pragma unroll
        for (int j = 0; j < kV; ++j) {
            const int i = threadIdx.x + 256 * j;
            if (i < nv) ((uint4*) s_x)[i] = v[j];
        }
        __syncthreads();
    }
}

// this thread's row against the staged n entries, the result written to row r of entries e..e+n-1
template<int TG, int AB>
__device__ __forceinline__ void nibx_run(const uint4 (&q)[5], const uint32_t (&sc)[5], const __half (&dh)[5],
                                         const block_q8_1* s_x, int e, int n, int t, float* dst, int r) {
    if (n == 1) {
        const int o[1] = { 0 }; float s[1];
        nibx_row_dot<TG, 1, AB>(q, sc, dh, s_x, o, t, s);
        if (t == 0) dst[(size_t) e * nibk::kNFf + r] = s[0];
    } else if (n == 2) {
        const int o[2] = { 0, 80 }; float s[2];
        nibx_row_dot<TG, 2, AB>(q, sc, dh, s_x, o, t, s);
        if (t < 2) dst[(size_t) (e + t) * nibk::kNFf + r] = t == 0 ? s[0] : s[1];
    } else if (n == 3) {
        const int o[3] = { 0, 80, 160 }; float s[3];
        nibx_row_dot<TG, 3, AB>(q, sc, dh, s_x, o, t, s);
        if (t < 3) dst[(size_t) (e + t) * nibk::kNFf + r] = t == 0 ? s[0] : t == 1 ? s[1] : s[2];
    } else {
        const int o[4] = { 0, 80, 160, 240 }; float s[4];
        nibx_row_dot<TG, 4, AB>(q, sc, dh, s_x, o, t, s);
        if (t < 4) dst[(size_t) (e + t) * nibk::kNFf + r] = t == 0 ? s[0] : t == 1 ? s[1] : t == 2 ? s[2] : s[3];
    }
}

// IL: nib_pass_s's compute on the staged entries (nib_row_dot_s, the production function), stored without the
// dynamically indexed s[t]
template<int TG>
__device__ __forceinline__ void nibx_run_il(const uint8_t* slot, int row, const block_q8_1* s_x, int e, int n, int t,
                                            float* dst, int r) {
    const uint4* crow = (const uint4*) (slot + nibk::kCodesOff) + (size_t) row * nibk::kItems;
    const uint8_t* srow = slot + nibk::kScOff + (size_t) row * nibk::kItems;
    const __half* drow = (const __half*) (slot + nibk::kDOff) + (size_t) row * nibk::kSuper;
    if (n == 1) {
        const int o[1] = { 0 }; float s[1];
        nib_row_dot_s<TG, 1>(crow, srow, drow, s_x, o, t, s);
        if (t == 0) dst[(size_t) e * nibk::kNFf + r] = s[0];
    } else if (n == 2) {
        const int o[2] = { 0, 80 }; float s[2];
        nib_row_dot_s<TG, 2>(crow, srow, drow, s_x, o, t, s);
        if (t < 2) dst[(size_t) (e + t) * nibk::kNFf + r] = t == 0 ? s[0] : s[1];
    } else if (n == 3) {
        const int o[3] = { 0, 80, 160 }; float s[3];
        nib_row_dot_s<TG, 3>(crow, srow, drow, s_x, o, t, s);
        if (t < 3) dst[(size_t) (e + t) * nibk::kNFf + r] = t == 0 ? s[0] : t == 1 ? s[1] : s[2];
    } else {
        const int o[4] = { 0, 80, 160, 240 }; float s[4];
        nib_row_dot_s<TG, 4>(crow, srow, drow, s_x, o, t, s);
        if (t < 4) dst[(size_t) (e + t) * nibk::kNFf + r] = t == 0 ? s[0] : t == 1 ? s[1] : t == 2 ? s[2] : s[3];
    }
}

template<int TG, int RB, int PF, int ST, int AB, int IL>
__global__ void __launch_bounds__(256) nibx_gu_kernel(const unsigned long long* __restrict__ grp_ptr,
                                                      const int32_t* __restrict__ grp_start,
                                                      const int32_t* __restrict__ n_groups,
                                                      const int32_t* __restrict__ ent_tok,
                                                      const block_q8_1* __restrict__ xq, float* __restrict__ gate,
                                                      float* __restrict__ up) {
    static_assert(80 % RB == 0, "row blocks per block divide 80");
    const int ng = *n_groups;
    if (blockIdx.y >= (unsigned) ng) return;
    if constexpr (AB == 7) return;
    STRATA_SHARED_ALIGN16_U32(s_buf, kNibSWords);
    block_q8_1* s_x = (block_q8_1*) s_buf;
    const int subwarp = threadIdx.x >> 4, t = threadIdx.x & 15;
    const int rb0 = blockIdx.x * RB;
    for (int g = blockIdx.y; g < ng; g += gridDim.y) {
        const uint8_t* slot = (const uint8_t*) grp_ptr[g];
        uint4 q[5]; uint32_t sc[5]; __half dh[5];
        if constexpr (PF) nibx_load<AB>(slot, rb0 * 16 + subwarp, t, q, sc, dh);
        const int e0 = AB == 6 ? 0 : grp_start[g], e1 = AB == 6 ? 1 : grp_start[g + 1];
        const int nt = e1 - e0;
        if (nt <= 0) continue;   // an empty group: nothing to write (block-uniform, no barrier skipped unevenly)
        if (nt <= GRP_NC) {
            nibx_stage<ST, AB>(s_x, ent_tok, xq, e0, nt);
#pragma unroll 1
            for (int j = 0; j < RB; ++j) {
                const int row = (rb0 + j) * 16 + subwarp;
                const bool is_up = row >= nibk::kNFf;
                const int r = is_up ? row - (int) nibk::kNFf : row;
                float* dst = is_up ? up : gate;
                if constexpr (IL) {   // the production row dot (nib_row_dot_s: each item loaded where it is used)
                    nibx_run_il<TG>(slot, row, s_x, e0, nt, t, dst, r);
                } else if constexpr (!PF) {
                    nibx_load<AB>(slot, row, t, q, sc, dh);
                    nibx_run<TG, AB>(q, sc, dh, s_x, e0, nt, t, dst, r);
                } else {
                    uint4 qn[5]; uint32_t scn[5]; __half dhn[5];
                    if (j + 1 < RB) nibx_load<AB>(slot, row + 16, t, qn, scn, dhn);
                    nibx_run<TG, AB>(q, sc, dh, s_x, e0, nt, t, dst, r);
                    if (j + 1 < RB) {
#pragma unroll
                        for (int i = 0; i < 5; ++i) { q[i] = qn[i]; sc[i] = scn[i]; dh[i] = dhn[i]; }
                    }
                }
            }
        } else {   // more entries than one pass holds: per row block, the passes as nib_pass_s runs them
#pragma unroll 1
            for (int j = 0; j < RB; ++j) {
                const int row = (rb0 + j) * 16 + subwarp;
                const bool is_up = row >= nibk::kNFf;
                const int r = is_up ? row - (int) nibk::kNFf : row;
                float* dst = is_up ? up : gate;
                if (!IL && (!PF || j > 0)) nibx_load<AB>(slot, row, t, q, sc, dh);
                for (int e = e0; e < e1; e += GRP_NC) {
                    const int n = min(GRP_NC, e1 - e);
                    nibx_stage<ST, AB>(s_x, ent_tok, xq, e, n);
                    if constexpr (IL) nibx_run_il<TG>(slot, row, s_x, e, n, t, dst, r);
                    else nibx_run<TG, AB>(q, sc, dh, s_x, e, n, t, dst, r);
                }
            }
        }
    }
}

// The configurations (id -> RB, PF, ST, AB, IL).  10..18 ablations of the base, 2x variants (r1: 20, 22..27, the
// prefetching ones, had 96-144 registers and were slower; dropped), 3x IL variants (the production row dot, the
// staging amortized over RB row blocks).
int g_nibx_occ = 0;   // the last attribute query's resident blocks per SM

template<int TG, int RB, int PF, int ST, int AB, int IL>
void nibx_launch_one(dim3 grid, cudaStream_t s, const unsigned long long* grp_ptr, const int32_t* grp_start,
                     const int32_t* n_groups, const int32_t* ent_tok, const block_q8_1* X, float* gate, float* up,
                     cudaFuncAttributes* attr) {
    if (attr) {
        cudaFuncGetAttributes(attr, nibx_gu_kernel<TG, RB, PF, ST, AB, IL>);
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&g_nibx_occ, nibx_gu_kernel<TG, RB, PF, ST, AB, IL>, 256, 0);
        return;
    }
    nibx_gu_kernel<TG, RB, PF, ST, AB, IL><<<dim3(80 / RB, grid.y), 256, 0, s>>>(grp_ptr, grp_start, n_groups, ent_tok, X, gate, up);
}

// false: not a gu-proto id.  attr != nullptr: only the kernel's attributes, nothing launched.
template<int TG>
bool nibx_launch_tg(int id, dim3 grid, cudaStream_t s, const unsigned long long* grp_ptr, const int32_t* grp_start,
                    const int32_t* n_groups, const int32_t* ent_tok, const block_q8_1* X, float* gate, float* up,
                    cudaFuncAttributes* attr) {
#define STRATA_NIBX(ID, RB, PF, ST, AB, IL) \
    case ID: nibx_launch_one<TG, RB, PF, ST, AB, IL>(grid, s, grp_ptr, grp_start, n_groups, ent_tok, X, gate, up, attr); return true;
    switch (id) {
        STRATA_NIBX(10, 1, 0, 0, 0, 0) STRATA_NIBX(11, 1, 0, 0, 1, 0) STRATA_NIBX(12, 1, 0, 0, 2, 0) STRATA_NIBX(13, 1, 0, 0, 3, 0)
        STRATA_NIBX(14, 1, 0, 0, 4, 0) STRATA_NIBX(15, 1, 0, 0, 5, 0) STRATA_NIBX(16, 1, 0, 0, 6, 0) STRATA_NIBX(17, 1, 0, 0, 7, 0)
        STRATA_NIBX(18, 1, 0, 0, 8, 0)
        STRATA_NIBX(21, 1, 0, 1, 0, 0) STRATA_NIBX(28, 2, 0, 1, 0, 0) STRATA_NIBX(29, 4, 0, 1, 0, 0)
        STRATA_NIBX(30, 1, 0, 1, 0, 1) STRATA_NIBX(31, 2, 0, 1, 0, 1) STRATA_NIBX(32, 4, 0, 1, 0, 1) STRATA_NIBX(33, 5, 0, 1, 0, 1)
        STRATA_NIBX(34, 8, 0, 1, 0, 1) STRATA_NIBX(35, 10, 0, 1, 0, 1) STRATA_NIBX(36, 4, 0, 0, 0, 1) STRATA_NIBX(37, 5, 0, 0, 0, 1)
        STRATA_NIBX(38, 16, 0, 1, 0, 1) STRATA_NIBX(39, 20, 0, 1, 0, 1)
        default: return false;
    }
#undef STRATA_NIBX
}

bool nibx_launch(int gu_type, int id, dim3 grid, cudaStream_t s, const unsigned long long* grp_ptr,
                 const int32_t* grp_start, const int32_t* n_groups, const int32_t* ent_tok, const block_q8_1* X,
                 float* gate, float* up, cudaFuncAttributes* attr = nullptr) {
    switch (gu_type) {
        case 22: return nibx_launch_tg<22>(id, grid, s, grp_ptr, grp_start, n_groups, ent_tok, X, gate, up, attr);
        case 18: return nibx_launch_tg<18>(id, grid, s, grp_ptr, grp_start, n_groups, ent_tok, X, gate, up, attr);
        case 21: return nibx_launch_tg<21>(id, grid, s, grp_ptr, grp_start, n_groups, ent_tok, X, gate, up, attr);
        default: return false;
    }
}
