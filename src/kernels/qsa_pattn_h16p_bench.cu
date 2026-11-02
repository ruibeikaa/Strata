// src/kernels/qsa_pattn_h16p_bench.cu - STRATA_PF_ATTN_PERSIST: the persistent FP16 score kernel (ps_score_hp_kernel) against the engine's ps_score_h_kernel on a PH402 die:
//   1. bitwise: the whole scratch of both kernels (WH true and false) over several shapes - full width, short width (partial and empty
//      chunks), masked pages, one query, a ragged number of queries - and, BIG=1, pools that straddle 4 GiB (pages from both sides);
//   2. speed: ABAB interleaved calls (events a call), medians.
// Pools are filled on the device (hash), so a 5 GiB pool costs nothing on the host.
//   build (not built by CMake): nvcc -O3 -arch=sm_60 -std=c++20 -DSTRATA_EXPERIMENTAL_SM60=1 -I include -o qsa_pattn_h16p_bench.exe src/kernels/qsa_pattn_h16p_bench.cu
//   run (CUDA_VISIBLE_DEVICES=<one PH402 die>): qsa_pattn_h16p_bench [reps=40] [big=0] [nq=100]
#include "cuda/qsa_pattn.cu"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

using namespace strata::kernels;

static void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) { std::fprintf(stderr, "bench: %s: %s\n", what, cudaGetErrorString(e)); std::exit(2); }
}
__global__ void fill_i8(int8_t* p, size_t n, uint32_t seed) {
    for (size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x) {
        uint32_t h = (uint32_t) (i * 2654435761ull) ^ (uint32_t) (i >> 32) * 40503u ^ seed;
        h ^= h >> 15; h *= 2246822519u; h ^= h >> 13;
        p[i] = (int8_t) ((int) (h % 255u) - 127);
    }
}
__global__ void fill_sc(uint16_t* p, size_t n, uint32_t seed) {
    for (size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x) {
        uint32_t h = (uint32_t) (i * 2654435761ull) ^ (uint32_t) (i >> 32) * 40503u ^ seed;
        h ^= h >> 15; h *= 2246822519u; h ^= h >> 13;
        p[i] = (uint16_t) ((9u << 10) | (h & 0x3FFu));
    }
}
template <typename T> static T* dalloc(size_t n) {
    T* p = nullptr;
    ck(cudaMalloc((void**) &p, n * sizeof(T) + 256), "cudaMalloc");
    ck(cudaMemset(p, 0, n * sizeof(T) + 256), "cudaMemset");
    return p;
}
template <typename T> static T* upload(const std::vector<T>& h) {
    T* d = dalloc<T>(h.size());
    ck(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice), "upload");
    return d;
}

int main(int argc, char** argv) {
    const int REPS = argc > 1 ? std::atoi(argv[1]) : 40;
    const int BIG = argc > 2 ? std::atoi(argv[2]) : 0;
    const int NQ0 = argc > 3 ? std::atoi(argv[3]) : 100;
    cudaDeviceProp prop{};
    ck(cudaGetDeviceProperties(&prop, 0), "props");
    std::printf("device: %s (cc %d.%d, %d SMs)  reps %d big %d nq %d  SP_NPC %d\n", prop.name, prop.major, prop.minor,
                prop.multiProcessorCount, REPS, BIG, NQ0, SP_NPC);
    if (!std::strstr(prop.name, "PH402")) { std::fprintf(stderr, "bench: device 0 is not a PH402 die (%s)\n", prop.name); return 4; }
    const QsaShapes s = qsa_real_shapes();
    const int64_t cap = 2112;
    const int NKV = (int) s.n_head_kv, NH = (int) s.n_head;
    const int PAGE = 4;
    const int PAGES = BIG ? 2600000 : 16384;   // BIG: 2.6 M pages * 2 KV heads * 4 cells * 256 B = 5.3 GB of K codes (rows * 256 passes 2^32)
    const int64_t rows_total = (int64_t) PAGES * NKV * PAGE;
    std::printf("K pool %.2f GiB (rows %lld)\n", (double) rows_total * HD / (1ull << 30), (long long) rows_total);
    QsaAttnPools pools;
    int8_t* kq = dalloc<int8_t>((size_t) rows_total * HD);
    int8_t* vq = dalloc<int8_t>(BIG ? 1 : (size_t) rows_total * HD);
    uint16_t* ksc = dalloc<uint16_t>((size_t) rows_total * NSC);
    uint16_t* vsc = dalloc<uint16_t>(BIG ? 1 : (size_t) rows_total * NSC);
    fill_i8<<<1024, 256>>>(kq, (size_t) rows_total * HD, 11u);
    if (!BIG) fill_i8<<<1024, 256>>>(vq, (size_t) rows_total * HD, 12u);
    fill_sc<<<1024, 256>>>(ksc, (size_t) rows_total * NSC, 13u);
    if (!BIG) fill_sc<<<1024, 256>>>(vsc, (size_t) rows_total * NSC, 14u);
    ck(cudaDeviceSynchronize(), "fill");
    pools.k_q = kq; pools.v_q = vq; pools.k_scale = ksc; pools.v_scale = vsc;
    std::mt19937 rng(7);
    std::vector<int32_t> pt(PAGES);
    for (int i = 0; i < PAGES; ++i) pt[i] = i;
    std::vector<int> masked;
    for (int i = 0; i < PAGES; ++i) if (rng() % 100 < 3) masked.push_back(i);
    const int NQ_MAX = std::max(NQ0, 130);
    const uint64_t per_q = (uint64_t) NKV * 33 * PS_BLK;
    float* scratchA = dalloc<float>(per_q * NQ_MAX);
    float* scratchB = dalloc<float>(per_q * NQ_MAX);
    const float scale = 1.0f / sqrtf((float) HD);
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0); cudaEventCreate(&e1);
    bool all_ok = true;
    // shapes: nq, width, masking (0 none, 1 3 % + a run of whole chunks), page range (BIG: pages around 2^32 / 256 / 2048 = 2.1 M)
    struct Shape { int nq, width, mask; const char* what; };
    const std::vector<Shape> shapes = {
        {NQ0, 2112, 0, "full width"}, {NQ0, 2112, 1, "full width, masked pages"}, {NQ0, 1000, 0, "width 1000 (partial last chunk, empty chunks)"},
        {NQ0, 4, 0, "width 4 (one page)"}, {1, 2112, 0, "one query"}, {37, 1536, 1, "37 queries, width 1536, masked"},
        {NQ0, 2108, 0, "width 2108"}, {NQ0, 64, 0, "width 64 (one chunk)"}, {NQ0, 0, 0, "width 0"}};
    for (const Shape& sh : shapes) {
        const int NQ = std::min(sh.nq, NQ_MAX);
        std::vector<int32_t> ptl = pt;
        if (sh.mask) {
            for (int x : masked) ptl[x] = -1;
            for (int i = 4096; i < 4096 + 64 && i < PAGES; ++i) ptl[i] = -1;
        }
        std::vector<int32_t> ids((size_t) NQ * cap, 0), steps((size_t) NQ * kStepCount, 0);
        // pages: BIG draws half of them from the top of the pool (rows * 256 > 2^32) and half from the bottom
        const int lo_hi = BIG ? PAGES / 2 : PAGES;
        std::vector<int> hot;
        {
            std::vector<int> all(PAGES);
            for (int i = 0; i < PAGES; ++i) all[i] = i;
            std::shuffle(all.begin(), all.end(), rng);
            hot.assign(all.begin(), all.begin() + std::max(0, (sh.width / PAGE) / 2));
        }
        (void) lo_hi;
        for (int z = 0; z < NQ; ++z) {
            std::vector<int> pg = hot;
            std::vector<char> in(PAGES, 0);
            for (int x : pg) in[x] = 1;
            while ((int) pg.size() < sh.width / PAGE) {
                const int x = BIG ? (int) ((rng() % 2) ? (2097152 - 200000 + rng() % 400000) : (rng() % 400000)) : (int) (rng() % PAGES);
                if (!in[x]) { in[x] = 1; pg.push_back(x); }
            }
            std::sort(pg.begin(), pg.end());
            int n = 0;
            for (int x : pg) for (int c = 0; c < PAGE; ++c) ids[(size_t) z * cap + n++] = x * PAGE + c;
            steps[(size_t) z * kStepCount + kStepWidth] = n;
        }
        std::vector<float> q((size_t) NQ * NH * HD);
        std::normal_distribution<float> nd(0.0f, 1.0f);
        for (auto& x : q) x = nd(rng);
        if (sh.nq == 1) for (auto& x : q) x *= 40.0f;   // a large q: the power of two e is not 0
        pools.page_table = upload(ptl);
        float* dq = upload(q);
        int32_t* dids = upload(ids);
        int32_t* dsteps = upload(steps);
        const int stride = (33 + SP_NPC - 1) / SP_NPC;
        const dim3 g1(33, NKV, NQ), gp(stride, NKV, NQ);
        for (int wh = 0; wh < 2; ++wh) {
            ck(cudaMemset(scratchA, 0x5a, per_q * NQ * 4), "memset");
            ck(cudaMemset(scratchB, 0x5a, per_q * NQ * 4), "memset");
            if (wh) {
                ps_score_h_kernel<true><<<g1, SH_T>>>(dq, pools, dids, dsteps, NKV, PAGE, scale, scratchA, 33, (int) cap);
                ps_score_hp_kernel<true><<<gp, SH_T>>>(dq, pools, dids, dsteps, NKV, PAGE, scale, scratchB, 33, (int) cap, stride);
            } else {
                ps_score_h_kernel<false><<<g1, SH_T>>>(dq, pools, dids, dsteps, NKV, PAGE, scale, scratchA, 33, (int) cap);
                ps_score_hp_kernel<false><<<gp, SH_T>>>(dq, pools, dids, dsteps, NKV, PAGE, scale, scratchB, 33, (int) cap, stride);
            }
            ck(cudaGetLastError(), "launch");
            ck(cudaDeviceSynchronize(), "run");
            std::vector<uint32_t> ha(per_q * NQ), hb(per_q * NQ);
            ck(cudaMemcpy(ha.data(), scratchA, ha.size() * 4, cudaMemcpyDeviceToHost), "dl");
            ck(cudaMemcpy(hb.data(), scratchB, hb.size() * 4, cudaMemcpyDeviceToHost), "dl");
            size_t nd_ = 0, first = (size_t) -1;
            for (size_t i = 0; i < ha.size(); ++i) if (ha[i] != hb[i]) { if (first == (size_t) -1) first = i; ++nd_; }
            std::printf("  [%-48s] WH=%d scratch words differing: %zu of %zu%s\n", sh.what, wh, nd_, ha.size(),
                        nd_ ? "  <-- MISMATCH" : "  identical");
            if (nd_) { all_ok = false; std::printf("      first at word %zu (query %zu of the layout): %08x vs %08x\n", first, first / per_q, ha[first], hb[first]); }
        }
        // timing for the full-width unmasked shape and the masked one only
        if (sh.width == 2112 && sh.nq == NQ0) {
            std::vector<float> ta, tb;
            auto one = [&](bool persist, float* scr) {
                cudaEventRecord(e0);
                if (persist) ps_score_hp_kernel<true><<<gp, SH_T>>>(dq, pools, dids, dsteps, NKV, PAGE, scale, scr, 33, (int) cap, stride);
                else ps_score_h_kernel<true><<<g1, SH_T>>>(dq, pools, dids, dsteps, NKV, PAGE, scale, scr, 33, (int) cap);
                cudaEventRecord(e1);
                cudaEventSynchronize(e1);
                float ms = 0; cudaEventElapsedTime(&ms, e0, e1);
                return ms;
            };
            for (int i = 0; i < 5; ++i) { one(false, scratchA); one(true, scratchB); }
            for (int i = 0; i < REPS; ++i) { ta.push_back(one(false, scratchA)); tb.push_back(one(true, scratchB)); }
            std::sort(ta.begin(), ta.end()); std::sort(tb.begin(), tb.end());
            const double fa = ta[ta.size() / 2], fb = tb[tb.size() / 2];
            std::printf("  TIMING nq %d %s: engine ps_score_h_kernel<true> median %.3f ms (min %.3f) = %.2f ms/1536q; persistent median %.3f ms (min %.3f) = %.2f ms/1536q; %+.1f%% (median)\n",
                        NQ, sh.mask ? "masked" : "plain", fa, ta[0], fa * 1536.0 / NQ, fb, tb[0], fb * 1536.0 / NQ, 100.0 * (fb - fa) / fa);
        }
        cudaFree((void*) pools.page_table); cudaFree(dq); cudaFree(dids); cudaFree(dsteps);
    }
    std::printf("%s\n", all_ok ? "ALL IDENTICAL" : "MISMATCH SOMEWHERE");
    return all_ok ? 0 : 1;
}
