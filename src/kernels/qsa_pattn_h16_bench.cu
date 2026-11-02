// src/kernels/qsa_pattn_h16_bench.cu - STRATA_PF_HFMA_ATTN: the FP16 score / value kernels against the bit-exact ones and against an FP64
// reference, and their speed (GPU, synthetic int8 K/V, no model).  Not built by CMake: it includes qsa_pattn.cu itself so it can time the
// kernels one by one,
//   nvcc -O3 -arch=sm_60 -std=c++20 -DSTRATA_EXPERIMENTAL_SM60=1 -I include -o qsa_pattn_h16_bench.exe src/kernels/qsa_pattn_h16_bench.cu
//   set CUDA_VISIBLE_DEVICES=<one PH402 die>;  qsa_pattn_h16_bench [nq=100] [width=2112] [reps=30] [qsigma=1.0] [seed=1]
//
// Data: random int8 codes, fp16 scales in [1/64, 1/32) (|x| < 4), q ~ N(0, qsigma), selections of whole 4-cell pages (the indexer's
// block): half of a query's pages are a set every query shares (neighbouring queries share 42-61 % of their cells), half its own.
// Reported per route: the largest |difference| and the RMS difference over the RMS of the FP64 reference, for 3 queries x 24 heads x
// 256 dimensions, and against the bit-exact route.
#include "cuda/qsa_pattn.cu"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

using namespace strata::kernels;

static void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) { std::fprintf(stderr, "bench: %s: %s\n", what, cudaGetErrorString(e)); std::exit(2); }
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
static float h2f(uint16_t h) {
    const int s = (h >> 15) & 1, e = (h >> 10) & 31, m = h & 1023;
    float v = e == 0 ? std::ldexp((float) m, -24) : std::ldexp((float) (m + 1024), e - 25);
    return s ? -v : v;
}

int main(int argc, char** argv) {
    const int NQ = argc > 1 ? std::atoi(argv[1]) : 100;
    const int WIDTH = argc > 2 ? std::atoi(argv[2]) : 2112;
    const int REPS = argc > 3 ? std::atoi(argv[3]) : 30;
    const float QSIG = argc > 4 ? (float) std::atof(argv[4]) : 1.0f;
    const unsigned SEED = argc > 5 ? (unsigned) std::atoi(argv[5]) : 1u;
    cudaDeviceProp prop{};
    ck(cudaGetDeviceProperties(&prop, 0), "props");
    std::printf("device: %s (cc %d.%d, %d SMs)  nq %d width %d reps %d qsigma %.2f\n", prop.name, prop.major, prop.minor,
                prop.multiProcessorCount, NQ, WIDTH, REPS, QSIG);
    if (!std::strstr(prop.name, "PH402")) {
        std::fprintf(stderr, "bench: device 0 is not a PH402 die (%s): set CUDA_VISIBLE_DEVICES\n", prop.name);
        return 4;
    }
    const QsaShapes s = qsa_real_shapes();
    const int64_t cap = 2112;
    const int NKV = (int) s.n_head_kv, NH = (int) s.n_head;
    const int PAGES = 16384, PAGE = 4;
    const int64_t rows_total = (int64_t) PAGES * NKV * PAGE;
    std::mt19937 rng(SEED);
    std::vector<int8_t> kq((size_t) rows_total * HD), vq((size_t) rows_total * HD);
    std::vector<uint16_t> ksc((size_t) rows_total * NSC), vsc((size_t) rows_total * NSC);
    for (auto& x : kq) x = (int8_t) ((int) (rng() % 255) - 127);
    for (auto& x : vq) x = (int8_t) ((int) (rng() % 255) - 127);
    for (auto& x : ksc) x = (uint16_t) ((9u << 10) | (rng() & 0x3FFu));
    for (auto& x : vsc) x = (uint16_t) ((9u << 10) | (rng() & 0x3FFu));
    std::vector<int32_t> pt(PAGES);
    for (int i = 0; i < PAGES; ++i) pt[i] = i;
    if (const char* mk = std::getenv("MASK"); mk && *mk == '1') {   // ~3 % unresident pages and a run of 64 of them (whole chunks masked)
        for (int i = 0; i < PAGES; ++i) if (rng() % 100 < 3) pt[i] = -1;
        for (int i = 4096; i < 4096 + 64; ++i) pt[i] = -1;
        std::printf("MASK=1: unresident pages in the page table\n");
    }
    std::vector<int32_t> ids((size_t) NQ * cap, 0), steps((size_t) NQ * kStepCount, 0);
    std::vector<int> hot;
    {
        std::vector<int> all(PAGES);
        for (int i = 0; i < PAGES; ++i) all[i] = i;
        std::shuffle(all.begin(), all.end(), rng);
        hot.assign(all.begin(), all.begin() + (WIDTH / PAGE) / 2);
    }
    for (int z = 0; z < NQ; ++z) {
        std::vector<int> pg = hot;
        std::vector<char> in(PAGES, 0);
        for (int x : pg) in[x] = 1;
        while ((int) pg.size() < WIDTH / PAGE) {
            const int x = (int) (rng() % PAGES);
            if (!in[x]) { in[x] = 1; pg.push_back(x); }
        }
        std::sort(pg.begin(), pg.end());
        int n = 0;
        for (int x : pg) for (int c = 0; c < PAGE; ++c) ids[(size_t) z * cap + n++] = x * PAGE + c;
        steps[(size_t) z * kStepCount + kStepWidth] = n;
    }
    std::vector<float> q((size_t) NQ * NH * HD);
    std::normal_distribution<float> nd(0.0f, QSIG);
    for (auto& x : q) x = nd(rng);
    QsaAttnPools pools;
    pools.k_q = upload(kq); pools.v_q = upload(vq); pools.k_scale = upload(ksc); pools.v_scale = upload(vsc);
    pools.page_table = upload(pt);
    float* dq = upload(q);
    int32_t* dids = upload(ids);
    int32_t* dsteps = upload(steps);
    const uint64_t per_q = (uint64_t) NKV * 33 * PS_BLK;
    const uint64_t scratch_floats = per_q * (uint64_t) NQ;
    float* scratch = dalloc<float>(scratch_floats);
    float* outA = dalloc<float>((size_t) NQ * NH * HD);
    float* outB = dalloc<float>((size_t) NQ * NH * HD);
    float* outC = dalloc<float>((size_t) NQ * NH * HD);
    auto run = [&](int mode, float* out) {   // 0 bit-exact, 1/2 FP16
        const bool ok = mode == 0 ? qsa_pattn_batch_any_arch(dq, pools, dids, dsteps, cap, s, scratch, scratch_floats, out, NQ, nullptr)
                                  : qsa_pattn_h16_batch_any_arch(dq, pools, dids, dsteps, cap, s, scratch, scratch_floats, out, NQ, nullptr, mode);
        if (!ok) { std::fprintf(stderr, "route %d refused\n", mode); std::exit(3); }
        ck(cudaDeviceSynchronize(), "run");
    };
    run(0, outA); run(1, outB); run(2, outC);
    std::vector<float> hA((size_t) NQ * NH * HD), hB(hA.size()), hC(hA.size());
    ck(cudaMemcpy(hA.data(), outA, hA.size() * 4, cudaMemcpyDeviceToHost), "dl");
    ck(cudaMemcpy(hB.data(), outB, hA.size() * 4, cudaMemcpyDeviceToHost), "dl");
    ck(cudaMemcpy(hC.data(), outC, hA.size() * 4, cudaMemcpyDeviceToHost), "dl");
    // FP64 reference
    const int zs[3] = {0, NQ / 2, NQ - 1};
    double e2[3][3] = {}, r2 = 0, d2[2] = {}, a2 = 0, emax[3] = {}, dmax[2] = {};
    for (int zi = 0; zi < 3; ++zi) {
        const int z = zs[zi];
        const int width = steps[(size_t) z * kStepCount + kStepWidth];
        for (int kvh = 0; kvh < NKV; ++kvh) for (int h = 0; h < 12; ++h) {
            const int hh = kvh * 12 + h;
            std::vector<double> sc(width);
            std::vector<char> ok(width, 1);
            double mx = -1e300;
            for (int i = 0; i < width; ++i) {
                const int cell = ids[(size_t) z * cap + i];
                if (pt[cell / PAGE] < 0) { ok[i] = 0; sc[i] = -1e300; continue; }
                const int64_t row = ((int64_t) pt[cell / PAGE] * NKV + kvh) * PAGE + cell % PAGE;
                double acc = 0;
                for (int d = 0; d < HD; ++d)
                    acc += (double) q[((size_t) z * NH + hh) * HD + d] * ((double) kq[row * HD + d] * (double) h2f(ksc[row * NSC + d / 64]));
                sc[i] = acc / 16.0;
                mx = std::max(mx, sc[i]);
            }
            double den = 0;
            std::vector<double> w(width);
            for (int i = 0; i < width; ++i) { w[i] = ok[i] ? std::exp(sc[i] - mx) : 0.0; den += w[i]; }
            for (int d = 0; d < HD; ++d) {
                double acc = 0;
                for (int i = 0; i < width; ++i) {
                    if (!ok[i]) continue;
                    const int cell = ids[(size_t) z * cap + i];
                    const int64_t row = ((int64_t) pt[cell / PAGE] * NKV + kvh) * PAGE + cell % PAGE;
                    acc += w[i] * ((double) vq[row * HD + d] * (double) h2f(vsc[row * NSC + d / 64]));
                }
                const double ref = den > 0 ? acc / den : 0.0;
                const size_t at = ((size_t) z * NH + hh) * HD + d;
                const double x[3] = {hA[at], hB[at], hC[at]};
                r2 += ref * ref;
                for (int r = 0; r < 3; ++r) { const double e = x[r] - ref; e2[zi][r] += e * e; emax[r] = std::max(emax[r], std::fabs(e)); }
                d2[0] += (x[1] - x[0]) * (x[1] - x[0]);
                d2[1] += (x[2] - x[0]) * (x[2] - x[0]);
                dmax[0] = std::max(dmax[0], std::fabs(x[1] - x[0]));
                dmax[1] = std::max(dmax[1], std::fabs(x[2] - x[0]));
                a2 += x[0] * x[0];
            }
        }
    }
    const char* names[3] = {"bit-exact", "FP16 score (=1)", "FP16 score+value (=2)"};
    std::printf("reference RMS %.4e\n", std::sqrt(r2 / (3.0 * NH * HD)));
    for (int r = 0; r < 3; ++r) {
        double t = 0;
        for (int zi = 0; zi < 3; ++zi) t += e2[zi][r];
        std::printf("%-22s vs FP64: max |err| %.3e   rel RMS %.3e\n", names[r], emax[r], std::sqrt(t / r2));
    }
    std::printf("vs bit-exact route:  =1 max %.3e rel RMS %.3e;  =2 max %.3e rel RMS %.3e\n", dmax[0], std::sqrt(d2[0] / a2), dmax[1],
                std::sqrt(d2[1] / a2));
    // timings: whole routes, then kernels one by one
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0); cudaEventCreate(&e1);
    auto timeit = [&](const char* name, auto&& fn) {
        for (int i = 0; i < 3; ++i) fn();
        ck(cudaDeviceSynchronize(), name);
        cudaEventRecord(e0);
        for (int i = 0; i < REPS; ++i) fn();
        cudaEventRecord(e1);
        ck(cudaEventSynchronize(e1), name);
        float ms = 0;
        cudaEventElapsedTime(&ms, e0, e1);
        ms /= REPS;
        std::printf("  %-34s %8.3f ms per %d queries   %7.2f us/query   %7.2f ms per 1536 queries\n", name, ms, NQ, 1000.0 * ms / NQ,
                    ms * 1536.0 / NQ);
        return ms;
    };
    std::printf("timings (the whole call: score + value):\n");
    timeit("bit-exact (score_x + value_d4)", [&] { qsa_pattn_batch_any_arch(dq, pools, dids, dsteps, cap, s, scratch, scratch_floats, outA, NQ, nullptr); });
    timeit("FP16 score + value_d4 (=1)", [&] { qsa_pattn_h16_batch_any_arch(dq, pools, dids, dsteps, cap, s, scratch, scratch_floats, outB, NQ, nullptr, 1); });
    timeit("FP16 score + FP16 value (=2)", [&] { qsa_pattn_h16_batch_any_arch(dq, pools, dids, dsteps, cap, s, scratch, scratch_floats, outC, NQ, nullptr, 2); });
    std::printf("timings (kernels alone, same scratch):\n");
    const float scale = 1.0f / sqrtf((float) HD);
    const dim3 g1(33, NKV, NQ), g2(NKV, NQ);
    timeit("ps_score_x_kernel", [&] { ps_score_x_kernel<1><<<g1, THREADS>>>(dq, pools, dids, dsteps, NKV, PAGE, scale, scratch, 33, (int) cap); });
    timeit("ps_value_d4_kernel", [&] { ps_value_d4_kernel<<<g2, 64>>>(pools, dsteps, scratch, NKV, 33, outA); });
    timeit("ps_score_h_kernel<false>", [&] { ps_score_h_kernel<false><<<g1, SH_T>>>(dq, pools, dids, dsteps, NKV, PAGE, scale, scratch, 33, (int) cap); });
    timeit("ps_score_h_kernel<true>", [&] { ps_score_h_kernel<true><<<g1, SH_T>>>(dq, pools, dids, dsteps, NKV, PAGE, scale, scratch, 33, (int) cap); });
    timeit("ps_value_h_kernel", [&] { ps_value_h_kernel<<<g2, VH_T>>>(pools, dsteps, scratch, NKV, 33, outC); });
    ck(cudaGetLastError(), "kernels");
    unsigned long long fb = 0;
    cudaMemcpyFromSymbol(&fb, g_ph16_value_fb, sizeof fb);
    std::printf("value CTAs redone in FP32: %llu\n", fb);
    return 0;
}
