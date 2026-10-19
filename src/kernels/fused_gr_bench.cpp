// src/kernels/fused_gr_bench.cpp - fused_gr_read_multi: the AMD fast kernels (STRATA_GR_FAST) against the old
// ones, bit for bit on every output (lo, rs, inject, mixed, the in-place R), and each one's time.
//
//     build/fused_gr_bench [iters] [T min] [T max]
//     build/fused_gr_bench parts [iters] [T min] [T max] [weight sets]   (each read and each kernel apart; below)
#include "strata/kernels/fused_gr.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace K = strata::kernels;

static uint16_t bf16(float x) {
    uint32_t u;
    std::memcpy(&u, &x, 4);
    return (uint16_t) ((u + 0x7fff + ((u >> 16) & 1)) >> 16);
}

// ---- parts mode (P1 part E gate, 2026-10-19) ----------------------------------------------------------------------
//
//     build/fused_gr_bench parts [iters] [T min] [T max] [weight sets]
//
// Times each read and each piece of it (FusedGrBenchPiece) at the engine's shape, with the production arguments
// (apply = the pending write folded in, inject rows on, the q8_1 tail of STRATA_QFUSE on or off), two ways:
//   stream - `iters` back-to-back launches, weight set i % sets (sets > L2 so every read comes from DRAM)
//   graph  - 24 reads (one 12-layer stage) captured once and replayed, as the engine's window graph runs them
// Before timing: the card's fused_gr_check verdict, and every read's outputs against the staged read's (bitwise
// for the reads that promise it, max |diff| for v3, which sums in another order).
namespace {

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e));
        std::exit(2);
    }
}

struct Piece { int id; const char* name; int t_max; bool read; };
const Piece kPieces[] = {
    {K::kGrBenchStaged, "staged read (production)", 8, true},
    {K::kGrBenchNormSplit, "  gr_norm_split", 8, false},
    {K::kGrBenchDownStaged, "  gr_down_staged<T,true>", 8, false},
    {K::kGrBenchUpMulti, "  gr_up_multi<T,true>", 8, false},
    {K::kGrBenchPlain, "plain read", 8, true},
    {K::kGrBenchSingle, "single read (norm folded)", 1, true},
    {K::kGrBenchSingleDown, "  gr_down_kernel (norm folded)", 1, false},
    {K::kGrBenchSingleUp, "  gr_up_kernel", 1, false},
    {K::kGrBenchV3S2, "v3<2> read (generic MAX_T 8)", 8, true},
    {K::kGrBenchV3S2Down, "  gr_down_v3<2>", 8, false},
    {K::kGrBenchV3S2Up, "  gr_up_v3<2>", 8, false},
    {K::kGrBenchV3S2Exact, "v3<2,T,true> read", 2, true},
    {K::kGrBenchV3S1Exact, "v3<1,T,true> read", 2, true},
};

int parts(int argc, char** argv) {
    const int iters = argc > 2 ? std::atoi(argv[2]) : 2400;
    const int t_lo = argc > 3 ? std::atoi(argv[3]) : 1, t_hi = argc > 4 ? std::atoi(argv[4]) : 2;
    const int NS = argc > 5 ? std::atoi(argv[5]) : 8;
    const int N = 2560, HC = 4, D = N * HC, LR = 320, TM = K::kFusedGrMaxT, Q8B = (N / 32) * 36;
    int dev = 0;
    ck(cudaGetDevice(&dev), "cudaGetDevice");
    cudaDeviceProp pr{};
    ck(cudaGetDeviceProperties(&pr, dev), "cudaGetDeviceProperties");
    std::printf("device %s  pci %04x:%02x:%02x  sm_%d%d  %d SMs  L2 %d KB  smem/block %zu  optin %zu\n", pr.name,
                pr.pciDomainID, pr.pciBusID, pr.pciDeviceID, pr.major, pr.minor, pr.multiProcessorCount,
                pr.l2CacheSize / 1024, pr.sharedMemPerBlock, pr.sharedMemPerBlockOptin);
    K::fused_gr_check();
    std::printf("fused_gr_check: variant %d (1 plain, 2 split, 3 staged)\n", K::fused_gr_variant());
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.f, 1.f);
    // weights: NS distinct layers (13.3 MB each), so a read never finds the previous one's weights in L2
    std::vector<uint16_t*> dwd(NS), dwu(NS), dwi(NS);
    std::vector<float*> dwn(NS);
    {
        std::vector<uint16_t> wd((size_t) LR * D), wu((size_t) D * LR), wi((size_t) HC * D);
        std::vector<float> wn(D);
        for (int s = 0; s < NS; ++s) {
            for (auto& x : wd) x = bf16(0.02f * nd(rng));
            for (auto& x : wu) x = bf16(0.05f * nd(rng));
            for (auto& x : wi) x = bf16(0.02f * nd(rng));
            for (auto& x : wn) x = 1.0f + 0.1f * nd(rng);
            ck(cudaMalloc((void**) &dwd[s], wd.size() * 2), "malloc");
            ck(cudaMalloc((void**) &dwu[s], wu.size() * 2), "malloc");
            ck(cudaMalloc((void**) &dwi[s], wi.size() * 2), "malloc");
            ck(cudaMalloc((void**) &dwn[s], wn.size() * 4), "malloc");
            cudaMemcpy(dwd[s], wd.data(), wd.size() * 2, cudaMemcpyHostToDevice);
            cudaMemcpy(dwu[s], wu.data(), wu.size() * 2, cudaMemcpyHostToDevice);
            cudaMemcpy(dwi[s], wi.data(), wi.size() * 2, cudaMemcpyHostToDevice);
            cudaMemcpy(dwn[s], wn.data(), wn.size() * 4, cudaMemcpyHostToDevice);
        }
    }
    std::vector<float> R((size_t) TM * D), bo((size_t) TM * N), inj((size_t) TM * HC);
    for (auto& x : R) x = nd(rng);
    for (auto& x : bo) x = 0.05f * nd(rng);
    for (auto& x : inj) x = 2.0f * nd(rng);
    float *dR, *dbo, *dinj, *dxn, *dlo, *drs, *dio, *dmix;
    uint8_t* dq8;
    unsigned* dcnt;
    ck(cudaMalloc((void**) &dR, R.size() * 4), "malloc");
    ck(cudaMalloc((void**) &dbo, bo.size() * 4), "malloc");
    ck(cudaMalloc((void**) &dinj, inj.size() * 4), "malloc");
    ck(cudaMalloc((void**) &dxn, (size_t) TM * D * 4), "malloc");
    ck(cudaMalloc((void**) &dlo, (size_t) TM * LR * 4), "malloc");
    ck(cudaMalloc((void**) &drs, (size_t) TM * HC * 4), "malloc");
    ck(cudaMalloc((void**) &dio, (size_t) TM * HC * 4), "malloc");
    ck(cudaMalloc((void**) &dmix, (size_t) TM * N * 4), "malloc");
    ck(cudaMalloc((void**) &dq8, (size_t) TM * Q8B), "malloc");
    ck(cudaMalloc((void**) &dcnt, (N / 32) * sizeof(unsigned)), "malloc");
    cudaMemset(dcnt, 0, (N / 32) * sizeof(unsigned));
    cudaMemcpy(dbo, bo.data(), bo.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dinj, inj.data(), inj.size() * 4, cudaMemcpyHostToDevice);
    cudaStream_t s;
    ck(cudaStreamCreate(&s), "stream");   // blocking: the legacy-stream memsets and copies order with it
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    auto args = [&](K::FusedGrArgs* a, int T, int set, bool q8) {
        for (int t = 0; t < T; ++t) {
            K::FusedGrArgs& x = a[t];
            x = K::FusedGrArgs{};
            x.R = dR + (size_t) t * D; x.R_out = dR + (size_t) t * D; x.apply = true;
            x.bo_prev = dbo + (size_t) t * N; x.inj_prev = dinj + (size_t) t * HC;
            x.w_norm = dwn[set]; x.w_down = dwd[set]; x.w_up = dwu[set]; x.w_inject = dwi[set];
            x.lo = dlo + (size_t) t * LR; x.rs = drs + (size_t) t * HC;
            x.inject_out = dio + (size_t) t * HC; x.mixed = dmix + (size_t) t * N;
            if (q8) { x.q8_cnt = dcnt; x.q8_mixed = dq8 + (size_t) t * Q8B; }
        }
    };
    auto run = [&](int which, int T, int set, bool q8) {
        K::FusedGrArgs a[K::kFusedGrMaxT];
        args(a, T, set, q8);
        return K::fused_gr_bench_piece(which, a, T, dxn, s);
    };
    // outputs of one read from a fresh R: R_out, lo, rs, inject, mixed (T tokens)
    auto outputs = [&](int which, int T) {
        cudaMemcpy(dR, R.data(), R.size() * 4, cudaMemcpyHostToDevice);
        cudaMemset(dlo, 0xff, (size_t) TM * LR * 4);
        cudaMemset(drs, 0xff, (size_t) TM * HC * 4);
        cudaMemset(dio, 0xff, (size_t) TM * HC * 4);
        cudaMemset(dmix, 0xff, (size_t) TM * N * 4);
        std::vector<float> o((size_t) T * (D + LR + HC + HC + N));
        if (!run(which, T, 0, false)) return std::vector<float>();
        ck(cudaStreamSynchronize(s), "read");
        float* p = o.data();
        cudaMemcpy(p, dR, (size_t) T * D * 4, cudaMemcpyDeviceToHost); p += (size_t) T * D;
        cudaMemcpy(p, dlo, (size_t) T * LR * 4, cudaMemcpyDeviceToHost); p += (size_t) T * LR;
        cudaMemcpy(p, drs, (size_t) T * HC * 4, cudaMemcpyDeviceToHost); p += (size_t) T * HC;
        cudaMemcpy(p, dio, (size_t) T * HC * 4, cudaMemcpyDeviceToHost); p += (size_t) T * HC;
        cudaMemcpy(p, dmix, (size_t) T * N * 4, cudaMemcpyDeviceToHost);
        return o;
    };
    int bad = 0;
    for (int T = t_lo; T <= t_hi; ++T) {
        std::printf("\n== T %d: outputs against the staged read (apply 1, inject 1) ==\n", T);
        const std::vector<float> ref = outputs(K::kGrBenchStaged, T);
        // mixed is the last T*N floats; lo follows R
        for (const Piece& p : kPieces) {
            if (!p.read || p.id == K::kGrBenchStaged || T > p.t_max) continue;
            const std::vector<float> o = outputs(p.id, T);
            if (o.empty()) { std::printf("  %-32s launch failed\n", p.name); ++bad; continue; }
            const bool same = std::memcmp(o.data(), ref.data(), o.size() * 4) == 0;
            double dmax = 0, mmax = 0;
            const size_t m0 = (size_t) T * (D + LR + HC + HC);
            for (size_t i = m0; i < o.size(); ++i) {
                dmax = std::fmax(dmax, std::fabs((double) o[i] - ref[i]));
                mmax = std::fmax(mmax, std::fabs((double) ref[i]));
            }
            double lmax = 0;
            for (size_t i = (size_t) T * D; i < (size_t) T * (D + LR); ++i) lmax = std::fmax(lmax, std::fabs((double) o[i] - ref[i]));
            std::printf("  %-32s %s  (mixed max|diff| %.3g of max %.3g; lo max|diff| %.3g)\n", p.name,
                        same ? "bitwise equal" : "differs", dmax, mmax, lmax);
            const bool must_equal = p.id == K::kGrBenchPlain || p.id == K::kGrBenchSingle;
            if (must_equal && !same) ++bad;
            if (!must_equal && !(dmax <= 1e-3 * mmax)) ++bad;   // v3: another order, the same maths
        }
    }
    cudaMemcpy(dR, R.data(), R.size() * 4, cudaMemcpyHostToDevice);
    constexpr int kStage = 24;   // HC reads of one 12-layer stage (2 per layer)
    for (int q8 = 1; q8 >= 0; --q8)
        for (int T = t_lo; T <= t_hi; ++T) {
            std::printf("\n== T %d, apply 1, inject 1, q8_1 tail %s: us per read (min / median of 7 rounds) ==\n", T,
                        q8 ? "on (STRATA_QFUSE=1, production)" : "off");
            std::printf("  %-32s %18s %18s\n", "", "stream", "graph x24");
            for (const Piece& p : kPieces) {
                if (T > p.t_max) continue;
                // the pieces that read an earlier read's outputs: one full read of that kind first
                const int prep = p.id == K::kGrBenchDownStaged || p.id == K::kGrBenchUpMulti ? K::kGrBenchStaged
                                 : p.id == K::kGrBenchSingleUp ? K::kGrBenchSingle
                                 : p.id == K::kGrBenchV3S2Up ? K::kGrBenchV3S2 : -1;
                if (prep >= 0) run(prep, T, 0, q8 != 0);
                for (int i = 0; i < 40; ++i)
                    if (!run(p.id, T, i % NS, q8 != 0)) { std::printf("  %-32s launch failed\n", p.name); ++bad; break; }
                ck(cudaStreamSynchronize(s), "warm-up");
                std::vector<double> st, gr;
                for (int round = 0; round < 7; ++round) {
                    cudaEventRecord(e0, s);
                    for (int i = 0; i < iters; ++i) run(p.id, T, i % NS, q8 != 0);
                    cudaEventRecord(e1, s);
                    cudaEventSynchronize(e1);
                    float ms = 0;
                    cudaEventElapsedTime(&ms, e0, e1);
                    st.push_back(1e3 * ms / iters);
                }
                cudaGraph_t g;
                cudaGraphExec_t ge;
                ck(cudaStreamBeginCapture(s, cudaStreamCaptureModeRelaxed), "capture");
                for (int i = 0; i < kStage; ++i) run(p.id, T, i % NS, q8 != 0);
                ck(cudaStreamEndCapture(s, &g), "end capture");
                ck(cudaGraphInstantiate(&ge, g, 0), "instantiate");
                const int reps = iters / kStage > 0 ? iters / kStage : 1;
                for (int i = 0; i < 5; ++i) cudaGraphLaunch(ge, s);
                for (int round = 0; round < 7; ++round) {
                    cudaEventRecord(e0, s);
                    for (int i = 0; i < reps; ++i) cudaGraphLaunch(ge, s);
                    cudaEventRecord(e1, s);
                    cudaEventSynchronize(e1);
                    float ms = 0;
                    cudaEventElapsedTime(&ms, e0, e1);
                    gr.push_back(1e3 * ms / (reps * kStage));
                }
                cudaGraphExecDestroy(ge);
                cudaGraphDestroy(g);
                ck(cudaGetLastError(), p.name);
                std::sort(st.begin(), st.end());
                std::sort(gr.begin(), gr.end());
                std::printf("  %-32s %7.1f / %7.1f   %7.1f / %7.1f\n", p.name, st[0], st[3], gr[0], gr[3]);
            }
        }
    std::printf("\nfused_gr_bench parts: %d problem(s)\n", bad);
    return bad ? 1 : 0;
}

}  // namespace

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    if (argc > 1 && std::strcmp(argv[1], "parts") == 0) return parts(argc, argv);
    const int iters = argc > 1 ? std::atoi(argv[1]) : 500;
    const int t_lo = argc > 2 ? std::atoi(argv[2]) : 1, t_hi = argc > 3 ? std::atoi(argv[3]) : 6;
    const int N = 2560, HC = 4, D = N * HC, LR = 320, TM = K::kFusedGrMaxT;
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.f, 1.f);
    auto fill = [&](std::vector<float>& v, float sc) { for (auto& x : v) x = sc * nd(rng); };
    std::vector<float> R((size_t) TM * D), bo((size_t) TM * N), inj((size_t) TM * HC), wn(D);
    fill(R, 1.0f); fill(bo, 0.5f); fill(inj, 1.0f); fill(wn, 0.3f);
    for (auto& x : wn) x += 1.0f;
    std::vector<uint16_t> wd((size_t) LR * D), wu((size_t) D * LR), wi((size_t) HC * D);
    for (auto& x : wd) x = bf16(0.02f * nd(rng));
    for (auto& x : wu) x = bf16(0.05f * nd(rng));
    for (auto& x : wi) x = bf16(0.02f * nd(rng));
    float *dR, *dbo, *dinj, *dwn, *dxn, *dlo, *drs, *dio, *dmix;
    uint16_t *dwd, *dwu, *dwi;
    cudaMalloc((void**) &dR, R.size() * 4);
    cudaMalloc((void**) &dbo, bo.size() * 4);
    cudaMalloc((void**) &dinj, inj.size() * 4);
    cudaMalloc((void**) &dwn, wn.size() * 4);
    cudaMalloc((void**) &dxn, (size_t) TM * D * 4);
    cudaMalloc((void**) &dlo, (size_t) TM * LR * 4);
    cudaMalloc((void**) &drs, (size_t) TM * HC * 4);
    cudaMalloc((void**) &dio, (size_t) TM * HC * 4);
    cudaMalloc((void**) &dmix, (size_t) TM * N * 4);
    cudaMalloc((void**) &dwd, wd.size() * 2);
    cudaMalloc((void**) &dwu, wu.size() * 2);
    cudaMalloc((void**) &dwi, wi.size() * 2);
    cudaMemcpy(dbo, bo.data(), bo.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dinj, inj.data(), inj.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dwn, wn.data(), wn.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dwd, wd.data(), wd.size() * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(dwu, wu.data(), wu.size() * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(dwi, wi.data(), wi.size() * 2, cudaMemcpyHostToDevice);
    cudaStream_t s;
    cudaStreamCreate(&s);
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    int failures = 0;
    for (int T = t_lo; T <= t_hi; ++T)
        for (int apply = 0; apply < 2; ++apply)
            for (int inject = 0; inject < 2; ++inject) {
                K::FusedGrArgs a[TM];
                for (int t = 0; t < T; ++t) {
                    a[t].R = dR + (size_t) t * D; a[t].R_out = dR + (size_t) t * D; a[t].apply = apply;
                    a[t].bo_prev = dbo + (size_t) t * N; a[t].inj_prev = dinj + (size_t) t * HC;
                    a[t].w_norm = dwn; a[t].w_down = dwd; a[t].w_up = dwu; a[t].w_inject = inject ? dwi : nullptr;
                    a[t].lo = dlo + (size_t) t * LR; a[t].rs = drs + (size_t) t * HC;
                    a[t].inject_out = dio + (size_t) t * HC; a[t].mixed = dmix + (size_t) t * N;
                }
                std::vector<float> out[2];
                for (int f = 0; f < 2; ++f) {
                    cudaMemcpy(dR, R.data(), R.size() * 4, cudaMemcpyHostToDevice);
                    cudaMemset(dlo, 0xff, (size_t) TM * LR * 4);
                    cudaMemset(drs, 0xff, (size_t) TM * HC * 4);
                    cudaMemset(dio, 0xff, (size_t) TM * HC * 4);
                    cudaMemset(dmix, 0xff, (size_t) TM * N * 4);
                    K::fused_gr_set_fast(f);
                    K::fused_gr_read_multi(a, T, dxn, s);
                    cudaStreamSynchronize(s);
                    auto& o = out[f];
                    o.resize((size_t) TM * (D + LR + HC + HC + N));
                    float* p = o.data();
                    cudaMemcpy(p, dR, (size_t) TM * D * 4, cudaMemcpyDeviceToHost); p += (size_t) TM * D;
                    cudaMemcpy(p, dlo, (size_t) TM * LR * 4, cudaMemcpyDeviceToHost); p += (size_t) TM * LR;
                    cudaMemcpy(p, drs, (size_t) TM * HC * 4, cudaMemcpyDeviceToHost); p += (size_t) TM * HC;
                    cudaMemcpy(p, dio, (size_t) TM * HC * 4, cudaMemcpyDeviceToHost); p += (size_t) TM * HC;
                    cudaMemcpy(p, dmix, (size_t) TM * N * 4, cudaMemcpyDeviceToHost);
                }
                const bool same = std::memcmp(out[0].data(), out[1].data(), out[0].size() * 4) == 0;
                double us[2] = {1e30, 1e30};
                for (int round = 0; round < 3; ++round)
                    for (int f = 0; f < 2; ++f) {
                        K::fused_gr_set_fast(f);
                        for (int i = 0; i < 20; ++i) K::fused_gr_read_multi(a, T, dxn, s);
                        cudaEventRecord(e0, s);
                        for (int i = 0; i < iters; ++i) K::fused_gr_read_multi(a, T, dxn, s);
                        cudaEventRecord(e1, s);
                        cudaEventSynchronize(e1);
                        float ms = 0;
                        cudaEventElapsedTime(&ms, e0, e1);
                        us[f] = std::fmin(us[f], 1e3 * ms / iters);
                    }
                std::printf("T %d apply %d inject %d | old %6.1f us  fast %6.1f us | %s\n", T, apply, inject, us[0], us[1],
                            same ? "bitwise equal" : "DIFFERS");
                if (!same) ++failures;
            }
    std::printf("fused_gr_bench: %d differing\n", failures);
    return failures ? 1 : 0;
}
