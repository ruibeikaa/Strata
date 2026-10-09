// src/kernels/fused_gr_bench.cpp - fused_gr_read_multi: the latency-hidden kernels (STRATA_GR_FAST; gfx906's, built
// for CUDA too) against the old ones, bit for bit on every output (lo, rs, inject, mixed, the in-place R; with q8 1
// also the STRATA_QFUSE q8_1 image), each read's time and the up projection's alone (stamps after the down
// projection and after the read).
//
//     build/fused_gr_bench [iters] [T min] [T max] [q8]
//
// The read runs as fused_gr_variant() says (no fused_gr_check here): plain unless STRATA_HC_SPLIT=1 or 2 names
// split or staged.  Under STRATA_HC_SPLIT=2 (staged, what the engine's check picks) the norm is the split kernel in
// both arms, so the up projection is the only kernel that differs; plain swaps the norm too.
#include "strata/kernels/fused_gr.hpp"
#include "strata/kernels/verify_kernels.hpp"

#include <cuda_runtime.h>

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

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    const int iters = argc > 1 ? std::atoi(argv[1]) : 500;
    const int t_lo = argc > 2 ? std::atoi(argv[2]) : 1, t_hi = argc > 3 ? std::atoi(argv[3]) : 6;
    const bool q8 = argc > 4 && std::atoi(argv[4]) != 0;
    const int N = 2560, HC = 4, D = N * HC, LR = 320, TM = K::kFusedGrMaxT;
    const size_t QB = (size_t) (N / 32) * 36;   // a token's q8_1 image: N / 32 blocks of half2 (d, sum) + 32 int8
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
    uint8_t* dq8 = nullptr;
    unsigned* dcnt = nullptr;
    unsigned long long* dst = nullptr;   // per iteration: after the norm, after the down projection, after the read
    cudaMalloc((void**) &dq8, (size_t) TM * QB);
    cudaMalloc((void**) &dcnt, (size_t) (N / 32) * sizeof(unsigned));
    cudaMemset(dcnt, 0, (size_t) (N / 32) * sizeof(unsigned));   // the read's group counters start at zero
    cudaMalloc((void**) &dst, (size_t) 3 * iters * sizeof(unsigned long long));
    std::vector<unsigned long long> hst((size_t) 3 * iters);
    {
        int dev = 0;
        cudaGetDevice(&dev);
        cudaDeviceProp prop{};
        cudaGetDeviceProperties(&prop, dev);
        static const char* const vname[4] = {"?", "plain", "split", "staged"};
        const int v = K::fused_gr_variant();
        std::printf("CUDA%d %s (sm_%d%d): the %s read (STRATA_HC_SPLIT)%s\n", dev, prop.name, prop.major, prop.minor,
                    vname[v >= 1 && v <= 3 ? v : 0], q8 ? ", with the q8_1 image" : "");
    }
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
                    a[t].q8_mixed = q8 ? dq8 + (size_t) t * QB : nullptr; a[t].q8_cnt = q8 ? dcnt : nullptr;
                }
                std::vector<float> out[2];
                std::vector<uint8_t> outq[2];
                for (int f = 0; f < 2; ++f) {
                    cudaMemcpy(dR, R.data(), R.size() * 4, cudaMemcpyHostToDevice);
                    cudaMemset(dlo, 0xff, (size_t) TM * LR * 4);
                    cudaMemset(drs, 0xff, (size_t) TM * HC * 4);
                    cudaMemset(dio, 0xff, (size_t) TM * HC * 4);
                    cudaMemset(dmix, 0xff, (size_t) TM * N * 4);
                    cudaMemset(dq8, 0xff, (size_t) TM * QB);
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
                    outq[f].resize((size_t) TM * QB);
                    cudaMemcpy(outq[f].data(), dq8, outq[f].size(), cudaMemcpyDeviceToHost);
                }
                const bool same = std::memcmp(out[0].data(), out[1].data(), out[0].size() * 4) == 0 &&
                                  std::memcmp(outq[0].data(), outq[1].data(), outq[0].size()) == 0;
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
                // the up projection alone: the read's own stamp after the down projection to one right after the read
                // (both arms pay the same stamp launch, so the difference is the kernel's)
                double up_us[2] = {1e30, 1e30};
                for (int round = 0; round < 3; ++round)
                    for (int f = 0; f < 2; ++f) {
                        K::fused_gr_set_fast(f);
                        for (int i = 0; i < iters; ++i) {
                            K::fused_gr_read_multi(a, T, dxn, s, dst, 3 * i);
                            K::gpu_stamp(dst, 3 * i + 2, s);
                        }
                        cudaStreamSynchronize(s);
                        cudaMemcpy(hst.data(), dst, hst.size() * sizeof(unsigned long long), cudaMemcpyDeviceToHost);
                        double ns = 0;
                        for (int i = 0; i < iters; ++i) ns += (double) (hst[(size_t) 3 * i + 2] - hst[(size_t) 3 * i + 1]);
                        up_us[f] = std::fmin(up_us[f], 1e-3 * ns / iters);
                    }
                std::printf("T %d apply %d inject %d | read: old %6.1f us  fast %6.1f us | up: old %6.2f us  fast %6.2f us | %s\n",
                            T, apply, inject, us[0], us[1], up_us[0], up_us[1], same ? "bitwise equal" : "DIFFERS");
                if (!same) ++failures;
            }
    std::printf("fused_gr_bench: %d differing\n", failures);
    return failures ? 1 : 0;
}
