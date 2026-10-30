// src/kernels/s2_parity.cpp - S2 line (campaign): the router's top-10 kernels as launched under STRATA_S2_ROUTE_SMALL (0 / 1 / 2 / 3), bit for
// bit.  The switch is read once per process, so run this once per value and compare the printed hash:
//
//     STRATA_S2_ROUTE_SMALL=0 s2_parity   STRATA_S2_ROUTE_SMALL=3 s2_parity   ...   (all hashes must be equal)
//
// It feeds native_router_top10 (one token) and native_router_top10_multi (1..8 tokens) logits of several kinds - normal, a handful of
// repeated values (ties), all equal, -inf entries, NaN entries, large magnitudes - and folds every id and weight bit into one FNV-1a hash.
#include "strata/kernels/native_router.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

int main() {
    int n_dev = 0;
    if (cudaGetDeviceCount(&n_dev) != cudaSuccess || n_dev == 0) { std::puts("s2_parity: no GPU"); return 77; }
    cudaStream_t s;
    cudaStreamCreate(&s);
    const int MAXT = 8, NE = 512, K = 10;
    float* dl; int32_t* di; float* dw;
    cudaMalloc((void**) &dl, (size_t) MAXT * NE * 4);
    cudaMalloc((void**) &di, (size_t) MAXT * K * 4);
    cudaMalloc((void**) &dw, (size_t) MAXT * K * 4);
    std::mt19937 rng(12345);
    std::normal_distribution<float> nd(0.f, 1.f);
    unsigned long long h = 1469598103934665603ull;
    auto fold = [&](const void* p, size_t n) { for (size_t b = 0; b < n; ++b) h = (h ^ ((const unsigned char*) p)[b]) * 1099511628211ull; };
    long calls = 0;
    for (int kind = 0; kind < 8; ++kind)
        for (int rep = 0; rep < 60; ++rep) {
            std::vector<float> L((size_t) MAXT * NE);
            for (auto& x : L) {
                switch (kind) {
                    case 0: x = nd(rng); break;
                    case 1: x = 3.0f * nd(rng); break;
                    case 2: x = (float) (rng() % 5) - 2.0f; break;                       // 5 distinct values: many ties
                    case 3: x = 0.25f; break;                                            // all equal
                    case 4: x = (rng() % 7 == 0) ? -std::numeric_limits<float>::infinity() : nd(rng); break;
                    case 5: x = (rng() % 50 == 0) ? std::numeric_limits<float>::quiet_NaN() : nd(rng); break;
                    case 6: x = 80.0f * nd(rng); break;                                  // exp underflow / dominance
                    default: x = (float) (rng() % 3) * 0.5f + ((rng() % 200 == 0) ? 7.0f : 0.f); break;
                }
            }
            cudaMemcpy(dl, L.data(), L.size() * 4, cudaMemcpyHostToDevice);
            for (int T = 1; T <= MAXT; ++T) {
                cudaMemset(di, 0xff, (size_t) MAXT * K * 4);
                cudaMemset(dw, 0xff, (size_t) MAXT * K * 4);
                if (T == 1 && (rep & 1)) strata::kernels::native_router_top10(dl, di, dw, s);
                else strata::kernels::native_router_top10_multi(dl, di, dw, T, s);
                cudaStreamSynchronize(s);
                std::vector<int32_t> hi((size_t) T * K);
                std::vector<float> hw((size_t) T * K);
                cudaMemcpy(hi.data(), di, hi.size() * 4, cudaMemcpyDeviceToHost);
                cudaMemcpy(hw.data(), dw, hw.size() * 4, cudaMemcpyDeviceToHost);
                fold(hi.data(), hi.size() * 4);
                fold(hw.data(), hw.size() * 4);
                ++calls;
            }
        }
    const cudaError_t e = cudaGetLastError();
    const char* v = std::getenv("STRATA_S2_ROUTE_SMALL");
    std::printf("s2_parity: STRATA_S2_ROUTE_SMALL=%s  %ld calls  hash %016llx  %s\n", v ? v : "(unset)", calls, h,
                e == cudaSuccess ? "ok" : cudaGetErrorString(e));
    return e == cudaSuccess ? 0 : 1;
}
