// exp_bench.cu - microbench of the engine's compiler-scheduled FP16 gate/up kernel (REF: moe_ws_hf.cuh's gu_kernel, the engine's own header)
// against the hand-scheduled SASS twin (MOD: a cubin from gen_gu.py), on one PH402 die, with real-format 4-bit blocks (random valid codes, scales,
// fp16 d): G MAC/s, share of the 6451 peak, and a BITWISE comparison of every output plus the non-finite counter.
//   exp_bench <hand.cubin> <fam 0..3> <tier 16|32> <ntok> <experts> [reps] [flags...]
//   fam: 0 IQ2_S/IQ2_XS  1 IQ2_XXS  2 IQ3_XXS  3 IQ3_S      ntok x 10 routed rows over `experts` slots (lognormal counts), chunks of <= 64 rows
//   flags: wild (inject inf / NaN / huge activations)  tail (many small chunks)  noref (time MOD only)
//   env:   PLACE=wxyt (operands astride a 4 GiB boundary: w arena, x rows, y output, tables)  PLACE_W_AT=<bytes of the arena below the boundary>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <omp.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

#ifndef HFHDR
#define HFHDR "moe_ws_hf.cuh"
#endif
#include HFHDR

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { std::fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #x, __FILE__, __LINE__, cudaGetErrorString(e_)); std::exit(1); } } while (0)
#define CKD(x) do { CUresult r_ = (x); if (r_ != CUDA_SUCCESS) { const char* m_ = nullptr; cuGetErrorString(r_, &m_); std::fprintf(stderr, "driver error %s at %s:%d: %s\n", #x, __FILE__, __LINE__, m_ ? m_ : "?"); std::exit(1); } } while (0)

namespace hf = strata::prefill::ws::hf;
namespace nb = strata::core::nibble;
using strata::prefill::ws::Chunk;

// ---- the prep kernel of the hand path: the compiler kernel's prologue (k, hs_tab, lut) written to a table in global memory -----------------------------
template <int FAM>
__global__ void __launch_bounds__(256) prep_kernel(uint32_t* __restrict__ tab, const int layer) {
    __shared__ uint32_t hs_tab[32];
    const int tid = threadIdx.x;
    const unsigned cur = *(volatile const unsigned*) &hf::g_hf.cur;
    const float xmax = __half2float(__ushort_as_half((unsigned short) cur));
    const float need = xmax * (hf::wmax(FAM) * (1.0f / 512.0f));
    int k;
    if (!(need <= 1e30f)) {
        k = hf::KMAX;
    } else if (need <= 0.0f) {
        k = hf::KMIN;
    } else {
        int e;
        (void) frexpf(need, &e);
        k = e < hf::KMIN ? hf::KMIN : e > hf::KMAX ? hf::KMAX : e;
    }
    if (tid == 0) {
        atomicAdd(&hf::g_hf.launches, 1u);
        atomicMax(&hf::g_hf.all_max, cur);
        atomicMax(&hf::g_hf.layer_max[layer & 63], cur);
        atomicAdd(&hf::g_hf.khist[k - hf::KMIN], 1u);
        if (cur >= 0x7C00u) atomicAdd(&hf::g_hf.bad_x, 1u);
    }
    const float kdn = ldexpf(1.0f, -k), kup = ldexpf(1.0f, k);
    if (tid < 32) {
        const int sv = (FAM == 1 || FAM == 3) ? tid : 2 * tid + 1;
        const uint32_t h = (uint32_t) __half_as_ushort(__float2half_rn((float) sv * kdn));
        hs_tab[tid] = h | (h << 16);
    }
    const uint32_t lutv = hf::pack_h2(hf::code_value<FAM>(tid & 15), hf::code_value<FAM>(tid >> 4));
    __syncthreads();
    uint32_t a, b;
    if (FAM == 0) { a = hs_tab[tid & 15]; b = hs_tab[(tid >> 4) & 15]; }
    else { a = hs_tab[tid & 31]; b = a; }
    tab[tid] = lutv;
    tab[256 + tid] = a;
    tab[512 + tid] = b;
    if (tid == 0) tab[768] = __float_as_uint(hf::cfmt(FAM) * kup);
}

static uint64_t splitmix(uint64_t& s) {
    uint64_t z = (s += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}
struct Rng {
    uint64_t s;
    explicit Rng(uint64_t seed) : s(seed) { splitmix(s); }
    uint64_t next() { return splitmix(s); }
    double u01() { return ((next() >> 11) + 0.5) * (1.0 / 9007199254740992.0); }
    double normal() { return std::sqrt(-2.0 * std::log(u01())) * std::cos(6.283185307179586 * u01()); }
};
static uint16_t f2h_host(float f) { __half h = __float2half_rn(f); uint16_t b; std::memcpy(&b, &h, 2); return b; }

static const int FAM_TYPE[4] = {nb::kIQ2_S, nb::kIQ2_XXS, nb::kIQ3_XXS, nb::kIQ3_S};

template <int TPP, int FAM>
static void launch_ref(unsigned grid, const Chunk* ch, const char* w, const int64_t* xoff, const uint4* xh, const int32_t* src, float* dst) {
    hf::gu_kernel<TPP, FAM><<<grid, hf::NT>>>(ch, w, xoff, xh, src, dst, 3);
}
static void launch_ref_any(int tier, int fam, unsigned grid, const Chunk* ch, const char* w, const int64_t* xoff, const uint4* xh, const int32_t* src, float* dst) {
#define L(T, F) if (tier == T && fam == F) { launch_ref<T, F>(grid, ch, w, xoff, xh, src, dst); return; }
    L(32, 0) L(32, 1) L(32, 2) L(32, 3) L(16, 0) L(16, 1) L(16, 2) L(16, 3)
#undef L
}
template <int FAM> static void launch_prep(uint32_t* tab) { prep_kernel<FAM><<<1, 256>>>(tab, 3); }
static void launch_prep_any(int fam, uint32_t* tab) {
    switch (fam) { case 0: launch_prep<0>(tab); break; case 1: launch_prep<1>(tab); break; case 2: launch_prep<2>(tab); break; default: launch_prep<3>(tab); }
}

static double bench_ms(const std::function<void()>& f, int reps, double* minms) {
    f(); f();
    CK(cudaDeviceSynchronize());
    cudaEvent_t a, b;
    CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
    std::vector<float> ms;
    for (int i = 0; i < reps; ++i) {
        CK(cudaEventRecord(a));
        f();
        CK(cudaEventRecord(b));
        CK(cudaEventSynchronize(b));
        float m; CK(cudaEventElapsedTime(&m, a, b));
        ms.push_back(m);
    }
    std::sort(ms.begin(), ms.end());
    *minms = ms.front();
    return ms[ms.size() / 2];
}

constexpr double kPeak = 6451.2;

int main(int argc, char** argv) {
    if (argc < 6) { std::fprintf(stderr, "usage: see the head of exp_bench.cu\n"); return 2; }
    const char* cubin = argv[1];
    const int fam = std::atoi(argv[2]), tier = std::atoi(argv[3]), ntok = std::atoi(argv[4]), E = std::atoi(argv[5]);
    int reps = argc > 6 && argv[6][0] >= '0' && argv[6][0] <= '9' ? std::atoi(argv[6]) : 5;
    bool wild = false, tail = false, noref = false, full = false;
    for (int a = 6; a < argc; ++a) { if (!std::strcmp(argv[a], "wild")) wild = true; if (!std::strcmp(argv[a], "tail")) tail = true; if (!std::strcmp(argv[a], "noref")) noref = true; if (!std::strcmp(argv[a], "full")) full = true; }
    CKD(cuInit(0));
    cudaDeviceProp P{};
    CK(cudaSetDevice(0)); CK(cudaFree(0));
    CK(cudaGetDeviceProperties(&P, 0));
    std::printf("device 0 = %s  cc %d.%d  %d SMs  PCI bus %02x\n", P.name, P.major, P.minor, P.multiProcessorCount, P.pciBusID);
    if (P.major != 6 || P.minor != 0 || std::strstr(P.name, "PH402") == nullptr) { std::fprintf(stderr, "NOT a PH402 die: refusing to run\n"); return 3; }
    CUmodule mod;
    CKD(cuModuleLoad(&mod, cubin));
    const bool paired = fam == 3;
    char kname[32];
    std::snprintf(kname, sizeof kname, "gu_t%d%s", tier, paired ? "p" : "");
    CUfunction fn;
    CKD(cuModuleGetFunction(&fn, mod, kname));
    int regs = 0, smem = 0;
    CKD(cuFuncGetAttribute(&regs, CU_FUNC_ATTRIBUTE_NUM_REGS, fn));
    CKD(cuFuncGetAttribute(&smem, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, fn));
    int occ = 0;
    CKD(cuOccupancyMaxActiveBlocksPerMultiprocessor(&occ, fn, 32, 0));
    std::printf("hand kernel %s: %d registers, %d B static shared, %d CTAs/SM\n", kname, regs, smem, occ);

    // ---- routing: ntok x 10 rows over E expert slots, lognormal counts -------------------------------------------------------------------------------
    const int topk = 10;
    const int64_t R = (int64_t) ntok * topk;
    std::vector<int> cnt(E, 0);
    {
        Rng r(0xABC123ULL + ntok * 7 + E);
        std::vector<double> wgt(E);
        double sum = 0;
        for (int e = 0; e < E; ++e) { wgt[e] = full ? 1.0 : std::exp((tail ? 1.6 : 0.8) * r.normal()); sum += wgt[e]; }
        int64_t left = R;
        for (int e = 0; e < E; ++e) { cnt[e] = (int) std::floor(R * wgt[e] / sum); left -= cnt[e]; }
        for (int e = 0; left > 0; e = (e + 1) % E) { ++cnt[e]; --left; }
    }
    std::vector<Chunk> chunks;
    std::vector<int32_t> srcv(R);
    {
        Rng r(0x777ULL + ntok);
        int64_t row = 0;
        for (int e = 0; e < E; ++e) {
            for (int c = 0; c < cnt[e]; c += 64) {
                const int rows = std::min(64, cnt[e] - c);
                chunks.push_back(Chunk{e, (int) row, rows, 0});
                row += rows;
            }
        }
        for (int64_t i = 0; i < R; ++i) srcv[i] = (int32_t) (r.next() % (uint64_t) ntok);
    }
    const int nchunks = (int) chunks.size();
    std::printf("fam %d tier %d: %d tokens x %d = %lld rows over %d slots, %d chunks (rows/expert mean %.1f)\n", fam, tier, ntok, topk, (long long) R, E, nchunks, (double) R / E);
    if (nchunks > 65535) { std::fprintf(stderr, "too many chunks\n"); return 2; }

    // ---- weights: valid 4-bit codes ---------------------------------------------------------------------------------------------------------------
    const size_t stride = nb::kGuBytes;   // 1,766,400 (no down part here)
    std::vector<uint8_t> hw((size_t) E * stride + 65536);   // the engine's slots carry the down part after the gate/up part: the hand kernel's last prefetch reads 20 bytes into it
#pragma omp parallel for schedule(static)
    for (int e = 0; e < E; ++e) {
        Rng r(0x5EED0000ULL + (uint64_t) e * 0x9E3779B1ULL + fam);
        uint8_t* slot = hw.data() + (size_t) e * stride;
        for (int row = 0; row < nb::kRows; ++row) {
            uint32_t* codes = (uint32_t*) (slot + nb::code_off(row, 0));
            for (int k = 0; k < nb::kItems; ++k) {
                for (int m = 0; m < 4; ++m) {
                    uint32_t w = 0;
                    for (int nib = 0; nib < 8; ++nib) {
                        uint32_t c;
                        if (fam <= 1) { static const uint32_t v[6] = {0, 1, 2, 4, 5, 6}; c = v[r.next() % 6]; }
                        else c = (uint32_t) (r.next() & 15);
                        w |= c << (4 * nib);
                    }
                    codes[k * 4 + m] = w;
                }
                uint8_t sc;
                if (fam == 0) sc = (uint8_t) (r.next() & 255);
                else if (fam == 1) sc = (uint8_t) (2 * (r.next() % 16) + 1);
                else if (fam == 2) sc = (uint8_t) (r.next() & 15);
                else sc = (uint8_t) (2 * (r.next() % 16) + 1);
                slot[nb::scale_off(row, k)] = sc;
            }
            for (int sb = 0; sb < nb::kSuper; ++sb) {
                const float dv = (float) (0.002 + 0.02 * r.u01());
                const uint16_t h = f2h_host(dv);
                std::memcpy(slot + nb::d_off(row, sb), &h, 2);
            }
        }
    }
    // ---- activations ------------------------------------------------------------------------------------------------------------------------------------
    std::vector<uint16_t> hx((size_t) ntok * 2560);
#pragma omp parallel for schedule(static)
    for (int i = 0; i < ntok; ++i) {
        Rng r(0x51ED0000ULL + (uint64_t) i * 2654435761ULL);
        const double ts = std::exp(0.3 * r.normal());
        for (int b = 0; b < 80; ++b) {
            const double bs = ts * std::exp(0.7 * r.normal());
            for (int k = 0; k < 32; ++k) {
                const double z = r.normal(), n1 = r.normal(), n2 = r.normal(), n3 = r.normal();
                const double t3 = z / std::sqrt((n1 * n1 + n2 * n2 + n3 * n3) / 3.0);
                hx[(size_t) i * 2560 + b * 32 + k] = f2h_host((float) (bs * t3 / std::sqrt(3.0)));
            }
        }
    }
    if (wild) {
        Rng r(0x1234);
        for (int i = 0; i < ntok; i += 37) hx[(size_t) i * 2560 + (r.next() % 2560)] = 0x7C00;          // +inf
        for (int i = 5; i < ntok; i += 53) hx[(size_t) i * 2560 + (r.next() % 2560)] = 0x7E01;          // NaN
        for (int i = 9; i < ntok; i += 11) hx[(size_t) i * 2560 + (r.next() % 2560)] = f2h_host(30000.0f);
    }
    // ---- device buffers (PLACE: astride 4 GiB boundaries) --------------------------------------------------------------------------------------------------
    const char* place = std::getenv("PLACE");
    CUdeviceptr pool = 0, b0 = 0;
    if (place != nullptr) {
        CKD(cuMemAlloc(&pool, (size_t) 21 << 30));
        b0 = ((pool + ((size_t) 64 << 20)) + 0xFFFFFFFFull) & ~0xFFFFFFFFull;
        std::printf("PLACE=%s: pool %llx, first boundary %llx\n", place, (unsigned long long) pool, (unsigned long long) b0);
    }
    auto astride = [&](int which, size_t below) { return b0 + (size_t) which * ((size_t) 4 << 30) - below; };
    const size_t w_bytes = hw.size(), x_bytes = hx.size() * 2;
    size_t w_below = std::getenv("PLACE_W_AT") ? (size_t) std::atoll(std::getenv("PLACE_W_AT")) : (size_t) (E / 2) * stride + 17 * 1280 + 640;
    size_t x_below = std::getenv("PLACE_X_AT") ? (size_t) std::atoll(std::getenv("PLACE_X_AT")) : (size_t) (ntok / 2) * 5120 + 2560 + 32;
    CUdeviceptr dW, dX, dY, dTab, dCh, dXo, dSrc;
    if (place && std::strchr(place, 'w')) dW = astride(0, w_below & ~(size_t) 255); else CKD(cuMemAlloc(&dW, w_bytes));
    if (place && std::strchr(place, 'x')) dX = astride(1, (x_below + 15) & ~(size_t) 15); else CKD(cuMemAlloc(&dX, x_bytes));
    const size_t ny = (size_t) R * 1280;
    const size_t GUARD = 1 << 16;
    CUdeviceptr dYr0, dYm0;
    CKD(cuMemAlloc(&dYr0, (ny + 2 * GUARD) * 4));
    if (place && std::strchr(place, 'y')) dYm0 = astride(2, (ny * 4 / 2 + 4 * GUARD) & ~(size_t) 3); else CKD(cuMemAlloc(&dYm0, (ny + 2 * GUARD) * 4));
    dY = dYm0 + GUARD * 4;
    CUdeviceptr dYr = dYr0 + GUARD * 4;
    if (place && std::strchr(place, 't')) dTab = astride(3, 1536); else CKD(cuMemAlloc(&dTab, 4096));
    CKD(cuMemAlloc(&dCh, chunks.size() * sizeof(Chunk)));
    CKD(cuMemAlloc(&dXo, E * 8));
    CKD(cuMemAlloc(&dSrc, R * 4));
    std::vector<int64_t> xoff(E);
    for (int e = 0; e < E; ++e) xoff[e] = (int64_t) e * stride;
    CKD(cuMemcpyHtoD(dW, hw.data(), w_bytes));
    CKD(cuMemcpyHtoD(dX, hx.data(), x_bytes));
    CKD(cuMemcpyHtoD(dCh, chunks.data(), chunks.size() * sizeof(Chunk)));
    CKD(cuMemcpyHtoD(dXo, xoff.data(), E * 8));
    CKD(cuMemcpyHtoD(dSrc, srcv.data(), R * 4));
    CKD(cuMemsetD32(dYr0, 0xFFFFFFFFu, ny + 2 * GUARD));
    CKD(cuMemsetD32(dYm0, 0xFFFFFFFFu, ny + 2 * GUARD));
    CKD(cuMemsetD32(dTab, 0, 1024));

    hf::HfState* g_state = nullptr;
    CK(cudaGetSymbolAddress((void**) &g_state, hf::g_hf));
    // the max-reduce of the layer's input (shared by both paths)
    auto xmax = [&] {
        CK(cudaMemset(g_state, 0, sizeof(unsigned)));
        const int64_t n16 = (int64_t) ntok * hf::XROW16;
        const unsigned blocks = (unsigned) std::max<int64_t>(1, std::min<int64_t>(2048, (n16 + 1023) / 1024));
        hf::xmax_kernel<<<blocks, 256>>>((const uint4*) dX, n16);
    };
    const unsigned refgrid = (unsigned) (nchunks * hf::PIECES);
    auto ref = [&] {
        xmax();
        launch_ref_any(std::getenv("REFTIER") ? std::atoi(std::getenv("REFTIER")) : tier, fam, refgrid, (const Chunk*) dCh, (const char*) dW, (const int64_t*) dXo, (const uint4*) dX, (const int32_t*) dSrc, (float*) dYr);
    };
    CUdeviceptr badp = (CUdeviceptr) ((char*) g_state + offsetof(hf::HfState, bad_y));
    CUdeviceptr pch = dCh, pw = dW, pxo = dXo, pxh = dX, psrc = dSrc, pdst = dY, ptab = dTab, pbad = badp;
    void* params[] = {&pch, &pw, &pxo, &pxh, &psrc, &pdst, &ptab, &pbad};
    const unsigned dynsm = std::getenv("DYN") ? (unsigned) std::atoi(std::getenv("DYN")) : 0u;
    auto mod_launch = [&] { CKD(cuLaunchKernel(fn, 20, (unsigned) nchunks, 1, 32, 1, 1, dynsm, 0, params, 0)); };
    auto modf = [&] {
        xmax();
        launch_prep_any(fam, (uint32_t*) dTab);
        mod_launch();
    };
    // ---- correctness ---------------------------------------------------------------------------------------------------------------------------------------
    hf::HfState zero{};
    auto read_state = [&](hf::HfState* st) { CK(cudaMemcpyFromSymbol(st, hf::g_hf, sizeof(*st))); };
    auto reset_state = [&] { CK(cudaMemcpyToSymbol(hf::g_hf, &zero, sizeof zero)); };
    hf::HfState sr{}, sm{};
    reset_state(); ref(); CK(cudaDeviceSynchronize()); read_state(&sr);
    reset_state(); modf(); CK(cudaDeviceSynchronize()); read_state(&sm);
    std::vector<float> yr(ny), ym(ny);
    CKD(cuMemcpyDtoH(yr.data(), dYr, ny * 4));
    CKD(cuMemcpyDtoH(ym.data(), dY, ny * 4));
    size_t diff = 0, first = (size_t) -1, nonfin = 0, unwritten = 0;
    for (size_t i = 0; i < ny; ++i) {
        if (std::memcmp(&yr[i], &ym[i], 4) != 0) { if (first == (size_t) -1) first = i; ++diff; }
        nonfin += !std::isfinite(yr[i]);
        uint32_t bits; std::memcpy(&bits, &ym[i], 4);
        unwritten += bits == 0xFFFFFFFFu;
    }
    size_t gbad = 0;
    for (int side = 0; side < 2; ++side) {
        std::vector<uint32_t> g(GUARD);
        CKD(cuMemcpyDtoH(g.data(), side == 0 ? dYm0 : dYm0 + (GUARD + ny) * 4, GUARD * 4));
        for (auto v : g) gbad += v != 0xFFFFFFFFu;
    }
    std::printf("    k histogram equal: %s | bad_x REF %u MOD %u | bad_y REF %u MOD %u\n", std::memcmp(sr.khist, sm.khist, sizeof sr.khist) == 0 ? "yes" : "NO", sr.bad_x, sm.bad_x, sr.bad_y, sm.bad_y);
    std::printf("    bitwise over %zu outputs: %s (%zu differ%s) | guard words touched: %zu | never written in MOD: %zu | non-finite outputs in REF: %zu\n", ny,
                diff == 0 ? "IDENTICAL" : "DIFFERENT", diff, first == (size_t) -1 ? "" : (" first at " + std::to_string(first / 1280) + ":" + std::to_string(first % 1280)).c_str(), gbad, unwritten, nonfin);
    if (diff != 0) {
        int shown = 0;
        for (size_t i = 0; i < ny && shown < 8; ++i)
            if (std::memcmp(&yr[i], &ym[i], 4) != 0) { std::printf("      row %zu col %zu: ref %.9g (%08x) mod %.9g (%08x)\n", i / 1280, i % 1280, yr[i], *(uint32_t*) &yr[i], ym[i], *(uint32_t*) &ym[i]); ++shown; }
        // by (row % 64)-ish structure
        std::vector<size_t> byrowmod(64, 0), bycol(64, 0);
        for (size_t i = 0; i < ny; ++i) if (std::memcmp(&yr[i], &ym[i], 4) != 0) { byrowmod[(i % 1280) % 64]++; }
        std::printf("      by col%%64:"); for (int i = 0; i < 64; ++i) if (byrowmod[i]) std::printf(" %d:%zu", i, byrowmod[i]);
        std::printf("\n");
    }
    // ---- timing --------------------------------------------------------------------------------------------------------------------------------------------
    double mn_r = 0, mn_m = 0, ms_r = 0, ms_m = 0;
    if (!noref) ms_r = bench_ms(ref, reps, &mn_r);
    ms_m = bench_ms(modf, reps, &mn_m);
    // the gu launches alone (xmax / prep are not in the engine's kernel time for REF; prep is a few us)
    double mn_k = 0, ms_k = bench_ms(mod_launch, reps, &mn_k);
    CUdeviceptr dummy = 0; (void) dummy;
    const double macs = (double) R * 1280.0 * 2560.0;
    if (!noref) std::printf("    REF compiler    %9.3f ms (min %9.3f)  %7.1f G MAC/s = %5.1f%% of %.0f\n", ms_r, mn_r, macs / ms_r * 1e-6, 100.0 * macs / ms_r * 1e-6 / kPeak, kPeak);
    std::printf("    MOD hand (+prep)%9.3f ms (min %9.3f)  %7.1f G MAC/s = %5.1f%% of %.0f", ms_m, mn_m, macs / ms_m * 1e-6, 100.0 * macs / ms_m * 1e-6 / kPeak, kPeak);
    if (!noref) std::printf("   x%.3f", ms_r / ms_m);
    std::printf("\n    MOD kernel alone%9.3f ms (min %9.3f)  %7.1f G MAC/s = %5.1f%% of %.0f\n", ms_k, mn_k, macs / ms_k * 1e-6, 100.0 * macs / ms_k * 1e-6 / kPeak, kPeak);
    return diff == 0 && gbad == 0 && unwritten == 0 && sr.bad_y == sm.bad_y ? 0 : 1;
}
