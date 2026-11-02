// src/prefill/hfma_small_parity.cu - STRATA_PF_HFMA_SMALL's kernels on one GP100 die (sm_60), synthetic data, no model.
//
//   hfma_small_parity --kernels       xcvt_kernel + cvt_fix_kernel against xmax_kernel + cvt_kernel: the FP16 image and the HcState counters bit for bit
//                                     (F16 / BF16 inputs, row strides, outliers that need a shift, inf, NaN); inj_kernel's error against a double reference
//                                     next to the up product's, the down product's and the GEMM kernel on a 128-wide tile of N = 4
//   hfma_small_parity --gemm <file>   Gemm::bf16_hfma for the down product with the inject companion and the up product, written to <file>
//                                     (run with STRATA_PF_HFMA_SMALL=0 and =1: the down and up files must be equal byte for byte)
//   hfma_small_parity --mix <file>    gr_mix_r's outputs, written to <file> (run with STRATA_PF_HFMA_SMALL=0 and =1: equal byte for byte)
// Exit 0 = every check passed; 77 = no GP100.
#include "gemm_hfma_hc.cuh"
#include "strata/prefill/gemm.hpp"
#include "strata/prefill/kernels.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <random>
#include <string>
#include <vector>

namespace hc = strata::prefill::hfma::hc;
namespace hf = strata::prefill::hfma;

#define CK(x) do { const cudaError_t e_ = (x); if (e_ != cudaSuccess) { std::printf("FAIL %s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_)); std::exit(1); } } while (0)

static uint16_t bf16_of(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    if ((u & 0x7FFFFFFFu) > 0x7F800000u) return (uint16_t) ((u >> 16) | 0x40u);   // NaN stays NaN
    u += 0x7fffu + ((u >> 16) & 1u);
    return (uint16_t) (u >> 16);
}
static float of_bf16(uint16_t b) {
    const uint32_t u = (uint32_t) b << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}
static uint16_t f16_of(float f) { return __half_as_ushort(__float2half_rn(f)); }

template <class T> struct Dev {
    T* p = nullptr;
    size_t n = 0;
    explicit Dev(size_t n_) : n(n_) { CK(cudaMalloc((void**) &p, n * sizeof(T))); }
    ~Dev() { cudaFree(p); }
    Dev(const Dev&) = delete;
    void up(const std::vector<T>& h) { CK(cudaMemcpy(p, h.data(), std::min(n, h.size()) * sizeof(T), cudaMemcpyHostToDevice)); }
    std::vector<T> down() const { std::vector<T> h(n); CK(cudaMemcpy(h.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost)); return h; }
};

// activations like a normalized hidden state: per-channel scales (log-normal), a few massive channels
static std::vector<float> make_x(std::mt19937_64& rng, int64_t rows, int64_t K, float massive) {
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> chan((size_t) K);
    for (auto& c : chan) c = std::exp(0.6f * nd(rng));
    for (int64_t k = 0; k < K; k += 997) chan[(size_t) k] *= massive;
    std::vector<float> x((size_t) (rows * K));
    for (int64_t r = 0; r < rows; ++r)
        for (int64_t k = 0; k < K; ++k) x[(size_t) (r * K + k)] = nd(rng) * chan[(size_t) k];
    return x;
}
static std::vector<float> make_w(std::mt19937_64& rng, int64_t N, int64_t K, float sigma) {
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> w((size_t) (N * K));
    for (auto& v : w) {
        const float g = nd(rng);
        v = sigma * g * (1.0f + 0.5f * std::fabs(nd(rng)));
    }
    return w;
}

struct Err {
    double rms = 0, mx = 0;   // RMS error / RMS reference, max |error| / RMS reference
};
// y[rows][N] (device image, row stride ldy) against the double products of the BF16 x[rows][K] and w[N][K], on `nsamp` evenly spread rows
static Err err_of(const std::vector<float>& y, int64_t ldy, const std::vector<uint16_t>& xb, const std::vector<uint16_t>& wb, int64_t rows, int64_t N, int64_t K, int nsamp) {
    double e2 = 0, r2 = 0, mx = 0;
    int64_t n = 0;
    for (int s = 0; s < nsamp; ++s) {
        const int64_t r = rows <= nsamp ? s : (int64_t) s * (rows - 1) / (nsamp - 1);
        if (r >= rows) break;
        for (int64_t c = 0; c < N; ++c) {
            double ref = 0;
            for (int64_t k = 0; k < K; ++k) ref += (double) of_bf16(xb[(size_t) (r * K + k)]) * (double) of_bf16(wb[(size_t) (c * K + k)]);
            const double d = (double) y[(size_t) (r * ldy + c)] - ref;
            e2 += d * d;
            r2 += ref * ref;
            mx = std::max(mx, std::fabs(d));
            ++n;
        }
    }
    Err e;
    const double rr = std::sqrt(r2 / (double) n);
    e.rms = std::sqrt(e2 / (double) n) / rr;
    e.mx = mx / rr;
    return e;
}

static int g_fail = 0;
static void check(bool ok, const char* what) {
    std::printf("%s  %s\n", ok ? "ok  " : "FAIL", what);
    if (!ok) ++g_fail;
}

// ---------------------------------------------------------------------------------------------------------------------------------------------
template <bool F16>
static bool image_case(std::mt19937_64& rng, int64_t rows, int64_t K, int64_t pad, int kind, float wmax, std::string& what) {
    const int64_t K8 = K / 8, ldx8 = K8 + pad, n8 = rows * K8;
    const size_t n16 = (size_t) (rows * ldx8) * 8;
    std::vector<uint16_t> xh(n16, 0);
    std::vector<float> xf = make_x(rng, rows, K, kind == 1 ? 3000.0f : 1.0f);
    for (int64_t r = 0; r < rows; ++r)
        for (int64_t k = 0; k < K; ++k) {
            float v = xf[(size_t) (r * K + k)];
            if (kind == 4 && (r * K + k) % 5003 == 17) v = INFINITY;
            if (kind == 5 && (r * K + k) % 7001 == 5) v = NAN;
            if (kind == 6) v = 0.0f;
            if (F16 && std::isfinite(v)) v = std::min(std::max(v, -65504.0f), 65504.0f);
            xh[(size_t) (r * ldx8 * 8 + k)] = F16 ? f16_of(v) : bf16_of(v);
        }
    Dev<uint16_t> X(n16), A(n16), B(n16);
    X.up(xh);
    CK(cudaMemset(A.p, 0xAB, n16 * 2));
    CK(cudaMemset(B.p, 0xAB, n16 * 2));
    hc::HcState zero{};
    const unsigned blocks = (unsigned) std::max<int64_t>(1, std::min<int64_t>(2048, (n8 + 1023) / 1024));
    hc::HcState sa{}, sb{};
    CK(cudaMemcpyToSymbol(hc::g_hc, &zero, sizeof zero));
    {
        unsigned* cur = nullptr;
        void* sp = nullptr;
        CK(cudaGetSymbolAddress(&sp, hc::g_hc));
        cur = (unsigned*) sp;
        CK(cudaMemset(cur, 0, 4));
        hc::xmax_kernel<F16><<<blocks, 256>>>(reinterpret_cast<const uint4*>(X.p), n8, K8, ldx8);
        hc::cvt_kernel<F16><<<blocks, 256>>>(reinterpret_cast<const uint4*>(X.p), reinterpret_cast<uint4*>(A.p), n8, K8, ldx8, wmax);
        CK(cudaDeviceSynchronize());
    }
    CK(cudaMemcpyFromSymbol(&sa, hc::g_hc, sizeof sa));
    CK(cudaMemcpyToSymbol(hc::g_hc, &zero, sizeof zero));
    {
        void* sp = nullptr;
        CK(cudaGetSymbolAddress(&sp, hc::g_hc));
        CK(cudaMemset(sp, 0, 4));
        hc::xcvt_kernel<F16><<<blocks, 256>>>(reinterpret_cast<const uint4*>(X.p), reinterpret_cast<uint4*>(B.p), (unsigned) n8, (unsigned) K8, (unsigned) ldx8);
        hc::cvt_fix_kernel<F16><<<std::min(blocks, 224u), 256>>>(reinterpret_cast<const uint4*>(X.p), reinterpret_cast<uint4*>(B.p), (unsigned) n8, (unsigned) K8, (unsigned) ldx8, wmax);
        CK(cudaDeviceSynchronize());
    }
    CK(cudaMemcpyFromSymbol(&sb, hc::g_hc, sizeof sb));
    const auto a = A.down(), b = B.down();
    const bool same_img = std::memcmp(a.data(), b.data(), n16 * 2) == 0;
    const bool same_state = std::memcmp(&sa, &sb, sizeof sa) == 0;
    int k_used = 0;
    for (int i = 0; i <= hc::KMAXH; ++i) if (sa.khist[i]) k_used = i;
    char buf[200];
    std::snprintf(buf, sizeof buf, "%s rows %lld K %lld pad %lld kind %d wmax %.1f: shift %d%s%s", F16 ? "F16" : "BF16", (long long) rows, (long long) K, (long long) pad, kind, wmax, k_used,
                  same_img ? "" : "  IMAGE DIFFERS", same_state ? "" : "  STATE DIFFERS");
    what = buf;
    return same_img && same_state;
}

static void run_kernels() {
    std::mt19937_64 rng(12345);
    int cases = 0, bad = 0, shifted = 0;
    for (int f16 = 0; f16 < 2; ++f16)
        for (int64_t rows : {1, 100, 1024})
            for (int64_t K : {10240, 320})
                for (int64_t pad : {0, 8, 3 * 0 + 24})
                    for (int kind = 0; kind <= 6; ++kind)
                        for (float wmax : {1.0f, 8.0f}) {
                            std::string what;
                            const bool ok = f16 ? image_case<true>(rng, rows, K, pad, kind, wmax, what) : image_case<false>(rng, rows, K, pad, kind, wmax, what);
                            ++cases;
                            if (what.find("shift 0") == std::string::npos) ++shifted;
                            if (!ok) { ++bad; std::printf("FAIL %s\n", what.c_str()); }
                        }
    std::printf("fused max + image: %d cases (%d with a shift above 0), %d differ\n", cases, shifted, bad);
    check(bad == 0, "xcvt_kernel + cvt_fix_kernel == xmax_kernel + cvt_kernel (image and counters, bit for bit)");
    check(shifted > 0, "the cases include slices that need a shift");

    // ---- the products' errors against a double reference
    struct Cfg { const char* name; float massive, sw; };
    for (const Cfg cfg : {Cfg{"gaussian channels, a few x3000-massive ones", 3000.0f / 30, 0.02f}, Cfg{"no massive channels", 1.0f, 0.02f}}) {
        const int64_t rows = 1024, K = 10240;
        auto xf = make_x(rng, rows, K, cfg.massive);
        std::vector<uint16_t> xb(xf.size());
        for (size_t i = 0; i < xf.size(); ++i) xb[i] = bf16_of(xf[i]);
        auto wi = make_w(rng, 4, K, cfg.sw), wd = make_w(rng, 320, K, cfg.sw), wu = make_w(rng, 10240, 320, 0.05f);
        std::vector<uint16_t> wib(wi.size()), wdb(wd.size()), wub(wu.size()), wih(wi.size()), wdh(wd.size()), wuh(wu.size());
        float wmax = 0;
        for (size_t i = 0; i < wi.size(); ++i) { wib[i] = bf16_of(wi[i]); wih[i] = f16_of(of_bf16(wib[i])); wmax = std::max(wmax, std::fabs(of_bf16(wib[i]))); }
        for (size_t i = 0; i < wd.size(); ++i) { wdb[i] = bf16_of(wd[i]); wdh[i] = f16_of(of_bf16(wdb[i])); wmax = std::max(wmax, std::fabs(of_bf16(wdb[i]))); }
        float wumax = 0;
        for (size_t i = 0; i < wu.size(); ++i) { wub[i] = bf16_of(wu[i]); wuh[i] = f16_of(of_bf16(wub[i])); wumax = std::max(wumax, std::fabs(of_bf16(wub[i]))); }
        // up's input: silu-like values of K = 320
        auto lof = make_x(rng, 128, 320, 1.0f);
        std::vector<uint16_t> lob(lof.size());
        for (size_t i = 0; i < lof.size(); ++i) { const float v = 0.5f * lof[i]; lob[i] = bf16_of(v / (1.0f + std::exp(-v))); }

        const int64_t n8 = rows * K / 8, K8 = K / 8;
        Dev<uint16_t> X(xb.size()), X16(xb.size()), WI(wih.size()), WD(wdh.size()), LO(lob.size()), LO16(lob.size()), WU(wuh.size());
        Dev<float> YI(rows * 4), YT(rows * 4), YD(rows * 320), YU(128 * 10240), PART(8 * rows * 320);
        X.up(xb); WI.up(wih); WD.up(wdh); LO.up(lob); WU.up(wuh);
        hc::HcState zero{};
        CK(cudaMemcpyToSymbol(hc::g_hc, &zero, sizeof zero));
        void* sp = nullptr;
        CK(cudaGetSymbolAddress(&sp, hc::g_hc));
        const unsigned blocks = (unsigned) std::min<int64_t>(2048, (n8 + 1023) / 1024);
        CK(cudaMemset(sp, 0, 4));
        hc::xcvt_kernel<false><<<blocks, 256>>>(reinterpret_cast<const uint4*>(X.p), reinterpret_cast<uint4*>(X16.p), (unsigned) n8, (unsigned) K8, (unsigned) K8);
        hc::cvt_fix_kernel<false><<<std::min(blocks, 224u), 256>>>(reinterpret_cast<const uint4*>(X.p), reinterpret_cast<uint4*>(X16.p), (unsigned) n8, (unsigned) K8, (unsigned) K8, wmax);
        hc::inj_kernel<<<(unsigned) ((rows + hc::kInjRows - 1) / hc::kInjRows), 256>>>(reinterpret_cast<const uint4*>(X16.p), reinterpret_cast<const uint4*>(WI.p), YI.p, (int) rows, 4, wmax);
        // the GEMM kernel on a 128-wide tile of the same N = 4, split in 4 (what a "padded tile" route would run), and the down product beside it
        const int Z = 4;
        hc::gemm_sk_kernel<<<dim3(1, (unsigned) ((rows + 127) / 128), Z), hf::kThreads>>>(reinterpret_cast<const uint4*>(X16.p), reinterpret_cast<const uint4*>(WI.p), PART.p, (int) rows, 4, (int) K, (int) K8, 4,
                                                                                         rows * 4, Z, wmax);
        hc::reduce_kernel<<<64, 256>>>(PART.p, YT.p, rows, 4, 4, Z, rows * 4);
        hc::gemm_sk_kernel<<<dim3(3, (unsigned) ((rows + 127) / 128), Z), hf::kThreads>>>(reinterpret_cast<const uint4*>(X16.p), reinterpret_cast<const uint4*>(WD.p), PART.p, (int) rows, 320, (int) K, (int) K8, 320,
                                                                                         rows * 320, Z, wmax);
        hc::reduce_kernel<<<256, 256>>>(PART.p, YD.p, rows, 320, 320, Z, rows * 320);
        // the up product: x = lo16 (K 320), its own shift
        const int64_t n8u = 128 * 320 / 8;
        CK(cudaMemset(sp, 0, 4));
        hc::xcvt_kernel<false><<<(unsigned) std::max<int64_t>(1, n8u / 1024), 256>>>(reinterpret_cast<const uint4*>(LO.p), reinterpret_cast<uint4*>(LO16.p), (unsigned) n8u, 40, 40);
        hc::cvt_fix_kernel<false><<<(unsigned) std::max<int64_t>(1, n8u / 1024), 256>>>(reinterpret_cast<const uint4*>(LO.p), reinterpret_cast<uint4*>(LO16.p), (unsigned) n8u, 40, 40, wumax);
        hc::gemm_sk_kernel<<<dim3(80, 1, 1), hf::kThreads>>>(reinterpret_cast<const uint4*>(LO16.p), reinterpret_cast<const uint4*>(WU.p), YU.p, 128, 10240, 320, 40, 10240, 128 * 10240, 1, wumax);
        CK(cudaDeviceSynchronize());
        // (the up product's k is its own; the down / inject ones above come from the x slice's max: read back the counters of the last slice only for the print)
        const auto yi = YI.down(), yt = YT.down(), yd = YD.down(), yu = YU.down();
        const Err ei = err_of(yi, 4, xb, wib, rows, 4, K, 64), et = err_of(yt, 4, xb, wib, rows, 4, K, 64), ed = err_of(yd, 320, xb, wdb, rows, 320, K, 24), eu = err_of(yu, 10240, lob, wub, 128, 10240, 320, 16);
        std::printf("[%s] rel RMS error (max error) vs double:  inject kernel %.3e (%.3e)   inject on a 128-wide tile %.3e (%.3e)   down %.3e (%.3e)   up %.3e (%.3e)\n", cfg.name, ei.rms, ei.mx, et.rms, et.mx, ed.rms,
                    ed.mx, eu.rms, eu.mx);
        check(ei.rms <= eu.rms, "inject kernel: RMS error no worse than the up product's");
        check(ei.mx <= eu.mx, "inject kernel: max error no worse than the up product's");
        check(ei.rms <= et.rms * 1.05, "inject kernel: RMS error no worse than the same N = 4 on the 128-wide GEMM tile");
        // the inject kernel is deterministic: a second launch gives the same bits
        Dev<float> YI2(rows * 4);
        hc::inj_kernel<<<(unsigned) ((rows + hc::kInjRows - 1) / hc::kInjRows), 256>>>(reinterpret_cast<const uint4*>(X16.p), reinterpret_cast<const uint4*>(WI.p), YI2.p, (int) rows, 4, wmax);
        CK(cudaDeviceSynchronize());
        check(std::memcmp(yi.data(), YI2.down().data(), yi.size() * 4) == 0, "inject kernel: two launches, the same bits");
        // a tail: rows not a multiple of the block's 8
        Dev<float> YI3(rows * 4);
        for (int64_t rt : {1, 7, 9, 1023}) {
            CK(cudaMemset(YI3.p, 0, rows * 16));
            hc::inj_kernel<<<(unsigned) ((rt + hc::kInjRows - 1) / hc::kInjRows), 256>>>(reinterpret_cast<const uint4*>(X16.p), reinterpret_cast<const uint4*>(WI.p), YI3.p, (int) rt, 4, wmax);
            CK(cudaDeviceSynchronize());
            const auto y3 = YI3.down();
            bool same = std::memcmp(y3.data(), yi.data(), (size_t) rt * 16) == 0;
            bool rest0 = true;
            for (size_t i = (size_t) rt * 4; i < y3.size(); ++i) rest0 = rest0 && y3[i] == 0.0f;
            char b[80];
            std::snprintf(b, sizeof b, "inject kernel: %lld rows == the first %lld rows of the 1024, nothing written past them", (long long) rt, (long long) rt);
            check(same && rest0, b);
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------------------------------------
static void write_file(const char* path, const std::vector<std::vector<float>>& parts) {
    std::FILE* f = std::fopen(path, "wb");
    if (!f) { std::printf("FAIL cannot write %s\n", path); std::exit(1); }
    for (const auto& p : parts) std::fwrite(p.data(), 4, p.size(), f);
    std::fclose(f);
}

static void run_gemm(const char* path) {
    using strata::prefill::Gemm;
    std::mt19937_64 rng(777);
    const int64_t T = 2300, K = 10240;
    cudaStream_t st;
    CK(cudaStreamCreate(&st));
    Gemm gm;
    std::string err;
    if (!gm.init(st, 1 << 20, err)) { std::printf("FAIL Gemm::init: %s\n", err.c_str()); std::exit(1); }
    auto xf = make_x(rng, T, K, 100.0f);
    auto wd = make_w(rng, 320, K, 0.02f), wi = make_w(rng, 4, K, 0.02f), wu = make_w(rng, 10240, 320, 0.05f);
    auto lof = make_x(rng, T, 320, 1.0f);
    std::vector<uint16_t> xb(xf.size()), wdb(wd.size()), wib(wi.size()), wub(wu.size()), lob(lof.size());
    for (size_t i = 0; i < xf.size(); ++i) xb[i] = bf16_of(xf[i]);
    for (size_t i = 0; i < wd.size(); ++i) wdb[i] = bf16_of(wd[i]);
    for (size_t i = 0; i < wi.size(); ++i) wib[i] = bf16_of(wi[i]);
    for (size_t i = 0; i < wu.size(); ++i) wub[i] = bf16_of(wu[i]);
    for (size_t i = 0; i < lof.size(); ++i) { const float v = 0.5f * lof[i]; lob[i] = bf16_of(v / (1.0f + std::exp(-v))); }
    Dev<uint16_t> X(xb.size()), WD(wdb.size()), WI(wib.size()), WU(wub.size()), LO(lob.size());
    Dev<float> YD(T * 320), YI(T * 4), YU(T * 10240);
    X.up(xb); WD.up(wdb); WI.up(wib); WU.up(wub); LO.up(lob);
    CK(cudaMemset(YD.p, 0, T * 320 * 4));
    CK(cudaMemset(YI.p, 0, T * 4 * 4));
    CK(cudaMemset(YU.p, 0, T * 10240 * 4));
    Gemm::HcCompanion c;
    c.W = WI.p; c.Y = YI.p; c.N = 4; c.ldy = 0; c.resident = true; c.name = "blk.1.hc_attn_inject.weight";
    const bool dn = gm.bf16_hfma(X.p, false, WD.p, YD.p, T, 320, K, 0, 0, true, false, "blk.1.hc_attn_down.weight", &c);
    if (!dn) { std::printf("FAIL the down product was refused (need STRATA_PF_HFMA_HC route conditions: GP100)\n"); std::exit(1); }
    std::printf("inject companion: %s\n", c.done ? "ran with the down product" : "not taken: the old route runs it");
    if (!c.done) gm.bf16(X.p, WI.p, YI.p, T, 4, K, 0, 0.0f, 0, false, true);
    if (!gm.bf16_hfma(LO.p, false, WU.p, YU.p, T, 10240, 320, 0, 0, true, false, "blk.1.hc_attn_up.weight")) { std::printf("FAIL the up product was refused\n"); std::exit(1); }
    CK(cudaStreamSynchronize(st));
    const auto yd = YD.down(), yi = YI.down(), yu = YU.down();
    const Err ed = err_of(yd, 320, xb, wdb, T, 320, K, 16), ei = err_of(yi, 4, xb, wib, T, 4, K, 64), eu = err_of(yu, 10240, lob, wub, T, 10240, 320, 8);
    std::printf("rel RMS error (max error) vs double: down %.3e (%.3e)  inject %.3e (%.3e)  up %.3e (%.3e)\n", ed.rms, ed.mx, ei.rms, ei.mx, eu.rms, eu.mx);
    write_file(path, {yd, yi, yu});
    // the written file is [down | inject | up]; the inject part is compared by error, the other two byte for byte
    std::printf("wrote %s\n", path);
    strata::prefill::gemm_hfma_hc_report(T);
}

static void run_mix(const char* path) {
    std::mt19937_64 rng(4242);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<std::vector<float>> out;
    cudaStream_t st;
    CK(cudaStreamCreate(&st));
    constexpr int64_t N = 2560, D = 10240;
    std::vector<float> wn(D);
    for (auto& v : wn) v = 1.0f + 0.3f * nd(rng);
    Dev<float> W(D);
    W.up(wn);
    for (int64_t T : {1, 7, 1000, 2300}) {
        std::vector<float> R((size_t) (T * D)), rs((size_t) (T * 4)), g((size_t) (T * D));
        for (auto& v : R) v = 2.0f * nd(rng);
        for (auto& v : rs) v = 0.4f + 0.3f * std::fabs(nd(rng));
        for (size_t i = 0; i < g.size(); ++i) g[i] = 4.0f * nd(rng) + (i % 997 == 0 ? 120.0f : 0.0f) - (i % 1009 == 0 ? 130.0f : 0.0f);
        Dev<float> dR(R.size()), dRs(rs.size()), dG(g.size()), dM(T * N);
        Dev<uint16_t> d16(T * N), dH(T * N), dLo(T * N);
        dR.up(R); dRs.up(rs); dG.up(g);
        for (int variant = 0; variant < 4; ++variant) {   // mixed only; + bf16; + bf16 + lo; + fp16 (all)
            CK(cudaMemset(dM.p, 0xEE, T * N * 4));
            CK(cudaMemset(d16.p, 0xEE, T * N * 2));
            CK(cudaMemset(dH.p, 0xEE, T * N * 2));
            CK(cudaMemset(dLo.p, 0xEE, T * N * 2));
            strata::prefill::gr_mix_r(dR.p, dRs.p, W.p, dG.p, dM.p, variant >= 1 ? d16.p : nullptr, T, st, variant == 0 || variant == 3 ? dH.p : nullptr, variant >= 2 ? dLo.p : nullptr);
            CK(cudaStreamSynchronize(st));
            out.push_back(dM.down());
            auto pack = [&](const Dev<uint16_t>& d) {
                const auto h = d.down();
                std::vector<float> f(h.size() / 2 + 1, 0.0f);
                std::memcpy(f.data(), h.data(), h.size() * 2);
                return f;
            };
            out.push_back(pack(d16));
            out.push_back(pack(dH));
            out.push_back(pack(dLo));
        }
    }
    write_file(path, out);
    std::printf("wrote %s\n", path);
}

// timings: the slice kernels of the hyper-connection read at 1024 rows (K 10240), and gr_mix_r at 2048 tokens (run with STRATA_PF_HFMA_SMALL=0 / 1 for the two mix kernels)
static float time_ms(int reps, const std::function<void()>& f) {
    cudaEvent_t a, b;
    CK(cudaEventCreate(&a));
    CK(cudaEventCreate(&b));
    for (int i = 0; i < 3; ++i) f();
    CK(cudaEventRecord(a));
    for (int i = 0; i < reps; ++i) f();
    CK(cudaEventRecord(b));
    CK(cudaEventSynchronize(b));
    float ms = 0;
    CK(cudaEventElapsedTime(&ms, a, b));
    return ms / reps;
}
static void run_bench() {
    std::mt19937_64 rng(99);
    const int64_t rows = 1024, K = 10240, n8 = rows * K / 8, K8 = K / 8;
    auto xf = make_x(rng, rows, K, 1.0f);
    std::vector<uint16_t> xb(xf.size());
    for (size_t i = 0; i < xf.size(); ++i) xb[i] = bf16_of(xf[i]);
    auto wi = make_w(rng, 4, K, 0.02f), wd = make_w(rng, 320, K, 0.02f);
    std::vector<uint16_t> wih(wi.size()), wdh(wd.size());
    for (size_t i = 0; i < wi.size(); ++i) wih[i] = f16_of(of_bf16(bf16_of(wi[i])));
    for (size_t i = 0; i < wd.size(); ++i) wdh[i] = f16_of(of_bf16(bf16_of(wd[i])));
    Dev<uint16_t> X(xb.size()), X16(xb.size()), WI(wih.size()), WD(wdh.size());
    Dev<float> YI(rows * 4), YD(rows * 320), PART(8 * rows * 320);
    X.up(xb); WI.up(wih); WD.up(wdh);
    void* sp = nullptr;
    CK(cudaGetSymbolAddress(&sp, hc::g_hc));
    const unsigned blocks = (unsigned) std::min<int64_t>(2048, (n8 + 1023) / 1024);
    const uint4* xs = reinterpret_cast<const uint4*>(X.p);
    uint4* x16 = reinterpret_cast<uint4*>(X16.p);
    const float wmax = 1.0f;
    std::printf("per 1024-row slice, K 10240 (ms):\n");
    std::printf("  xmax_kernel                         %.4f\n", time_ms(50, [&] { hc::xmax_kernel<false><<<blocks, 256>>>(xs, n8, K8, K8); }));
    std::printf("  cvt_kernel                          %.4f\n", time_ms(50, [&] { hc::cvt_kernel<false><<<blocks, 256>>>(xs, x16, n8, K8, K8, wmax); }));
    std::printf("  xcvt_kernel (max + image)           %.4f\n", time_ms(50, [&] { hc::xcvt_kernel<false><<<blocks, 256>>>(xs, x16, (unsigned) n8, (unsigned) K8, (unsigned) K8); }));
    std::printf("  cvt_fix_kernel (shift 0: counts)    %.4f\n", time_ms(50, [&] { hc::cvt_fix_kernel<false><<<std::min(blocks, 224u), 256>>>(xs, x16, (unsigned) n8, (unsigned) K8, (unsigned) K8, wmax); }));
    std::printf("  inj_kernel (N 4)                    %.4f\n", time_ms(50, [&] { hc::inj_kernel<<<(unsigned) (rows / hc::kInjRows), 256>>>(reinterpret_cast<const uint4*>(X16.p), reinterpret_cast<const uint4*>(WI.p), YI.p, (int) rows, 4, wmax); }));
    std::printf("  gemm_sk_kernel, N 4 on a tile, Z 4  %.4f (+ reduce)\n", time_ms(30, [&] { hc::gemm_sk_kernel<<<dim3(1, 8, 4), hf::kThreads>>>(reinterpret_cast<const uint4*>(X16.p), reinterpret_cast<const uint4*>(WI.p), PART.p, (int) rows, 4, (int) K, (int) K8, 4, rows * 4, 4, wmax); }));
    std::printf("  gemm_sk_kernel, N 320 (down), Z 4   %.4f (+ reduce; the compiler kernel; the hand-scheduled one is ~20%% faster)\n", time_ms(30, [&] { hc::gemm_sk_kernel<<<dim3(3, 8, 4), hf::kThreads>>>(reinterpret_cast<const uint4*>(X16.p), reinterpret_cast<const uint4*>(WD.p), PART.p, (int) rows, 320, (int) K, (int) K8, 320, rows * 320, 4, wmax); }));
    // gr_mix_r
    const int64_t T = 2048, N = 2560, D = 10240;
    Dev<float> dR(T * D), dRs(T * 4), dG(T * D), dM(T * N), dW(D);
    Dev<uint16_t> d16(T * N), dH(T * N);
    CK(cudaMemset(dR.p, 0, T * D * 4)); CK(cudaMemset(dRs.p, 0, T * 16)); CK(cudaMemset(dG.p, 0, T * D * 4)); CK(cudaMemset(dW.p, 0, D * 4));
    std::printf("gr_mix_r at %lld tokens (STRATA_PF_HFMA_SMALL=%s): %.4f ms, %.0f GB/s\n", (long long) T, std::getenv("STRATA_PF_HFMA_SMALL") ? std::getenv("STRATA_PF_HFMA_SMALL") : "unset",
                time_ms(50, [&] { strata::prefill::gr_mix_r(dR.p, dRs.p, dW.p, dG.p, dM.p, d16.p, T, nullptr, dH.p, nullptr); }),
                (T * (D * 4.0 * 2 + N * 4.0 + N * 4.0 + 16)) / 1e9 / (time_ms(50, [&] { strata::prefill::gr_mix_r(dR.p, dRs.p, dW.p, dG.p, dM.p, d16.p, T, nullptr, dH.p, nullptr); }) / 1e3));
}

int main(int argc, char** argv) {
    int dev = 0;
    cudaDeviceProp p;
    if (cudaGetDevice(&dev) != cudaSuccess || cudaGetDeviceProperties(&p, dev) != cudaSuccess || p.major != 6 || p.minor != 0) {
        std::printf("not a GP100 (sm_60): skipped\n");
        return 77;
    }
    std::printf("device %d: %s\n", dev, p.name);
    if (argc >= 2 && std::strcmp(argv[1], "--kernels") == 0) run_kernels();
    else if (argc >= 2 && std::strcmp(argv[1], "--bench") == 0) run_bench();
    else if (argc >= 3 && std::strcmp(argv[1], "--gemm") == 0) run_gemm(argv[2]);
    else if (argc >= 3 && std::strcmp(argv[1], "--mix") == 0) run_mix(argv[2]);
    else { std::printf("usage: hfma_small_parity --kernels | --bench | --gemm <file> | --mix <file>\n"); return 2; }
    if (g_fail) { std::printf("%d check(s) FAILED\n", g_fail); return 1; }
    return 0;
}
