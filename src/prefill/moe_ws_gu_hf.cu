// src/prefill/moe_ws_gu_hf.cu - STRATA_PF_HFMA_EXP (moe_ws.hpp): the host side of moe_ws_hf.cuh's gate/up kernels - the
// switch, the per-layer decision (format, tier, why not), the launches, and the once-per-read report with what ran, what
// fell back and why, the range of the input and the weights' shifts.  Opt-in; with the switch off nothing here is called.
#include "moe_ws_hf.cuh"
#include "moe_ws_hf_r2.cuh"
#include "strata/core/expert_nibble.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#if STRATA_EXPERT_NIBBLE_BUILT
namespace strata::prefill::ws {
namespace {

namespace nb = strata::core::nibble;

// tokens a pass by rows an expert u (moe_ws_hf.cuh's microbenchmark, lognormal counts: tier 16 wins to u ~ 15, tier 32 from
// ~ 20; below ~ 7 rows an expert the int8 8-token kernels are faster)
constexpr double kU16 = 7.0, kU32 = 20.0;
constexpr double kMacsPerRow = 1280.0 * 2560.0;

// STRATA_PF_HFMA_R2's tier rule: the two-rows-a-thread kernel (moe_ws_hf_r2.cuh) against the compiler kernel's 16- and 32-token tiers,
// measured on one PH402 die (kf-work, 512 experts, lognormal counts, chunks largest first, rows an expert u = 7 .. 90, all four
// formats).  The 32-token tier of the new kernel wins from u 11 for IQ2_S / IQ2_XS / IQ2_XXS / IQ3_XXS (u 11: +0.6 .. +1.6 %, u 14:
// +5.6 %, u 20: +11 %, u 90: +16 % on the best compiler tier) and from u 9 for IQ3_S (u 9: +6 %, u 20: +19 %, u 90: +23 %); below that
// the compiler kernel's 16-token tier is faster for the first four (-1 .. -4 % for the new one at tier 16 or 32) and the new kernel's
// 16-token tier for IQ3_S in u 7 .. 8 (+4 %, the compiler kernel's 16-token tier being the other choice).  0 = the compiler kernel.
int r2_pick(const int fam, const double u) {
    if (fam == 3) return u < 8.5 ? 16 : 32;
    return u < 11.0 ? 0 : 32;
}

struct Dev {
    bool seen = false;
    int64_t taken[32][2] = {};        // [type & 31][tier 16 / 32] launches
    uint64_t layers[32] = {};         // [type & 31] bit (layer & 63)
    double macs_taken[32] = {}, macs_left[32] = {};
    int64_t left_n[32] = {};
    std::map<std::string, int64_t> left;   // "FMT: reason" -> launches
    int64_t r2_taken[32][2] = {};     // STRATA_PF_HFMA_R2: [type & 31][tier 16 / 32] launches of the two-rows kernel
    double r2_macs[32] = {};          // ... their MACs
    int64_t r2_left_n[32] = {};       // launches of a taken layer that stayed on the compiler kernel (r2_pick said 0)
    double r2_macs_left[32] = {};
    std::map<std::string, int64_t> r2_left;   // "FMT: reason" -> launches
};
std::mutex g_mu;
Dev g_dev[64];
void* g_state_ptr[64] = {};

// STRATA_PREFILL_TIMING: an event pair round each launch (per device, reused), summed by format and tier in the report - the
// kernels' own time and so their real G MAC/s
struct Tm {
    cudaEvent_t a = nullptr, b = nullptr;
    int type = 0, tier = 0, r2 = 0;
    double macs = 0.0;
};
std::vector<Tm> g_tm[64];
size_t g_tm_n[64] = {};
bool timing_on() {
    static const bool on = std::getenv("STRATA_PREFILL_TIMING") != nullptr;
    return on;
}

int cur_dev() {
    int d = 0;
    if (cudaGetDevice(&d) != cudaSuccess || d < 0 || d >= 64) { cudaGetLastError(); return 0; }
    return d;
}
void* state_ptr(int dev) {
    std::lock_guard<std::mutex> lk(g_mu);
    if (g_state_ptr[dev] == nullptr) {
        void* p = nullptr;
        if (cudaGetSymbolAddress(&p, hf::g_hf) != cudaSuccess) {
            std::fprintf(stderr, "prefill ws: STRATA_PF_HFMA_EXP: the device state symbol is missing on GPU %d\n", dev);
            std::exit(1);
        }
        g_state_ptr[dev] = p;
    }
    return g_state_ptr[dev];
}
void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "prefill ws: %s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}
const char* fname(int t) { return nb::fmt_name(t); }
int popcount64(uint64_t v) {
    int n = 0;
    for (; v != 0; v &= v - 1) ++n;
    return n;
}

void note_left(int dev, int gu_type, int64_t rows, const char* why) {
    std::lock_guard<std::mutex> lk(g_mu);
    Dev& d = g_dev[dev];
    d.seen = true;
    d.macs_left[gu_type & 31] += (double) rows * kMacsPerRow;
    ++d.left_n[gu_type & 31];
    ++d.left[std::string(fname(gu_type)) + ": " + why];
}

}  // namespace

namespace {
// STRATA_PF_HFMA_R2=1 (opt-in, read once, with STRATA_PF_HFMA_EXP): the 4-bit gate/up runs moe_ws_hf_r2.cuh's two-rows-a-thread kernel
// at the tier r2_pick() gives; where it says 0 the compiler kernel runs, counted with its reason.
int hfma_r2() {
    static const int on = [] {
        const char* e = std::getenv("STRATA_PF_HFMA_R2");
        if (e == nullptr || std::atoi(e) == 0) return 0;
        const char* x = std::getenv("STRATA_PF_HFMA_EXP");
        if (x == nullptr || std::atoi(x) == 0) {
            std::fprintf(stderr, "strata prefill: STRATA_PF_HFMA_R2=1 needs STRATA_PF_HFMA_EXP=1: ignored\n");
            return 0;
        }
        std::fprintf(stderr, "strata prefill: STRATA_PF_HFMA_R2=1: the 4-bit gate/up of the prompt on the two-rows-a-thread FP16 HFMA2 kernel (the same "
                             "FP16 sums of 64 products widened to FP32 every 128 values; IQ3_S decoded by shift + LOP3 + one HFMA2, its activations "
                             "re-paired; not bitwise the compiler kernel's for IQ3_S, bitwise for the other formats)\n");
        return 1;
    }();
    return on;
}
}  // namespace

int hfma_exp() {
    static const int on = [] {
        const char* e = std::getenv("STRATA_PF_HFMA_EXP");
        if (e == nullptr || std::atoi(e) == 0) return 0;
        if (mode() == 0) {
            std::fprintf(stderr, "strata prefill: STRATA_PF_HFMA_EXP=1 needs STRATA_MMQ_WS=1 on compute capability 6.0: ignored\n");
            return 0;
        }
        std::fprintf(stderr, "strata prefill: STRATA_PF_HFMA_EXP=1: the 4-bit gate/up of the prompt on FP16 HFMA2 (FP16 input rows, weights "
                             "decoded in the kernel, FP16 sums of 64 products widened to FP32 every 128 values, the weights' shift "
                             "picked per launch from the layer's max |x|; not bitwise the int8 kernels' GU)\n");
        return 1;
    }();
    return on;
}

bool hfma_exp_plan(Layer& L) {
    L.hf_tier = 0;
    const int dev = cur_dev();
    const double rows = (double) L.rows, u = L.n > 0 ? rows / (double) L.n : 0.0;
    const char* why = nullptr;
    if (!L.gu_nib) why = "GGUF-block slot (not 4-bit codes)";
    else if (hf::family(L.gu_type) < 0) why = "no FP16 kernel for the format";
    else if (L.hf_xh == nullptr || L.hf_src == nullptr || ((uintptr_t) L.hf_xh & 15) != 0 || L.hf_tokens <= 0)
        why = "FP16 input missing or misaligned";
    else if (u < kU16) why = "u below 7 rows an expert (the int8 8-token tier is faster)";
    if (why != nullptr) {
        note_left(dev, L.gu_type, L.rows, why);
        return false;
    }
    L.hf_tier = u >= kU32 ? 32 : 16;
    L.hf_r2 = 0;
    bool r2_stay = false;
    if (hfma_r2() != 0) {
        const int pick = r2_pick(hf::family(L.gu_type), u);
        if (pick != 0) { L.hf_r2 = 1; L.hf_tier = pick; }
        else r2_stay = true;
    }
    std::lock_guard<std::mutex> lk(g_mu);
    Dev& d = g_dev[dev];
    d.seen = true;
    ++d.taken[L.gu_type & 31][L.hf_tier == 32 ? 1 : 0];
    d.macs_taken[L.gu_type & 31] += rows * kMacsPerRow;
    if (L.hf_r2 != 0) {
        ++d.r2_taken[L.gu_type & 31][L.hf_tier == 32 ? 1 : 0];
        d.r2_macs[L.gu_type & 31] += rows * kMacsPerRow;
    } else if (r2_stay) {
        ++d.r2_left_n[L.gu_type & 31];
        d.r2_macs_left[L.gu_type & 31] += rows * kMacsPerRow;
        ++d.r2_left[std::string(fname(L.gu_type)) + ": u below the two-rows kernel's range, the compiler kernel's 16-token tier is faster there"];
    }
    if (L.hf_layer >= 0) d.layers[L.gu_type & 31] |= 1ull << (L.hf_layer & 63);
    return true;
}

void hfma_exp_left(int gu_type, bool nibble, int64_t rows, const char* why) {
    (void) nibble;
    note_left(cur_dev(), gu_type, rows, why);
}

void hfma_exp_prepass(const Layer& L, void* stream) {
    const cudaStream_t s = (cudaStream_t) stream;
    void* st = state_ptr(cur_dev());
    ck(cudaMemsetAsync(st, 0, sizeof(unsigned), s), "hfma_exp: memset");   // HfState::cur is its first member
    const int64_t n16 = L.hf_tokens * hf::XROW16;
    const unsigned blocks = (unsigned) std::max<int64_t>(1, std::min<int64_t>(2048, (n16 + 1023) / 1024));
    hf::xmax_kernel<<<blocks, 256, 0, s>>>((const uint4*) L.hf_xh, n16);
    ck(cudaGetLastError(), "hfma_exp: max-reduce");
}

namespace detail {
void gu_hf(const Layer& L, cudaStream_t s) {
    if (L.nchunks <= 0) return;
    static const bool said = [] {
        std::fprintf(stderr, "strata prefill: STRATA_PF_HFMA_EXP: a 4-bit layer's gate/up runs the FP16 HFMA2 kernels\n");
        return true;
    }();
    (void) said;
    const unsigned grid = (unsigned) (L.nchunks * hf::PIECES);
    const uint4* xh = (const uint4*) L.hf_xh;
    const char* w = (const char*) L.w;
    cudaEvent_t ev_b = nullptr;
    if (timing_on()) {
        const int dev = cur_dev();
        cudaEvent_t ev_a = nullptr;
        {
            std::lock_guard<std::mutex> lk(g_mu);
            auto& v = g_tm[dev];
            if (g_tm_n[dev] == v.size()) {
                Tm t;
                ck(cudaEventCreate(&t.a), "hfma_exp: event");
                ck(cudaEventCreate(&t.b), "hfma_exp: event");
                v.push_back(t);
            }
            Tm& t = v[g_tm_n[dev]++];
            t.type = L.gu_type;
            t.tier = L.hf_tier;
            t.r2 = L.hf_r2;
            t.macs = (double) L.rows * kMacsPerRow;
            ev_a = t.a;
            ev_b = t.b;
        }
        cudaEventRecord(ev_a, s);
    }
#define STRATA_HF_LAUNCH(TPP)                                                                                          \
    switch (hf::family(L.gu_type)) {                                                                                   \
        case 0: hf::gu_kernel<TPP, 0><<<grid, hf::NT, 0, s>>>(L.chunks, w, L.xoff, xh, L.hf_src, L.gu, L.hf_layer); break; \
        case 1: hf::gu_kernel<TPP, 1><<<grid, hf::NT, 0, s>>>(L.chunks, w, L.xoff, xh, L.hf_src, L.gu, L.hf_layer); break; \
        case 2: hf::gu_kernel<TPP, 2><<<grid, hf::NT, 0, s>>>(L.chunks, w, L.xoff, xh, L.hf_src, L.gu, L.hf_layer); break; \
        case 3: hf::gu_kernel<TPP, 3><<<grid, hf::NT, 0, s>>>(L.chunks, w, L.xoff, xh, L.hf_src, L.gu, L.hf_layer); break; \
        default: std::fprintf(stderr, "prefill ws: 4-bit gate/up type %d has no FP16 kernel\n", L.gu_type); std::exit(1); \
    }
#define STRATA_HF_LAUNCH_R2(TPP)                                                                                       \
    switch (hf::family(L.gu_type)) {                                                                                   \
        case 0: hf::gu2_kernel<TPP, 0><<<grid2, hf::R2_NT, 0, s>>>(L.chunks, w, L.xoff, xh, L.hf_src, L.gu, L.hf_layer); break; \
        case 1: hf::gu2_kernel<TPP, 1><<<grid2, hf::R2_NT, 0, s>>>(L.chunks, w, L.xoff, xh, L.hf_src, L.gu, L.hf_layer); break; \
        case 2: hf::gu2_kernel<TPP, 2><<<grid2, hf::R2_NT, 0, s>>>(L.chunks, w, L.xoff, xh, L.hf_src, L.gu, L.hf_layer); break; \
        case 3: hf::gu2_kernel<TPP, 3><<<grid2, hf::R2_NT, 0, s>>>(L.chunks, w, L.xoff, xh, L.hf_src, L.gu, L.hf_layer); break; \
        default: std::fprintf(stderr, "prefill ws: 4-bit gate/up type %d has no FP16 kernel\n", L.gu_type); std::exit(1); \
    }
    if (L.hf_r2 != 0) {   // STRATA_PF_HFMA_R2: 256 rows a CTA, five CTAs a chunk
        const unsigned grid2 = (unsigned) (L.nchunks * hf::R2_PIECES);
        if (L.hf_tier == 32) { STRATA_HF_LAUNCH_R2(32) }
        else { STRATA_HF_LAUNCH_R2(16) }
    } else if (L.hf_tier == 32) { STRATA_HF_LAUNCH(32) }
    else { STRATA_HF_LAUNCH(16) }
#undef STRATA_HF_LAUNCH_R2
#undef STRATA_HF_LAUNCH
    ck(cudaGetLastError(), "gate/up 4-bit FP16");
    if (ev_b != nullptr) cudaEventRecord(ev_b, s);
}
}  // namespace detail

void hfma_exp_report(int64_t tokens) {
    if (hfma_exp() == 0) return;
    std::lock_guard<std::mutex> lk(g_mu);
    const int back = cur_dev();
    for (int dev = 0; dev < 64; ++dev) {
        Dev& d = g_dev[dev];
        if (!d.seen) continue;
        hf::HfState st{};
        const bool have_state = g_state_ptr[dev] != nullptr;
        if (have_state) {
            cudaSetDevice(dev);
            if (cudaMemcpyFromSymbol(&st, hf::g_hf, sizeof st) != cudaSuccess) { cudaGetLastError(); std::memset(&st, 0, sizeof st); }
            hf::HfState zero{};
            if (cudaMemcpyToSymbol(hf::g_hf, &zero, sizeof zero) != cudaSuccess) cudaGetLastError();
        }
        // the kernels' own time by format and tier (STRATA_PREFILL_TIMING)
        std::string kt;
        if (g_tm_n[dev] > 0) {
            struct Acc { double ms = 0.0, macs = 0.0; int64_t n = 0; };
            std::map<std::pair<int, int>, Acc> acc;   // (type, tier + 100 for the two-rows kernel)
            double ms_all = 0.0, macs_all = 0.0;
            cudaSetDevice(dev);
            for (size_t i = 0; i < g_tm_n[dev]; ++i) {
                float ms = 0.0f;
                if (cudaEventSynchronize(g_tm[dev][i].b) == cudaSuccess && cudaEventElapsedTime(&ms, g_tm[dev][i].a, g_tm[dev][i].b) == cudaSuccess) {
                    Acc& a = acc[{g_tm[dev][i].type, g_tm[dev][i].tier + (g_tm[dev][i].r2 != 0 ? 100 : 0)}];
                    a.ms += ms; a.macs += g_tm[dev][i].macs; ++a.n;
                    ms_all += ms; macs_all += g_tm[dev][i].macs;
                } else {
                    cudaGetLastError();
                }
            }
            g_tm_n[dev] = 0;
            for (const auto& [key, a] : acc) {
                char b[160];
                std::snprintf(b, sizeof b, "%s%s t%d%s x%lld %.1f ms %.0f GMAC/s", kt.empty() ? "" : "; ", fname(key.first), key.second % 100,
                              key.second >= 100 ? "r2" : "", (long long) a.n, a.ms, a.ms > 0 ? a.macs / a.ms * 1e-6 : 0.0);
                kt += b;
            }
            char b[120];
            std::snprintf(b, sizeof b, "; all %.1f ms %.0f GMAC/s", ms_all, ms_all > 0 ? macs_all / ms_all * 1e-6 : 0.0);
            kt += b;
        }
        int64_t n_taken = 0, n_left = 0;
        double m_taken = 0.0, m_left = 0.0;
        std::string by;
        for (int t = 0; t < 32; ++t) {
            const int64_t tk = d.taken[t][0] + d.taken[t][1];
            n_taken += tk;
            n_left += d.left_n[t];
            m_taken += d.macs_taken[t];
            m_left += d.macs_left[t];
            if (tk == 0 && d.left_n[t] == 0) continue;
            char b[200];
            std::snprintf(b, sizeof b, "%s%s %lld launches on HFMA2 (%d layers; tier 16 x%lld, tier 32 x%lld; %.1f%% of its %lld launches' MACs)",
                          by.empty() ? "" : "; ", fname(t), (long long) tk, popcount64(d.layers[t]),(long long) d.taken[t][0],
                          (long long) d.taken[t][1], d.macs_taken[t] + d.macs_left[t] > 0 ? 100.0 * d.macs_taken[t] / (d.macs_taken[t] + d.macs_left[t]) : 0.0,
                          (long long) (tk + d.left_n[t]));
            by += b;
        }
        std::string lft;
        for (const auto& [what, n] : d.left) lft += (lft.empty() ? "" : ", ") + what + " x" + std::to_string((long long) n);
        std::string r2s;   // STRATA_PF_HFMA_R2: what ran on the two-rows kernel, what stayed on the compiler kernel and why
        if (hfma_r2() != 0) {
            int64_t r2_n = 0, r2_l = 0;
            double r2_m = 0.0, r2_lm = 0.0;
            std::string rb;
            for (int t = 0; t < 32; ++t) {
                const int64_t tk = d.r2_taken[t][0] + d.r2_taken[t][1];
                r2_n += tk; r2_l += d.r2_left_n[t]; r2_m += d.r2_macs[t]; r2_lm += d.r2_macs_left[t];
                if (tk == 0 && d.r2_left_n[t] == 0) continue;
                char b[200];
                std::snprintf(b, sizeof b, "%s%s %lld (tier 16 x%lld, tier 32 x%lld)", rb.empty() ? "" : "; ", fname(t), (long long) tk, (long long) d.r2_taken[t][0],
                              (long long) d.r2_taken[t][1]);
                rb += b;
            }
            std::string rl;
            for (const auto& [what, n] : d.r2_left) rl += (rl.empty() ? "" : ", ") + what + " x" + std::to_string((long long) n);
            char b[300];
            std::snprintf(b, sizeof b, "; STRATA_PF_HFMA_R2: %lld launches on the two-rows kernel (%.1f%% of the HFMA2 MACs), %lld stayed on the compiler kernel",
                          (long long) r2_n, r2_m + r2_lm > 0 ? 100.0 * r2_m / (r2_m + r2_lm) : 0.0, (long long) r2_l);
            r2s = b;
            r2s += "; by format: " + (rb.empty() ? std::string("-") : rb) + "; left: " + (rl.empty() ? std::string("nothing") : rl);
        }
        auto h2f = [](unsigned bits) { return __half2float(__ushort_as_half((unsigned short) bits)); };
        std::string lm;
        for (int l = 0; l < 64; ++l)
            if (st.layer_max[l] != 0) {
                char b[40];
                std::snprintf(b, sizeof b, "%sL%d %.1f", lm.empty() ? "" : ", ", l, h2f(st.layer_max[l]));
                lm += b;
            }
        std::string kh;
        for (int i = 0; i <= hf::KMAX - hf::KMIN; ++i)
            if (st.khist[i] != 0) {
                char b[40];
                std::snprintf(b, sizeof b, "%s%d x%u", kh.empty() ? "" : ", ", i + hf::KMIN, st.khist[i]);
                kh += b;
            }
        std::fprintf(stderr, "strata prefill: STRATA_PF_HFMA_EXP: read of %lld tokens, CUDA%d: gate/up %lld launches on HFMA2, %lld left to the int8 kernels "
                             "(%.1f%% of %.3g MACs on HFMA2); by format: %s; left: %s; input max |x| %.2f (per layer: %s); weights' shift k "
                             "(2^-k) per launch: %s; non-finite: input in %u launches, output %u values; kernel time: %s%s\n",
                     (long long) tokens, dev, (long long) n_taken, (long long) n_left,
                     m_taken + m_left > 0 ? 100.0 * m_taken / (m_taken + m_left) : 0.0, m_taken + m_left, by.empty() ? "-" : by.c_str(),
                     lft.empty() ? "nothing" : lft.c_str(), h2f(st.all_max), lm.empty() ? "-" : lm.c_str(), kh.empty() ? "-" : kh.c_str(), st.bad_x,
                     st.bad_y, kt.empty() ? "-" : kt.c_str(), r2s.c_str());
        d = Dev{};
    }
    cudaSetDevice(back);
}

}  // namespace strata::prefill::ws
#else  // STRATA_EXPERT_NIBBLE_BUILT is 0 (a build without the 4-bit kernels): the switch does not exist
namespace strata::prefill::ws {
int hfma_exp() { return 0; }
bool hfma_exp_plan(Layer& L) { L.hf_tier = 0; return false; }
void hfma_exp_left(int, bool, int64_t, const char*) {}
void hfma_exp_prepass(const Layer&, void*) {}
void hfma_exp_report(int64_t) {}
}  // namespace strata::prefill::ws
#endif
