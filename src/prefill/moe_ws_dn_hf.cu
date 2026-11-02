// src/prefill/moe_ws_dn_hf.cu - STRATA_PF_HFMA_DOWN (moe_ws.hpp): the host side of moe_ws_dn_hf.cuh's kernels - the switch,
// the per-layer decision (format, tier, why not), the launches, and the once-per-read report with what ran, what fell back
// and why, the range of H and the weights' shifts.  Opt-in; with the switch off nothing here is called.
#include "common.cuh"

#include "moe_ws_dn_hf.cuh"

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

namespace strata::prefill::ws {
namespace {

// tokens a pass by rows an expert u: the 16-token kernels from 7 rows (the int8 kernels have 8-token tiers below that).  The
// 32-token kernels are built and tested but not picked: on a GP100 die (dn_test.exe, 128 experts of 16 / 32 / 64 rows) they
// run at 1.6-2.2 TMAC/s against the 16-token kernels' 2.7-3.2 (168 registers a thread and a few spills at 32 tokens).
constexpr double kU16 = 7.0, kU32 = 1e30;
constexpr double kMacsPerRow = (double) dn::DN_ROWS * dn::DN_K;

int fmt_idx(const int d_type) { return d_type == GGML_TYPE_Q2_0 ? 0 : d_type == GGML_TYPE_IQ4_NL ? 1 : 2; }
std::string fmt_name(const int d_type) {
    return d_type == GGML_TYPE_Q2_0 ? "Q2_0" : d_type == GGML_TYPE_IQ4_NL ? "IQ4_NL" : "GGML type " + std::to_string(d_type);
}

struct Dev {
    bool seen = false;
    int64_t taken[3][2] = {};          // [format][tier 16 / 32] launches
    uint64_t layers[3] = {};           // [format] bit (layer & 63)
    double macs_taken[3] = {}, macs_left[3] = {};
    int64_t left_n[3] = {};
    std::map<std::string, int64_t> left;   // "FMT: reason" -> launches
};
std::mutex g_mu;
Dev g_dev[64];
void* g_state_ptr[64] = {};

// STRATA_PREFILL_TIMING: an event triple round each layer's swiglu + down (per device, reused), summed by format and tier in
// the report - the kernels' own time and so their real G MAC/s
struct Tm {
    cudaEvent_t a = nullptr, b = nullptr, c = nullptr;
    int fmt = 0, tier = 0;
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
void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "prefill ws: %s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}
int popcount64(uint64_t v) {
    int n = 0;
    for (; v != 0; v &= v - 1) ++n;
    return n;
}
void note_left(const int dev, const int d_type, const int64_t rows, const char* why) {
    std::lock_guard<std::mutex> lk(g_mu);
    Dev& d = g_dev[dev];
    d.seen = true;
    const int f = fmt_idx(d_type);
    d.macs_left[f] += (double) rows * kMacsPerRow;
    ++d.left_n[f];
    ++d.left[fmt_name(d_type) + ": " + why];
}

// the constant table the IQ4_NL kernel looks codes up in against ggml's, once
__global__ void kv_check2_kernel(int* __restrict__ bad) {
    if (threadIdx.x < 16 && dn::kKv[threadIdx.x] != kvalues_iq4nl[threadIdx.x]) atomicAdd(bad, 1);
}
void kv_check() {
    int* d_bad = nullptr;
    ck(cudaMalloc((void**) &d_bad, sizeof(int)), "STRATA_PF_HFMA_DOWN: kv check");
    ck(cudaMemset(d_bad, 0, sizeof(int)), "STRATA_PF_HFMA_DOWN: kv check");
    kv_check2_kernel<<<1, 32>>>(d_bad);
    int bad = 0;
    ck(cudaMemcpy(&bad, d_bad, sizeof(int), cudaMemcpyDeviceToHost), "STRATA_PF_HFMA_DOWN: kv check");
    cudaFree(d_bad);
    if (bad != 0) {
        std::fprintf(stderr, "prefill ws: STRATA_PF_HFMA_DOWN: the IQ4_NL value table differs from ggml's (%d entries)\n", bad);
        std::exit(1);
    }
}
void* state_ptr(const int dev) {
    std::lock_guard<std::mutex> lk(g_mu);
    if (g_state_ptr[dev] == nullptr) {
        void* p = nullptr;
        if (cudaGetSymbolAddress(&p, dn::g_dn) != cudaSuccess) {
            std::fprintf(stderr, "prefill ws: STRATA_PF_HFMA_DOWN: the device state symbol is missing on GPU %d\n", dev);
            std::exit(1);
        }
        g_state_ptr[dev] = p;
    }
    return g_state_ptr[dev];
}

}  // namespace

int hfma_down() {
    static const int on = [] {
        const char* e = std::getenv("STRATA_PF_HFMA_DOWN");
        if (e == nullptr || std::atoi(e) == 0) return 0;
        if (mode() == 0) {
            std::fprintf(stderr, "strata prefill: STRATA_PF_HFMA_DOWN=1 needs STRATA_MMQ_WS=1 on compute capability 6.0: ignored\n");
            return 0;
        }
        kv_check();
        std::fprintf(stderr, "strata prefill: STRATA_PF_HFMA_DOWN=1: the down of a weight-stationary layer (Q2_0, IQ4_NL) on FP16 HFMA2 (H as FP16 "
                             "rows scaled by a power of two a row, the weights times their block scale as FP16 shifted per row and step, FP16 "
                             "sums of 64 products widened to FP32 every 128 values; not bitwise the int8 kernels' Dm)\n");
        return 1;
    }();
    return on;
}

bool hfma_down_plan(Layer& L) {
    L.hd_tier = 0;
    const int dev = cur_dev();
    const double rows = (double) L.rows, u = L.n > 0 ? rows / (double) L.n : 0.0;
    const int f = fmt_idx(L.d_type);
    const char* why = nullptr;
    if (f > 1) why = "no FP16 kernel for the format";
    else if (L.gu == nullptr || ((uintptr_t) L.gu & 15) != 0 || L.rows <= 0) why = "the GU rows are missing or misaligned";
    else if (u < kU16) why = "u below 7 rows an expert (the int8 8-token tier is faster)";
    if (why != nullptr) {
        note_left(dev, L.d_type, L.rows, why);
        return false;
    }
    L.hd_tier = u >= kU32 ? 32 : 16;
    std::lock_guard<std::mutex> lk(g_mu);
    Dev& d = g_dev[dev];
    d.seen = true;
    ++d.taken[f][L.hd_tier == 32 ? 1 : 0];
    d.macs_taken[f] += rows * kMacsPerRow;
    if (L.hd_layer >= 0) d.layers[f] |= 1ull << (L.hd_layer & 63);
    return true;
}

void hfma_down_left(const int d_type, const int64_t rows, const char* why) { note_left(cur_dev(), d_type, rows, why); }

void down_hf_prep(const Layer& L, void* stream) {
    if (L.rows <= 0) return;
    const cudaStream_t s = (cudaStream_t) stream;
    (void) state_ptr(cur_dev());
    if (timing_on()) {
        const int dev = cur_dev();
        std::lock_guard<std::mutex> lk(g_mu);
        auto& v = g_tm[dev];
        if (g_tm_n[dev] == v.size()) {
            Tm t;
            ck(cudaEventCreate(&t.a), "hfma_down: event");
            ck(cudaEventCreate(&t.b), "hfma_down: event");
            ck(cudaEventCreate(&t.c), "hfma_down: event");
            v.push_back(t);
        }
        Tm& t = v[g_tm_n[dev]++];
        t.fmt = fmt_idx(L.d_type);
        t.tier = L.hd_tier;
        t.macs = (double) L.rows * kMacsPerRow;
        cudaEventRecord(t.a, s);
    }
    dn::swiglu_h16_kernel<<<(unsigned) L.rows, 160, 0, s>>>(L.gu, L.hd_layer);
    ck(cudaGetLastError(), "down FP16: swiglu");
    if (timing_on()) {
        const int dev = cur_dev();
        std::lock_guard<std::mutex> lk(g_mu);
        cudaEventRecord(g_tm[dev][g_tm_n[dev] - 1].b, s);
    }
}

void down_hf(const Layer& L, void* stream) {
    if (L.nchunks <= 0) return;
    const cudaStream_t s = (cudaStream_t) stream;
    static const bool said = [] {
        std::fprintf(stderr, "strata prefill: STRATA_PF_HFMA_DOWN: a weight-stationary layer's down runs the FP16 HFMA2 kernels\n");
        return true;
    }();
    (void) said;
    const unsigned grid = (unsigned) (L.nchunks * dn::PIECES);
    const uint4* hb = (const uint4*) L.gu;
    const char* w = (const char*) L.w;
    if (L.d_type == GGML_TYPE_Q2_0) {
        if (L.hd_tier == 32) dn::q20::dn_kernel<32><<<grid, dn::NT, 0, s>>>(L.chunks, w, L.xoff, L.down_off, hb, L.dm);
        else dn::q20::dn_kernel<16><<<grid, dn::NT, 0, s>>>(L.chunks, w, L.xoff, L.down_off, hb, L.dm);
    } else if (L.d_type == GGML_TYPE_IQ4_NL) {
        if (L.hd_tier == 32) dn::nl4::dn_kernel<32><<<grid, dn::NT, 0, s>>>(L.chunks, w, L.xoff, L.down_off, hb, L.dm);
        else dn::nl4::dn_kernel<16><<<grid, dn::NT, 0, s>>>(L.chunks, w, L.xoff, L.down_off, hb, L.dm);
    } else {
        std::fprintf(stderr, "prefill ws: down type %d has no FP16 kernel\n", L.d_type);
        std::exit(1);
    }
    ck(cudaGetLastError(), "down FP16");
    if (timing_on()) {
        const int dev = cur_dev();
        std::lock_guard<std::mutex> lk(g_mu);
        if (g_tm_n[dev] > 0) cudaEventRecord(g_tm[dev][g_tm_n[dev] - 1].c, s);
    }
}

void hfma_down_report(const int64_t tokens) {
    if (hfma_down() == 0) return;
    std::lock_guard<std::mutex> lk(g_mu);
    const int back = cur_dev();
    for (int dev = 0; dev < 64; ++dev) {
        Dev& d = g_dev[dev];
        if (!d.seen) continue;
        dn::DnState st{};
        if (g_state_ptr[dev] != nullptr) {
            cudaSetDevice(dev);
            if (cudaMemcpyFromSymbol(&st, dn::g_dn, sizeof st) != cudaSuccess) { cudaGetLastError(); std::memset(&st, 0, sizeof st); }
            dn::DnState zero{};
            if (cudaMemcpyToSymbol(dn::g_dn, &zero, sizeof zero) != cudaSuccess) cudaGetLastError();
        }
        // the kernels' own time by format and tier (STRATA_PREFILL_TIMING): swiglu (prep) and down
        std::string kt;
        if (g_tm_n[dev] > 0) {
            struct Acc { double prep = 0.0, dn = 0.0, macs = 0.0; int64_t n = 0; };
            std::map<std::pair<int, int>, Acc> acc;
            Acc all;
            cudaSetDevice(dev);
            for (size_t i = 0; i < g_tm_n[dev]; ++i) {
                float p = 0.0f, q = 0.0f;
                if (cudaEventSynchronize(g_tm[dev][i].c) == cudaSuccess && cudaEventElapsedTime(&p, g_tm[dev][i].a, g_tm[dev][i].b) == cudaSuccess &&
                    cudaEventElapsedTime(&q, g_tm[dev][i].b, g_tm[dev][i].c) == cudaSuccess) {
                    Acc& a = acc[{g_tm[dev][i].fmt, g_tm[dev][i].tier}];
                    a.prep += p; a.dn += q; a.macs += g_tm[dev][i].macs; ++a.n;
                    all.prep += p; all.dn += q; all.macs += g_tm[dev][i].macs; ++all.n;
                } else {
                    cudaGetLastError();
                }
            }
            g_tm_n[dev] = 0;
            auto line = [&](const std::string& name, const Acc& a) {
                char b[220];
                std::snprintf(b, sizeof b, "%s%s x%lld swiglu %.1f ms + down %.1f ms: %.0f GMAC/s down alone, %.0f with the swiglu", kt.empty() ? "" : "; ",
                              name.c_str(), (long long) a.n, a.prep, a.dn, a.dn > 0 ? a.macs / a.dn * 1e-6 : 0.0,
                              a.prep + a.dn > 0 ? a.macs / (a.prep + a.dn) * 1e-6 : 0.0);
                kt += b;
            };
            for (const auto& [key, a] : acc) line(std::string(key.first == 0 ? "Q2_0" : "IQ4_NL") + " t" + std::to_string(key.second), a);
            line("all", all);
        }
        int64_t n_taken = 0, n_left = 0;
        double m_taken = 0.0, m_left = 0.0;
        std::string by;
        const int types[3] = {GGML_TYPE_Q2_0, GGML_TYPE_IQ4_NL, -1};
        for (int f = 0; f < 3; ++f) {
            const int64_t tk = d.taken[f][0] + d.taken[f][1];
            n_taken += tk;
            n_left += d.left_n[f];
            m_taken += d.macs_taken[f];
            m_left += d.macs_left[f];
            if (tk == 0 && d.left_n[f] == 0) continue;
            char b[220];
            std::snprintf(b, sizeof b, "%s%s %lld launches on HFMA2 (%d layers; tier 16 x%lld, tier 32 x%lld; %.1f%% of its %lld launches' MACs)",
                          by.empty() ? "" : "; ", f == 2 ? "other" : fmt_name(types[f]).c_str(), (long long) tk, popcount64(d.layers[f]),
                          (long long) d.taken[f][0], (long long) d.taken[f][1],
                          d.macs_taken[f] + d.macs_left[f] > 0 ? 100.0 * d.macs_taken[f] / (d.macs_taken[f] + d.macs_left[f]) : 0.0,
                          (long long) (tk + d.left_n[f]));
            by += b;
        }
        std::string lft;
        for (const auto& [what, n] : d.left) lft += (lft.empty() ? "" : ", ") + what + " x" + std::to_string((long long) n);
        auto f32 = [](unsigned bits) { float f; std::memcpy(&f, &bits, 4); return f; };
        std::string lm;
        for (int l = 0; l < 64; ++l)
            if (st.layer_max[l] != 0) {
                char b[40];
                std::snprintf(b, sizeof b, "%sL%d %.1f", lm.empty() ? "" : ", ", l, f32(st.layer_max[l]));
                lm += b;
            }
        std::string kh;
        for (int i = 0; i <= dn::KMAX - dn::KMIN; ++i)
            if (st.khist[i] != 0) {
                char b[40];
                std::snprintf(b, sizeof b, "%s%d x%u", kh.empty() ? "" : ", ", i + dn::KMIN, st.khist[i]);
                kh += b;
            }
        std::fprintf(stderr, "strata prefill: STRATA_PF_HFMA_DOWN: read of %lld tokens, CUDA%d: down %lld launches on HFMA2, %lld left to the int8 kernels "
                             "(%.1f%% of %.3g MACs on HFMA2); by format: %s; left: %s; max |H| %.1f (per layer: %s); weights' shift k "
                             "(2^-k), one sample a CTA: %s; non-finite: H %u values, output %u values; kernel time: %s\n",
                     (long long) tokens, dev, (long long) n_taken, (long long) n_left,
                     m_taken + m_left > 0 ? 100.0 * m_taken / (m_taken + m_left) : 0.0, m_taken + m_left, by.empty() ? "-" : by.c_str(),
                     lft.empty() ? "nothing" : lft.c_str(), f32(st.hmax_all), lm.empty() ? "-" : lm.c_str(), kh.empty() ? "-" : kh.c_str(),
                     st.bad_h, st.bad_y, kt.empty() ? "-" : kt.c_str());
        d = Dev{};
    }
    cudaSetDevice(back);
}

}  // namespace strata::prefill::ws
