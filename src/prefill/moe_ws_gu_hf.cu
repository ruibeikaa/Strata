// src/prefill/moe_ws_gu_hf.cu - STRATA_PF_HFMA_EXP (moe_ws.hpp): the host side of moe_ws_hf.cuh's gate/up kernels - the
// switch, the per-layer decision (format, tier, why not), the launches, and the once-per-read report with what ran, what
// fell back and why, the range of the input and the weights' shifts.  Opt-in; with the switch off nothing here is called.
#include "moe_ws_hf.cuh"
#include "moe_ws_hf_r2.cuh"
#include "moe_ws_hf_cls.cuh"
#include "strata/core/expert_nibble.hpp"
#include "strata/core/pinned_ring.hpp"

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
    int64_t cls_taken[32] = {};       // STRATA_PF_HFMA_CLASSES: [type & 31] launches of the pass-class kernels
    double cls_macs[32] = {};         // ... their MACs
    int64_t cls_items[4] = {};        // ... their passes by class (32 rows, 25..31, 17..24, 1..16)
    int64_t cls_left_n[32] = {};      // launches that took the compiler kernel instead (a reason below, planned or at launch)
    double cls_macs_left[32] = {};
    std::map<std::string, int64_t> cls_left;   // "FMT: reason" -> launches
};
std::mutex g_mu;
Dev g_dev[64];
void* g_state_ptr[64] = {};

// STRATA_PREFILL_TIMING: an event pair round each launch (per device, reused), summed by format and tier in the report - the
// kernels' own time and so their real G MAC/s
struct Tm {
    cudaEvent_t a = nullptr, b = nullptr;
    int type = 0, tier = 0, r2 = 0;   // r2: 0 the compiler kernel, 1 the R2 kernel, 2 the pass-class kernels
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

namespace {
// STRATA_PF_HFMA_CLASSES=1 (opt-in, read once, with STRATA_PF_HFMA_EXP): the 4-bit gate/up of the prompt runs moe_ws_hf_cls.cuh's pass-class kernels
// (the host splits the layer's chunk list into passes of 32 / 25..31 / 17..24 / 1..16 rows, each class on its own kernel and stream) wherever
// r2_pick() says the two-rows kernel would have run (the same tier rule); it replaces the R2 kernel there.  Where it cannot run (no host chunk list,
// no device buffer, a launch error) the compiler kernel runs, counted with its reason.
int hfma_cls() {
    static const int on = [] {
        const char* e = std::getenv("STRATA_PF_HFMA_CLASSES");
        if (e == nullptr || std::atoi(e) == 0) return 0;
        const char* x = std::getenv("STRATA_PF_HFMA_EXP");
        if (x == nullptr || std::atoi(x) == 0) {
            std::fprintf(stderr, "strata prefill: STRATA_PF_HFMA_CLASSES=1 needs STRATA_PF_HFMA_EXP=1: ignored\n");
            return 0;
        }
        std::fprintf(stderr, "strata prefill: STRATA_PF_HFMA_CLASSES=1: the 4-bit gate/up of the prompt on the pass-class FP16 HFMA2 kernels (passes of 32 / 25..31 / "
                             "17..24 / 1..16 rows on direct-code, direct-code, 24-token and staged-code kernels; the same FP16 sums of 64 products widened to FP32 "
                             "every 128 values; bitwise the compiler kernel's for IQ2_S, IQ2_XS, IQ2_XXS and IQ3_XXS, the R2 kernel's for IQ3_S)\n");
        return 1;
    }();
    return on;
}

// per device: the layer's pass items on the device (uploaded by the prepass, on the compute stream, so a layer's lists are rewritten only after the
// previous layer's kernels), the side streams and the fork / join events
struct Cls {
    std::vector<Chunk> host;          // this layer's items, [full | 25..31 | 17..24 | 1..16], from plan to prepass
    Chunk* d = nullptr;
    size_t d_cap = 0;
    strata::core::PinnedRing ring;
    cudaStream_t st[3] = {nullptr, nullptr, nullptr};
    cudaEvent_t e0 = nullptr, ej[3] = {nullptr, nullptr, nullptr};
    bool ready = false;               // the lists are on the device (the prepass's upload succeeded)
    bool init = false;
};
Cls* g_cls[64] = {};
Cls& cls_dev(int dev) {
    std::lock_guard<std::mutex> lk(g_mu);
    if (g_cls[dev] == nullptr) g_cls[dev] = new Cls();   // never freed: the process exit owns the device state
    return *g_cls[dev];
}

// the plan: split the layer's chunks into pass items by class; nullptr = taken, else why not
const char* cls_plan(Layer& L) {
    if (L.chunks_host == nullptr || L.nchunks <= 0) return "no host copy of the chunk list";
    Cls& S = cls_dev(cur_dev());
    static thread_local std::vector<Chunk> v[4];
    for (auto& x : v) x.clear();
    for (int64_t ci = 0; ci < L.nchunks; ++ci) {
        const Chunk& c = L.chunks_host[ci];
        for (int p0 = 0; p0 < c.rows; p0 += 32) {
            const int nt = std::min(32, c.rows - p0);
            v[nt == 32 ? 0 : nt > hf::W_MID_MIN ? 1 : nt > hf::W_SMALL_MAX ? 2 : 3].push_back(Chunk{c.j, c.row0 + p0, nt, 0});
        }
    }
    const auto bigger = [](const Chunk& a, const Chunk& b) { return a.rows != b.rows ? a.rows > b.rows : a.row0 < b.row0; };
    for (int c = 1; c < 4; ++c) std::sort(v[c].begin(), v[c].end(), bigger);   // the big remainders first, as the full passes are in chunk order
    S.host.clear();
    for (int c = 0; c < 4; ++c) {
        L.hf_cls_n[c] = (int) v[c].size();
        S.host.insert(S.host.end(), v[c].begin(), v[c].end());
    }
    if (S.host.empty()) return "no passes";
    if (S.d_cap < S.host.size()) {   // grow-only; cudaFree waits for the device, so this is rare (4096 items = 64 KiB at first)
        const size_t cap = std::max<size_t>(S.host.size() + S.host.size() / 2, 4096);
        if (S.d != nullptr) cudaFree(S.d);
        S.d = nullptr;
        S.d_cap = 0;
        if (cudaMalloc((void**) &S.d, cap * sizeof(Chunk)) != cudaSuccess) {
            cudaGetLastError();
            S.d = nullptr;
            return "no device buffer for the pass lists";
        }
        S.d_cap = cap;
    }
    return nullptr;
}

// the prepass: the lists up, on the compute stream
void cls_upload(const Layer& L, cudaStream_t s) {
    Cls& S = cls_dev(cur_dev());
    S.ready = false;
    size_t n = 0;
    for (int c = 0; c < 4; ++c) n += (size_t) L.hf_cls_n[c];
    if (n == 0 || n != S.host.size() || S.d == nullptr || n > S.d_cap) return;
    S.ready = S.ring.h2d(S.d, S.host.data(), n * sizeof(Chunk), s) == cudaSuccess;
}

bool cls_init(Cls& S) {
    if (S.init) return true;
    for (int i = 0; i < 3; ++i)
        if (cudaStreamCreateWithFlags(&S.st[i], cudaStreamNonBlocking) != cudaSuccess || cudaEventCreateWithFlags(&S.ej[i], cudaEventDisableTiming) != cudaSuccess) {
            cudaGetLastError();
            return false;
        }
    if (cudaEventCreateWithFlags(&S.e0, cudaEventDisableTiming) != cudaSuccess) { cudaGetLastError(); return false; }
    S.init = true;
    return true;
}

// the four class kernels of a layer: full passes on the compute stream, the others on side streams that fork from it and join back to it
template <int FAM>
const char* cls_launch_fam(const Layer& L, Cls& S, const char* w, const uint4* xh, cudaStream_t s) {
    const int nf = L.hf_cls_n[0], nm = L.hf_cls_n[1], nm2 = L.hf_cls_n[2], ns = L.hf_cls_n[3];
    const Chunk* pf = S.d;
    const Chunk* pm = pf + nf;
    const Chunk* pm2 = pm + nm;
    const Chunk* ps = pm2 + nm2;
    cudaEventRecord(S.e0, s);
    for (int i = 0; i < 3; ++i) cudaStreamWaitEvent(S.st[i], S.e0, 0);
    int note = 1;   // the launch is counted by ONE kernel of the layer
    if (nf > 0) { hf::gd_kernel<32, FAM, 2, true><<<(unsigned) nf * hf::W_PIECES, hf::W_NT, 0, s>>>(pf, w, L.xoff, xh, L.hf_src, L.gu, L.hf_layer, note); note = 0; }
    if (nm > 0) { hf::gd_kernel<32, FAM, 2, false><<<(unsigned) nm * hf::W_PIECES, hf::W_NT, 0, S.st[0]>>>(pm, w, L.xoff, xh, L.hf_src, L.gu, L.hf_layer, note); note = 0; }
    if (nm2 > 0) { hf::gd_kernel<24, FAM, 2, false><<<(unsigned) nm2 * hf::W_PIECES, hf::W_NT, 0, S.st[1]>>>(pm2, w, L.xoff, xh, L.hf_src, L.gu, L.hf_layer, note); note = 0; }
    if (ns > 0) { hf::gs_kernel<16, FAM><<<(unsigned) ns * hf::W_PIECES, hf::W_NT, 0, S.st[2]>>>(ps, w, L.xoff, xh, L.hf_src, L.gu, L.hf_layer, note); note = 0; }
    for (int i = 0; i < 3; ++i) {
        cudaEventRecord(S.ej[i], S.st[i]);
        cudaStreamWaitEvent(s, S.ej[i], 0);
    }
    return cudaGetLastError() == cudaSuccess ? nullptr : "a class kernel's launch failed";
}

// nullptr: launched; else why it was not (nothing, or all but a failed part, is enqueued: the caller runs the compiler kernel after the join)
const char* cls_launch(const Layer& L, const char* w, const uint4* xh, cudaStream_t s) {
    Cls& S = cls_dev(cur_dev());
    if (!S.ready) return "the pass lists' upload failed";
    if (!cls_init(S)) return "the side streams could not be created";
    switch (hf::family(L.gu_type)) {
        case 0: return cls_launch_fam<0>(L, S, w, xh, s);
        case 1: return cls_launch_fam<1>(L, S, w, xh, s);
        case 2: return cls_launch_fam<2>(L, S, w, xh, s);
        case 3: return cls_launch_fam<3>(L, S, w, xh, s);
        default: return "no class kernel for the format";
    }
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
    L.hf_cls = 0;
    bool r2_stay = false;
    const char* cls_why = nullptr;
    if (hfma_cls() != 0 || hfma_r2() != 0) {
        const int pick = r2_pick(hf::family(L.gu_type), u);
        if (pick == 0) {
            r2_stay = true;
        } else if (hfma_cls() != 0) {   // the pass-class kernels replace the R2 kernel where it would have run; the compiler kernel is their counted fallback
            cls_why = cls_plan(L);
            if (cls_why == nullptr) { L.hf_cls = 1; L.hf_tier = pick; }
        } else {
            L.hf_r2 = 1;
            L.hf_tier = pick;
        }
    }
    std::lock_guard<std::mutex> lk(g_mu);
    Dev& d = g_dev[dev];
    d.seen = true;
    ++d.taken[L.gu_type & 31][L.hf_tier == 32 ? 1 : 0];
    d.macs_taken[L.gu_type & 31] += rows * kMacsPerRow;
    if (hfma_cls() != 0) {
        if (L.hf_cls != 0) {
            ++d.cls_taken[L.gu_type & 31];
            d.cls_macs[L.gu_type & 31] += rows * kMacsPerRow;
            for (int c = 0; c < 4; ++c) d.cls_items[c] += L.hf_cls_n[c];
        } else {
            ++d.cls_left_n[L.gu_type & 31];
            d.cls_macs_left[L.gu_type & 31] += rows * kMacsPerRow;
            ++d.cls_left[std::string(fname(L.gu_type)) + ": " +
                         (cls_why != nullptr ? cls_why : "u below the pass-class kernels' range, the compiler kernel's 16-token tier is faster there")];
        }
    } else if (L.hf_r2 != 0) {
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
    if (L.hf_cls != 0) cls_upload(L, s);   // STRATA_PF_HFMA_CLASSES: this layer's pass lists up on the compute stream, behind the previous layer's kernels
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
    size_t tm_i = 0;
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
            tm_i = g_tm_n[dev];
            Tm& t = v[g_tm_n[dev]++];
            t.type = L.gu_type;
            t.tier = L.hf_tier;
            t.r2 = L.hf_cls != 0 ? 2 : L.hf_r2;
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
    const char* cls_why = nullptr;
    if (L.hf_cls != 0) cls_why = cls_launch(L, w, xh, s);   // STRATA_PF_HFMA_CLASSES: four class kernels; nullptr = launched
    if (cls_why != nullptr) {   // the compiler kernel, at the tier its own rule gives (everything the class kernels wrote is rewritten behind them on this stream)
        const int dev = cur_dev();
        const double macs = (double) L.rows * kMacsPerRow;
        const int ftier = (double) L.rows >= kU32 * (double) L.n ? 32 : 16;
        {
            std::lock_guard<std::mutex> lk(g_mu);
            Dev& d = g_dev[dev];
            --d.cls_taken[L.gu_type & 31];
            d.cls_macs[L.gu_type & 31] -= macs;
            for (int c = 0; c < 4; ++c) d.cls_items[c] -= L.hf_cls_n[c];
            ++d.cls_left_n[L.gu_type & 31];
            d.cls_macs_left[L.gu_type & 31] += macs;
            ++d.cls_left[std::string(fname(L.gu_type)) + ": " + cls_why + " (at launch)"];
            if (ev_b != nullptr) { g_tm[dev][tm_i].r2 = 0; g_tm[dev][tm_i].tier = ftier; }
        }
        if (ftier == 32) { STRATA_HF_LAUNCH(32) }
        else { STRATA_HF_LAUNCH(16) }
    } else if (L.hf_cls != 0) {
        // launched above
    } else if (L.hf_r2 != 0) {   // STRATA_PF_HFMA_R2: 256 rows a CTA, five CTAs a chunk
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
                    Acc& a = acc[{g_tm[dev][i].type, g_tm[dev][i].tier + 100 * g_tm[dev][i].r2}];
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
                              key.second >= 200 ? "cls" : key.second >= 100 ? "r2" : "", (long long) a.n, a.ms, a.ms > 0 ? a.macs / a.ms * 1e-6 : 0.0);
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
        std::string cls_s;   // STRATA_PF_HFMA_CLASSES: what ran on the pass-class kernels, what stayed on the compiler kernel and why
        if (hfma_cls() != 0) {
            int64_t c_n = 0, c_l = 0;
            double c_m = 0.0, c_lm = 0.0;
            std::string cb;
            for (int t = 0; t < 32; ++t) {
                c_n += d.cls_taken[t]; c_l += d.cls_left_n[t]; c_m += d.cls_macs[t]; c_lm += d.cls_macs_left[t];
                if (d.cls_taken[t] == 0 && d.cls_left_n[t] == 0) continue;
                char b[160];
                std::snprintf(b, sizeof b, "%s%s %lld on the class kernels, %lld on the compiler kernel", cb.empty() ? "" : "; ", fname(t), (long long) d.cls_taken[t],
                              (long long) d.cls_left_n[t]);
                cb += b;
            }
            std::string cl;
            for (const auto& [what, n] : d.cls_left) cl += (cl.empty() ? "" : ", ") + what + " x" + std::to_string((long long) n);
            char b[400];
            std::snprintf(b, sizeof b, "; STRATA_PF_HFMA_CLASSES: %lld launches on the class kernels (%.1f%% of the HFMA2 MACs; passes: 32 rows x%lld, 25..31 x%lld, 17..24 x%lld, 1..16 x%lld), %lld on the compiler kernel",
                          (long long) c_n, c_m + c_lm > 0 ? 100.0 * c_m / (c_m + c_lm) : 0.0, (long long) d.cls_items[0], (long long) d.cls_items[1], (long long) d.cls_items[2],
                          (long long) d.cls_items[3], (long long) c_l);
            cls_s = b;
            cls_s += "; by format: " + (cb.empty() ? std::string("-") : cb) + "; left: " + (cl.empty() ? std::string("nothing") : cl);
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
                             "(2^-k) per launch: %s; non-finite: input in %u launches, output %u values; kernel time: %s%s%s\n",
                     (long long) tokens, dev, (long long) n_taken, (long long) n_left,
                     m_taken + m_left > 0 ? 100.0 * m_taken / (m_taken + m_left) : 0.0, m_taken + m_left, by.empty() ? "-" : by.c_str(),
                     lft.empty() ? "nothing" : lft.c_str(), h2f(st.all_max), lm.empty() ? "-" : lm.c_str(), kh.empty() ? "-" : kh.c_str(), st.bad_x,
                     st.bad_y, kt.empty() ? "-" : kt.c_str(), r2s.c_str(), cls_s.c_str());
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
