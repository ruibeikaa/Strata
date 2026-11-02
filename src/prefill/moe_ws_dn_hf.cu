// src/prefill/moe_ws_dn_hf.cu - STRATA_PF_HFMA_DOWN (moe_ws.hpp): the host side of moe_ws_dn_hf.cuh's kernels - the switch,
// the per-layer decision (format, tier, why not), the launches, and the once-per-read report with what ran, what fell back
// and why, the range of H and the weights' shifts.  Opt-in; with the switch off nothing here is called.
#include "common.cuh"

#include "moe_ws_dn_hf.cuh"
#include "moe_ws_dn_hf_r2.cuh"
#include "moe_ws_dn_hf_cls.cuh"
#include "strata/core/pinned_ring.hpp"

#include <algorithm>
#include <chrono>
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

// STRATA_PF_HFMA_R2_DOWN's rule: the two-rows-a-thread kernels (moe_ws_dn_hf_r2.cuh) against the engine's 16-token kernels, measured on one PH402 die (kf-work\kfdown,
// 512 experts, lognormal counts, chunks largest first, rows an expert u = 7 .. 90).  Q2_0: the 16-token two-rows kernel is faster at every u from 7 (+8 % at u 7, +10.5 %
// at u 20, +12 % from u 45).  IQ4_NL: the 32-token two-rows kernel is slower below u 11 (-5 % at u 7, -4 % at u 9), level at u 11 and faster from u 14 (+5 %; u 20:
// +12.5 %, u 45: +20.7 %, u 90: +22.5 %), so the engine's kernel keeps u < 12.  Returns the tier label (16 / 32) of the two-rows kernel, 0 = the engine kernel.
int r2_pick(const int fmt, const double u) {
    if (fmt == 0) return 16;
    return u >= 12.0 ? 32 : 0;
}

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
    int64_t r2_taken[3] = {};          // STRATA_PF_HFMA_R2_DOWN: [format] launches of the two-rows kernel
    double r2_macs[3] = {};
    int64_t r2_left_n[3] = {};         // launches of a taken layer that stayed on the engine kernel (r2_pick said 0)
    double r2_macs_left[3] = {};
    std::map<std::string, int64_t> r2_left;   // "FMT: reason" -> launches
    int64_t cls_taken[3] = {};         // STRATA_PF_HFMA_DOWN_CLASSES: [format] launches of the pass-class kernels
    double cls_macs[3] = {};           // ... their MACs
    int64_t cls_items[3] = {};         // ... their passes by class (32 rows, 17..31, 1..16)
    int64_t cls_left_n[3] = {};        // launches that took the one-row kernel instead (a reason below, planned or at launch)
    double cls_macs_left[3] = {};
    std::map<std::string, int64_t> cls_left;   // "FMT: reason" -> launches
};
std::mutex g_mu;
Dev g_dev[64];
void* g_state_ptr[64] = {};

// STRATA_PREFILL_TIMING: an event triple round each layer's swiglu + down (per device, reused), summed by format and tier in
// the report - the kernels' own time and so their real G MAC/s
struct Tm {
    cudaEvent_t a = nullptr, b = nullptr, c = nullptr;
    cudaEvent_t u = nullptr;                                   // after the pass lists' upload (before the swiglu)
    cudaEvent_t cs0[3] = {nullptr, nullptr, nullptr};          // the pass-class streams: after the fork wait / after the last class kernel
    cudaEvent_t cs1[3] = {nullptr, nullptr, nullptr};
    int fmt = 0, tier = 0, r2 = 0;   // r2: 0 the one-row kernel, 1 the two-rows kernel, 2 the pass-class kernels
    double macs = 0.0;
    bool cls_timed = false;          // the class marks above were recorded for this launch
};
// STRATA_PREFILL_TIMING: the host's side around the down's pass classes, per device (summed over a read, reset by the report)
struct HostT {
    double plan_ms = 0.0, up_ms = 0.0, prep_ms = 0.0, launch_ms = 0.0;
    int64_t n_plan = 0, n_up = 0, n_prep = 0, n_launch = 0, idle_prep = 0, idle_launch = 0;
};
HostT g_ht[64];
double host_now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
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

namespace {
// STRATA_PF_HFMA_R2_DOWN=1 (opt-in, read once, with STRATA_PF_HFMA_DOWN): the down runs moe_ws_dn_hf_r2.cuh's two-rows-a-thread kernels at the tier r2_pick() gives;
// where it says 0 the engine's kernel runs, counted with its reason.
int hfma_r2_down() {
    static const int on = [] {
        const char* e = std::getenv("STRATA_PF_HFMA_R2_DOWN");
        if (e == nullptr || std::atoi(e) == 0) return 0;
        const char* x = std::getenv("STRATA_PF_HFMA_DOWN");
        if (x == nullptr || std::atoi(x) == 0) {
            std::fprintf(stderr, "strata prefill: STRATA_PF_HFMA_R2_DOWN=1 needs STRATA_PF_HFMA_DOWN=1: ignored\n");
            return 0;
        }
        std::fprintf(stderr, "strata prefill: STRATA_PF_HFMA_R2_DOWN=1: the FP16 HFMA2 down on the two-rows-a-thread kernels (Q2_0 at 16 tokens a pass, IQ4_NL at 32 "
                             "from u 12; the same arithmetic and bits as the one-row kernels)\n");
        return 1;
    }();
    return on;
}
}  // namespace

namespace {
// STRATA_PF_HFMA_DOWN_CLASSES=1 (opt-in, read once, with STRATA_PF_HFMA_DOWN): the down splits the layer's passes into size classes (32 rows, 17..31, 1..16) on
// moe_ws_dn_hf_cls.cuh's kernels, wherever r2_pick() says the two-rows kernel would have run (the same tier rule); it replaces the R2 kernel there.  Where it cannot run (no host chunk
// list, no device buffer, a failed upload or launch) the one-row kernel runs, counted with its reason.
int hfma_cls_down() {
    static const int on = [] {
        const char* e = std::getenv("STRATA_PF_HFMA_DOWN_CLASSES");
        if (e == nullptr || std::atoi(e) == 0) return 0;
        const char* x = std::getenv("STRATA_PF_HFMA_DOWN");
        if (x == nullptr || std::atoi(x) == 0) {
            std::fprintf(stderr, "strata prefill: STRATA_PF_HFMA_DOWN_CLASSES=1 needs STRATA_PF_HFMA_DOWN=1: ignored\n");
            return 0;
        }
        std::fprintf(stderr, "strata prefill: STRATA_PF_HFMA_DOWN_CLASSES=1: the FP16 HFMA2 down on pass-class kernels (the host splits each layer's passes into 32 / 17..31 / "
                             "1..16 rows, each class on its own kernel and stream; the same arithmetic and bits as the one-row and two-rows kernels)\n");
        return 1;
    }();
    return on;
}

// STRATA_PF_HFMA_DOWN_UP2=1 (opt-in, read once, with STRATA_PF_HFMA_DOWN_CLASSES): the pass lists go up from hfma_down_plan on their own stream, into one of two alternating device
// buffers, and only the class streams wait for them - the compute stream no longer carries the copy between the gate/up and the swiglu.  The buffer a layer fills is the one the layer
// before the previous one read: the upload waits for the compute stream's mark behind that layer's join.  The same bytes in the same kernels: the outputs are bit-identical.
int hfma_dn_up2() {
    static const int on = [] {
        const char* e = std::getenv("STRATA_PF_HFMA_DOWN_UP2");
        if (e == nullptr || std::atoi(e) == 0) return 0;
        if (hfma_cls_down() == 0) {
            std::fprintf(stderr, "strata prefill: STRATA_PF_HFMA_DOWN_UP2=1 needs STRATA_PF_HFMA_DOWN_CLASSES=1: ignored\n");
            return 0;
        }
        std::fprintf(stderr, "strata prefill: STRATA_PF_HFMA_DOWN_UP2=1: the down's pass lists upload on their own stream into two alternating device buffers, from the plan "
                             "(behind the gate/up), and only the class streams wait for them (the same kernels and bytes)\n");
        return 1;
    }();
    return on;
}

// per device: the layer's pass items on the device (uploaded by down_hf_prep on the compute stream, so a layer's lists are rewritten only after the previous layer's kernels;
// with STRATA_PF_HFMA_DOWN_UP2 by the plan on up_st into the buffer par of two),
// the side streams and the fork / join events
struct Cls {
    std::vector<Chunk> host;          // this layer's items, [full | 17..31 | 1..16], from plan to prep
    Chunk* d = nullptr;               // UP2: two buffers of d_cap items, back to back
    size_t d_cap = 0;
    strata::core::PinnedRing ring;
    cudaStream_t st[3] = {nullptr, nullptr, nullptr};
    cudaEvent_t e0 = nullptr, ej[3] = {nullptr, nullptr, nullptr};
    bool ready = false;               // the lists are on the device (the upload succeeded)
    bool init = false;
    cudaStream_t up_st = nullptr;     // UP2: the upload stream
    cudaEvent_t eu[2] = {nullptr, nullptr};   // UP2: buffer p holds this layer's lists (recorded on up_st)
    cudaEvent_t ef[2] = {nullptr, nullptr};   // UP2: buffer p is free (recorded on the compute stream behind the join of the layer that read it)
    int par = 0, next = 0;            // UP2: the buffer of the layer in flight / of the next one
    Chunk* cur = nullptr;             // the lists of the layer in flight on the device (d, or d + par * d_cap)
    bool up_init = false;
};
Cls* g_cls[64] = {};
Cls& cls_dev(int dev) {
    std::lock_guard<std::mutex> lk(g_mu);
    if (g_cls[dev] == nullptr) g_cls[dev] = new Cls();   // never freed: the process exit owns the device state
    return *g_cls[dev];
}

// the plan: split the layer's chunks into pass items by class; nullptr = taken, else why not
const char* cls_plan(Layer& L) {
    L.hd_cls_n[0] = L.hd_cls_n[1] = L.hd_cls_n[2] = 0;
    if (L.chunks_host == nullptr || L.nchunks <= 0) return "no host copy of the chunk list";
    Cls& S = cls_dev(cur_dev());
    static thread_local std::vector<Chunk> v[3];
    for (auto& x : v) x.clear();
    for (int64_t ci = 0; ci < L.nchunks; ++ci) {
        const Chunk& c = L.chunks_host[ci];
        for (int p0 = 0; p0 < c.rows; p0 += 32) {
            const int nt = std::min(32, c.rows - p0);
            v[nt == 32 ? 0 : nt > 16 ? 1 : 2].push_back(Chunk{c.j, c.row0 + p0, nt, p0 == 0 ? 1 : 0});   // pad: the first pass of its chunk (one khist sample a chunk piece)
        }
    }
    const auto bigger = [](const Chunk& a, const Chunk& b) { return a.rows != b.rows ? a.rows > b.rows : a.row0 < b.row0; };
    for (int c = 1; c < 3; ++c) std::sort(v[c].begin(), v[c].end(), bigger);   // the big remainders first, as the full passes are in chunk order
    S.host.clear();
    for (int c = 0; c < 3; ++c) {
        L.hd_cls_n[c] = (int) v[c].size();
        S.host.insert(S.host.end(), v[c].begin(), v[c].end());
    }
    if (S.host.empty()) return "no passes";
    if (S.d_cap < S.host.size()) {   // grow-only; cudaFree waits for the device, so this is rare (4096 items = 64 KiB at first)
        const size_t cap = std::max<size_t>(S.host.size() + S.host.size() / 2, 4096);
        if (S.d != nullptr) cudaFree(S.d);
        S.d = nullptr;
        S.d_cap = 0;
        if (cudaMalloc((void**) &S.d, cap * sizeof(Chunk) * (hfma_dn_up2() != 0 ? 2 : 1)) != cudaSuccess) {
            cudaGetLastError();
            S.d = nullptr;
            return "no device buffer for the pass lists";
        }
        S.d_cap = cap;
    }
    return nullptr;
}

// the lists up, on the compute stream (down_hf_prep)
void cls_upload(const Layer& L, cudaStream_t s) {
    Cls& S = cls_dev(cur_dev());
    S.ready = false;
    const size_t n = (size_t) L.hd_cls_n[0] + (size_t) L.hd_cls_n[1] + (size_t) L.hd_cls_n[2];
    if (n == 0 || n != S.host.size() || S.d == nullptr || n > S.d_cap) return;
    const double t0 = timing_on() ? host_now_ms() : 0.0;
    S.ready = S.ring.h2d(S.d, S.host.data(), n * sizeof(Chunk), s) == cudaSuccess;
    S.cur = S.d;
    if (timing_on()) {
        std::lock_guard<std::mutex> lk(g_mu);
        HostT& h = g_ht[cur_dev()];
        h.up_ms += host_now_ms() - t0;
        ++h.n_up;
    }
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

bool cls_up_init(Cls& S) {
    if (S.up_init) return true;
    if (cudaStreamCreateWithFlags(&S.up_st, cudaStreamNonBlocking) != cudaSuccess) { cudaGetLastError(); return false; }
    for (int i = 0; i < 2; ++i)
        if (cudaEventCreateWithFlags(&S.eu[i], cudaEventDisableTiming) != cudaSuccess || cudaEventCreateWithFlags(&S.ef[i], cudaEventDisableTiming) != cudaSuccess) {
            cudaGetLastError();
            return false;
        }
    S.up_init = true;
    return true;
}

// STRATA_PF_HFMA_DOWN_UP2: this layer's lists up on up_st (called by the plan, behind the gate/up already enqueued), into the buffer the layer before the previous one read
void cls_upload_ahead(const Layer& L) {
    Cls& S = cls_dev(cur_dev());
    S.ready = false;
    const size_t n = (size_t) L.hd_cls_n[0] + (size_t) L.hd_cls_n[1] + (size_t) L.hd_cls_n[2];
    if (n == 0 || n != S.host.size() || S.d == nullptr || n > S.d_cap) return;
    if (!cls_init(S) || !cls_up_init(S)) return;
    const double t0 = timing_on() ? host_now_ms() : 0.0;
    const int p = S.next;
    Chunk* dst = S.d + (size_t) p * S.d_cap;
    if (cudaStreamWaitEvent(S.up_st, S.ef[p], 0) != cudaSuccess) { cudaGetLastError(); return; }   // the layer two back is done with this buffer (an event not yet recorded: no wait)
    if (S.ring.h2d(dst, S.host.data(), n * sizeof(Chunk), S.up_st) != cudaSuccess) return;
    if (cudaEventRecord(S.eu[p], S.up_st) != cudaSuccess) { cudaGetLastError(); return; }
    S.par = p;
    S.next = p ^ 1;
    S.cur = dst;
    S.ready = true;
    if (timing_on()) {
        std::lock_guard<std::mutex> lk(g_mu);
        HostT& h = g_ht[cur_dev()];
        h.up_ms += host_now_ms() - t0;
        ++h.n_up;
    }
}

// the three class kernels of a layer, each on a side stream that forks from the compute stream and joins back to it; nullptr: launched; else why it was not (nothing, or all but a
// failed part, is enqueued: the caller runs the one-row kernel after the join, which rewrites every output behind them on the compute stream)
const char* cls_launch(const Layer& L, cudaStream_t s) {
    Cls& S = cls_dev(cur_dev());
    if (!S.ready) return "the pass lists' upload failed";
    if (!cls_init(S)) return "the side streams could not be created";
    const int nf = L.hd_cls_n[0], nm = L.hd_cls_n[1], ns = L.hd_cls_n[2];
    const bool up2 = hfma_dn_up2() != 0;
    const Chunk* pf = up2 ? S.cur : S.d;
    const Chunk* pm = pf + nf;
    const Chunk* ps = pm + nm;
    const char* w = (const char*) L.w;
    const uint4* hb = (const uint4*) L.gu;
    Tm* tm = nullptr;   // STRATA_PREFILL_TIMING: this launch's marks on the class streams (the vector is only grown by this thread, in down_hf_prep)
    if (timing_on()) {
        const int dev = cur_dev();
        std::lock_guard<std::mutex> lk(g_mu);
        if (g_tm_n[dev] > 0) tm = &g_tm[dev][g_tm_n[dev] - 1];
    }
    cudaEventRecord(S.e0, s);
    for (int i = 0; i < 3; ++i) cudaStreamWaitEvent(S.st[i], S.e0, 0);
    if (up2) for (int i = 0; i < 3; ++i) cudaStreamWaitEvent(S.st[i], S.eu[S.par], 0);   // the lists are in (the upload has had the gate/up and the swiglu to finish)
    if (tm != nullptr) {
        for (int i = 0; i < 3; ++i) cudaEventRecord(tm->cs0[i], S.st[i]);
        tm->cls_timed = true;
    }
    if (L.d_type == GGML_TYPE_Q2_0) {
        if (nf > 0) dn::cls::q::kernel<32, true, 2><<<(unsigned) nf * dn::P2, dn::NT, 0, S.st[2]>>>(pf, w, L.xoff, L.down_off, hb, L.dm);
        if (nm > 0) dn::cls::q::kernel<32, false, 2><<<(unsigned) nm * dn::P2, dn::NT, 0, S.st[1]>>>(pm, w, L.xoff, L.down_off, hb, L.dm);
        if (ns > 0) dn::cls::q::kernel<16, false, 2><<<(unsigned) ns * dn::P2, dn::NT, 0, S.st[0]>>>(ps, w, L.xoff, L.down_off, hb, L.dm);
    } else if (L.d_type == GGML_TYPE_IQ4_NL) {
        if (nf > 0) dn::cls::n::kernel<32, true, 2><<<(unsigned) nf * dn::P2, dn::NT, 0, S.st[2]>>>(pf, w, L.xoff, L.down_off, hb, L.dm);
        if (nm > 0) dn::cls::n::kernel<32, false, 2><<<(unsigned) nm * dn::P2, dn::NT, 0, S.st[1]>>>(pm, w, L.xoff, L.down_off, hb, L.dm);
        if (ns > 0) dn::cls::n::kernel<16, false, 2><<<(unsigned) ns * dn::P2, dn::NT, 0, S.st[0]>>>(ps, w, L.xoff, L.down_off, hb, L.dm);
    } else {
        for (int i = 0; i < 3; ++i) { cudaEventRecord(S.ej[i], S.st[i]); cudaStreamWaitEvent(s, S.ej[i], 0); }
        if (up2) cudaEventRecord(S.ef[S.par], s);
        return "no class kernel for the format";
    }
    if (tm != nullptr)
        for (int i = 0; i < 3; ++i) cudaEventRecord(tm->cs1[i], S.st[i]);
    for (int i = 0; i < 3; ++i) {
        cudaEventRecord(S.ej[i], S.st[i]);
        cudaStreamWaitEvent(s, S.ej[i], 0);
    }
    if (up2) cudaEventRecord(S.ef[S.par], s);   // behind the join: the next upload into this buffer may go
    return cudaGetLastError() == cudaSuccess ? nullptr : "a class kernel's launch failed";
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

namespace {
// STRATA_PREFILL_TIMING: the host time in hfma_down_plan (the pass-class split, and with UP2 the upload)
struct PlanClock {
    double t0;
    int dev;
    PlanClock() : t0(timing_on() ? host_now_ms() : 0.0), dev(cur_dev()) {}
    ~PlanClock() {
        if (!timing_on()) return;
        std::lock_guard<std::mutex> lk(g_mu);
        g_ht[dev].plan_ms += host_now_ms() - t0;
        ++g_ht[dev].n_plan;
    }
};
}  // namespace

bool hfma_down_plan(Layer& L) {
    PlanClock plan_clock;
    L.hd_tier = 0;
    L.hd_r2 = 0;
    L.hd_cls = 0;
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
    L.hd_r2 = 0;
    bool r2_stay = false;
    const char* cls_why = nullptr;
    if (hfma_cls_down() != 0 || hfma_r2_down() != 0) {
        const int pick = r2_pick(f, u);
        if (pick == 0) {
            r2_stay = true;
        } else if (hfma_cls_down() != 0) {   // the pass-class kernels replace the two-rows kernel where it would have run; the one-row kernel is their counted fallback
            cls_why = cls_plan(L);
            if (cls_why == nullptr) {
                L.hd_cls = 1;
                L.hd_tier = pick;
                if (hfma_dn_up2() != 0) cls_upload_ahead(L);   // the lists go up now, behind the gate/up that is already queued
            }
        } else {
            L.hd_r2 = 1;
            L.hd_tier = pick;
        }
    }
    std::lock_guard<std::mutex> lk(g_mu);
    Dev& d = g_dev[dev];
    d.seen = true;
    ++d.taken[f][L.hd_tier == 32 ? 1 : 0];
    d.macs_taken[f] += rows * kMacsPerRow;
    if (hfma_cls_down() != 0) {
        if (L.hd_cls != 0) {
            ++d.cls_taken[f];
            d.cls_macs[f] += rows * kMacsPerRow;
            for (int c = 0; c < 3; ++c) d.cls_items[c] += L.hd_cls_n[c];
        } else {
            ++d.cls_left_n[f];
            d.cls_macs_left[f] += rows * kMacsPerRow;
            ++d.cls_left[fmt_name(L.d_type) + ": " +
                         (cls_why != nullptr ? cls_why : "u below the pass-class kernels' range, the engine's 16-token kernel is faster there")];
        }
    } else if (L.hd_r2 != 0) {
        ++d.r2_taken[f];
        d.r2_macs[f] += rows * kMacsPerRow;
    } else if (r2_stay) {
        ++d.r2_left_n[f];
        d.r2_macs_left[f] += rows * kMacsPerRow;
        ++d.r2_left[fmt_name(L.d_type) + ": u below the two-rows kernel's range, the engine's 16-token kernel is faster there"];
    }
    if (L.hd_layer >= 0) d.layers[f] |= 1ull << (L.hd_layer & 63);
    return true;
}

void hfma_down_left(const int d_type, const int64_t rows, const char* why) { note_left(cur_dev(), d_type, rows, why); }

void down_hf_prep(const Layer& L, void* stream) {
    if (L.rows <= 0) return;
    const cudaStream_t s = (cudaStream_t) stream;
    (void) state_ptr(cur_dev());
    const double th0 = timing_on() ? host_now_ms() : 0.0;
    if (timing_on() && L.hd_cls != 0) {   // is the compute stream already idle? then the host is what the GPU waits for from here
        const bool idle = cudaStreamQuery(s) == cudaSuccess;
        cudaGetLastError();
        std::lock_guard<std::mutex> lk(g_mu);
        g_ht[cur_dev()].idle_prep += idle ? 1 : 0;
        ++g_ht[cur_dev()].n_prep;
    }
    if (timing_on()) {
        const int dev = cur_dev();
        std::lock_guard<std::mutex> lk(g_mu);
        auto& v = g_tm[dev];
        if (g_tm_n[dev] == v.size()) {
            Tm t;
            ck(cudaEventCreate(&t.a), "hfma_down: event");
            ck(cudaEventCreate(&t.b), "hfma_down: event");
            ck(cudaEventCreate(&t.c), "hfma_down: event");
            ck(cudaEventCreate(&t.u), "hfma_down: event");
            for (int i = 0; i < 3; ++i) {
                ck(cudaEventCreate(&t.cs0[i]), "hfma_down: event");
                ck(cudaEventCreate(&t.cs1[i]), "hfma_down: event");
            }
            v.push_back(t);
        }
        Tm& t = v[g_tm_n[dev]++];
        t.fmt = fmt_idx(L.d_type);
        t.tier = L.hd_tier;
        t.r2 = L.hd_cls != 0 ? 2 : L.hd_r2;
        t.macs = (double) L.rows * kMacsPerRow;
        t.cls_timed = false;
        cudaEventRecord(t.a, s);
    }
    if (L.hd_cls != 0 && hfma_dn_up2() == 0) cls_upload(L, s);   // STRATA_PF_HFMA_DOWN_CLASSES: this layer's pass lists up on the compute stream, behind the previous layer's kernels
    if (timing_on()) {
        const int dev = cur_dev();
        std::lock_guard<std::mutex> lk(g_mu);
        cudaEventRecord(g_tm[dev][g_tm_n[dev] - 1].u, s);
    }
    dn::swiglu_h16_kernel<<<(unsigned) L.rows, 160, 0, s>>>(L.gu, L.hd_layer);
    ck(cudaGetLastError(), "down FP16: swiglu");
    if (timing_on()) {
        const int dev = cur_dev();
        std::lock_guard<std::mutex> lk(g_mu);
        cudaEventRecord(g_tm[dev][g_tm_n[dev] - 1].b, s);
        g_ht[dev].prep_ms += host_now_ms() - th0;
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
    const char* cls_why = nullptr;
    if (L.hd_cls != 0) {
        const double tl0 = timing_on() ? host_now_ms() : 0.0;
        if (timing_on()) {
            const bool idle = cudaStreamQuery(s) == cudaSuccess;
            cudaGetLastError();
            std::lock_guard<std::mutex> lk(g_mu);
            g_ht[cur_dev()].idle_launch += idle ? 1 : 0;
            ++g_ht[cur_dev()].n_launch;
        }
        cls_why = cls_launch(L, s);   // STRATA_PF_HFMA_DOWN_CLASSES: three class kernels; nullptr = launched
        if (timing_on()) {
            std::lock_guard<std::mutex> lk(g_mu);
            g_ht[cur_dev()].launch_ms += host_now_ms() - tl0;
        }
    }
    if (cls_why != nullptr) {   // the one-row kernel at the tier its own rule gives (everything the class kernels wrote is rewritten behind them on this stream)
        const int dev = cur_dev();
        const int f = fmt_idx(L.d_type);
        const double macs = (double) L.rows * kMacsPerRow;
        {
            std::lock_guard<std::mutex> lk(g_mu);
            Dev& d = g_dev[dev];
            --d.cls_taken[f];
            d.cls_macs[f] -= macs;
            for (int c = 0; c < 3; ++c) d.cls_items[c] -= L.hd_cls_n[c];
            ++d.cls_left_n[f];
            d.cls_macs_left[f] += macs;
            ++d.cls_left[fmt_name(L.d_type) + ": " + cls_why + " (at launch)"];
            if (timing_on() && g_tm_n[dev] > 0) { g_tm[dev][g_tm_n[dev] - 1].r2 = 0; g_tm[dev][g_tm_n[dev] - 1].tier = 16; }
        }
        if (L.d_type == GGML_TYPE_Q2_0) dn::q20::dn_kernel<16><<<grid, dn::NT, 0, s>>>(L.chunks, w, L.xoff, L.down_off, hb, L.dm);
        else dn::nl4::dn_kernel<16><<<grid, dn::NT, 0, s>>>(L.chunks, w, L.xoff, L.down_off, hb, L.dm);
    } else if (L.hd_cls != 0) {
        // launched above
    } else if (L.hd_r2 != 0) {   // STRATA_PF_HFMA_R2_DOWN: 256 rows a CTA, ten CTAs a chunk
        const unsigned grid2 = (unsigned) (L.nchunks * dn::P2);
        if (L.d_type == GGML_TYPE_Q2_0) dn::r2q::dn2_kernel<16><<<grid2, dn::NT, 0, s>>>(L.chunks, w, L.xoff, L.down_off, hb, L.dm);
        else if (L.d_type == GGML_TYPE_IQ4_NL) dn::r2n::dn2_kernel<32><<<grid2, dn::NT, 0, s>>>(L.chunks, w, L.xoff, L.down_off, hb, L.dm);
        else {
            std::fprintf(stderr, "prefill ws: down type %d has no FP16 kernel\n", L.d_type);
            std::exit(1);
        }
    } else if (L.d_type == GGML_TYPE_Q2_0) {
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
        std::string kt, cl_t;
        if (g_tm_n[dev] > 0) {
            struct Acc { double prep = 0.0, dn = 0.0, macs = 0.0; int64_t n = 0; };
            struct ClsAcc { double up = 0.0, swi = 0.0, fork = 0.0, span = 0.0, join = 0.0, kern[3] = {0.0, 0.0, 0.0}, ksum = 0.0; int64_t n = 0; } ca;   // the pass-class launches
            std::map<std::pair<int, int>, Acc> acc;   // (format, tier + 100 for the two-rows kernel, + 200 for the pass-class kernels)
            Acc all;
            cudaSetDevice(dev);
            for (size_t i = 0; i < g_tm_n[dev]; ++i) {
                float p = 0.0f, q = 0.0f;
                if (cudaEventSynchronize(g_tm[dev][i].c) == cudaSuccess && cudaEventElapsedTime(&p, g_tm[dev][i].a, g_tm[dev][i].b) == cudaSuccess &&
                    cudaEventElapsedTime(&q, g_tm[dev][i].b, g_tm[dev][i].c) == cudaSuccess) {
                    Acc& a = acc[{g_tm[dev][i].fmt, g_tm[dev][i].tier + 100 * g_tm[dev][i].r2}];
                    a.prep += p; a.dn += q; a.macs += g_tm[dev][i].macs; ++a.n;
                    all.prep += p; all.dn += q; all.macs += g_tm[dev][i].macs; ++all.n;
                    if (g_tm[dev][i].cls_timed && g_tm[dev][i].r2 == 2) {   // a..u upload, u..b swiglu, b..cs0 fork, cs0..cs1 the class kernels, cs1..c join
                        const Tm& t = g_tm[dev][i];
                        float up = 0.0f, sw = 0.0f, f0 = 1e30f, f1 = -1e30f, ksum = 0.0f, kk[3] = {0.0f, 0.0f, 0.0f};
                        bool ok = cudaEventElapsedTime(&up, t.a, t.u) == cudaSuccess && cudaEventElapsedTime(&sw, t.u, t.b) == cudaSuccess;
                        for (int k = 0; k < 3 && ok; ++k) {
                            float x = 0.0f, y = 0.0f;
                            ok = cudaEventElapsedTime(&x, t.b, t.cs0[k]) == cudaSuccess && cudaEventElapsedTime(&y, t.b, t.cs1[k]) == cudaSuccess &&
                                 cudaEventElapsedTime(&kk[k], t.cs0[k], t.cs1[k]) == cudaSuccess;
                            f0 = std::min(f0, x);
                            f1 = std::max(f1, y);
                            ksum += kk[k];
                        }
                        if (ok) {
                            ca.up += up; ca.swi += sw; ca.fork += f0; ca.span += f1 - f0; ca.join += q - f1; ca.ksum += ksum;
                            for (int k = 0; k < 3; ++k) ca.kern[k] += kk[k];
                            ++ca.n;
                        } else {
                            cudaGetLastError();
                        }
                    }
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
            if (ca.n > 0 || g_ht[dev].n_plan > 0) {
                const HostT& h = g_ht[dev];
                char b[900];
                std::snprintf(b, sizeof b, "; down pass classes, GPU (x%lld): upload %.1f ms, swiglu %.1f ms, fork %.1f ms, class kernels span %.1f ms (full %.1f / mid %.1f / small %.1f ms, sum %.1f), join %.1f ms; "
                                           "host: plan %.1f ms (x%lld), upload %.1f ms (x%lld), prep %.1f ms (x%lld), launch %.1f ms (x%lld); compute stream already idle at prep %lld, at launch %lld",
                              (long long) ca.n, ca.up, ca.swi, ca.fork, ca.span, ca.kern[2], ca.kern[1], ca.kern[0], ca.ksum, ca.join, h.plan_ms, (long long) h.n_plan, h.up_ms, (long long) h.n_up,
                              h.prep_ms, (long long) h.n_prep, h.launch_ms, (long long) h.n_launch, (long long) h.idle_prep, (long long) h.idle_launch);
                cl_t = b;
            }
            for (const auto& [key, a] : acc) line(std::string(key.first == 0 ? "Q2_0" : "IQ4_NL") + " t" + std::to_string(key.second % 100) + (key.second >= 200 ? "cls" : key.second >= 100 ? "r2" : ""), a);
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
        std::string r2s;   // STRATA_PF_HFMA_R2_DOWN: what ran on the two-rows kernels, what stayed on the engine's kernel and why
        if (hfma_r2_down() != 0) {
            int64_t r2_n = 0, r2_l = 0;
            double r2_m = 0.0, r2_lm = 0.0;
            std::string rb;
            for (int f = 0; f < 3; ++f) {
                r2_n += d.r2_taken[f]; r2_l += d.r2_left_n[f]; r2_m += d.r2_macs[f]; r2_lm += d.r2_macs_left[f];
                if (d.r2_taken[f] == 0 && d.r2_left_n[f] == 0) continue;
                char b[160];
                std::snprintf(b, sizeof b, "%s%s %lld", rb.empty() ? "" : "; ", f == 2 ? "other" : fmt_name(types[f]).c_str(), (long long) d.r2_taken[f]);
                rb += b;
            }
            std::string rl;
            for (const auto& [what, n] : d.r2_left) rl += (rl.empty() ? "" : ", ") + what + " x" + std::to_string((long long) n);
            char b[300];
            std::snprintf(b, sizeof b, "; STRATA_PF_HFMA_R2_DOWN: %lld launches on the two-rows kernels (%.1f%% of the HFMA2 MACs), %lld stayed on the engine kernel",
                          (long long) r2_n, r2_m + r2_lm > 0 ? 100.0 * r2_m / (r2_m + r2_lm) : 0.0, (long long) r2_l);
            r2s = b;
            r2s += "; by format: " + (rb.empty() ? std::string("-") : rb) + "; left: " + (rl.empty() ? std::string("nothing") : rl);
        }
        std::string cls_s;   // STRATA_PF_HFMA_DOWN_CLASSES: what ran on the pass-class kernels, what stayed on the one-row / two-rows kernel and why
        if (hfma_cls_down() != 0) {
            int64_t c_n = 0, c_l = 0;
            double c_m = 0.0, c_lm = 0.0;
            std::string cb;
            for (int f = 0; f < 3; ++f) {
                c_n += d.cls_taken[f]; c_l += d.cls_left_n[f]; c_m += d.cls_macs[f]; c_lm += d.cls_macs_left[f];
                if (d.cls_taken[f] == 0 && d.cls_left_n[f] == 0) continue;
                char b[200];
                std::snprintf(b, sizeof b, "%s%s %lld on the class kernels, %lld on the one-row / two-rows kernel", cb.empty() ? "" : "; ", f == 2 ? "other" : fmt_name(types[f]).c_str(),
                              (long long) d.cls_taken[f], (long long) d.cls_left_n[f]);
                cb += b;
            }
            std::string cl;
            for (const auto& [what, n] : d.cls_left) cl += (cl.empty() ? "" : ", ") + what + " x" + std::to_string((long long) n);
            char b[400];
            std::snprintf(b, sizeof b, "; STRATA_PF_HFMA_DOWN_CLASSES: %lld launches on the class kernels (%.1f%% of the HFMA2 MACs; passes: 32 rows x%lld, 17..31 x%lld, 1..16 x%lld), %lld on the other kernels",
                          (long long) c_n, c_m + c_lm > 0 ? 100.0 * c_m / (c_m + c_lm) : 0.0, (long long) d.cls_items[0], (long long) d.cls_items[1], (long long) d.cls_items[2],
                          (long long) c_l);
            cls_s = b;
            cls_s += "; by format: " + (cb.empty() ? std::string("-") : cb) + "; left: " + (cl.empty() ? std::string("nothing") : cl);
        }
        cls_s += cl_t;
        g_ht[dev] = HostT{};
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
                             "(2^-k), one sample a CTA: %s; non-finite: H %u values, output %u values; kernel time: %s%s%s\n",
                     (long long) tokens, dev, (long long) n_taken, (long long) n_left,
                     m_taken + m_left > 0 ? 100.0 * m_taken / (m_taken + m_left) : 0.0, m_taken + m_left, by.empty() ? "-" : by.c_str(),
                     lft.empty() ? "nothing" : lft.c_str(), f32(st.hmax_all), lm.empty() ? "-" : lm.c_str(), kh.empty() ? "-" : kh.c_str(),
                     st.bad_h, st.bad_y, kt.empty() ? "-" : kt.c_str(), r2s.c_str(), cls_s.c_str());
        d = Dev{};
    }
    cudaSetDevice(back);
}

}  // namespace strata::prefill::ws
