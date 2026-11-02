// include/strata/kernels/ktime.hpp - STRATA_PREFILL_TIMING=2: the prompt path's GPU time by KERNEL for the three phases
// "qsa select", "qsa attn" and "gdn recurrence" (STRATA_PREFILL_TIMING=1 stays the phase table; 2 adds one line a chunk).
//
// begin(id, stream) records an event on the compute stream; the time to the NEXT begin on the same device is charged to id
// (kOther = not counted).  Launch gaps between two kernels therefore land on the kernel before.  The events are folded where
// prefill.cpp prints its phase table (the stream was synchronized there).  Header only (C++17 inline), host only: with the
// switch off every call is one test of a cached bool and no device code differs.
#pragma once

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace strata::kernels::ktime {

enum Id : int {
    kOther = 0,
    kSelScores,    // qsa_block_scores: the thread-per-head scorer (prompt calls)
    kSelTail,      // block_scores_tail_kernel: the query's own unfinished block
    kSelTopk,      // qsa_block_topk (block_topk_wide_kernel on sm_60)
    kAttnScore,    // ps_score_x_kernel (STRATA_PATTN_SM60) or the FP16 twin
    kAttnValue,    // ps_value_d4_kernel or the FP16 twin
    kGdnRec,       // gdn_rec_cols_pipe_kernel (the token chain)
    kGdnNorm,      // gdn_out_norm_kernel (output normalisation + gate, y16)
    kIdCount
};
inline const char* name(int id) {
    static const char* const n[kIdCount] = {"other", "select scores", "select tail", "select top-k", "attn score",
                                            "attn value", "gdn recurrence", "gdn out norm"};
    return id >= 0 && id < kIdCount ? n[id] : "?";
}
inline bool on() {
    static const bool v = [] {
        const char* e = std::getenv("STRATA_PREFILL_TIMING");
        return e != nullptr && std::atoi(e) >= 2;
    }();
    return v;
}
struct Dev {
    std::vector<cudaEvent_t> ev;
    std::vector<int> id;
    size_t used = 0;
    double ms[kIdCount] = {};
    long long n[kIdCount] = {};
};
inline Dev& dev_state() {
    static Dev d[64];
    int dv = 0;
    cudaGetDevice(&dv);
    return d[dv & 63];
}
inline void begin(int id, void* stream) {
    if (!on()) return;
    Dev& d = dev_state();
    if (d.used == d.ev.size()) {
        cudaEvent_t e = nullptr;
        cudaEventCreate(&e);
        d.ev.push_back(e);
        d.id.push_back(0);
    }
    d.id[d.used] = id;
    cudaEventRecord(d.ev[d.used], (cudaStream_t) stream);
    ++d.used;
}
/// Charge the recorded gaps (every event has completed or is waited for) and keep the last event as the start of the next run.
inline void fold() {
    Dev& d = dev_state();
    if (d.used < 2) return;
    cudaEventSynchronize(d.ev[d.used - 1]);
    for (size_t i = 0; i + 1 < d.used; ++i) {
        float t = 0.0f;
        if (d.id[i] != kOther && cudaEventElapsedTime(&t, d.ev[i], d.ev[i + 1]) == cudaSuccess) {
            d.ms[d.id[i]] += t;
            d.n[d.id[i]] += 1;
        }
    }
    std::swap(d.ev[0], d.ev[d.used - 1]);
    std::swap(d.id[0], d.id[d.used - 1]);
    d.used = 1;
}
/// The line for the chunk just read on this device (and the totals reset).
inline void print_line(long long tokens, int cuda_dev) {
    if (!on()) return;
    fold();
    Dev& d = dev_state();
    std::string s;
    char b[96];
    for (int i = 1; i < kIdCount; ++i) {
        if (d.n[i] == 0) continue;
        std::snprintf(b, sizeof b, " %s %.2f ms (%lld)", name(i), d.ms[i], d.n[i]);
        s += b;
        d.ms[i] = 0.0;
        d.n[i] = 0;
    }
    std::fprintf(stderr, "strata prefill timing (kernels): %lld tokens:%s (CUDA%d)\n", tokens, s.c_str(), cuda_dev);
}

}  // namespace strata::kernels::ktime
