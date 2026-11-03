// include/strata/prefill/pf_small2.hpp - four default-off switches for the small kernels of the FP16 prompt read on GP100 (sm_60), found with the Kernel Forge harness
// (kf-work\ksel, ktopk, kgdn, kcomb), meant to ride with STRATA_PF_HFMA=1 / _EXP / _DOWN / _HC / _SASS / _SMALL / _R2:
//   STRATA_PF_HFMA_SEL=1    the prompt's select scores (qsa_block_scores) on FP16 HFMA2 pairs of key blocks (power-of-two scaled per block and per query head, FP16
//                           accumulate over <= 32 products a lane, widened to FP32 for the relu / head sum): the selection changes only for blocks within rounding of the threshold
//   STRATA_PF_TOPK_SEL=1    the prompt's select top-k (qsa_block_topk) by a sample bound and ordered survivors instead of the six-pass radix: the ids are EXACTLY the wide kernel's
//   STRATA_PF_GDN_FUSED=1   the GDN recurrence in one pass over the state registers a token, one barrier a token, two columns a thread: BIT-IDENTICAL to gdn_rec_cols_pipe_kernel
//   STRATA_PF_COMBINE_V2=1  the MoE combine + hyper-connection write (moe_combine_write_norm_rs): the 4 rows loaded together, one block sum, prefetched wn: BIT-IDENTICAL
//   STRATA_PF_ROUTER_V2=1   the MoE layer's router logits and the shared expert's scalar gate from ONE FP32 widening of the layer input a slice (the engine widened it four times: router hi / lo,
//                           shared gate hi / lo) and the shared expert's SwiGLU four columns a thread: the same cublasSgemm calls with the same arguments, BIT-IDENTICAL (kf-work\krouter)
//   STRATA_PF_SEL_EXACT=1   the prompt's select (scores + top-k) as ONE exact path on the FP16 pair scorer: the FP16 scores are only a filter - a score is trusted outside a PROVEN worst-case error
//                           band around the top-k threshold and recomputed with the FP32 kernel's own arithmetic inside it, so the selected ids are IDENTICAL to the FP32 scorer + top-k's
//                           (kf-work\kexact\BOUND.md).  Takes the place of STRATA_PF_HFMA_SEL (which it overrides) and STRATA_PF_TOPK_SEL for the prompt calls it takes.
// Off, every call is what it was.  A call the new path cannot take (another architecture, a row too long, a misaligned buffer ...) runs the old kernel and is COUNTED by reason;
// the counts are printed once when the process ends (and the first time a switch is used).
#pragma once

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace strata::prefill {

inline bool pf_env_on(const char* name) {
    const char* e = std::getenv(name);
    return e != nullptr && std::atoi(e) != 0;
}
inline bool pf_hfma_sel() { static const bool on = pf_env_on("STRATA_PF_HFMA_SEL"); return on; }
inline bool pf_topk_sel() { static const bool on = pf_env_on("STRATA_PF_TOPK_SEL"); return on; }
inline bool pf_gdn_fused() { static const bool on = pf_env_on("STRATA_PF_GDN_FUSED"); return on; }
inline bool pf_combine_v2() { static const bool on = pf_env_on("STRATA_PF_COMBINE_V2"); return on; }
inline bool pf_sel_exact() { static const bool on = pf_env_on("STRATA_PF_SEL_EXACT"); return on; }
inline bool pf_router_v2() { static const bool on = pf_env_on("STRATA_PF_ROUTER_V2"); return on; }
// STRATA_PF_ROUTER_F16=1 (default off, GP100 + STRATA_PF_HFMA_SASS): the MoE router's logits on the hand-scheduled dense HFMA2 kernel (N 512, K 2560) from the layer input's FP16 image and the
// router weight's resident FP16 copy, instead of cublasSgemm on the BF16-rounded input widened to FP32.  NOT bitwise the FP32 router's: the logits change by rounding and a near-tied expert id can
// change.  STRATA_PF_ROUTER_F16_CHECK=1 (diagnostic, only with the switch above): also runs the FP32 router into scratch after every chunk and layer and counts the ids that differ.
inline bool pf_router_f16() { static const bool on = pf_env_on("STRATA_PF_ROUTER_F16"); return on; }
inline bool pf_router_f16_check() { static const bool on = pf_router_f16() && pf_env_on("STRATA_PF_ROUTER_F16_CHECK"); return on; }
// STRATA_PF_PLE_F16=1 (default off, GP100): the PLE block's key (N 10240, K 2560) and value (N 2560, K 2560) projections - the first stage's layer 1, one block per chunk - from the FP16 image of the dequantized
// n-gram rows and the BF16 weights' resident FP16 copies on the dense HFMA2 kernels (hand-scheduled for the key where STRATA_PF_HFMA_SASS covers it), instead of the BF16-rounded rows widened to FP32 and cublasSgemm
// on the BF16 weights widened to FP32.  NOT bitwise the FP32 route's: the activation keeps 11 mantissa bits (the FP32 route 8) and the sums are FP16 per 64 products widened to FP32.
inline bool pf_ple_f16() { static const bool on = pf_env_on("STRATA_PF_PLE_F16"); return on; }
// STRATA_PF_PLE_F16=2: each image is scaled by a power of two (its largest |x| to [128, 256): the n-gram rows are 0.06 at most) and the product scaled back by the exact inverse
inline bool pf_ple_f16_scale() { static const bool on = [] { const char* e = std::getenv("STRATA_PF_PLE_F16"); return e != nullptr && std::atoi(e) >= 2; }(); return on; }
// STRATA_PF_PLE_PEER=1 (default off, the first stage of a layer split whose next stage's die is its NVLink pair): the PLE block's key projection for the NEXT chunk runs on that die, on a stream of its own
// (highest priority), while this stage reads the current chunk, and lands in this die's second key buffer over NVLink (cudaMemcpyPeerAsync).  The same raw rows, kernels, cuBLAS calls and weights
// as the local product: the key is bit for bit the local key.  The first chunk of a call, a read of one chunk, a helped read and any launch that fails keep the local product (counted by reason, printed per read).
inline bool pf_ple_peer() { static const bool on = pf_env_on("STRATA_PF_PLE_PEER"); return on; }

enum Pf2Switch { kPf2Sel = 0, kPf2Topk, kPf2Gdn, kPf2Comb, kPf2SelX, kPf2Router, kPf2Count };
enum Pf2Reason { kPf2Arch = 0, kPf2Geometry, kPf2Align, kPf2Length, kPf2ReasonCount };

struct Pf2Stats {
    std::atomic<uint64_t> launches[kPf2Count];
    std::atomic<uint64_t> fallbacks[kPf2Count][kPf2ReasonCount];
    std::atomic<uint64_t> flagged[kPf2Count];   // top-k only (STRATA_PF_TOPK_SEL_STATS=1): queries the kernel handed to the wide kernel
    std::atomic<uint64_t> selx_queries{0}, selx_blocks{0}, selx_survivors{0}, selx_band{0}, selx_slow{0};   // STRATA_PF_SEL_EXACT_STATS=1
    std::atomic<uint64_t> selx_why[6]{};   // why a query took the all-blocks path: bound, sample, too many survivors, non-finite score, LB, T
    Pf2Stats() {
        for (int i = 0; i < kPf2Count; ++i) {
            launches[i] = 0;
            flagged[i] = 0;
            for (int r = 0; r < kPf2ReasonCount; ++r) fallbacks[i][r] = 0;
        }
    }
    ~Pf2Stats() {
        static const char* names[kPf2Count] = {"STRATA_PF_HFMA_SEL", "STRATA_PF_TOPK_SEL", "STRATA_PF_GDN_FUSED", "STRATA_PF_COMBINE_V2", "STRATA_PF_SEL_EXACT", "STRATA_PF_ROUTER_V2"};
        static const char* what[kPf2Count] = {"calls on the FP16 pair scorer", "calls on the sample-bound top-k", "recurrence calls on the fused kernel", "calls on the v2 combine", "calls on the exact FP16-filtered select", "layer calls on the shared-widening router + shared gate"};
        static const char* why[kPf2ReasonCount] = {"architecture", "geometry", "alignment", "row length"};
        for (int i = 0; i < kPf2Count; ++i) {
            const uint64_t n = launches[i].load();
            uint64_t fb = 0;
            for (int r = 0; r < kPf2ReasonCount; ++r) fb += fallbacks[i][r].load();
            if (n == 0 && fb == 0) continue;
            std::fprintf(stderr, "strata prefill: %s: %llu %s, %llu fell back to the old kernel (", names[i], (unsigned long long) n, what[i], (unsigned long long) fb);
            bool any = false;
            for (int r = 0; r < kPf2ReasonCount; ++r)
                if (const uint64_t v = fallbacks[i][r].load()) { std::fprintf(stderr, "%s%s %llu", any ? ", " : "", why[r], (unsigned long long) v); any = true; }
            if (!any) std::fprintf(stderr, "none");
            std::fprintf(stderr, ")");
            if (flagged[i].load()) std::fprintf(stderr, "; %llu queries handed to the wide kernel inside the launches", (unsigned long long) flagged[i].load());
            if (i == kPf2SelX && selx_queries.load())
                std::fprintf(stderr, "; %llu queries, %llu blocks scored, %.2f%% of them rescored with the FP32 kernel's arithmetic (survivors %.1f a query), %llu queries took the all-blocks FP32 path [bound %llu, sample %llu, survivors %llu, non-finite %llu, LB %llu, T %llu]",
                             (unsigned long long) selx_queries.load(), (unsigned long long) selx_blocks.load(), 100.0 * (double) selx_band.load() / (double) (selx_blocks.load() ? selx_blocks.load() : 1),
                             (double) selx_survivors.load() / (double) selx_queries.load(), (unsigned long long) selx_slow.load(), (unsigned long long) selx_why[0].load(), (unsigned long long) selx_why[1].load(),
                             (unsigned long long) selx_why[2].load(), (unsigned long long) selx_why[3].load(), (unsigned long long) selx_why[4].load(), (unsigned long long) selx_why[5].load());
            std::fprintf(stderr, "\n");
        }
    }
};
inline Pf2Stats g_pf2;   // one for the process (inline variable)

inline void pf2_launched(Pf2Switch s) {
    if (g_pf2.launches[s].fetch_add(1) == 0)
        std::fprintf(stderr, "strata prefill: %s is on: first call on the new kernel\n",
                     s == kPf2Sel ? "STRATA_PF_HFMA_SEL" : s == kPf2Topk ? "STRATA_PF_TOPK_SEL" : s == kPf2Gdn ? "STRATA_PF_GDN_FUSED" : s == kPf2Comb ? "STRATA_PF_COMBINE_V2" : s == kPf2SelX ? "STRATA_PF_SEL_EXACT" : "STRATA_PF_ROUTER_V2");
}
inline void pf2_fallback(Pf2Switch s, Pf2Reason r) { g_pf2.fallbacks[s][r].fetch_add(1); }

}  // namespace strata::prefill
