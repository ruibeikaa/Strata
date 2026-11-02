// include/strata/prefill/pf_small.hpp - STRATA_PF_HFMA_SMALL (default off): one switch for a batch of small savings in the FP16 prompt read
// on GP100 (sm_60), meant to ride with STRATA_PF_HFMA=1 / _EXP / _DOWN / _HC / _SASS:
//   - the hyper-connection read's activation: the max |x| and the FP16 image in one pass (a second pass only for a slice that needs a shift),
//   - gr_mix_r as four values a thread (same bits),
//   - the inject product (N 4, K 10240) on FP16 HFMA2 pairs, sharing the down product's activation image,
//   - the layer split's hand-off over NVLink between the two dies of a board (a direct peer copy in place of a download and an upload).
// Off, every call is what it was.
#pragma once

#include <cstdlib>

namespace strata::prefill {

inline bool pf_hfma_small() {
    static const bool on = [] {
        const char* e = std::getenv("STRATA_PF_HFMA_SMALL");
        return e != nullptr && std::atoi(e) != 0;
    }();
    return on;
}

}  // namespace strata::prefill
