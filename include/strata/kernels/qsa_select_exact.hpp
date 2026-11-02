// include/strata/kernels/qsa_select_exact.hpp - STRATA_PF_SEL_EXACT (default off), declared on its own so that the files that read qsa_select.hpp keep the text they had
#pragma once
#include <cstdint>

#include "strata/kernels/qsa.hpp"

namespace strata::kernels {
/// STRATA_PF_SEL_EXACT=1 (CUDA, sm_60, STRATA_EXPERIMENTAL_SM60): the prompt's scores + top-k of one call in one go, ids [nq, cap] IDENTICAL to qsa_block_scores + qsa_block_topk's with the FP32
/// scorer: an FP16 pair scorer filters, only the blocks inside a proven worst-case error band of the threshold are scored again with the FP32 kernel's arithmetic
/// (kf-work\kexact\BOUND.md).  `scores` is the [nq, max_blocks] scratch of qsa_block_scores.  False (nothing launched) when the switch is off, `active_blocks` <= 0, the card is not
/// compute capability 6.x, or the call is outside the kernels' limits (a row of more than 40,960 blocks ...): the caller then runs qsa_block_scores + qsa_block_topk.
bool qsa_select_exact(const float* pooled, const float* dead, const float* q_idx, const int32_t* steps, int64_t nq, int64_t max_blocks, int64_t cap, const QsaShapes& s,
                      float* scores, int32_t* ids, void* stream, int64_t active_blocks);
}  // namespace strata::kernels
