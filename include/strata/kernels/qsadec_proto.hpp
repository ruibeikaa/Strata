// include/strata/kernels/qsadec_proto.hpp - STRATA_QSADEC_PROTO: the decode window's QSA select and attention on
// sm_60, bit for bit the long-context select (qsa_block_scores_long + qsa_block_topk_long) and qsa_decode_attn_batch.
//
//   select : one kernel scores the blocks (one lane per block, all four indexer heads and every query of the call, the
//            key read straight into registers - no shared-memory staging, no barriers), scores the tail block and
//            builds the top-k's first radix histogram; the multi-CTA top-k's passes 1-3, count and emit follow.
//            6 kernels instead of 8.  Every score keeps block_scores_dx_kernel's tree and relu order, the ids follow
//            from the same integer histograms (the threshold and the cells above it do not depend on the order the
//            counts are summed in).
//   attn   : the chunk kernel with its loads issued ahead of the barriers (ids, q, K for all of a warp's cells, V into
//            shared memory with 16-byte loads), the same arithmetic in the same order; the merge is attn_merge_v2's
//            (bitwise attn_merge_kernel's).
//
// Off unless STRATA_QSADEC_PROTO=1.  Bench entry points (variants, ablations) are for qsadec_bench only.
#pragma once

#include "strata/kernels/qsa.hpp"
#include "strata/kernels/qsa_decode_attn.hpp"

#include <cstdint>

namespace strata::kernels {

/// STRATA_QSADEC_PROTO: 0 off (default), 1 the select and the attention, 2 the select only, 3 the attention only.
int qsadec_proto_mode();

/// STRATA_QSASEL (campaign QSA-SELECT, opt-in, needs STRATA_QSADEC_PROTO=1/2): the proto select's scorer, then the
/// multi top-k's passes 1-3, count and emit in ONE kernel (a barrier among a query's CTAs where the kernel boundaries
/// were); the same scores and ids.  1 = variant 33 (a warp's first 8 rows of keys kept in registers: the deploy
/// setting), 2 = variant 30 (keys read again on every pass), 3 = variant 40 (two barriers; slower).  0 off (default).
/// STRATA_QSASEL_CHECK=1 (diagnostic): every call is also run on the production long path and compared in the graph.
int qsasel_mode();
/// the qsadec_select variant STRATA_QSASEL picks: 0 (off), 1 -> 33, 2 -> 30, 3 -> 40
int qsasel_variant();

/// Scratch the proto select needs per call of up to `nq` queries (int32); 0 when it is not built.  It must be zero
/// before the first call (the select leaves its own histogram row zeroed for the next one).
int64_t qsadec_select_scratch_ints(int64_t nq);

/// The long-context decode select (scores rows [0, n_bid] and ids as the production long path).  `variant` 0 is the
/// default; others are bench variants.  False: not applicable (nothing launched).
bool qsadec_select(const float* pooled, const float* dead, const float* q_idx, const int32_t* steps, int64_t nq,
                   int64_t max_blocks, int64_t cap, const QsaShapes& s, float* scores, int32_t* ids, int32_t* scratch,
                   void* stream, int variant = 0);

/// qsa_decode_attn_batch (INT8 K/V only).  False: not applicable (nothing launched).
bool qsadec_attn(const float* q, const QsaAttnPools& pools, const int32_t* ids, const int32_t* steps, int64_t cap,
                 const QsaShapes& s, float* scratch, float* attn, int64_t nq, void* stream, int variant = 0);

// ---- bench only: copies of the production kernels with ablations, one kernel at a time
/// select stages: 0 dx scorer copy (+ tail), 1..4 multi hist pass 0..3 copies, 5 count, 6 emit; abl: see the .cu
void qsadec_bench_select_stage(int stage, int abl, const float* pooled, const float* dead, const float* q_idx,
                               const int32_t* steps, int64_t nq, int64_t max_blocks, int64_t cap, float* scores,
                               int32_t* ids, int32_t* scratch, void* stream);
/// attention: the production chunk kernel's copy with ablation bits `abl` (0: the copy), then merge `merge` (0 v1, 1 v2,
/// -1 none)
void qsadec_bench_attn_abl(int abl, int merge, const float* q, const QsaAttnPools& pools, const int32_t* ids,
                           const int32_t* steps, int64_t cap, const QsaShapes& s, float* scratch, float* attn,
                           int64_t nq, void* stream);
/// the top-k alone on given score rows (a prep kernel first builds the fused scorer's pass-0 row); variant 0 = the
/// proto's five kernels, 30 / 33 / 40 = QSASEL's; prep_only: the prep alone
void qsadec_bench_topk(int variant, bool prep_only, const float* scores, const int32_t* steps, int64_t nq,
                       int64_t max_blocks, int64_t cap, int32_t* ids, int32_t* scratch, void* stream);
/// registers / CTAs per SM of a bench kernel (for the report)
void qsadec_bench_attrs(char* buf, int len);

}  // namespace strata::kernels
