// include/strata/prefill/moe_ws.hpp - PH402 (sm_60): the prompt path's expert products of a layer whose experts are all
// read in place (mmq::inplace), weight-stationary: ONE launch for the layer's gate/up and ONE for its down instead of a
// gate/up, swiglu and down MMQ launch per 16-expert group.  A CTA holds 128 (or 256) weight rows of one expert and runs
// them against up to 64 of the expert's activation rows, a thread per weight row: each weight block is decoded once per
// pass instead of once per MMQ tile, and an expert's rows are its own count, not the group's largest.  Every product
// is MMQ's (llama.cpp-ph402 6dd634b mul_mat_q_case_inplace: its int8 values, scales and per-32-value epilogue in its K
// order), so GU, H and Dm are next13's bits.  Swift Flash's shapes and formats only: gate/up [1280 x 2560] IQ2_XXS,
// IQ2_XS, IQ2_S, IQ3_XXS or IQ3_S, down [2560 x 640] Q2_0 or IQ4_NL (the K loop stops at 640).
#pragma once

#include <cstddef>
#include <cstdint>

namespace strata::prefill::ws {

/// The products of a layer, with STRATA_MMQ_WS: 0 never (the per-group MMQ, the A/B), 1 (the default) gate/up and down,
/// 2 the down product only (gate/up stays on the per-group MMQ).  0 also where the build has no kernels (HIP, no ggml
/// sources) or a visible GPU is not sm_60 (the kernels were tuned and checked on the PH402 only).
int mode();
/// mode() != 0 and both formats have kernels.
bool supported(int gu_type, int d_type);
/// The weight-stationary product of `ggml_type` beats the per-group in-place MMQ at u rows per routed expert (the
/// measured ratios, moe_ws.cu; u is clamped to the measured 2.5 .. 160).
bool faster(int ggml_type, double u);
/// Every gate/up and down matrix starts this aligned (the IQ4_NL down reads 8-byte pieces; the expert cache's slots are
/// 256-byte aligned and the Swift gate/up sizes multiples of 8).
constexpr uint64_t kAlign = 8;

/// The CTAs' work: rows [row0, row0 + rows) of Xq / GU / Hq / Dm, all of the expert at position j of the layer's order
/// (MMQ's: its xoff entry), at most kChunkRows.  A CTA takes one chunk and one 128- or 256-row piece of the weights.
struct alignas(16) Chunk {
    int32_t j, row0, rows, pad;
};
constexpr int kChunkRows = 64;
/// The most chunks a layer of `rows` routed rows over at most `n_expert` experts makes.
inline size_t max_chunks(int64_t rows, int64_t n_expert) { return (size_t) (rows / kChunkRows + n_expert); }
/// The chunks of the n experts of `order` (expert e has cnt[e] rows from bounds[j], j its position), the largest
/// expert first (the long CTAs start first); returns their number (<= max_chunks).
size_t chunks(const int32_t* order, size_t n, const int32_t* cnt, const int32_t* bounds, Chunk* out);

/// One layer: its n routed experts' matrices at w + xoff[j] (gate/up, rows 0-639 gate then up) and w + xoff[j] +
/// down_off (down), xoff on the device in the layer's order (Product::xoff's table, all n entries); `rows` = T*K.
struct Layer {
    int gu_type = -1, d_type = -1;
    const void* w = nullptr;
    const int64_t* xoff = nullptr;
    int64_t down_off = 0;
    const Chunk* chunks = nullptr;   ///< on the device
    int64_t nchunks = 0;
    int64_t rows = 0;
    int n = 0;
    const void* xq = nullptr;        ///< the gate/up's q8_1 rows (mmq::quantize_scatter's, rows of them)
    float* gu = nullptr;             ///< [rows][1280], the rows MMQ writes through identity ids
    const void* hq = nullptr;        ///< H's q8_1 rows: ONE mmq::swiglu_quant over all `rows` rows of gu
    float* dm = nullptr;             ///< [rows][2560]
};
/// The layer's gate/up into gu, one launch; false (nothing launched) where the per-group MMQ is to run instead (mode 2,
/// or not faster at this u) - the same GU rows either way.
bool gate_up(const Layer& L, void* stream);
/// The layer's down from hq into dm, one launch.
void down(const Layer& L, void* stream);

}  // namespace strata::prefill::ws
