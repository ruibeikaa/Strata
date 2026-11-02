// include/strata/prefill/moe_ws.hpp - GP100 (sm_60, the STRATA_EXPERIMENTAL_SM60 build): the prompt path's expert
// products of a layer whose experts are all resident in VRAM, read in place there and weight-stationary: ONE launch
// for the layer's gate/up and ONE for its down instead of a gate/up, swiglu and down MMQ launch per 16-expert group.
// A CTA holds 128 (or 256) weight rows of one expert and runs them against up to 64 of the expert's activation rows, a
// thread per weight row: each weight block is decoded once per pass instead of once per MMQ tile, and an expert's
// rows are its own count, not the group's largest.  Every product keeps MMQ's int8 values, scales and per-32-value
// epilogue in its K order (the down K loop stops at 640), so GU, H and Dm are the per-group MMQ's bits.  Swift
// Flash's shapes and formats only: gate/up [1280 x 2560] IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS or IQ3_S, down [2560 x 640]
// Q2_0 or IQ4_NL.
#pragma once

#include <cstddef>
#include <cstdint>

namespace strata::prefill::ws {

/// The products of a layer: 1 with STRATA_MMQ_WS=1 (opt-in) gate/up and down, else 0 (the per-group MMQ).  0 also where
/// the build has no kernels (HIP, no ggml sources, not the STRATA_EXPERIMENTAL_SM60 build) or a visible GPU is not
/// cc 6.0 (the kernels were tuned and checked on GP100 only).
int mode();
/// mode() != 0 and both formats have kernels.
bool supported(int gu_type, int d_type);
/// STRATA_WS_FAST=1 (opt-in, read once): the 32-token tier's gate/up on 4-bit slots (IQ2_S, IQ3_XXS, IQ3_S) and its
/// IQ4_NL down run kernels that issue fewer instructions per pass - 16-byte activation staging, a switch into the
/// unrolled token groups, a per-token tail, dy transposed, IQ3_S's decode fused - with every token's operations and
/// their order unchanged, so GU and Dm are the same bits.  0 (unset): the kernels as before.
int fast();
/// Benches and tests: 0 / 1 overrides fast(), -1 goes back to the environment.
void set_fast(int v);
/// STRATA_WS_V2=1 (opt-in, read once): the gate/up on 4-bit slots (IQ2_S, IQ3_XXS, IQ3_S) takes moe_ws_v2.cuh's
/// 32-token kernels from lower rows per expert than the 32-token tier did - IQ2_S from u 17 (was 25) with each thread
/// reading its row's item codes from global memory, IQ3_XXS from u 9 (was 18), IQ3_S from u 7 (was 18), its codes read
/// per thread from u 16, four CTAs an SM - where they beat the 16-token kernels / STRATA_WS_FAST's (WS2.md).  Every
/// output's operations and their order are unchanged, so GU is the same bits.  Independent of fast() (the down keeps
/// fast()'s kernel); where both are set the V2 picks run for the 4-bit gate/up.  0 (unset): as before.
int v2();
/// Benches and tests: 0 / 1 overrides v2(), -1 goes back to the environment.
void set_v2(int v);
/// STRATA_WS_V3=1 (opt-in, read once): the 16-token tier (gate/up on the 4-bit slots below the 32-token tier's
/// thresholds, and the IQ4_NL down below u 20) runs STRATA_WS_FAST's kernels - 16-byte activation staging, a switch
/// into the unrolled token groups, dy transposed, whole token pairs plus a one-token tail - so a pass of 15 rows runs
/// 15 tokens instead of 16.  Every output's operations and their order are unchanged, so GU and Dm are the same bits.
/// 0 (unset): the kernels as before.
int v3();
/// Benches and tests: 0 / 1 overrides v3(), -1 goes back to the environment.
void set_v3(int v);
/// The weight-stationary product of `ggml_type` beats the per-group MMQ at u rows per routed expert (the measured
/// ratios, moe_ws.cu; u is clamped to the measured 2.5 .. 160).
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
    /// STRATA_EXPERT_NIBBLE=1: the gate/up experts are 4-bit codes in their slots (strata/core/expert_nibble.hpp's
    /// layout; down_off = its kDownOff), read by moe_ws_gu_nib.cu's kernels - the GGUF kernels' GU rows, bit for bit.
    /// false: the GGUF blocks, as before.
    bool gu_nib = false;
    /// STRATA_PF_HFMA_EXP=1 (hfma_exp() below): the layer's input as FP16 rows, one row a token ([hf_tokens][2560], the
    /// prompt path's `mixed_h`), the routed row -> token table (`hf_src`, rows of it) and the layer's number (the report's
    /// per-layer maximum).  Set by the caller before hfma_exp_plan; `hf_tier` is its answer: 0 = the int8 kernels run
    /// (and `xq` must have been quantized), 16 / 32 = gate_up runs the FP16 route (Xq is not read).
    const void* hf_xh = nullptr;
    const int32_t* hf_src = nullptr;
    int64_t hf_tokens = 0;
    int hf_layer = -1;
    int hf_tier = 0;
    /// STRATA_PF_HFMA_R2=1 (with STRATA_PF_HFMA_EXP): hfma_exp_plan's answer for a layer it took - 1 = the two-rows-a-thread kernel
    /// of moe_ws_hf_r2.cuh runs it at `hf_tier`, 0 = the compiler kernel of moe_ws_hf.cuh (what the layer's u favours).
    int hf_r2 = 0;
    /// STRATA_PF_HFMA_DOWN=1 (hfma_down() below): the layer's number for the report (set by the caller before
    /// hfma_down_plan) and the plan's answer `hd_tier`: 0 = the int8 swiglu_quant + down run, 16 / 32 = down_hf_prep +
    /// down_hf (H as FP16 rows in `gu`'s own rows).
    int hd_layer = -1;
    int hd_tier = 0;
    /// STRATA_PF_HFMA_R2_DOWN=1 (with STRATA_PF_HFMA_DOWN): hfma_down_plan's answer for a layer it took - 1 = the two-rows-a-thread
    /// kernel of moe_ws_dn_hf_r2.cuh runs it (Q2_0 at 16 tokens a pass, IQ4_NL at 32; `hd_tier` says which), 0 = the one-row kernel
    /// of moe_ws_dn_hf.cuh.
    int hd_r2 = 0;
};
/// The layer's gate/up into gu, one launch; false (nothing launched) where the per-group MMQ is to run instead (mode 0,
/// or not faster at this u) - the same GU rows either way.  A gu_nib layer has no per-group MMQ (the caller stops on
/// false).
bool gate_up(const Layer& L, void* stream);

/// STRATA_PF_HFMA_EXP=1 (opt-in, read once; default off; needs mode() != 0, i.e. STRATA_MMQ_WS=1 on cc 6.0, and the 4-bit slots
/// of STRATA_EXPERT_NIBBLE; independent of STRATA_PF_HFMA, which it works with): the gate/up of a 4-bit layer on FP16 HFMA2
/// pairs (moe_ws_hf.cuh) - the FP16 image of the layer's input against weights decoded in the kernel, 64-product FP16 lane
/// sums widened to FP32 every 128 K values; the same GU rows' contract (an FP32 [rows][1280] for swiglu_quant), not their
/// bits.  Down, decode and everything else keep their kernels.  0 = off: every function below is then not called.
int hfma_exp();
/// For a layer that took the weight-stationary route: whether its gate/up runs the FP16 route (sets L.hf_tier and counts the
/// taken launch, or the fallback with its reason and format).
bool hfma_exp_plan(Layer& L);
/// A layer of `gu_type` that did not reach hfma_exp_plan (counted with its reason: the per-group MMQ ran it).
void hfma_exp_left(int gu_type, bool nibble, int64_t rows, const char* why);
/// The max-reduce of the layer's FP16 input that sets the weights' shift (before gate_up, on the same stream).
void hfma_exp_prepass(const Layer& L, void* stream);
/// One line per device with the launches (by format and tier), what fell back (by format and reason), max |x|, the shifts
/// and the non-finite counts since the last call; `tokens` = the read's length.  Prints nothing when no layer was seen.
void hfma_exp_report(int64_t tokens);
/// The layer's down from hq into dm, one launch.
void down(const Layer& L, void* stream);

/// STRATA_PF_HFMA_DOWN=1 (opt-in, read once; default off; needs mode() != 0, i.e. STRATA_MMQ_WS=1 on cc 6.0; independent of
/// STRATA_PF_HFMA_EXP and STRATA_PF_HFMA): the down of a weight-stationary layer (Q2_0 or IQ4_NL experts) on FP16 HFMA2 pairs
/// (moe_ws_dn_hf.cuh): H, the swiglu of the layer's gate/up rows, as FP16 rows (scaled by a power of two a row) in `gu`'s own
/// rows instead of q8_1 blocks, the weights times their block scales as FP16 (shifted by a power of two a row and step so no
/// lane of 64 products can overflow), the sums FP16 per 128 values and widened to FP32.  The same Dm contract (an FP32
/// [rows][2560]), not its bits.  0 = off: every function below is then not called.
int hfma_down();
/// For a layer that took the weight-stationary route (after gate_up): whether its swiglu + down run the FP16 route (sets
/// L.hd_tier and counts the taken launch, or the fallback with its reason and format).
bool hfma_down_plan(Layer& L);
/// A layer of `d_type` that did not reach hfma_down_plan (counted with its reason).
void hfma_down_left(int d_type, int64_t rows, const char* why);
/// swiglu of `gu`'s rows into FP16 H in place (replaces mmq::swiglu_quant for a layer with hd_tier != 0).
void down_hf_prep(const Layer& L, void* stream);
/// The FP16 down from the rows down_hf_prep left in `gu` into dm, one launch.
void down_hf(const Layer& L, void* stream);
/// One line per device with the launches (by format and tier), the fallbacks, the largest |H|, the weights' shifts, the
/// non-finite counts and (with STRATA_PREFILL_TIMING) the kernels' own time and G MAC/s since the last call; `tokens` = the
/// read's length.  Prints nothing when no layer was seen.
void hfma_down_report(int64_t tokens);

}  // namespace strata::prefill::ws
