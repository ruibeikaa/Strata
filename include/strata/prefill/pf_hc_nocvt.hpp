// include/strata/prefill/pf_hc_nocvt.hpp - STRATA_PF_HC_NOCVT (default off; needs STRATA_PF_HFMA_HC and STRATA_PF_HFMA_SMALL, GP100):
// the hyper-connection read's two FP16 GEMM activations (down: xn16, T x 10240; up: lo16, T x 320) need their max |x| per 1024-row slice and an
// FP16 image.  Both used to be found by a pass over the BF16 image (xcvt_kernel: read it, convert, write the FP16 image, atomicMax; then
// cvt_fix_kernel).  With the switch the kernels that WRITE the image (gr_norm_rs, gr_write_norm_rs, moe_combine_write_norm_rs and its v2,
// gr_silu) also write the FP16 image (the very conversion xcvt_kernel does: half_rn(float(bf16))) and fold the max BF16 magnitude bits of each
// slice into a slot (an atomicMax of the warp's max, taken only when it beats the slot); the product's prep kernel (cvt_slot_kernel) is
// cvt_fix_kernel reading the slot and the product reads the FP16 image where it lies.
// xn16 (down side) is then NOT written as BF16 at all (the FP16 image replaces it, so the writers move the bytes they always did): its only
// readers are the HFMA products, and everything else that reads it as BF16 (an old-route product, the layer-major loop, a declined call) goes
// through Gemm::bf16, which first rebuilds it with hc_img_materialize from R, the row scales and the norm weights - bf16(R * rs * w), the very
// expression the writers rounded (gr_mix_r relies on the same identity) - and a slice that needs a shift is made from those the same way.
// lo16 (up side, 1.5 MB a chunk) keeps its BF16 image.  STRATA_PF_HC_NOCVT_CHECK=1 keeps the BF16 image too and runs the old route beside the new one,
// comparing the max, the image and the rebuilt BF16 image.
// A writer that does not record (the cvec variant, the unfused gr_norm, a padded stride, T past 16384, f16img, xn16_lo) drops the record of its image
// (and writes the BF16 image), and a product whose image has no valid record (other writer, other rows, other stream) takes the old route, counted per read.
// Off, every call is what it was.
#pragma once

#include <cstdint>
#include <cstdlib>

#include "strata/prefill/pf_small.hpp"

namespace strata::prefill {

inline bool pf_hc_nocvt() {
    static const bool on = [] {
        const char* e = std::getenv("STRATA_PF_HC_NOCVT");
        const char* h = std::getenv("STRATA_PF_HFMA_HC");
        return e != nullptr && std::atoi(e) != 0 && h != nullptr && std::atoi(h) != 0 && pf_hfma_small();
    }();
    return on;
}
inline bool pf_hc_nocvt_check() {
    static const bool on = [] {
        const char* e = std::getenv("STRATA_PF_HC_NOCVT_CHECK");
        return e != nullptr && std::atoi(e) != 0;
    }();
    return on;
}

constexpr int kHcImgSlots = 16;   // 1024-row slices: chunks up to 16384 rows

struct HcImg {
    uint16_t* xh = nullptr;     // rows x k FP16 images (half_rn(float(bf16)) of the BF16 image), token stride k
    unsigned* slots = nullptr;  // kHcImgSlots words: max BF16 magnitude bits << 16 (abs_bits<false>) of each 1024-row slice, 0 = all zero
    const float* R = nullptr;   // the down image's source: xn16 = bf16(R[i] * rs[i / 2560] * wn[i % 10240]) (null for lo16)
    const float* rs = nullptr;
    const float* wn = nullptr;
    bool bf16_ok = true;        // the BF16 image is in memory (else hc_img_materialize makes it)
    int keep = 1;               // writer: also write the BF16 image
};
/// Writer side (the current device, `stream`): the record of the image at `x` (rows x k) is made valid with its slots cleared; false (and the record dropped)
/// where it cannot be (sizes, allocation).  R / rs / wn: its source (null for lo16, whose BF16 image is always written).  `drop`: a writer that does not
/// record wrote `x` (BF16): its record, if any, is stale.
bool hc_img_begin(const void* x, int64_t rows, int64_t k, void* stream, HcImg* out, const float* R, const float* rs, const float* wn);
void hc_img_drop(const void* x);
/// Consumer side: the valid record of `x` for exactly these rows, on this stream; `why` says why not.
bool hc_img_take(const void* x, int64_t rows, int64_t k, void* stream, HcImg* out, const char** why);
/// Anything that reads the BF16 image at (or inside) `x` calls this first, on its stream: a valid record without its BF16 image gets it rebuilt.
void hc_img_materialize(const void* x, void* stream);
long long hc_img_materialized();   // how many times, since the start

}  // namespace strata::prefill
