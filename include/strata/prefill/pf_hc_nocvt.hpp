// include/strata/prefill/pf_hc_nocvt.hpp - STRATA_PF_HC_NOCVT (default off; rides with STRATA_PF_HFMA_HC + STRATA_PF_HFMA_SMALL, GP100):
// the hyper-connection read's two FP16 GEMM activations (down: xn16, T x 10240; up: lo16, T x 320) need their max |x| per 1024-row slice and an
// FP16 image.  Both used to be found by a pass over the BF16 image (xcvt_kernel: read it, convert, write the FP16 image, atomicMax; then
// cvt_fix_kernel).  With the switch the kernels that WRITE the BF16 image (gr_norm_rs, gr_write_norm_rs, moe_combine_write_norm_rs and its v2,
// gr_silu) also write the FP16 image (the very conversion xcvt_kernel does: half_rn(float(bf16))) and the max BF16 magnitude bits of each slice into
// a slot (an atomicMax of the block's max, taken only when it beats the slot), and the product's prep kernel (cvt_slot_kernel) is cvt_fix_kernel
// reading the slot: the BF16 image stays as it was, so a slice that needs a shift is rewritten from it by the same code as before.  Bit for bit the
// old route's image, max, shift and counts (STRATA_PF_HC_NOCVT_CHECK=1 runs the old route beside it and compares).
// A writer that does not record (the cvec variant, the unfused gr_norm, a padded stride, T past 16384, f16img, xn16_lo) drops the record of its image, and a
// product whose image has no valid record (other writer, other rows, other stream) takes the old route, counted per read.
// Off, every call is what it was.
#pragma once

#include <cstdint>
#include <cstdlib>

namespace strata::prefill {

inline bool pf_hc_nocvt() {
    static const bool on = [] {
        const char* e = std::getenv("STRATA_PF_HC_NOCVT");
        return e != nullptr && std::atoi(e) != 0;
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
};
/// Writer side (the current device, `stream`): the record of the BF16 image at `x` (rows x k) is made valid with its slots cleared; false (and the record
/// dropped) where it cannot be (sizes, allocation).  `drop`: a writer that does not record wrote `x`: its record, if any, is stale.
bool hc_img_begin(const void* x, int64_t rows, int64_t k, void* stream, HcImg* out);
void hc_img_drop(const void* x);
/// Consumer side: the valid record of `x` for exactly these rows, on this stream; `why` says why not.
bool hc_img_take(const void* x, int64_t rows, int64_t k, void* stream, HcImg* out, const char** why);

}  // namespace strata::prefill
