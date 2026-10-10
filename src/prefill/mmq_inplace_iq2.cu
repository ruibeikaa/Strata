// src/prefill/mmq_inplace_iq2.cu - llama.cpp's MMQ with the experts read in place (mmq.cuh mul_mat_q_case_inplace; see
// moe_mmq.hpp mmq::inplace): IQ2_XXS, IQ2_XS and IQ2_S gate/up.
// A few formats per file, so the instances compile in parallel.
#include "mmq.cuh"

#if !defined(__HIPCC__) && defined(DECL_MMQ_CASE_INPLACE)   // a ggml checkout without it: nothing here
DECL_MMQ_CASE_INPLACE(GGML_TYPE_IQ2_XXS, false);
DECL_MMQ_CASE_INPLACE(GGML_TYPE_IQ2_XS, false);
DECL_MMQ_CASE_INPLACE(GGML_TYPE_IQ2_S, false);
#endif
