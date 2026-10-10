// src/prefill/mmq_inplace_iq3.cu - llama.cpp's MMQ with the experts read in place (mmq.cuh mul_mat_q_case_inplace; see
// moe_mmq.hpp mmq::inplace): IQ3_XXS and IQ3_S gate/up.
// A few formats per file, so the instances compile in parallel.
#include "mmq.cuh"

#if !defined(__HIPCC__) && defined(DECL_MMQ_CASE_INPLACE)   // a ggml checkout without it: nothing here
DECL_MMQ_CASE_INPLACE(GGML_TYPE_IQ3_XXS, false);
DECL_MMQ_CASE_INPLACE(GGML_TYPE_IQ3_S, false);
#endif
