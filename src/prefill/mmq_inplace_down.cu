// src/prefill/mmq_inplace_down.cu - llama.cpp's MMQ with the experts read in place (mmq.cuh mul_mat_q_case_inplace; see
// moe_mmq.hpp mmq::inplace): Q2_0 and IQ4_NL down (K 640: the K loop stops at the matrix's end).
// A few formats per file, so the instances compile in parallel.
#include "mmq.cuh"

#if !defined(__HIPCC__) && defined(DECL_MMQ_CASE_INPLACE)   // a ggml checkout without it: nothing here
DECL_MMQ_CASE_INPLACE(GGML_TYPE_Q2_0, true);
DECL_MMQ_CASE_INPLACE(GGML_TYPE_IQ4_NL, true);
#endif
