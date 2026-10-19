// src/prefill/moe_ws_gu_nib.cu - P1 part D (K1), M3: the weight-stationary gate/up product of a layer whose experts
// are 4-bit codes in their VRAM slots (STRATA_EXPERT_NIBBLE=1; the layout and the code: strata/core/expert_nibble.hpp).
//
// To come (D_DESIGN.md section 6): one template nib::gu_kernel<TPP, CVT, TG, FMT> for IQ2_S, IQ3_XXS and IQ3_S, the
// three TPP tiers each, on gate G2's S1W staging - a lane per (row lane >> 2, quarter lane & 3), one LDG.128 of codes
// each, rows 20 words apart in shared memory (STS.128 / LDS.128); the scale word and d read by the row's own thread.
// FMT changes only the nibble -> int8 word (IQ2_S one PRMT, IQ3_XXS magnitude PRMT + sign mask, IQ3_S 2u - 15 per
// byte) and dx from (ls, d); split2 -> mad2 -> mac16, stage_act and the epilogue are moe_ws.cuh's, so GU is the
// production WS kernels' bit for bit (gate G2: 2,872,320 rows, 0 different).
//
// Empty until M3: nothing here is linked into the engine, and with the switch off nothing would call it.
#include "strata/core/expert_nibble.hpp"
#include "strata/prefill/moe_ws.hpp"
