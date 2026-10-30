# PH402 baseline (2026-10-30, r14 on upstream v0.2.7.1)

The engine this branch builds is the one serving Swift 1.5 IQ3_XXS (route 8110, 1M context) and the abliterated
Swift 1.5 IQ3_S (route 8111) on 4× GP100 (2× PH402 SKU 200, sm_60, 1050 MHz / 140 W, Windows 11, driver 581.80, TCC).
It is upstream v0.2.7.1 (46c152a13, engine 0.1.53) plus the PH402 patches below. The previous baseline
(`ph402-baseline-0141`, 2026-10-09) was 0.1.41 + the same family of patches; everything that branch carried was
re-applied here (38 cherry-picks, two merge fixes), then the 10-29/10-30 releases were added on top. Every release
on this branch was gated on content + token + **state** hashes equal to the previous release on both models.

Production releases on this branch: `ph402-r13-p53-20261029` (868c98dc3, the rebase) and `ph402-r14-route-20261030`
(204a10c4a, + route_loop). The local branches these came from (`p53` fb59bec0a, `s2-fixedcost` 9b04f033a) have the same
trees; the hashes differ only because one early commit's author e-mail was normalised before publishing.

## What is on top of v0.2.7.1

| what | switch (production value) | notes |
|---|---|---|
| Pascal Q8_0 decode GEMVs for the dense shards and the MTP drafter (loads-first q8sm60p) | `STRATA_Q8_SM60=1 STRATA_DENSE_PROTO=1 STRATA_MTP_Q8_SM60=1 STRATA_Q8F=1 STRATA_ARGMAX_ROWS_SM60=1` | `tools/q8_dense_gguf.py` makes the Q8_0 dense shards |
| 4-bit gate/up expert slots (IQ3_S / IQ3_XXS, and IQ2_XS / IQ2_XXS since r11), weight-stationary prompt experts | `STRATA_EXPERT_NIBBLE=1 STRATA_EXPERT_NIBBLE_XS=1 STRATA_NIB_GU_PROTO=1 STRATA_MMQ_WS=1 STRATA_WS_FAST=1 STRATA_WS_V3=1 STRATA_DOWN_WS=1` | mutually exclusive with upstream's MMQ_TAB: build with `-DSTRATA_MMQ_TAB=OFF` |
| multi-row verify kernels, HC read variants chosen by a per-card bitwise self-check | `STRATA_HC_MROW=3 STRATA_GR_UP75_MIN=2 STRATA_GR_NORM_FAST=1` | |
| QSA: long-context select in 2 kernels, decode-window select + attention prototype, prompt attention on the sm_6x score / value kernels | `STRATA_QSASEL=1 STRATA_QSADEC_PROTO=1 STRATA_SELECT_LONG=1 STRATA_SELECT_LONG_AT=8192 STRATA_SCORES_THREAD=1 STRATA_PATTN_SM60=1 STRATA_ATTN_MERGE_V2=1` | |
| prompt read: pipeline-sized chunks (b/a 70, floor 256), FP32 resident weight copies, cuBLAS ALGO6 where bitwise, resident-expert sort, fused swiglu / combine writes | `STRATA_PREFILL_PIPE=70 STRATA_PREFILL_PIPE_FLOOR=256 STRATA_DENSE_F32=1 STRATA_DENSE_F32_RESIDENT=1 STRATA_F32_RESIDENT_RESERVE_MIB=1536 STRATA_GEMM_ALGO6=2 STRATA_MMQ_RESIDENT_SORT_NE=1 STRATA_PREFILL_SWIGLU_QUANT=1 STRATA_PREFILL_COMBINE_WRITE=1 STRATA_IQ_STAGE_TABLES=1` | |
| pipelined decode windows at depth 3 on the 4-stage split (pl_sched), speculative launch gate, GDN snapshot ring, per-stage async commit, the layer split's draft K/V batched | `--pipeline-windows 3 STRATA_PIPELINE_DEBUG=1 STRATA_PIPELINE_THETA=0.1 STRATA_PIPELINE_SNAP_RING=1 STRATA_SPLIT_COMMIT_ASYNC=1 STRATA_MTP_PF_BATCH=1 STRATA_DF_BRANCH=1` | lookup-chain code is present but `STRATA_LKCHAIN` must stay unset |
| the session a request leaves does not depend on the window depth (r12) | `STRATA_STATE_CLEAN=4` | the fix that let 8110 run depth 3 with the state hash unchanged |
| conversation cache: parks reuse the checkpoint buffers they hold, uniform checkpoints, MSVC fill / copy on memset again (r13) | `STRATA_CKPT_PARK_REUSE=1 STRATA_CKPT_UNIFORM=16` | the v0.2.7.1 `DefaultInit` allocator made MSVC fill byte by byte (parks 20-33% slower) - fixed on this branch |
| the decode route (router top-10) kernel keeps its rank loop: 402 SASS instructions instead of 1518, same operations in the same order (r14) | `STRATA_S2_ROUTE_SMALL=2` | the other `STRATA_S2_*` switches are present, default off and not used |
| trim stage weights, free the host expert arena once every expert is resident, PLE table in RAM | `STRATA_STAGE_TRIM=1 STRATA_ARENA_FREE=1 --ple-io ram` | |
| sm_60 dp4a fallback with an early-clobber accumulator (upstream #4274) | always | |

Default-off and unused on this branch (kept for the record): `STRATA_LKCHAIN*` (lookup chaining: copy-class requests
+7%, ordinary -1%), `STRATA_SPEC_CUMP` / `STRATA_SPEC_CAP` (draft-row gates: seesaw across request classes),
`STRATA_S2_TOPK_PSCAN` / `STRATA_S2_HC_DOWN` / `STRATA_S2_HC_PF` (no gain), `STRATA_PARK_RAMLIKE`, `STRATA_STATE_HASH_*`
diagnostics. The `--tp2-pairs` work (board-internal two-die pairs, "P2") lives on separate local branches and is not here:
bit-identical, but 15% slower than the 4-stage pipeline.

## Build (sm_60)

CUDA 12.9, Visual Studio 2022 Build Tools (the device code needs C++20), Ninja, the pinned llama.cpp checkout:

    cmake -G Ninja -S . -B build-sm60 -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_TESTS=OFF
          -DSTRATA_BUILD_CONVERSATION_TESTS=OFF -DCMAKE_CUDA_STANDARD=20 -DCMAKE_CUDA_STANDARD_REQUIRED=ON
          -DCMAKE_CUDA_RUNTIME_LIBRARY=Shared -DGGML_CCACHE=OFF -DSTRATA_EXPERIMENTAL_SM60=ON -DSTRATA_MMQ_TAB=OFF
          -DCMAKE_CUDA_ARCHITECTURES=60 -DSTRATA_GGML_DIR=<the llama.cpp checkout setup uses>
    cmake --build build-sm60 --target strata

## How it is served

Serve from the `strata-serve` tree (0.1.41 serve with the graceful-exit and reasoning-close-retry patches). Engine
arguments on top of the setup template (both routes; 8110 at 1M context with yarn rope scale 4):

    --layer-split 12,24,36 --trim-stage-weights --kv int8 --spec 4 --spec-min-p 0.85 --mtp-window 8192
    --pipeline-windows 3 --short-read 24 --prefill auto --ple-io ram --conversation-cache-mib 32768
    --conversation-cache-slots 16 --mtp <rt-owner-v1 draft head> --mtp-draft-vocab <68K owner vocabulary>

plus every switch in the table above. The dies' application clocks are locked at 715,1050 at boot.

## Measured (2026-10-29/30, port 8190 A/B against the previous release, replies and state identical)

Decode, tok/s (16-request set: 8 ordinary answers, 6 real file edits / writes, 2 copy-class):

| | 8110 Swift IQ3_XXS | 8111 SwiftAbl IQ3_S |
|---|---|---|
| ordinary answers | 76.3 (cached-tail 79.3) | 69.4 (70.3) |
| file edits / writes | 116 | 116 |
| copy-class | 143 | 131 |
| r14 route_loop on top | +0.95..+1.58% every request (+1.30% weighted) | +0.84..+1.90% (+1.39%) |

Prompt reads, tok/s: 1.1K 554 / 531, 7.5K 928 / 870, 33K 1094 / 1071, 108K 1166 / 1148, 202K 1151 / 1136.

Against the 2026-10-24 baseline (`ph402-stack-main` era, same 16 requests, same harness) 8110 decode is +16% token-weighted
(ordinary +12%, edits +20%, copy +23%) before r14; reads +2..6%.

## What was tried on this hardware and closed (so it is not re-done)

Board-internal TP2 / two-stage pairs (-15% in the real pipeline); second-candidate speculation (the pipeline windows
share the K/V); head split across dies; moving the draft chain; changing the split points (reads -7%); persistent / grid-barrier
kernels; HC prefetch; FP16 prompt GEMMs (not bit-exact); per-row draft gates (seesaw); multi-stream batch (-23..43% single
stream). Instruction-cache cold starts exist only on single-block kernels (route, block_topk, the draft-step graph); the
multi-block kernels run at their bench speed in the graph.
