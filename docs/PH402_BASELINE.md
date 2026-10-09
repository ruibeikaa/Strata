# PH402 baseline (2026-10-09, next8)

The engine this branch builds is the one serving Swift 1.5 IQ3_XXS (1M context) and Qwen3.8-Flash-Next
IQ3_XXS (32K context) on 4× GP100 (2× PH402 SKU 200, sm_60, 1050 MHz / 140 W). It is upstream 0.1.41 (fb58e0d) plus two
open upstream PRs for pipelined decode windows and the PH402 patches below, some of which may never be merged upstream;
this branch keeps them together as a known-good baseline. Branch ph402-next has the same PH402 patches on 0.1.40.4
(next7), without the pipelined windows.

## What is on top of 0.1.41

| commit(s) | what | upstream |
|---|---|---|
| 9e67094a | a pipelined verifier leaves one-token commits to the host verdict (CC-David-CC) | PR #1674 (open) |
| c467317a | `--pipeline-windows 2` on three or more layer-split stages (Cass67) | PR #1656 (open) |
| c8fb4876 | QSA block scorer, tiled and bit for bit the warp kernel's (below sm_80; default on 6.x) | local only |
| 02ad1ee8 24f7f712 ffb6fcd9 80131a7d 3a70155d | IQ3_XXS / IQ3_S codebooks + sign masks in shared memory (default on 6.0); Pascal Q8_0 decode GEMV (`STRATA_Q8_SM60=1`); `tools/q8_dense_gguf.py` | PR #1424 (open) |
| 1801d32a 6a48d8c4 1c8cc07c | `--prefill-pipe` / `STRATA_PREFILL_PIPE`: prompt chunks sized for a layer split's pipeline, b/a from setup's `--calibrate` (the setup/calibrate tool changes are not carried: 0.1.41's own are kept) | PR #1441 (open; the PR has since moved on) |
| 0f9beb97 | `STRATA_MMQ_RESIDENT_SORT_NE` for native-layout packs too | PR #1660 (open) |
| 72d1acac | MoE input quantized once per token and scattered (sergqwer) | PR #1368 (open) |
| f3f04d9b | dense Q8_0 / IQ4_XS to FP16 with 16-byte stores, fused swiglu + q8_1 and combine + hc write (sergqwer); merged with 0.1.41's peer-sum combines (the fused combine+write runs in the plain branch only) | PR #1525 (open) |
| 34d090ac | async commit on the layer split, as one GPU does (`STRATA_COMMIT_SYNC=1` restores the host wait) | local only |
| 60113077 | the MTP drafter's draft head (packed in place) and its 9 projections on the Pascal Q8_0 GEMV (`STRATA_MTP_Q8_SM60=0` off) | local only |
| e344d920 98ea126f ba156160 | long-context QSA select: decode top-k over up to 32 CTAs per query and a shuffle-free decode block scorer, the same ids and score bits; only in window graphs past 65536 cells, pipelined windows included (both variants captured up front) (`STRATA_SELECT_LONG_AT`; `STRATA_TOPK_MULTI=0` / `STRATA_SCORES_DX=0` off) | local only |

GR_FAST (upstream aaa323fe) is in 0.1.41 itself; off on sm_60 (bitwise, but flat on GP100).

## Build (sm_60)

CUDA 12.9, Visual Studio 2022 host compiler (the device code needs C++20), Ninja:

    cmake -G Ninja -S . -B build-sm60 -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_TESTS=OFF
          -DCMAKE_CUDA_STANDARD=20 -DCMAKE_CUDA_STANDARD_REQUIRED=ON -DCMAKE_CUDA_RUNTIME_LIBRARY=Shared
          -DGGML_CCACHE=OFF -DSTRATA_EXPERIMENTAL_SM60=ON -DCMAKE_CUDA_ARCHITECTURES=60
          -DSTRATA_GGML_DIR=<the llama.cpp checkout setup uses>
    cmake --build build-sm60 --target strata

The RTX 3090 Ti routes run an sm_86 build of ph402-next (CUDA 13.0, not portable); the Pascal-only patches are inactive
there and the long-context select is turned on by their launcher (`STRATA_TOPK_MULTI=1 STRATA_SCORES_DX=1`).

## How it is served

Serve 0.1.41. Dense matrices and the head come from Q8_0 copies of the shards (`tools/q8_dense_gguf.py`; the experts
keep their offsets, so the pack is reused). Engine arguments on top of the setup template (Swift 1.5):

    --layer-split 12,24,36 --max-context 1048576 --rope-scaling yarn --rope-scale 4 --kv int8
    --spec 4 --spec-min-p 0.85 --mtp-window 8192 --pipeline-windows 2 --prefill auto
    --conversation-cache-mib 16384 --conversation-cache-slots 3

Flash-Next the same without the conversation cache, at 32K context. Environment (both):

    STRATA_Q8_SM60=1                 Pascal Q8_0 GEMV for the Q8_0 dense shards
    STRATA_PREFILL_PIPE=768          b/a from setup's calibration on this rig (+62% on 2K + 8K reads)
    STRATA_MMQ_RESIDENT_SORT_NE=1    row-count sort of resident experts before MMQ grouping
    STRATA_DF_BRANCH=1               the mixer's side branches inside the verify graph

The dies' application clocks are locked at 715,1050 (`nvidia-smi -ac`) at boot; without it the idle stages of a layer
split fall to 759 MHz. The launchers read the PLE shard once in the background after start (cold, the first replies
decode up to 25% slower).

## Measured (Swift 1.5, 2026-10-09, warm; every reply identical bit for bit across the three arms)

| | next7 (0.1.40.4) | next8, `--pipeline-windows 0` | next8 (served) |
|---|---|---|---|
| decode, short code prompt | 40.3 tok/s | 40.5 | **51.2** |
| decode, short reasoning prompt | 42.5 | 42.7 | **51.2** |
| decode, 512-token code answer | 47.5 | 47.6 | **61.3** |
| decode after a 35K read | 43.0 | 43.2 | **49.5** |
| decode at 300K | 41.8 | 41.7 | **50.3** |
| 1.2K / 35K / 301K prompt, fresh | 5.1 / 51.0 / 368 s | 5.0 / 51.1 / 371 s | 5.0 / 51.1 / 371 s |

Pipeline statistics (`STRATA_DECODE_TIMING=1`): about half the guessed windows hold; a held window costs 21-25 ms
against 38-49 ms for a fresh one, and 25-30% of the windows find the draft chain late (the draft layer runs on the last
stage's die beside its verify). Flash-Next, warm: 46.9-49.9 tok/s on the short prompts, 45.3 on 512-token code
(its drafts are accepted less often: 63-69% against Swift's ~90%).
