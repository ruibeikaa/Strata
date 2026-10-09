# PH402 baseline (2026-10-09)

The engine this branch builds is the one serving Swift 1.5 IQ3_XXS on 4× GP100 (2× PH402 SKU 200, sm_60, 1050 MHz
/ 140 W) at 1M context. It is upstream 0.1.40.4 (6674a00) plus the commits below, some of which may never be merged
upstream; this branch keeps them together as a known-good baseline.

## What is on top of 0.1.40.4

| commit(s) | what | upstream |
|---|---|---|
| 5d55847f | QSA block scorer, tiled and bit for bit the warp kernel's (below sm_80; default on 6.x) | local only |
| b898925f 4883097d 431cec86 7ff55e3f 3246508e | IQ3_XXS / IQ3_S codebooks + sign masks in shared memory (default on 6.0); Pascal Q8_0 decode GEMV (`STRATA_Q8_SM60=1`); `tools/q8_dense_gguf.py` | PR #1424 (open) |
| 79efc0a3 9e12a657 9f9a637c 1cc5dbec cb772657 | `--prefill-pipe` / `STRATA_PREFILL_PIPE`: prompt chunks sized for a layer split's pipeline, b/a from setup's `--calibrate` | PR #1441 (open; the PR has since moved on: one plan per segment, value 1 = `STRATA_PREFILL_EQUAL`'s rule) |
| f10c6ecf | `STRATA_MMQ_RESIDENT_SORT_NE` for native-layout packs too | PR #1660 (open) |
| c3117f22 | MoE input quantized once per token and scattered (sergqwer) | PR #1368 (open) |
| 72223b10 | dense Q8_0 / IQ4_XS to FP16 with 16-byte stores, fused swiglu + q8_1 and combine + hc write (sergqwer) | PR #1525 (open) |
| 1ce05bb0 | async commit on the layer split, as one GPU does (`STRATA_COMMIT_SYNC=1` restores the host wait) | local only |
| 299782c4 | the MTP drafter's draft head (packed in place) and its 9 projections on the Pascal Q8_0 GEMV (`STRATA_MTP_Q8_SM60=0` off) | local only |
| a7334265 a01a789e | upstream aaa323fe (`STRATA_GR_FAST`) and a bench mode for it; off on sm_60: bitwise, but flat on GP100 (T=1 +2 µs, T=4 −5..−9 µs per read) | upstream (main) |
| c194a3b6 61f87598 | long-context QSA select: decode top-k over up to 32 CTAs per query (six capturable kernels, caller scratch) and a shuffle-free decode block scorer, the same ids and score bits; only in window graphs past 65536 cells (`STRATA_SELECT_LONG_AT`; `STRATA_TOPK_MULTI=0` / `STRATA_SCORES_DX=0` off) | local only |

## Build (sm_60)

CUDA 12.9, Visual Studio 2022 host compiler (the device code needs C++20), Ninja:

    cmake -G Ninja -S . -B build-sm60 -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_TESTS=OFF
          -DCMAKE_CUDA_STANDARD=20 -DCMAKE_CUDA_STANDARD_REQUIRED=ON -DCMAKE_CUDA_RUNTIME_LIBRARY=Shared
          -DGGML_CCACHE=OFF -DSTRATA_EXPERIMENTAL_SM60=ON -DCMAKE_CUDA_ARCHITECTURES=60
          -DSTRATA_GGML_DIR=<the llama.cpp checkout setup uses>
    cmake --build build-sm60 --target strata

The two RTX 3090 Ti routes run an sm_86 build of this branch from before f10c6ecf (CUDA 13.0,
`-DCMAKE_CUDA_ARCHITECTURES=86`, not portable); the Pascal-only patches are inactive there.

## How it is served

Dense matrices and the head come from Q8_0 copies of both shards (`tools/q8_dense_gguf.py`, once per shard; the experts
keep their offsets, so the pack is reused). Engine arguments on top of the setup template:

    --layer-split 12,24,36 --max-context 1048576 --rope-scaling yarn --rope-scale 4 --kv int8
    --spec 4 --spec-min-p 0.85 --mtp-window 8192 --prefill auto
    --conversation-cache-mib 16384 --conversation-cache-slots 3

Environment:

    STRATA_Q8_SM60=1                 Pascal Q8_0 GEMV for the Q8_0 dense shards
    STRATA_PREFILL_PIPE=768          b/a from setup's calibration on this rig (+62% on 2K + 8K reads)
    STRATA_MMQ_RESIDENT_SORT_NE=1    row-count sort of resident experts before MMQ grouping
    STRATA_DF_BRANCH=1               the mixer's side branches inside the verify graph

The dies' application clocks are locked at 715,1050 (`nvidia-smi -ac`) at boot; without it the idle stages of a layer
split fall to 759 MHz.

## Measured (2026-10-09, warm, replies identical bit for bit to the runs without each change)

| | |
|---|---|
| 1.2K prompt, fresh | 4.9 s |
| 35K prompt, fresh | about 52 s |
| 301K prompt, fresh | about 369 s (815 tok/s) |
| decode, short context | 40–45 tok/s (next6: 1.4–2.1 ms less per window than without 1ce05bb0 + 299782c4) |
| decode, 300K context | about 41.8 tok/s (next7: 51.9 -> 47.1 ms a window; per QSA call on one die at 300K, T=2: top-k 390 -> 72 µs, scores 217 -> 154 µs) |

Right after a start the PLE table is not yet in the OS file cache, and the first replies decode up to 25% slower until
it is.
