# PH402 / GP100 stack on upstream main

This branch (`ph402-stack-main`) is upstream `main` at 011c9e60 (v0.2.2, engine 0.1.47) plus the whole
GP100 (sm_60) patch stack used on 4x GP100 (two PH402 SKU 200 boards, 32 GB HBM2 per die) in one line of
commits. It is a reference branch, not a pull request: the parts that are under review upstream have their
own branches and PRs (below).

Every item is opt-in. Without its switch the default path is unchanged: no new allocations, captures or log
lines, and the new kernels are compiled only with `-DSTRATA_EXPERIMENTAL_SM60=ON` (not under HIP). HIP, SYCL
and Metal keep their guards; Metal gets stubs for the new CUDA-side symbols.

## Commits (oldest first)

| Commit | Item | Switch |
|---|---|---|
| 595662a0, e8c5526f, 0258ee67, 66955ab6, 9e7e4595, b8725208, 367a50d9 | D: resident IQ2_S / IQ3_XXS / IQ3_S gate/up experts as 4-bit slots (decode and prompt), parity tool, docs | `STRATA_EXPERT_NIBBLE=1` |
| 926d08e2 | D1: a layer split commits without the host wait | `STRATA_SPLIT_COMMIT_ASYNC=1` |
| 70dddf3d | D2: the MTP draft head and projections on the Pascal Q8_0 GEMV | `STRATA_MTP_Q8_SM60=1` (needs `STRATA_Q8_SM60=1`) |
| a10ed9d8, d981a471, c67c18f5 | D3-D5: long-context QSA select (multi-CTA top-k, shuffle-free decode scores) in separate window graphs past 64K cells, pipelined windows included | `STRATA_SELECT_LONG=1` (`STRATA_SELECT_LONG_AT`, default 65536) |
| 5e0bfe58, 9d240808, 68a57fcd, b8c1b2c1, 0b7d9840, 2e6f7cae, 2ce1bdf0 | B1-B7: a third decode window in flight - 11-output draft chain, host-side scheduler (`pl_sched`, CPU test `pl_sched_test`), slot pool, per-stage undo | `--pipeline-windows 3` |
| 2a5b3446 | P1: smallest pipelined prompt chunk | `STRATA_PREFILL_PIPE_FLOOR=N` (default 512) |
| 11b4839d | P2 (author sergiywith): swiglu + q8_1, and the MoE combine + hyper-connection write, fused | `STRATA_PREFILL_SWIGLU_QUANT=1`, `STRATA_PREFILL_COMBINE_WRITE=1` |
| 793207a6, e189b1c2, 6a19d236 | P3-P5: weight-stationary expert kernels for a fully resident layer's prompt pass, one launch per product, 4-bit slots too | `STRATA_MMQ_WS=1` |
| b85c231f | P6: free the host expert arena once every expert is VRAM-resident | `STRATA_ARENA_FREE=1` |
| 87a6146e, dc842d8f | (branch pf-batch1) the WS 32-token tier's gate/up on 4-bit slots and IQ4_NL down with fewer instructions per pass, same bits; `moe_ws_prof_bench` | `STRATA_WS_FAST=1` |
| 84d57f48 | (branch pf-batch1) conversation checkpoints kept evenly spaced over the conversation | `--prompt-cache-uniform` (`--prompt-cache-recent N`), or `STRATA_CKPT_UNIFORM=N` |
| 3a36fb00 | (branch dense-proto) decode Q8_0 projections of the mixers loads-first (q8sm60p), the projections reading the same q8_1 input as one launch (GDN qkv + z, QSA q + k + v), same bits | `STRATA_DENSE_PROTO=1` (needs `STRATA_Q8_SM60=1`; `STRATA_DENSE_PROTO_GROUP=0` keeps one launch each) |
| 26de30af | (branch qsadec-proto) decode-window QSA select (split-tree scorer with the tail and first radix pass fused) and attention (loads-first chunk kernel + merge v2), same bits; `qsadec_bench` (`-DSTRATA_QSADEC_BENCH=ON`) | `STRATA_QSADEC_PROTO=1` (2 select only, 3 attention only; the select part needs `STRATA_SELECT_LONG=1`) |
| 4a287011 | (branch pf-batch2 9b9d00ad) a layer split's draft layer prompt K/V queued on the drafter's stream behind the next chunk, same K/V | `STRATA_MTP_PF_BATCH=1` (`STRATA_MTP_PF_BATCH_ROWS`, default 2048) |
| 6dd7ac30 | (branch ws2) the 4-bit gate/up's 32-token WS kernels from lower rows per expert (IQ2_S u 17, IQ3_XXS 9, IQ3_S 7), IQ2_S / IQ3_S reading their item codes per thread; same bits; `moe_ws_prof_bench --arms v2 / v2r2` | `STRATA_WS_V2=1` |

Open upstream PRs carrying parts of this stack (each rebased on its own, so its hashes differ from the ones above):

- #3455 `nibble-experts`: D (the first seven commits).
- #3751 `mtp-q8-sm60`: D2.
- #3752 `prefill-ws-gp100`: P3-P4 and the swiglu half of P2.
- #3760 `pipeline-windows-3`: B1-B7.

D1, D3-D5, P1, P2's combine write, P5 and P6 are not posted.

## Launch settings for the measured speed

4x GP100, `--layer-split 12,24,36`, every expert resident, the Q8_0 dense shards (`tools/q8_dense_gguf.py`).
Stack switches:

```
STRATA_EXPERT_NIBBLE=1
STRATA_SPLIT_COMMIT_ASYNC=1
STRATA_Q8_SM60=1
STRATA_MTP_Q8_SM60=1
--pipeline-windows 3
STRATA_SELECT_LONG=1
STRATA_MMQ_WS=1
STRATA_PREFILL_PIPE=70            (or --prefill-pipe 70: b/a measured on this rig)
STRATA_PREFILL_PIPE_FLOOR=256
STRATA_PREFILL_SWIGLU_QUANT=1
STRATA_PREFILL_COMBINE_WRITE=1
STRATA_ARENA_FREE=1
```

Upstream switches that matter on GP100:

```
STRATA_SM60_TABLE=0               expert mode 0 instead of upstream's sm_60 default mode 8 (+2.0% decode here)
STRATA_IQ_STAGE_TABLES=1
STRATA_SCORES_THREAD=1            #1825's per-thread block scorer
STRATA_MMQ_RESIDENT_SORT_NE=1
STRATA_STAGE_TRIM=1               (= --trim-stage-weights)
--mtp-window 8192
--mtp-hnorm stream
```

The runs below also had `STRATA_DF_BRANCH=1`; on sm_60 the default `STRATA_SIDE_MIXER` takes precedence over it.

## Measured

Model: Swift 1.5 IQ3_S (abliterated). Revision: these same commits applied to upstream e88ee39d (0.1.45.3),
one exe for every row, 2026-10-21/22. This branch is the same content on 011c9e60 and was not re-measured.
Decode is token-weighted tok/s over eight prompts (five short ones, a 9.8K read, a 26K read, an 88K append);
one engine start per arm.

Ladder (switches added one at a time, in this order):

| Arm | Decode tok/s | Change |
|---|---:|---:|
| upstream e88ee39d (two runs) | 47.27 | - |
| this stack, no switches | 47.10 | -0.2% |
| + D | 51.73 | +9.8% |
| + `--pipeline-windows 3` (before D2) | 51.81 | +0.2% |
| + D1 | 52.14 | +0.6% |
| + D2 | 56.20 | +7.8% (includes the depth-3 share, see below) |
| + D3-D5 | 56.39 | +0.3% (115K +2.9%) |
| + P1, P2, P3-P5, P6, `STRATA_SCORES_THREAD=1` (two runs) | 56.36 | +19.2% total |

Prompt reads, upstream to the full ladder (ms):

| Read | Upstream | Stack | Change |
|---|---:|---:|---:|
| 1.1K fresh | 4,248 | 2,164 | -49% |
| 7.7K fresh | 15,178 | 9,596 | -37% |
| 33K fresh | 49,000 | 33,708 | -31% |
| 1.1K appended at 26K | 4,895 | 2,496 | -49% |
| 88K appended | 123,744 | 84,888 | -31% |

P3-P5 give most of it (-27% to -44%); P1 -10% on 1K reads.

- `--pipeline-windows 3` pays only with D2 on (the drafter's cost per step): with D, D1 and D2 on, 2 -> 3 was
  54.22 -> 56.12 tok/s (+3.5%, A/B/B/A), every reply bit-identical.
- `STRATA_SM60_TABLE=0` on top of D, D1, D2 and `--pipeline-windows 3`: 56.09 -> 57.24 tok/s (+2.0%); with
  `STRATA_IQ_STAGE_TABLES=1` and `STRATA_SELECT_LONG=1` as well, 57.54. The same replies, bit for bit, in every arm.
- Every reply in the decode set was identical across arms; one appended-read reply changes its bits once
  `STRATA_PREFILL_PIPE_FLOOR=256` changes its chunks.

## Build

```
cmake -DSTRATA_ENABLE_CUDA=ON -DSTRATA_EXPERIMENTAL_SM60=ON -DCMAKE_CUDA_ARCHITECTURES=60 ...
cmake --build <dir> --target strata iq_nibble_parity native_expert_bench pl_sched_test
```

Without `STRATA_EXPERIMENTAL_SM60` (for example sm_86) the branch builds the default engine.
