# Attention decode on the final day: what was adopted, what was rejected

Written when the kernel track closed (2026-11-05). Commit hashes refer to our private working repo; the code they
point at is in `kernels/decode_layer_src/src/attention.cu` (final state) and the switches are named in the tables.
Receipt paths (`receipts/...`) point into the private workspace and are not part of this repo.

## Confirmed outcomes

These were reported adopted by the pipeline. The exact frozen file, its selected flags, and final submission
result are recorded in the adoption ledger; source presence alone does not establish that a variant was enabled.

| Change | Original source commit | Recorded result / invariant |
|---|---|---|
| Stable attention RMS | `e278980` | Normalize the softmax numerator before squaring. Fixed the planted aligned-key failure; 16 standalone TASK tests, 13 layer TASK tests, and six layer-derived aligned-key paths passed remotely. |
| Coherent asynchronous epochs | `d1fc085` | Adopted as `epoch_f`, cf1018. All variants share padded global epoch slots; asynchronous copies are aligned 16-byte `.cg` operations. |
| Full verify-16 WST | `181cce4` | Adopted in bundle3 (`ad087b7`). Keell's measured FP32-partial WST path, adapted to the layer ABI/PDL and coherent QKV reads; separate shape-local scratch. |
| Early shared gates | `a0e1ddf` | `ATT_V1_GATE_EARLY=2` adopted in bundle6; supersedes register-only value 1. Warp 0 computes owned gates while warp 7 handles the counter. |
| No poll sleep | `c21bdc1` | `ATT_V1_POLL_NS=0` reported adopted through bundle7. The same commit also contains the rejected, separate return-free-arrival experiment. |
| BF16 verify-one partials | `ba5f73c` | Adopted in bundle8 after full checks. Reuses the existing BF16 reduction/fold; denominators remain FP32, workspace remains zero at rest. |
| Wrap-to-zero CTA barrier | `4e84b6a` | Adopted as `ctabar1_r1` after its individual confirmation. Dedicated slot `ws_cnt[16384 + bp]`; the last of `nact` arrivals wraps it to zero. |
| Relaxed unsuccessful polls | `19c659d` | Later integration is recorded through `ce3c807` (`rlxpoll1_r4`); final validation receipts are under `submit_ready/cf1084_rlxpoll1_r5/`. Terminal observation still performs an acquire load and recheck. |

The first RMS failure was real: old standalone edge relative L2 was 0.3454, and eight planted aligned heads in
the layer-derived reproducer collapsed to zero. The repaired standalone edge was 0.005429. Do not restore the
old algebra that squares unnormalized fixed-reference numerators, even if a normal random-input test passes.

WST overwrites its partials and keeps monotonic tickets. Its `att_wst_ws` must remain separate from base
attention's zero-at-rest `att_ws`. Selection is limited to verify-16 full, page 1; page-size fallbacks retain
the base workspace. The standalone extraction helper has a separate WST scratch cache for the same reason.

## Rejected or superseded experiments

| Experiment | Source / reading | Disposition |
|---|---|---|
| Flat decode register lookahead | `8b2f148`, repaired off state `afef845` | d32f screen behind. Retired by `945455f` / integrated `a8f493c`. |
| Independent v1s cluster kernel | `b6c25ee` | Light checks passed; approximately +2.42 us versus the control-derived parent equivalent. Closed. |
| Public VP2 layer port | `949bc3d` | Light checks passed; approximately +2.46 us versus the control-derived parent equivalent. Closed. Standalone speed did not transfer to this layer port. |
| Finalizer inactive-warp trim | `8242090` | cf1027 behind: 1.0299 versus 1.0312. Closed. |
| New-key output ownership / inactive CTA32 | `9fbf307` | v1s -0.52%, about +0.53 us. Closed; do not resurrect it through a cleanup merge. |
| Return-free arrival | `c21bdc1`, `ATT_V1_RED_ARRIVE=1` | v1s -0.95%. Closed. |
| Explicit counter election | `19c659d`, `ATT_V1_ELECT_COUNTER=1` | Re-judged target ratio 0.9923, behind. Closed. |
| One warp per output row | `a18bb16` | Re-judged target ratio 0.9895, behind. Closed. |
| Pair-major shared gates | `e034a1b`, gate value 3 | Within at -0.16%, but did not beat the adopted gate value 2's selection criterion. Closed/dominated. |

The cluster deltas above are estimates from established controls, not directly timed matched-parent pairs.
An apparent v1clu gain on untouched v16f was invalid evidence; its mixed confirmation was withdrawn and not adopted.

## Work left unconfirmed at the close

The following are last readings received or recorded, not permission to promote. Consult the final ledger/frozen
point before resuming: later bundle outcomes can supersede a screen. No further runs are requested by this note.

| Flag / source | Last known reading |
|---|---|
| `ATT_V1_FINAL_INT=1`, `cad2dd5` | Within, corrected target ratio 0.9965; bundle9's joint performance failed and members were split. |
| `ATT_V1_COPIES=2`, `b441571` | Within, corrected target ratio 1.0014. Same grid/tile plan; two workspace copies. |
| `ATT_V1_QPAD=1`, `c3523dd` | Within, -0.48%. 144-byte raw query rows fit the existing 18 KB region. |
| `ATT_V1_QPAD=2`, `00c335d` | Within, -0.05%; preferred over value 1. Swaps adjacent 16-byte chunks on odd rows while retaining 128-byte rows. |
| `ATT_V1_SIGM_CORE=1`, `279a2cd` | Within, -0.27%. Direct EX2/add-one/RCP core; no runtime numerical guard or tanh approximation. |
| `ATT_V1_NORM_CORE=1`, `a7d54c2` | Within, +0.12%. Q/K normalization only; normal-input reciprocal root. |
| `ATT_V1_QSUM_PAIR=1`, `2e460a6` | Within, +0.29%. Two FP32 query-norm accumulation chains. |
| `ATT_V1_TILE64=1`, `02652ab` | Ledger records within, +0.30%. Propagates the ONE kernel's fixed 64-key chunk. |
| `ATT_V1_FINAL_RSQ=1`, `1cadfc4` | Source committed and sent; no screen reading received before the stop. Mean-before-squaring remains unchanged. |

`f97f8fc` records actual nonzero build defaults (`POLL_NS=64`, `COPIES=1`), preventing a per-case zero-poll build
from being incorrectly aliased to an absent-flag/default-64 build. Preserve this when keeping these knobs.

## Evidence and interpretation

- `receipts/evidence/lane_screens.jsonl` and `lane_reports.jsonl` hold readings; final validation receipts are in
  `receipts/evidence/submit_ready/cf1084_rlxpoll1_r5/`.
- `receipts/evidence/profile/inc/ncu_verify1slidi_{raw,details,summary,attn_sass}.csv`: M143, 33 CTAs, 88 registers,
  zero local spills. NCU replay duration 9.376 us at 1.43 GHz is not a screen time.
- That capture had only 169 PC samples, at a 2048-cycle interval over two passes. The largest site had 37 barrier
  samples immediately after the counter/gate CTA join (SASS CSV row 1773); the earlier prologue join had 11.
  There were 612 poll loads and 612 `CCTL.IVALL` operations. L2 atomic-input utilization was 1.18%, with no LG throttle:
  this did not justify claiming bulk-memory saturation or a private-partials rewrite.
- Raw-Q QDIR loads at SASS CSV rows 560/561/698/699 each had 2112 shared wavefronts versus 1056 ideal, totaling 4224
  excess. Gate2's four float2 loads contributed another 768 excess wavefronts. These motivated QPAD and gate3;
  reduced conflict counts alone are not a demonstrated layer speedup.
- The original `profile/inc/` nsys trace was eager, with null graph IDs and large host gaps. Do not treat it as the
  judge's PDL chain. `profile/inc_graph/nsys_graph.sqlite` is the graph-replay capture: over ten v1s traces, attention
  averaged 8.086 us, overlapped QKV by 2.170 us, and had 5.917 us after QKV ended. O began 5.517 us before attention
  ended and finished 3.123 us after it. Global sampled GPU bandwidth is not per-kernel bandwidth.
- Early standalone measurement was 2.572792x SGLang / 1454.085 us weighted time on one physical GPU of unknown serial.
  Do not mix it with later chain stamps or Keell runs. The public Keell snapshot comparison used one later physical
  GPU: weighted time ratio 1.05123, not the uniform-12-case ratio 1.058. Its main gains were verify sliding, not d32f
  gather. Public aggregate leader numbers do not reveal their per-case times or prove a higher gather rate.

Native off-state comparisons on a box found zero differing kernels for finint, relaxed/elected counter, CTA barrier,
gate3, QPAD1, and the copy-count parent rebuilt with the same adopted flags. Historical-point comparisons are not
off-state proofs: the earlier copy-count mismatch was v1BF16; the four-kernel poll-parent mismatch remained confounded.
