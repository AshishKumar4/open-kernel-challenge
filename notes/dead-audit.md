# Dead list by the evidence that killed it (2026-11-02)

Classes: (a) H200 measurement, effect well beyond run-to-run noise (~1-2%; production scored 2.238 and checked 2.265);
(b) H200 measurement within noise; (c) arithmetic or a model; (e) the base (keell0renz) measured it dead
in its own context; (k) compile-time fact (ptxas / SASS), not a speed claim.
Weights: the true objective (load = 46 x sliding + 16 x full; value per us in notes/avenues.md).

| item | evidence | class | status |
|---|---|---|---|
| DT (WGS decoder warps + tensor warpgroups), layer and MoE, 3 variants each | 1767c9b5 etc.: layer 1.62-1.68 vs 2.238 | a | dead |
| ROUTE_FOLD / FOLD_FIN at T = 1 / 8 | fd55a7c3 vs 1767c9b5: +21-27 us | a | dead |
| ROUTE_FOLD at T = 16 | same runs: -3.5 / -1.0 us | b | re-test |
| WGS at T <= 16 (A7) | 834aefc7: d1 sliding 136 us | a | dead |
| CTA-local MoE (T <= 16) | x0.57-0.65 (ba182e44, 72b10e79) | a | dead |
| MoE phase-B L2 prefetch after routing, N = 4-16 units | bs1 x0.886-0.917 (189ce74f) | a | dead |
| same, N = 2 at bs 8 / 16 | x0.993 | b | low value |
| phase-B prefetch when phase-A units run out | x0.99-1.013 (5f261c0c) | b | re-test (weights) |
| SC6S 6-bit scales in the tiny builds | bs1 x0.991, bs8 x0.995 (1dcd30af), MoE problem only | b | re-test in the layer |
| round-robin MoE unit order | base x1.11-1.24 faster (45f0c34b) | a | dead |
| attention q8 f16-pair exponentials | full x0.904-0.917 (f455d935); sliding x0.98-1.0 | a (full) / b (sliding) | dead on full; sliding low value |
| attention cp.async depth | L8 full x0.66-0.80 (397f34f4) | a | dead |
| layer attention OKC_PDLKV = 1 | x0.984-1.005 (6d40b31c) | b | re-test (candidate pdlkv) |
| qkv GEMV on 128 CTAs | x0.88-0.92 (9a67faf0) | a | dead |
| our C-CTA boundary kernel | base 0.5-2% faster (3ff79d0b) | b | low value |
| 2-CTA cluster GEMVs at 16 / 32 | x0.66-0.91 (f2e7d9d1) | a | dead |
| dense GEMM at 128 / 256 (CUTLASS, own wgmma) | never beat the packed base (03776d81) | a | dead |
| old fused route (in the MoE prologue) | x0.895-0.936 (af2abf4a) | a | dead |
| 16-token skinny jobs | x0.811-0.885 (af2abf4a) | a | dead |
| three-launch small MoE | x0.913-0.958 (af2abf4a) | a | dead |
| 16-warp small MoE | x0.960-0.990 (af2abf4a) | a / b | dead at bs 1 (4%); edge |
| attention-window L2 prefetch kernel | x0.929-1.004 (af2abf4a) | a / b | dead |
| rq programmatic launch | +0.1-0.5% | b | in production |
| attention B=1 split / merge plans, cluster modes (A3) | probe 3fabb07d: <= x1.005 | b | re-test on the pod (plans) |
| persistent hop 1.4-1.6 us -> no megakernel | 63936027 (hop) | a | dead |
| prefill: timing switches, x-multicast cluster | 414f7154 | b / a | prefill not scored for rank |
| rq for o / pooler at T = 128 / 256 | the base's judge sweep (13.06 vs 11.17 us) | e | dead |
| finer MoE B units | balance arithmetic | c | dead (balance, not weights) |
| qkv ramp (A12) | model: first kernel, nothing to overlap | c | dead by structure (L2 is flushed between replays) |
| MoE decode below 13 ops / 8 codes | exhaustive search / bit algebra | c | floor holds (math) |
| route in the MoE prologue (froute) | the judged older form x0.9 | a | dead |
| setmaxnreg producer warpgroup | ptxas keeps 96 registers, C7512 stays | k | dead (compile) |
| two wgmma groups in flight | C7511 | k | dead (compile) |
| next-stage WGS prefetch, slot-counter golf | spills / uniform datapath lost | k | dead (compile) |
| q_len 1 FP8 via mma.sync | emulated on sm_90 (F2FP + HMMA) | k | dead (compile) |
| skinny x residency, accumulator zeroing | already free in the base (SASS) | k | void premise |
| GS / NW / NACC sweeps, ring / unit / helper knobs, WSB = 0, WGS2, PUB, speculative expert prefetch, L2 windows, lookahead rings, fused-slice MoE | keell0renz LESSONS | e | top by potential: helper / ring knobs and WSB at T <= 16 (skinny MoE 5.4% weighted potential) |

## (c) under the true weights

- finer MoE B units: the argument is balance (max 4 vs mean 3.9 units per CTA at bs 1), independent of weights. Dead.
- qkv ramp: worth 0.35% at d1 sliding alone (1.9 us x 0.182) but has no mechanism (first kernel of the step; the judge
  flushes L2 between replays). Dead by structure.
- The A3 / A4 attention estimates (1.5-2%) were 12-shape numbers: weighted, B1 attention's gap is 0.64% (d1 sliding) +
  0.57% (v1 sliding) + 0.26 / 0.29% (full): ~1.8% potential, measured plans gained nothing beyond 0.5%.

## Predictions against outcomes

| prediction | basis | predicted | outcome | |
|---|---|---|---|---|
| DT at T >= 32 | issue-bound premise | +2-3% layer | -25% layer | wrong sign |
| route / finish folds | one hop saved | +0.3-0.5% | -20% at T = 1 / 8 | wrong sign |
| CTA-local MoE | A->B hand-off removed | gain | x0.57-0.65 | wrong sign |
| phase-B prefetch | bytes earlier | gain | x0.89-0.99 | wrong sign |
| qkv on 128 CTAs, 2-CTA GEMVs, boundary kernel | more parallelism | gain | x0.66-0.98 | wrong sign |
| SC6S | 2.8% fewer bytes | ~1-2% | x0.99 | wrong (not byte-bound) |
| f16 exponentials, cp.async depth | fewer MUFU ops, deeper pipeline | gain | loss | wrong sign |
| attention B1 plans | gap table | 1.5-2% | <= 0.5% | overestimate |
| setmaxnreg, slot counter, prefetch, q1 FP8 mma.sync | SASS reasoning | fewer instructions / no serialization | compile showed otherwise | wrong until compiled |
| HBM floor (4.6 TB/s ring slope) | judge measurement | | matches judge receipts | right |
| overall rank = layer rank | board fetch | | exact for 521 rows | right |
| production score from per-case times with 46/16 weights | TASK.md | 2.2376 | 2.2376 | exact |

What to trust: direct measurements on the H200 (judge or pod, with paired repetitions), compile / SASS facts once
compiled, and accounting identities (floors, the score formula). Not to trust: any gain model for a structural change
(new kernel boundaries, fusions, warp-role splits, prefetch traffic, access-order changes): the record is 0 for 12 on
sign. Those get built and measured, never ranked by estimate alone.
