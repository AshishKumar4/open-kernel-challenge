# H200 theory for Muse decode (v3, 2026-11-03)

This file is the readable summary of the H200 model in our Lean repo (h200-lean): proofs of our synchronization protocols (Lean, kernel-checked), a cited
parameter table (`calibration/params.md`), a timing simulator driven by the real SASS
(`H200/Sim`), its validation table (`calibration/validation.md`) and ranked predictions
(`Docs/predictions.md`). Marks: **Proven** = a Lean theorem (name given); **F** = measured; **I** =
inference; **P** = model prediction.

## Model status (v3)

**Proven** (any parameters in the class; `README.md` of h200-lean lists hypotheses): the MoE TMA ring
with mbarrier phases is safe for every slot/consumer/producer/byte/stage count with matching barrier
counts (`Ring.ring_safe`); the split-KV merge's last split sees every split's partial for any number
of splits, key partition and threads, in the PTX memory model (`SplitMerge.merge_sees`). Rejected as
bugs with kernel-checked traces: the joined-TMA count 1 vs 2, awready + wjoin's mismatched barrier-2
counts, the dry warm-up warp's second arrival, and hwarm v1's start join (deadlock).

**F/P, timing** (validation table): TMA streaming rates 28/30 within 10% and first-landing/ring
latencies 45/45 within 20%; MoE steady paces within 8%; MoE kernels end to end predicted 16% faster
than measured on average (worst v32f, −32%).

**What the model refutes or settles from v2:**
- *Settled:* the MoE and WGS steady stages are memory-bound with instruction slack (simulated issue of
  the clean build's SASS: 925 ns small-MoE unit, 760 ns WGS N=32 stage, against a 1055 ns HBM share
  per 36 KB at 1.755 GHz). Static instruction cuts there cannot gain until compute exceeds the HBM
  share (≈12% and ≈28% of slack).
- *Settled:* more in-flight bytes cannot speed the MoE: 108 KB/SM is in flight, the plateau is reached
  at ≥48 KB/SM. v2's Little's-law figure of ≈83 KB/SM needed is replaced by the measured curve below.
- *Corrected:* the TMA plateau is 4.61 TB/s (not ≈4.4): `receipts/h200bench/mr_benchB.txt`. One
  issuing thread that waits and reissues costs ≈571 cycles per copy, so small copies need more
  issuers (two CTAs per SM at 8 KB reach 4.57 TB/s, one reaches 3.75).
- *Corrected:* `MUFU.EX2` on f16x2 has no per-element throughput gain over f32 (16 vs 8 cycles per
  instruction for 2 vs 1 values, `mr_benchA.txt:66-71`); FP16 exp is not a lever.
- *Accounted (the gap):* the MoE's end-to-end residual (≈16%) is neither bytes nor instructions.
  Per-CTA stamps against per-part floors (h200-lean `calibration/moe-sync.md`): A→B edge + phase B
  4.3% score recoverable (in the stamped gbest-era build: credits only at each CTA's end of phase A, which the current source keeps at T = 1 only;
  the helper's cold epilogue; job-change demand loads ≈4 µs), phase-A imbalance 1.5% (WGS),
  small-MoE unit pace 1.1%, tail 0.4% (its prologue was double-counted: the first stamp precedes
griddepcontrol.wait). The incumbent, from fresh post-wait stamps at d1/v1: 2.0% recoverable (unit
pace 0.74, phase B 0.35, tail 0.30, T = 1 credit 0.27, start 0.26, imbalance 0.10); WGS per-CTA stall
budget 7.7–16.3 µs (judge-facts.md:414-417, 665-668). The steady DRAM rate equals the bare ring
(LESSONS.md §4): the loss is events.


Scope: the 12 scored cases on the judge's H200 SXM (sm_90a). **F** = observed fact (receipt cited); **I** = inference; **P** = prediction to test; **U** = unknown, with what would pin it. Public numbers measured on other GPUs are labelled *context*, never H200 constants. v1 is kept as `notes/gpu-theory-v1.md` (it was never committed). No GPU work or code edits were done for this note.

Changes from v1: a score accounting identity that re-ranks everything (§0); a parameter table of the machine (§1); per-sub-partition issue arithmetic (§2); the code and constant delivery mechanism, now backed by hodry3 and by ncu stall counters already in our exports (§3); pinned async and cross-SM costs (§4); a re-ranked experiment list (§6); the timing-model brief (§7). Corrected from v1: the TMA bench reports medians of per-CTA first/last landings, not the earliest landing; the v32-full "L2 84%" is not evidence of an L2 limit (§2); `native_identity.json` lives in a side worktree, not the main checkout; the "attention 2000→1192" static count has no receipt in `receipts/astra/attention_family_codegen.json` and is dropped until one is produced.

## 0. How microseconds become score

**F.** Score = geomean over the 6 loads of the load speedup. A load's time is 46 × sliding-layer time + 16 × full-layer time (`receipts/vast/our GPU box/cold_nf_clean13.log`, tail: d1 3905.1 ≈ 46 × 61.30 + 16 × 67.82; `notes/avenues.md:3-6`). So the local, first-order sensitivity of score to 1 µs on one case is w / (6 · T_load):

| % score per µs | d1 | v1 | d16 | d32 | v16 | v32 |
|---|---|---|---|---|---|---|
| sliding (w = 46) | 0.196 | 0.120 | 0.088 | 0.071 | 0.049 | 0.037 |
| full (w = 16) | 0.068 | 0.042 | 0.030 | 0.025 | 0.017 | 0.013 |

(T_load of a recent clean build, the same log; `notes/avenues.md:5` has the older table. These are derivatives, not finite-change gains.) **I.** One µs off decode-1 sliding is worth as much as 15 µs off verify-32 full. Every proposal below states its µs and case.

## 1. The machine: parameters and what pins them

| Parameter | Value | Source | Status |
|---|---|---|---|
| SMs, sub-partitions | 132 SMs × 4 SMSPs; each SMSP has its own scheduler (≤1 warp-instruction issued/cycle), a quarter of the 64K-register file, an L0 instruction cache, and one tensor core | NVIDIA H100 whitepaper SM diagram; [Hopper in-depth](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/) | public |
| Per-SM units | 128 FP32, 64 INT32, 4 tensor cores, 256 KB L1/shared | same | public; exact opcode→pipe mapping (LOP3/SHF/PRMT/IMAD/HMUL2) on sm_90 is **U**: pin with ncu `sm__inst_executed_pipe_*` on our loops |
| SM clock | max 1980 MHz; WGS cases run at 1.68–1.78 GHz median under load; ncu runs at 1.64–1.81 GHz | `judge-facts.md:282,437`; `fresh.md:83-93`; `ncu_multi.details.txt:4867`; `fresh.md:177-180` | F; models must take the per-run clock |
| Tensor (dense BF16) | ≈2048 FMA/clk/SM | `judge-facts.md:273-274` (from the 989-TFLOPS spec) | public-derived |
| Registers / smem | WGS 96 regs × 576 thr; small MoE 145–153 regs, 1 CTA/SM; v32f attention 80 regs × 384 thr, 2 CTAs/SM (register-limited), 112 KB dyn smem; ≤227 KB smem per CTA | `judge-facts.md:269-271,409-410`; `ncu_multi.raw.csv` (launch__*) | F |
| HBM | judge slope 4.58–4.70 TB/s + 2.5–2.8 µs; TMA ring plateau 4.61 TB/s with ≥48 KB in flight per SM; 8/16/32 KB per SM → 1.82/3.00/4.28 TB/s (`receipts/h200bench/mr_benchB.txt`, v3) | `judge-facts.md:17-18,211-213` | F |
| L2 | 50 MB, two partitions (H100-SXM5 configuration) | whitepaper | public, not measured by us. **F:** at v32f, 10.8 M of 31.9 M L2 sectors arrived over the inter-partition fabric (`lts__t_sectors_srcunit_ltcfabric` vs `_tex`, `ncu_multi.raw.csv`) |
| TMA bulk copy | one 36 KB copy alone 0.80 µs; 132 × 36 KB: per-CTA median 2.37 µs (max 2.75); 2/SM last 3.71; 3/SM last 5.12; one CTA issues bulk prefetches at 250–280 GB/s (≈128 B/clk) | `receipts/tmabench_h200.txt:2-16` | F (cold L2) |
| TMA path | global → L2 → shared; no L1 allocation; completion via mbarrier tx-count | PTX ISA `cp.async.bulk`; CUDA Programming Guide (TMA) | public |
| Latency context (H800 PCIe, *not H200*) | shared 29 cyc; L2 near/far hit 258/414; near/far miss 556/744; global 656; TMA + mbarrier ≈170 cyc above an ordinary load | [Luo et al. 2025](https://arxiv.org/html/2501.12084v2) Tables 3–4, §5.1 | context |
| Loaded-system latencies (ours) | demand L2 miss during streaming ≈4 µs; remote poller sees relaxed stores 1.6–2 µs after the writer; returned atomic 1–2 µs; grid data-flag hop 0.8 µs idle, 1.4–1.6 µs loaded | `judge-facts.md:582-584,359-361,179,187-189,78-79` | F |
| PDL | overlap is opportunistic (dependents launch only when resources free up); d1s attention starts 1.7 µs before qkv ends | [CUDA guide, PDL](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/programmatic-dependent-launch.html); `judge-facts.md:128` | public + F |
| Instruction/constant delivery | L0 i-cache per SMSP → SM-level instruction cache → L2 → HBM; constants via the immediate-constant cache (ncu stall `imc_miss`) | SM diagram; Jarmusch et al. 2025 §III (qualitative) | sizes, line size, miss latency: **U** (§7 microbenchmarks) |
| Register banks / reuse | Volta: two 64-bit banks, conflict only when all three sources share a bank; `.reuse` flags in control bits | Jia et al. 2018 (arXiv 1804.06826) §3, *context* | Hopper **U**; never claim a bank conflict without SASS operand evidence |

**I (Little's law).** Sustaining 4.4 TB/s at ≈2.5 µs loaded latency needs ≈11 MB in flight, ≈83 KB per SM. This matches the small-MoE ring (3 × 36 KB per SM) running at ≈99% of 4.4 TB/s (`judge-facts.md:556-558`). In the MoE and GEMV streams, which already sit at that point, every tested extra-in-flight or prefetch variant lost: NS=4 0.991, wns 4-slot −0.5…−2%, l2pf 0.974, gsk2l2pf −1…−12% (`judge-facts.md:155,331-334,365-367,622-626`). That is evidence about those queues, not a universal law; a kernel below the point (§2 attention) can still gain from more concurrency.

## 2. The model: critical path through streaming stages, per SMSP

**I (kept from v1).** A kernel = startup + steady stages + dependent transitions + tail; only *exposed* waits cost. **F:** joint ownership cut sampled d32 B waits 9.63→6.82 µs, yet B ended only 0.55 µs earlier (`fresh.md:136-140`). The sliding skeleton without math takes 5.1/7.7/10.0 µs against 5.7/8.6/10.8 real (`judge-facts.md:61-64`).

**WGS (T ≥ 17), new arithmetic.** 16 consumer warps = 4 per SMSP. The historical (pre-dqx) 304 instructions/warp/stage give an *ideal lower bound* of 1216 consumer issue cycles per SMSP per stage: 0.61 µs at 1.98 GHz, 0.69 µs at 1.75 GHz. That excludes producer warps and per-pipe acceptance limits (`judge-facts.md:269-273`). The measured stage is 0.96–1.0 µs uninstrumented (`fresh.md:136-138`; `judge-facts.md:627-634`), against a 1.07–1.1 µs per-SM share of 4.4 TB/s; ncu shows 0.58 issued/scheduler and 42% no-eligible (`judge-facts.md:640-645`). **I:** the steady stage already runs at memory pace with issue to spare, so faster consumers mainly pay at boundaries and tails (≈2.3% score pool: splits 0.9, A→B 0.6, whole units 0.6; `judge-facts.md:414-417`). The tested evidence: scaled3/4 at 0.9992/0.9985 (`fresh.md:226-230`), GS=4 neutral (`judge-facts.md:146-148`), dqx +0.2–0.4% (`evidence-ledger.md:46`; so schedule changes *can* move endpoints). This does not prove stage work exhausted.

**Small MoE (T ≤ 16).** 8 compute warps = 2 per SMSP. 700 instructions/warp/unit → an ideal 1400 issue cycles per SMSP = 0.71 µs at 1.98 GHz (0.80 at 1.75), against 1.05 µs measured per unit, the same warm or cold (`judge-facts.md:269-272,579-581`); 2.23 active warps/scheduler, 47–54% issue (`judge-facts.md:388-391`). **I:** at d1 the 4 serial A units are most likely latency-bound per SMSP (2 warps cannot hide dependency chains). Adding warps adds instructions and loses (gnw16 0.9885, `judge-facts.md:401-404`). At d16 the ring is HBM-paced (4.34 TB/s, `judge-facts.md:556-558`).

**Verify-full attention.** **F** (v32f, `ncu_multi.raw.csv` / `.details.txt:4856-4876`): DRAM 70.2%; L2 hit 7% (loads 1%); 659.6 MB read ≈ the KV floor (no over-fetch); L2 sector throughput 35%, data 50%, fills 65%. The 84.5% "L2 throughput" is the maximum over sub-units that our export does not break out. Issue 51%, warps active 36.7%; per issued instruction, 2.72 cycles long-scoreboard and 2.19 wait out of 11.43. **I (hypothesis, not isolated):** KV delivery is limited by bytes in flight (2 CTAs/SM, cp.async depth tied to registers and occupancy; deeper cp.async lost 20–34%, `judge-facts.md:47-49`), not by MUFU (f16 exponentials lost 8–10%, `judge-facts.md:41-42`). Dependency scheduling, partition hotspots and producer issue would produce the same counters. One testable design: FA3's answer on Hopper ([Shah et al. 2024](https://arxiv.org/abs/2407.08608), warp specialization) is a producer warp issuing bulk copies into an mbarrier ring, which costs no consumer registers. With 1-token pages and no TMA gather on Hopper, that means one `cp.async.bulk` per KV row (16-B aligned, ≥16 B), whose issue and mbarrier overhead must be budgeted.

**F (kept).** Launch overlap is real, and post-MoE tails are ≈16/21/33 µs at d32/v16/v32 (`receipts/fresh/timelines/fresh_scaled_nsys.summary.json`). All three tail/route fusions lost because they replaced PDL overlap with grid hops (`judge-facts.md:296-302,520-524,601-610`).

## 3. Code and constant delivery: why one-shot code and dead source cost time

**F (hodry3, new).** The helper's unit 0 ran twice at one call site with identical math (`src/py/_v_hodry3.py`). The reduction reads only shared memory. Same-CTA medians, first pass vs second pass:

- d1 sliding: reduce 1.09 vs 0.10 µs; SwiGLU (one `gsA` scalar load + `expf`) 1.44 vs 0.16; store 0.06 vs 0.10.
- d1 full: 0.51 vs 0.10; 0.58 vs 0.16; 0.26 vs 0.13.

Source: `receipts/vast/our GPU box/eat_hodry3.txt:59-60,82-83`. A further 1.18 µs separates the end of the d1s dry pass (8.90) from the real pass's readiness stamp (10.08); its cause is **U** (needs the loaded SASS).

**F (ncu, cold replay, already in our exports).** Warp cycles per issued instruction spent in `no_instruction` (instruction fetch) plus `imc_miss` (constant cache), as a share of all warp cycles per issued instruction (the denominator includes the 1 selected cycle): d1s attention 23% (1.04 + 1.27 of 10.03); v1s attention 18%; d32s attention 13%; d1s MoE 8.6%; v1s MoE 5%; v32f attention 3.7%; WGS 2.6–3%. These are whole-kernel normalized warp-cycle shares, not time or recoverable speedup, and they do not locate the helper interval (`ncu_multi.raw.csv`, `smsp__average_warps_issue_stalled_{no_instruction,imc_miss}_per_issue_active`, `smsp__average_warp_latency_per_inst_issued`; case mapping in `harness/h200/ncu_multi.sh`).

**F.** Every attempt to warm the code early lost or was neutral: hwarm 0.985, hwcode 0.986, warmw 0.9953, hwpre8 0.9955, hwarm2 0.998, sk2warm 0.998; hwarm v2 (before the PDL wait) was within noise (`judge-facts.md:240-257,305-310,348-351`; commit 2adb810). With the warm-up, the first-unit span is still 3.4–3.8 µs, against 4.16 without it (`eat_hodbg_hwarm4.txt:61`, `eat_hodbg_hwarm5.txt:61`, `eat_hodbg_cf010.txt:58`), while a true second pass takes 0.35 µs. Code cannot be named as data for prefetch (`judge-facts.md:655-657`).

**F.** The same never-taken branch written three ways (trap, `(void)nF`, `__builtin_assume`) changes the 7 tiny builds by 192–320 SASS instructions. With assume, d1f read 1.0096 vs 1.0212 (3/3 screens lower) and v1s read 1.0319 vs 1.0263 (3/3 higher) (commits 0ad2dd9, b90da43). Restoring the trap restored identical SASS and the ≈0.3% (3e0a1e8). An unpaired NOFLUSH timeline (n = 3 replays) shows no warm advantage (`tl_cf089_{flush,warm}.txt`). It is underpowered below ≈2 µs.

**I (leading hypothesis, upgraded from v1's "unresolved" but not established).** The first execution of a code region on an SM misses the SMSP and SM instruction caches and, for kernel parameters, the constant cache. Those misses are served from L2 and, after the judge's flush or the layer's own >50 MB of streaming, from HBM, at loaded latency; the hot loops amortize it. hodry3 does not prove this. Its first pass also warms data, address translations and other state; it changes buffer release and backpressure; it adds global stamps; and its two passes are source-identical, not verified loaded-SASS-identical. The per-region cost (≈0.5–2 µs in hodry3 and `judge-facts.md:297-302`) is a range, not a constant. The data read-flush does not architecturally invalidate the SM instruction caches. Other kernels' code displaces them between replays, and the L2 backing is gone.

Why warming fails is **U**. The leading candidates:

- warmed lines are evicted before use (capacity: the compute loops alone are ≈11 KB/warp-path);
- L0 is per-SMSP, so a warm-up warp on another SMSP warms only the SM level;
- the warm-up's own misses land on the same HBM queue during FAST1's 9.7 MB burst.

**I (why dead source moves time).** The candidates now are: (a) changed scheduling and allocation (v1); (b) a changed set and order of cold instruction lines on one-shot critical paths. (b) fits the facts: only d1/v1 move, the sign flips between shapes, and the size is ≈1 µs. (c) For parameter-slot removal, a change in which constant-cache lines are first touched on the critical path (`parameter_reads.json` shows only ULDC/payload changes). c[0] is not shared-memory banking, and uniform offsets do not serialize lanes (critique correct). In score terms each such swing is small: d1f 0.77 µs ≈ 0.05%.

**I (static cuts ≠ gains, kept from v1).** Deleting unexecuted code removes no dynamic work. It can still move layout (b). Judge a change by executed hot-path work on the critical CTA, plus its one-shot cold regions.

## 4. Asynchronous handoffs: what each step costs

**Public semantics** (PTX ISA, [wgmma](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-instructions), [mbarrier](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier); [CUTLASS wgmma guide](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/guides/mma/wgmma_programming.html)):

- Generic-proxy shared stores read by wgmma (SS operands) or by TMA need `fence.proxy.async` before the mbarrier arrive.
- `wgmma.fence` orders register and accumulator accesses.
- `commit_group`/`wait_group` retire groups.
- Semantics are not latencies.

**F:** ptxas serializes RS wgmma at depth > 1 (C7511/C7512; `judge-facts.md:105-108,116,407-408`).

**F (RS→SS).** HGMMA counts are equal, yet the kernel slows 60.6→96.1 µs and 154.7→269.3 µs. Shared wavefronts rise 3.67× with no excess wavefronts. First-probe long-scoreboard samples on decoded-full/empty: 2,885/8,881. Retry loops: 10/15 (`fresh.md:173-190`). **I:** publication/handshake latency leads, but it is not isolated from the cost of conflict-free extra traffic or from changed roles. Split it into stores, `fence.proxy.async`, arrive→observe, `wgmma.fence`, commit/wait, and slot reuse before any redesign. SS stays parked.

**Pinned cross-SM costs (F):** grid hop 1.4–1.6 µs loaded; relaxed-store visibility 1.6–2 µs; returned atomic 1–2 µs; release-red fire-and-forget is cheap (designated finishing; `judge-facts.md:187-193`); demand miss under streaming ≈4 µs. **I:** any design that adds one of these to a critical path must save more than its cost. This rule priced every loss in `dead-audit.md` (0/12 on sign, which mixes signs, compile failures and overestimates).

**Small-MoE d1 chain (F, `eat_hodbg_cf010.txt:58`, instrumented).** Compute unit 0 is ready at 5.31 µs; the helper picks it up at 6.85 and finishes it at 11.01. Units 1–4 are ready at 6.3–9.3 µs but are picked up at 11.2–13.1 µs: the first-unit cold cost turns into a backlog through to the A credit (13.70). **I:** that backlog, not polling or fences, is the d1 A→B edge. One-poller and release-add tested neutral (`judge-facts.md:156-169`).

## 5. Noise

**F (kept):** pooled screen sd 0.09–0.10%, difference sd ≈0.14%; Lium weighted repeat ±0.0003; a single 200-rep vast run can be ≈1% off at d1; selection shrank +0.66% to +0.28% (`evidence-ledger.md:12-16`; `judge-facts.md:513-514,530-533`). Incumbent control sd ranges 0.085–0.22% (`receipts/evidence/confirm.jsonl`, cf117). **Correction:** √2·sd assumes independent arms. Blocks are inputs/seeds/runs, never graph replays. Judge ±2% is seed and harness variation, not evidence against a paired win.

## 6. Ranked experiments (act now)

Gains are **P**, in % score via §0. Each item: changed-case confirmation, both roles, fresh blocks; unchanged cases are sentinels.

| # | Experiment | Mechanism | Expected | Decisive measurement |
|---|---|---|---|---|
| 1 | Per-shape attention split/merge/work plans on the incumbent, extending split11 (v16s 1.0247 vs control 1.0164, `confirm.jsonl` cf117) and d16f boundcost (+0.55% on its case, `receipts/astra/shape_confirmations.json`) to every case, sliding first | merge and work balance; no new hops | +0.1–0.4 | targeted confirmations; per-case merge-cost stamps to explain the winners; judge by endpoint, not by merge time |
| 2 | Sliding attention at d1/v1: stamp the critical CTA's hops (PDL wait → page-table read → first KV landing → last split → merge counter → finalize), then remove one dependent hop. Example: read page-table entries (an input, independent of qkv) before `griddepcontrol.wait`; KV itself pre-wait was neutral to −0.26% (`judge-facts.md:54-55`, `evidence-ledger.md:79`) | each dependent global hop ≈0.5–2 µs loaded; skeleton = real − 0.6 µs | 1 µs off d1s + v1s = 0.32; expect 0–0.4 | per-hop same-CTA stamps, then the uninstrumented endpoint at d1s/v1s |
| 3 | v16f/v32f KV delivery: a producer warp issuing per-row bulk copies into an mbarrier ring, raising bytes in flight without consumer registers | Little's law (§1–2) | copy-model reference gaps 37/50 µs = 0.63 + 0.64 (a reference, not an achievable bound); expect 0.2–0.5 | endpoint at equal work first; DRAM 70% → ≥85% and per-SM bytes in flight explain it |
| 4 | Code/constant delivery study, done the owner's way. (a) Microbenchmarks (§7). (b) 2×2×2 factorial (trap/assume × ABI padding × hot one-shot alignment) with ncu `no_instruction` and `imc_miss` per kernel, loaded-SASS layout of the helper's first-unit path, 3-vs-3 d1f/v1s screens. Done after (a); this one item replaces v1 #1/#2 | §3 (b)/(c) | direct 0–0.15 (d1f swing ≈ 0.05). Ranked here on the owner's priority, not on expected return; value: predicts one-shot regions and blocks layout regressions | stall counts that move with the endpoint across the factorial; a true A/A |
| 5 | Post-MoE tail plans at T = 128/256 (unpooler, finish; 21/33-µs tails): fixed-seed plan manifests and actual launched kernels | tail overlap and plan choice | 2 µs off v16s + v32s ≈ 0.17 | launched-kernel manifest, layer endpoint |
| 6 | d1s helper first unit: shrink the cold path the first unit executes (no stamps; fewest distinct lines/branches; share code with paths already run), gated on #4(a) | §4 backlog: ≈2 µs at 0.196%/µs | 0–0.4 | first-unit span at d1s without stamps (nsys MoE duration), then endpoint |

Deprioritized, with reasons:

- WGS steady-stage speedups: the stage already runs at memory pace (§2). Only boundaries pay, and recent boundary attempts were neutral (gdfa/gdfd, `judge-facts.md:687-689`).
- SS transpose: parked (§4).
- B-job cohorts: the control itself costs 0.19% (`evidence-ledger.md:54`).
- More in-flight or prefetch traffic in the MoE/GEMV streams (12+ losses; #3 is a different, under-saturated kernel), and code warming (7 losses).
- Fusions that add grid hops.

## 6a. Model ranking (v3)

`Docs/predictions.md` in h200-lean, summarized (P, % score). Refuted by measurement: full-attention
K/V by a bulk-copy ring (page size 1 makes one 128 B copy per row, issue-bound below `cp.async`'s 3.80
TB/s gather ceiling, which the incumbent already reaches; cute_gather1). Ranked: (1)
MoE events, ≈ +2.0 at d1/v1 and up to ≈ +2.4 at T ≥ 32; (2) at d32s, the WGS cost weights: the critical CTA held the under-costed 3-tile job; the fix
(2ed6d5b) measured 2.4 µs end-only against a predicted 9–10, within noise in the paired screen. Retired: the CK MoE design (no-go; clustered grids hold 120 SMs,
`.cluster` mbarrier forms drain reductions through MEMBAR.GPU, its stream paced 1.7 µs per stage); (the gbest-era split, 8.5 without its double-counted prologue, is historical); (3) FP16 attention accumulation ≈ 0 to +0.05 (64 of the 68 F2FP per tile are FP8 → F16 K/V
conversions that stay; exp and tensor rate gain nothing); (4) one dependent HBM hop off sliding attention +0.06 per hop at d1s; (5) cold first-unit code
0 to +0.2; (6) PDL on the remaining plain boundaries (T = 128/256 only) ≈ +0.01 each; (7) BF16 partials ≈0 to +0.05. Not
recommended: steady-stage instruction cuts, deeper MoE rings or prefetch, FP16 exp.

## 7. Brief for the timing model 

v3 status: microbenchmarks 1, 4, 6 done (`receipts/h200bench/mr_bench{A,B}.txt`: instruction
delivery sweep, mbarrier/named-barrier/fence/atomic costs, `%globaltimer` resolution 32 ns); 2 was
defective (chase optimized away); 3 and 5 are covered by ptxas facts  and remain otherwise
open. The owner closed the microbenchmark campaign: further probes only for a parameter a prediction
needs that nothing public covers, via a separate agent. The original brief:

Fit from §1, with the clock as an input. Validate against these receipts (target ≤10% on steady rates, ≤20% on first-arrival latencies):

- TMA latency grid: `tmabench_h200.txt:2-12`.
- Small-MoE steady unit: issue→land 2.56 µs, 1.12 µs/unit/CTA; T = 1 compute 1.05 µs/unit (`judge-facts.md:556-558,579-581`).
- WGS stage: 0.96–1.0 µs uninstrumented, 1.22 with stamps (`fresh.md:136-138`; `judge-facts.md:627-634`).
- Fused-pooler stage: consumer 0.32–0.48 µs, issue→data 2.5–3.2 µs, kernel 16.5 µs (`judge-facts.md:242-244`).
- d1s kernel chain: `judge-facts.md:452-454`; `tl_cf089_flush.txt`.
- RS vs SS at 60.6/96.1 and 154.7/269.3 µs (`fresh.md:177-180`).
- v32f attention: 196 µs at 1.64 GHz, DRAM 70% (`ncu_multi.details.txt:4863-4877`).
- hodry3 first and second pass (§3).
- The pinned hops (§4).

Microbenchmarks still needed (through the GPU pipeline):

1. Instruction cache: straight-line code of S KB executed once after a flush, then again. Sweep S, then evict with X KB of other code. This gives line size, sequential prefetch, SM-level and L0 capacity (run warm-up and use on the same vs different SMSPs), and per-miss latency, idle and under a 4.4-TB/s background stream.
2. Constant cache: the same, over N distinct c[] lines via ULDC.
3. Per-opcode issue rate for LOP3/SHF/PRMT/IMAD/HMUL2/HGMMA, from our WGS and tiny loops' ncu pipe counters.
4. Same-CTA mbarrier arrive→try_wait observe latency, and the cost of `fence.proxy.async`.
5. Register-bank conflicts on sm_90 (3-source FFMA bank permutations).
6. `%globaltimer` resolution (smallest non-zero delta), which bounds every stamp receipt.
