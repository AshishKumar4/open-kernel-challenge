# ptxas 13.0.88 reverse engineering: what it does to our kernels, and the source-level levers

**F** marks a fact, with its evidence. **I** marks an inference. **U** marks something not yet established.

All analysis ran on a CPU build box (cores 24–31). The ptxas there is identical to our GPU boxes' and the judge's
(`784ce3eb4e00d4aabf6479b8258ea600`, 37,741,528 bytes, `cuda_13.0.r13.0/compiler.36424714_0`). Builds use each
candidate's exact per-build flags. `harness/ptxas/buildx.py` runs the candidate's own `BUILDS` section and adds
the tvm_ffi wrapper flags `-Xcompiler -fPIC -std=c++17 -O2 -gencode=arch=compute_90a,code=sm_90a`.
nvcc invokes ptxas as `ptxas -arch=sm_90a -m64 x.ptx`, so every option there is a default.
Evidence lives in `receipts/evidence/gp_ptxas/`.

Leads, used as leads only: the public "PTXAS v13.0.88 RE reference"
(gh.evko.io/crucible-notes/ptxas). It targets our exact
binary. Some of its claims are wrong for the release build: `-knob` does not exist.

## 0. How to check a candidate's codegen yourself

The tools live in `harness/ptxas/`. Run them on a Linux box with the CUDA 13.0 toolkit (`export PATH=/usr/local/cuda/bin:$PATH`).

```
python3 harness/ptxas/buildx.py candidates/layer_X.py out/X [build-prefix ...]   # e.g. moe_d1 moe_big attention
```

- `buildx.py` executes the candidate's own `BUILDS` section, including flags appended at the end of the file, and
  adds the tvm_ffi wrapper flags. tvm_ffi headers come from `$TVM_FFI_INCLUDE`, default
  `/root/gp/pylib/tvm_ffi/include`.
- Per build it prints: ptxas C7xxx codes (C7507 = setmaxnreg dropped), instruction count, bad/total waterfall loops,
  REG, STACK, and a SASS hash. It leaves `mod.{cu,ptx,o,cubin,sass}` in `out/X/<build>/`.
- Compare against the parent by hash.
- Per-loop tools, all taking the `mod.cubin`:

  | tool               | reports                                                                  |
  | ------------------ | ------------------------------------------------------------------------ |
  | `hgloop.py`        | wgmma stage loops: size, cycles, R2UR, mix                               |
  | `hmloop.py`        | the same for mma.sync loops                                              |
  | `ntt.py`           | per-N stage-loop table                                                   |
  | `regmax.py`        | setmaxnreg regions and the highest register used                         |
  | `wgsan.py <dir>`   | spills in and out of loops; R2UR feeding HGMMA                           |
  | `blocks.py`        | a loop split into guarded blocks vs the per-iteration remainder          |
  | `mix.py`           | opcode diff of two loops                                                 |
  | `elect.py`         | waterfall loops; takes `mod.sass`                                        |
  | `invr2ur.py`       | loop-invariant R2UR                                                      |

- Source attribution needs a `-lineinfo` rebuild (`nvcc … $(cat flags.txt) -lineinfo -cubin`) and
  `nvdisasm -gi -c`. Then use `sites.py` (waterfall call sites), `bssy.py` (BSSY/STL/LDL → line) or `region.py`
  (all code from a source line range, plus its loops).

## 1. The never-taken `if (nF > 0) __trap();`

**F. NVVM produces most of the difference, not ptxas.** On the MoE tiny build (`layer_ex_trap_*`, then stk_s8 and
arch5_c1_1420):

- `if (nF > 0) __builtin_unreachable();` and `__builtin_assume(nF <= 0);` give identical PTX and SASS.
  LLVM uses nF ≤ 0 to delete skinny_split's fat-share arithmetic: PTX 4591 vs 4767 ops; SASS 5072 vs 5328 (no hint).
- cicc lowers `__trap()` to inline asm `trap;`. Its SASS is md5-identical to `if (nF > 0) asm("trap;")`, with or
  without volatile and with or without a memory clobber. To LLVM an asm is not noreturn, so the trap gives it no
  range fact and the code stays (T PTX 4782 ops ≈ N 4767).
- The incumbent's comment "this trap lets the compiler drop their share" is therefore false.

**F. In ptxas, `trap` is a CFG terminator.** All of these give md5-identical SASS: `@!p trap`, a branch over `trap`,
`brkpt` in place of `trap`, and two consecutive traps (the second is dead). A branch over an empty block is folded away.

**F. The trap block, though never executed, perturbs list scheduling and register allocation.** Removing only the
trap blocks from T's PTX leaves the instruction multiset of the hot loops unchanged, but it reorders them, swaps
SHF↔IMAD.SHL pipe-balancing choices, and changes registers (maxR 139→141).
**I:** the exit edge changes block numbering or tie-breaking. **U:** the exact heuristic.

**F. The real hot-path mechanism is divergence.** In the assume and no-check forms, NVVM merges
skinny_split's `(float)gridDim.x` with the same read inside an earlier thread-dependent branch:
`%r2 = (tid+256) % 288`, then `setp.eq %r2, 0`. The result is a PTX phi on a divergent condition. ptxas's divergence
analysis marks rG→v0/v1→the ring's `cp.async.bulk` operands as divergent. Each of the 4 ring-refill copies then runs
as a waterfall loop with 4 R2URs inside: `ELECT; 4×R2UR; 2×PLOP3; UBLKCP; BRA.U.ANY`, 9 instructions instead of 5.
The asm `trap;` call blocked the merge by accident.

Minimal PTX reproduction (`phi.py`). A value defined identically in both arms of a branch:

| branch condition                      | copy loops     |
| ------------------------------------- | -------------- |
| `tid == 64`                           | all 5 bad      |
| `(tid >> 5) == 2`                     | all 5 bad      |
| `shfl.sync.idx(tid >> 5, 0) == 2`     | ok             |
| kernel parameter                      | ok             |

**F. Recipe without a runtime check.** In run_phase, `int G; asm volatile("mov.u32 %0, %%nctaid.x;" : "=r"(G));`.
Combined with `__builtin_assume(nF <= 0)`, that gives the good elect loops plus the deletion, with no check
(`candidates/layer_gp_ra_1420.py`). Instruction counts with exact flags, against arch5_c1_1420:

| build | incumbent | RA   |
| ----- | --------- | ---- |
| d1f   | 5264      | 5000 |
| v1s   | 5304      | 5040 |
| d16s  | 5200      | 4928 |

REG goes down by ≤4. d32s/d32f/big are byte-identical. Controls: `layer_gp_a_1420.py` (assume only: 4 bad ring loops)
and `layer_gp_rt_1420.py` (trap + opaque G).

**F (screens: one GPU, 200 reps, full checks PASS, vs matched gp_ctl_1420 = arch5_c1_1420).**

| candidate                      | overall | notable cases                          |
| ------------------------------ | ------- | -------------------------------------- |
| A (assume only, bad loops)     | −0.57%  | d1f −2.28, v1s −1.63, v1f −1.70        |
| RA (assume + opaque G)         | −0.24%  | d1f −0.67, d1s −1.39                   |
| RT (trap + opaque G)           | −0.05%  | neutral                                |

RA vs A is +0.33%: the divergence fix recovers about 60% of the trap→assume loss, which confirms the mechanism.
The rest (RA vs the trap form, −0.24%) comes from deleting the fat-share code itself. **I:** a layout/schedule
effect of dead-code removal (gpu-theory §3 "dead source moves time"). The trap form stays. Absolute readings:
control 1.0248, A 1.0190, RA 1.0223, RT 1.0242.

**F. PTX carries no branch weights.** `__builtin_expect` and `[[likely]]` do not reach ptxas: NVVM emits no frequency
pragma for them. The only channel is `.pragma "frequency n"` (§4).

## 2. Divergence rules that cost us bulk/TMA issue latency

**F (minimal PTX/CUDA, `uni.cu`, `phi.py`).**

- ptxas treats anything derived from `%tid` as divergent, including `tid >> 5`.
  `__shfl_sync(~0u, tid >> 5, 0)` is treated as uniform.
- A phi at the join of a divergent branch is divergent, even when both incoming values are equal.
- A C-level spin `do { asm("try_wait … selp") } while (!ok);` makes the enclosing loop's carried values divergent
  (sk2). The same spin written inside the asm (`W: mbarrier.try_wait… p; @!p bra W;`) does not.
- When a converged warp issues under `elect.sync`, ptxas emits `UBLKCP` once, with no ELECT/BRA.U.ANY loop and no R2UR.
  `uni.cu` k4 is the template.

**F (scan of arch5_c1_1420, `elect.py`/`sites.py`).** Bad waterfall loops are those with R2UR inside:

| build              | bad / total | sites                                                       |
| ------------------ | ----------- | ----------------------------------------------------------- |
| sk2                | 52 / 118    | producer refill, SK2_CU lines 468/469, 474/475              |
| sk2_gle            | 42 / 54     |                                                             |
| sk2_narrow         | 4 / 10      |                                                             |
| rq128 / rq256      | 1 / 2       | UTMAREDG, line 691; fixed by the shfl warp index            |
| attention          | 4 / 10      | fetch_pages bulk_load, `bytes` operand only                 |
| MoE skinny builds  | 1 each      | tail L2 prefetch                                            |
| MoE d16            | +1          | first-units issue                                           |

Owners have been sent the sites and fixes.

**I.** The cost is the R2UR→issue latency, exposed in front of each copy: about 10–20 cycles per refill.

## 3. Scheduler model and control codes (sm_90a)

Control bits sit in the high 64-bit word of each instruction: stall = bits 41–44, yield = 45, write-SB = 46–48
(7 = none), read-SB = 49–51, wait mask = 52–57, reuse = 58–61. Decoder: `harness/ptxas/ctl.py`.

**F. Dependent issue distance** (stall sums in pure chains; `gen.py`, `an.py`, `pairs.txt`):

| path                                                                 | cycles |
| -------------------------------------------------------------------- | ------ |
| same "FMA-type" group (FFMA/FADD/FMUL/IMAD/IMAD.HI)                  | 4      |
| same "ALU-type" group (IADD3/LOP3/PRMT/SHF/SEL/FMNMX/F2FP/I2FP)      | 4      |
| across the two groups                                                | 5      |
| into fp16x2 ops                                                      | 5      |
| into MUFU/POPC/FLO/BREV/HFMA2.MMA                                    | 6      |
| uniform→vector (ULEA→IADD3)                                          | 6      |
| HFMA2.MMA → anything                                                 | 10     |
| DFMA→DFMA                                                            | 8      |
| HMMA.16816 accumulator chain                                         | 24     |
| HMMA → FADD                                                          | 24     |
| ISETP → dependent BRA (one sample)                                   | 13     |

Variable latency, tracked by scoreboard: MUFU, POPC/FLO/BREV, I2F/F2I/F2F64, LDS/LDSM/LDG/LDC, S2R/S2UR, SHFL.

**F. Issue interval** between independent instructions of the same kind:

| instruction                                                  | cycles                                   |
| ------------------------------------------------------------ | ---------------------------------------- |
| FFMA, FADD, FMUL                                             | 1                                        |
| IMAD, IADD3, LOP3, PRMT, SHF, FMNMX, DFMA                    | 2                                        |
| IMAD.HI                                                      | 4                                        |
| MUFU, POPC, FLO, BREV                                        | 8                                        |
| LDS                                                          | 4                                        |
| HMMA.16816                                                   | about 6                                  |
| fp16x2 streams (HFMA2 on the FMA pipe / HFMA2.MMA on tensor) | 2, alternating between the two pipes     |

**F.** Waits over 15 cycles are encoded as producer stall 15 plus a NOP carrying the remainder. Nearly all
mid-function NOPs are of this kind.

**F. wgmma.** `wgmma.fence` → `WARPGROUP.ARRIVE`. Back-to-back HGMMAs issue with S01; the last carries `gsb0`.
`wait_group 0` → `WARPGROUP.DEPBAR.LE gsb0, 0x0`. No register scoreboard is used for accumulators.

**F. FP8 mma.sync on sm_90a** (`lat/fp8mma.py`). `mma.sync m16n8k32 .e4m3/.e5m2` (f32 or f16 accumulator) is
accepted but emulated. Each one becomes 12× `F2FP.F16.E4M3.UNPACK_B` plus 2× `HMMA.16816.F32/F16`. FP8 tensor
throughput exists only through wgmma. An f16-accumulator `HMMA.16816.F16` is native.

**F. Code motion around waits** (`lat/order.py`).

- mbarrier spin loop: ptxas rotates it. The first TRYWAIT stays inline; the retry block moves to the function's end.
  Independent address math and LDC fill the gap before `@!P0 BRA`. Loads never move above the wait.
- wgmma `wait_group`: independent `ld.global` is hoisted above ARRIVE/HGMMA/DEPBAR; `ld.shared` stays after.
- `fence.proxy.async.shared::cta` → `MEMBAR.ALL.CTA` + `FENCE.VIEW.ASYNC.S`, i.e. a full CTA membar.

**F. setmaxnreg.**

- `.inc N` → a retry loop: `USETMAXREG.TRY_ALLOC.CTAPOOL UP0, N` + PLOP3 + BRA, about 28 issue cycles per try.
- `.dec N` → `USETMAXREG.DEALLOC.CTAPOOL N`.
- The kernel's reported register count stays at the launch cap, while ptxas allocates each region up to its target.
  With `inc 232` the consumer uses R0..R228 and doesn't spill; without it, the same code spills 732 B at 168
  (`lat/smr.cu`).

**F. HGMMA descriptors.** In the incumbent WGS builds every descriptor is born in UR, with no R2UR. the
transpose port has 48 HGMMAs per build whose descriptor comes from an R2UR ≤15 cycles earlier (`work/wgsan.py`).

**U.** ptxas's internal latency tables. The reference names the descriptor table at `0x2297C00` (sm_8x/9x family) and
per-SM dependency rules (sm_86 ≡ sm_90). Not extracted.

## 4. Knobs, pragmas and register caps

**F. Knobs are closed in release builds.**

- `ptxas -knob X=Y` fails with `Unknown option 'knob'`.
- `.pragma "func knob X=Y"` / `"next knob X=Y"` are parsed, then warn "unsupported".
- `.pragma "global knob …"` is a hard error.
- `DUMP_KNOBS_TO_FILE` does nothing.
- Conclusion: in the release build, knobs are reachable through none of CLI, env or pragma. The documented options
  and the two pragmas are the levers.

**F. `.pragma "frequency n"`** (PTX ISA 9.0 §12.4; n ≥ 1, 0 is rejected) is a block execution-count hint. From CUDA:
`asm volatile(".pragma \"frequency N\";");` as the first statement of a block. It lands after the block label.

- Values are relative. A lone hint gives the same SASS for any n.
- On raw PTX of T: entry=1 + hot ring loop=1000 gives 5120 instructions (vs 5272); entry=1000 + trap block=1 gives 5960.
- On RA d1f from CUDA, entry 1 + unit loop N=1/16/1000 gives 5024/5016/5112 instructions.
- On rq128, entry 1 + consumer loop N=4/64/4096 gives 1456/1464/1480 (vs 1432).
- Perf, measured on RA + entry 1, against the matched control:
  - Unit loop N=16: −0.27% overall, flat vs RA.
  - N=1000: −0.74% overall (d1s −2.43, d1f −2.69). In d1f, N=1000 restructured two other loops (new 224/258-instruction loops).
  - Up-weighting hot loops hurts.
  - The inverse (entry 16, loop 1): −0.13% vs a fresh control, −0.08% vs RA (d1f −1.00 vs RA).
  - No gain in either direction, so it is not a study dimension. Receipts: `receipts/evidence/gp_screens/`.
- **F. Why raf1000 lost** (code: `harness/ptxas/region.py`). route() ran 1.5–1.9 µs
  slower. In the control, ptxas lowers `__match_any_sync` / `__syncwarp` in route's loops as `BRA.DIV` → native
  `MATCH.ANY`, with the `WARPSYNC.COLLECTIVE…ENDCOLLECTIVE` emulation out of line (16 BRA.DIV). With entry=1 and the
  unit loop=1000, route's blocks count as cold:
  - the BRA.DIV fast paths vanish (2 left in the build);
  - the emulation runs inline every iteration;
  - route's loops grow 69→120 and 143→258 instructions.
- **Rule:** block weights change collective lowering. Cold blocks lose the converged fast path. Raising the entry
  weight to ≥100, or giving route's loops `frequency 1000`, restores BRA.DIV = 16.

**F. `.pragma "local_maxnreg N"`** (undocumented, accepted) is a per-function register cap. On the MoE T build
(148 regs, launch-bounds cap 168):

| N   | registers used | spill stores |
| --- | -------------- | ------------ |
| 128 | 126            | 0            |
| 96  | 96             | 152 B        |
| 64  | 64             | 756 B        |

**HAZARD:** with N above the cap (200), ptxas warns "Ignoring", then allocates 200 registers. At 384 threads that
kernel cannot launch.

**F. Other pragmas.**

- Accepted with no change here: `nounroll`, `used_bytes_mask`, `noPointerTrunc`, `sw4166479`.
- Unsupported: `predict`, `confirm`, `unroll N`, `nvopt 0|1` (other `nvopt` arguments are errors).
- `enable_smem_spilling` is not allowed with dynamic shared memory.

**F. ptxas options usable from a submission** (`"-Xptxas", "<opt>"` in a build's flags; `work/opts.py`).
On arch5_c1_1420 moe_d1f (5264 instructions, 148 regs):

| option                                            | effect                                                    |
| ------------------------------------------------- | --------------------------------------------------------- |
| `--register-usage-level=N` (documented BETA)      | tiers: 0–3 → 150 regs / 5424; 4–5 = default; 6–10 → 156 / 5256 |
| `--override-directive-values --maxrregcount=R`    | the cap overrides launch bounds (128 → 126 regs)          |
| `--opt-pointers`                                  | 143 / 5416                                                |
| `--cloning=no`, `--fast-compile`                  | 5352                                                      |
| `--def-load-cache` / `--def-store-cache`          | cache modifiers on default LDG/STG                        |
| `-O1`                                             | 128 regs / 5224                                           |

**F (study harvest h9b, commit 5497996).** These read as clear losers and were withdrawn from free draws:

- `--def-load-cache=cg`: MoE skinny −1.04%, MoE WGS −0.37%.
- `--register-usage-level` 3 and 7: MoE skinny −0.20%.
- `--register-usage-level` 3: sk2 and MoE WGS.

No dial value has shown a gain so far.

Accepted with no effect here: allow-expensive-optimizations, dont-merge-basicblocks, return-at-end, fastimul,
no-fastreg, disable-optimizer-constants, extensible-whole-program.

**F. Register caps.** The cap is min(255, floor8(65536 / (threads × minBlocks))):
512 threads → 128, 384 → 168, ≤256 → 255, 640 → 96, 768 → 80 (`regcap.py`).
`-maxrregcount` is ignored when `__launch_bounds__` is present. A 128 cap left the 148-register kernel at 148.
For setmaxnreg see §3.

**F. C7507** (the `layer_ex_wtrace_reg_1`). The message is `ptxas info: (C7507) Potential Performance
Loss: 'setmaxnreg' ignored to maintain minimum register requirements`. It is only visible with `-Xptxas -v`.

- When code after a dec/inc `setmaxnreg` split is reachable from both budgets, ptxas drops all the setmaxnregs and
  stays at the launch cap. Spills follow. In that build, the WTRACE init plus `__syncthreads` came after the
  if/else donation.
- Raising the dec value doesn't help.
- Moving the shared block before the donation restores USETMAXREG ×2.
- **Rule:** after setmaxnreg, the role paths must not rejoin. Check every build's `-v` output for C7507.

## 4b. Scans of other code

**F. the ports (exact flags).** No new false divergence. In the WGS builds, stage_after_commit,
clean_dsmem_tail_v2 and clean_transpose_k64_v2 spill at the 96-register cap (incumbent: 0):

| port                   | build | spill stores / loads | LDL in loops |
| ---------------------- | ----- | -------------------- | ------------ |
| stage_after_commit     | big   | 24 / 36 B            | 8            |
| clean_dsmem_tail_v2    | big   | 36 / 136 B           | 33           |
| clean_transpose_k64_v2 | big   | 32 / 68 B            | 16           |

**I.** This fits their measured losses.

**F. The CK MoE.**

- Fence costs, measured on microkernels (`lat/scope.py`):

  | PTX form                                     | SASS                                                    |
  | -------------------------------------------- | ------------------------------------------------------- |
  | remote arrive `.release.cluster`             | MEMBAR.ALL.CTA + MEMBAR.ALL.GPU + ERRBAR + CGAERRBAR + SYNCS.ARRIVE.RED |
  | remote arrive, default form                  | SYNCS.ARRIVE.RED alone                                  |
  | try_wait `.acquire.cluster`                  | adds CCTL.IVALL                                         |
  | `ld.acquire.gpu`                             | LDG.STRONG + CCTL.IVALL                                 |
  | `red.release.gpu`                            | MEMBAR.ALL.GPU + ERRBAR + CGAERRBAR + RED               |
  | `barrier.cluster.arrive.release`             | adds the MEMBAR.ALL.GPU trio                            |
  | `fence.proxy.async`                          | MEMBAR.ALL.CTA + FENCE.VIEW.ASYNC.S                     |

- Stage loop: 367–384 instructions / 551–594 static cycles, vs the incumbent's 288–293 / 411–452. It carries 32 R2URs
  (descriptors in vector registers).
- Cause, part 1: NVVM CSEs every `smem_raw` reference into one kernel-entry `mov`. ptxas keeps that kernel-long value
  in a vector register and R2URs it at each use. A local `asm volatile("mov.u32 %0, <smem_raw symbol>;")` in the stage
  function cuts total R2URs 546→420.
- Cause, part 2: the x-slot term derives from vector counters.
- URF pressure is not the cause: the HK_PIN=1 build has UR max UR23 and still 32 R2URs.
- **F. Fixed in aac95f9:** local base remat + `__shfl_sync`'d slot. The stage loop drops to 303–305 instructions /
  449–505 cycles, with 15 UIADD3 + 4 R2UR. The spill is gone.
- **Rule:** a uniform value with a kernel-long live range can end up in vector registers and be R2UR'd at every use.
  NVVM's CSE of `smem_raw` / shared-symbol addresses to kernel entry causes exactly this. Rematerialize it locally
  with `asm volatile`.

**F (follow-up, CK MoE 43d32b9..f3bedfd).** Adding the dedicated owner warp (HK_OWARP=1) flipped the stage loops back
to 32 R2UR / 365–370 instructions, even with the base remat in place.

- The x-slot `__shfl_sync` is deleted by ptxas once it proves the value uniform, and an `asm volatile` shfl is
  deleted too: volatile binds only NVVM.
- Each of these still gives 32 R2UR: an opaque running descriptor add, `__noinline__` owarp, NTL restored, the pown
  wait removed.
- Only HK_OWARP=0 gives 311–313 instructions with 13 UIADD3 + 8 R2UR.
- **I:** uniform-register allocation is a whole-kernel heuristic that small structural changes flip. Check the
  stage-loop shape (`lat/hgloop.py`) after every structural change.
- **F (c72f7d3).** Keeping the never-taken `owner()` path in the k-loop (pend opaque) restores the good allocation
  with the owner warp on: 312–314 instructions, 13 UIADD3 + 8 R2UR, no spill.
- **F:** a `__noinline__` call outside the wgmma groups (or even between groups) does not serialize SS wgmma: same
  DEPBAR.LE 0x1 pattern, no warning (`lat/wgcall.cu`).

**F (MID16 reviews, hk_m16b..m16j).** Patterns to look for:

- Shared unit loops with tile-count branches cost the one-tile path +36% static cycles. Separate loop
  instantiations fix it: m16e one-tile is 667/1214 vs the parent's 674/1279.
- A non-shfl'd tile predicate inside a helper epilogue made ptxas add BRA.DIV + WARPSYNC around its shuffles
  (+71 cycles per unit).
- Barrier inits in a divergent region became 23 SYNCS.EXCH waterfall loops.
- Peeling the first MMA with a literal-zero C makes ptxas encode `HMMA …, RZ`, so there are no accumulator zeroing
  instructions (m16k).
- Correction: the 64 `IMAD.U32 Rx, RZ, RZ, URZ` were per-job x0 padding zeroing in a job-change block, not
  per-unit work.
- Always split a loop into its guarded blocks before attributing cost (`lat/blocks.py`). Per-unit:
  - two-tile: 582 instructions / 919 cycles;
  - one-tile: 515 / 776;
  - per-job blocks: 363 / 718.
- LDC.64 parameter pointers reloaded inside the loop.
- Tool for BSSY → source line mapping: `lat/bssy.py`. Tool for loop mix diffs: `lat/mix.py`.

**F (CK MoE WIDE, 019db0a → 3c209fe).**

- A max-sized accumulator array (d[12][4]) shared by all NTT instantiations grew every narrow stage loop by
  +16 instructions, and it spilled 76/268 B at the 96-register cap.
- Keying the wide path off its own stage code (n % 3 barriers, (n & 1) × 48 KB x slots) restored the narrow loops
  to the parent's 288–296.
- Spills that remain with T = 96/88/80: frames of 48/40/16 B. They sit at segment/stage boundaries (wgs_phase slot
  setup, stage-loop control, the flush), not in HGMMA loop bodies.
- `lat/ntt.py` reports the per-N stage-loop table.
- **F (25a02c7, WIDE + REG_DONATE: 640 threads, consumers inc 112, producers dec 32).** STACK 0, no LDL/STL.
  - The consumers use up to R109 of 112.
  - All stage loops are 289–299 instructions at 417–462 cycles, wide N=72..96 included.
  - No C7507. `lat/regmax.py` reports the register actually used per region.

**F. Scan of incumbent arch5_c1_1646 for that rule** (`lat/invr2ur.py`, `loopr2ur.py`).

- MoE WGS/skinny/MID, attention and glue are clean: WGS HGMMA loops have 0 R2UR.
- sk2: 4 loop-invariant R2URs per producer refill (tensor-map operand bounced UR→R→UR; zero TMA coordinates passed
  as registers).
- rq: 2–3 per refill.
- Fix sent: local rematerialization of the operand, and immediate zero coordinates.

## 5. Alignment and placement

**F.** Kernel `.text` sections have sh_addralign 128, so function entry is 128-byte aligned. Loop heads and branch
targets are not aligned: the instruction after any NOP falls at 16/32/64/128 in roughly the uniform ratio, across
moe/sk2/attention/rq (`nops.py`). There is no alignment directive for blocks. Code placement can be moved only
indirectly: through code before the block (the factorial's 128-byte padding) and through the frequency pragma's
layout effect (**U**).
