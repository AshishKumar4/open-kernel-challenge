# The testing pipeline

This is the code we ran during the challenge, cleaned up for reading: I replaced machine paths, key paths and account
details with environment variables, and removed agent names from comments. It isn't a packaged tool. It expects the
workspace the challenge's `okc init` creates, and the comments still carry a lot of history from the run.

## What it expects

- A workspace root with `muse_<problem>/` directories holding the challenge's files (`TASK.md`, `reference.py`,
  `checks.py`, `baseline.py`, `nvfp4.py`) and a scored profile in `muse_<problem>/profiles/`. These come from the
  challenge, so they aren't in this repo.
- `candidates/` (one `layer_<name>.py` per candidate), `reviewed/muse_decode_layer.py` (the production file every bench
  runs against), and `receipts/` for results. The harness takes the workspace root to be the parent of `harness/`.
- The challenge's Docker image on the GPU boxes (`lmsysorg/sglang:dev-muse-spark-open`), with this directory at
  `/root/okc/harness`.
- `optuna` and `sqlalchemy` on the machine that runs the study, the `vastai` and `nebius` CLIs for the box tooling, and
  the challenge's `okc` CLI for judge checks.
- `OKC_STATE` (default `~/.okc`): the PC-side state directory (the relay's queue, known_hosts, logs). `OKC_SSH_KEY`
  (optional): the private key the boxes accept; without it, ssh uses its own defaults.

## The flow

```
agent or TPE study -> submit.py -> relay.py (on the PC) -> build box (box_build.sh, cpu_prebuild.py)
                   -> box_ship.sh -> GPU box queue (vqg.sh) -> benchd.py or okc_harness.py -> receipts
```

| File | What it does |
|---|---|
| `okc_harness.py` | correctness tests and judge-style timing for the five problems: CUDA-graph replay after a read of twice the L2, alternating with the other side on fresh inputs, empty-graph replay subtracted, median of paired ratios |
| `benchd.py` | a persistent bench server per GPU slot: reference, weights and production's prepared state made once, then one candidate per request |
| `tvm_publish.py` | builds each tvm-ffi module in a private directory and publishes it by an atomic rename (a shared build directory once hung every bench) |
| `submit.py` | the single entry point for GPU work, with priority levels 0 to 4 kept end to end |
| `relay.py` | the PC-side relay: build queue, shipping, every job's state, moving jobs off stopped or draining GPUs |
| `vast_hosts.py` | the box registry and SSH options shared by everything else |
| `confirm.py` | screens and confirmations against the incumbent's established reading and hourly controls, bundles, adoption |
| `lane_asm.py`, `lane_report.py` | assemble an agent's change on the current incumbent, and report its screen back |
| `clean_study.py` | the Optuna TPE study over the decode layer's compile-time switches, per case; uses `kernels/decode_layer_src/assemble.py` |
| `harvest.py`, `composite.py` | regress each switch's per-case effect from the study's trials, and build a per-case composite from it |
| `identity.py` | file and device-source md5s, so a reading is tied to the exact program it measured |
| `freeze.py`, `unifdef.py`, `cxxconst.py`, `noassert.py` | turn a built candidate into the final source: resolve its patches, drop dead `#if` groups and constant branches, strip asserts |
| `submit_validate.py`, `cold_prepare.py`, `soak.py`, `import_check.py` | validation before a scored run: cold `prepare` under the judge's 300 s cap, ordered gates, a 12 x 200 judge-style soak |
| `judge_check.py` | one unscored judge check of a candidate with a large confirmed win, keeping a reserve |
| `h200/` | box scripts: setup bundles, build boxes, GPU runners, telemetry, Vast onboarding and guard, the Nebius spot watcher |
| `ptxas/` | the SASS tools behind [`../notes/ptxas-re.md`](../notes/ptxas-re.md): control-bit decoding, waterfall-loop scans, per-build compiles with exact flags |

## What I'd keep if I did it again

The harness, the single queue with priorities, the build boxes, the bench servers with cached inputs, screens at 200
repetitions against an established reading, confirmations at 400 with full checks, and the Nebius watcher. I'd drop most
of the guards and special cases that grew around them.
