# Data

Small receipts from the run, in place of the 24 GB of raw results.

| File | What it is |
|---|---|
| `board_runs.csv` | every scored decode-layer run after the move to the public base: time (UTC), job, file, µs per decode step, speedup (earlier own-kernel runs appear only in `rank_history.csv`, as our 1.16x row) |
| `rank_history.csv` | our decode-board rank and best score each time we read the leaderboard, with board #1 where we saw it |
| `per_case_final.csv` | per case: the final file's 400-rep paired gain over the Nov 1 file, its single cold validation pass, first-run and final-11-run-mean judge speedups, and the judge's speed-of-light bound |
| `paired_ladder.csv` | the incumbent's whole-layer paired score against the Nov 1 file over time |
| `gap_ledger.csv` | per case and kernel slot, µs above the model floor on the Nov 4 incumbent, and each case's weighted share of the score |
| `time_shares.csv` | share of GPU kernel time per family in the best run's judge profile, per load |
| `pdl_d1s.csv` | start and end of each kernel in one decode-1 sliding graph replay (nsys) |
| `wgs_stage_bench.csv` | the big MoE's stage-loop microbenchmark, clocks per stage |
| `hbm_ring_bench.csv` | TMA ring streaming rate against bytes in flight per SM |
| `pipeline_throughput_hourly.csv` | measured 200-rep units per hour by job type |
| `lanes.csv` | the expert agents' working periods and areas |
| `adopted_changes.json` | the adoption ledger for the last night: every change adopted into the incumbent, its switches, raw flags, per-case overrides, commit and time |
| `incumbents.jsonl` | every incumbent promotion, oldest first (64 rows) |
| `final_incumbent_point.json` | the final incumbent's switch values and per-case overrides |
| `screened_candidates.md` | every candidate the pipeline screened (6,722 by 02:35 UTC on November 5), ranked by screen score against the Nov 1 file |

Notes on reading them:

- Speedups are against SGLang as the judge measures them, unless a file says "against reviewed production" (our
  pipeline's local reference).
- The judge's run-to-run noise is about ±1.5%, so single scored runs differ by more than most changes.
- Ranks come from the leaderboard reads in the run's logs. The leaderboard also showed teams' best runs only.
- Times in `adopted_changes.json` and `incumbents.jsonl` are Unix seconds.
