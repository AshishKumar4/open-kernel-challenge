"""Soak (before any submission): each scored case's graph replayed the way the judge times it (run_bench: fresh
inputs rotated into the captured buffers, a cold-L2 read before each replay, the reference's graph taking turns with
ours, the order swapped every repetition), every one of our replays' outputs checked against the reference. A failure
is blocking: an early-launch PDL chain once passed its first checks and failed on the judge at timed repetition 5, a
dependent having read its predecessor's data through a stale L1 (muse-spark-open EXPERIMENTS.md, L11/L12).

    python3 harness/soak.py <layer file> [--reps 200] [--fresh 20] [--cases 0,1,...] [--json out]
"""
import argparse
import json
import pathlib
import re
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import okc_harness as H  # noqa: E402

ERR = re.compile(r"(update|attention input) relative L2 error ([0-9.e+-]+) \(worst row ([0-9.e+-]+)\) \[bound ([0-9.e+-]+)\]")


def soak(prob, case, reps, fresh, flush):
    """case -> {replays, failures, worst errors per output and their bounds}"""
    datas = [prob.inputs(case, case["seed"] + k) for k in range(fresh)]
    ident = repr(sorted(case.items()))
    d_sub, d_base, held = H.clone_inputs(datas[0]), H.clone_inputs(datas[0]), {}
    gs = H.capture(lambda: held.__setitem__("o", prob.call(prob.sub, prob.sub_state, d_sub)))
    gb = H.capture(lambda: prob.call(prob.base, prob.base_state, d_base))
    out = {"replays": 0, "failures": [], "worst": {}}
    with H.gpu_timing():
        for r in range(reps):
            k = (r // 2) % len(datas)
            for dst in (d_sub, d_base):
                for key, v in datas[k].items():
                    if isinstance(v, torch.Tensor):
                        dst[key].copy_(v)
            torch.cuda.synchronize()
            for g in ((gs, gb) if r % 2 == 0 else (gb, gs)):
                H.time_graph(g, flush, 1)
            ok, msg = prob.check(d_sub, held["o"], key=(ident, case["seed"] + k), clean=datas[k])
            out["replays"] += 1
            for what, err, row, bound in ERR.findall(msg):
                w = out["worst"].setdefault(what, {"error": 0.0, "worst_row": 0.0, "bound": float(bound)})
                w["error"], w["worst_row"] = max(w["error"], float(err)), max(w["worst_row"], float(row))
            if not ok:
                out["failures"].append({"repetition": r, "input": k, "check": msg})
                print(f"  FAIL at repetition {r} (input {k}): {msg}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--fresh", type=int, default=20)
    ap.add_argument("--cases", default=None, help="indices into the scored cases")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    prob = H.Problem("layer", pathlib.Path(args.file))
    cases = H.scored_cases(prob.dir)
    pick = [int(i) for i in args.cases.split(",")] if args.cases else range(len(cases))
    flush, rows = H.l2_flusher(), []
    for i in pick:
        case = cases[i]
        res = soak(prob, case, args.reps, args.fresh, flush)
        name = f"{case['load']} {case['layer']}"
        print(f"{name}: {res['replays']} replays checked, {len(res['failures'])} failed; worst " +
              "; ".join(f"{w} {v['error']:.3e} (row {v['worst_row']:.3e}, bound {v['bound']:.3g})"
                        for w, v in res["worst"].items()), flush=True)
        rows.append({"case": name, **res})
        torch.cuda.empty_cache()
    if args.json:
        pathlib.Path(args.json).write_text(json.dumps({"file": args.file, "reps": args.reps, "fresh": args.fresh,
                                                       "cases": rows}, indent=1))
    sys.exit(3 if any(r["failures"] for r in rows) else 0)


if __name__ == "__main__":
    main()
