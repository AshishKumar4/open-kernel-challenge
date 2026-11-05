"""The clean study's harvest (the owner: the study's value is the ideas taken from it into the incumbent): each
switch value's effect per case, from every measured clean1 trial, as a regression of log case speedup on the switches
(one indicator per non-default value) with a per-GPU intercept, so each effect is matched against the trials without it
at the same other switches and GPU. Effects are % per case and on the weighted overall score, with their t (effect / SE).

    python3 harness/harvest.py run                     (extract here, fit on a build box: receipts/harvest/h<n>.md)
    python3 harness/harvest.py extract <out.json>    (on the PC: the study's measured trials, light)
    python3 harness/harvest.py fit <in.json> <out.md>  (on a build box: the fit and the table)"""
import json
import math
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
BLOCK_S = 3600   # the GPU x time block of the fit's drift fixed effects
WITHDRAWN = {"tanhsilu"}   # (its value 1 broke the accuracy rule: kept as a regressor, never proposed)


def extract(out):
    import sqlite3
    db = sqlite3.connect(f"file:{ROOT}/receipts/abo.db?mode=ro", uri=True)
    sid = db.execute("select study_id from studies where study_name = 'clean1'").fetchone()[0]
    rows = []
    import bisect, datetime, subprocess
    ts = lambda x: datetime.datetime.fromisoformat(x).replace(tzinfo=datetime.timezone.utc).timestamp() if x else None
    # the clean tip over time (its reflog): a trial without a recorded revision was assembled from the tip of its start
    log = sorted((int(l.split()[1].split("{")[1].rstrip("}")), l.split()[0]) for l in subprocess.run(
        ["git", "-C", str(ROOT), "reflog", "show", "clean", "--date=unix", "--format=%h %gd"],
        capture_output=True, text=True).stdout.split("\n") if l.strip())
    at = lambda t: log[max(0, bisect.bisect_right([x for x, _ in log], t) - 1)][1] if t and log else None
    for tid, num, start, done in db.execute("select trial_id, number, datetime_start, datetime_complete from trials "
                                            "where study_id = ? and state = 'COMPLETE'", (sid,)):
        a = {k: json.loads(v) for k, v in db.execute("select key, value_json from trial_user_attributes where trial_id = ?", (tid,))}
        if a.get("outcome") != "measured" or not a.get("cases"):
            continue
        p = {}
        for name, value, dist in db.execute("select param_name, param_value, distribution_json from trial_params where trial_id = ?", (tid,)):
            p[name] = json.loads(dist)["attributes"]["choices"][int(value)]
        p = {**a.get("fixed", {}), **p}   # (the adopted switches it was built with, fixed: not sampled)
        if not a.get("clean"):   # (only trials with the revision they actually assembled: the reflog's tip at a trial's
            continue             # start mismatched the recorded one on 59 % of trials, the study caching its source)
        rows.append({"trial": num, "host": a.get("host"), "params": p, "cases": a["cases"], "rev": a["clean"],
                     "t": ts(done)})   # (its source revision and completion time: the fit's cohort and GPU x hour fixed effects)
    import ast   # each switch's default: its first value in the study's DIMS
    src = (ROOT / "harness/clean_study.py").read_text()
    i = src.index("DIMS = {") + len("DIMS = ")
    depth, j = 0, i
    for j, ch in enumerate(src[i:], i):
        depth += ch == "{"
        depth -= ch == "}"
        if depth == 0:
            break
    dims = ast.literal_eval(src[i:j + 1])
    json.dump({"defaults": {k: v[0] for k, v in dims.items()}, "rows": rows}, open(out, "w"))
    print(f"{len(rows)} measured trials -> {out}")


def fit(inp, out):
    import numpy as np
    data = json.load(open(inp))
    rows, given = (data["rows"], data["defaults"]) if isinstance(data, dict) else (data, {})
    dims = {}
    for r in rows:
        for k, v in r["params"].items():
            dims.setdefault(k, set()).add(v)
    # the default (a value every dimension has first in DIMS) is the value most trials share at the study's start: take
    # the smallest for the binary switches, and the clean defaults for the many-valued ones
    default = {"stages": 3, "vunroll": 2, "nsplit16": 11, "v32s": 4, **given}   # (the study's own defaults: the ptxas dials'
    # are 5 and "none", and a non-default baseline made every other value of that switch read against it)
    cols = [(k, v) for k in sorted(dims) for v in sorted(dims[k]) if v != default.get(k, 0)]
    hosts = sorted({r["host"] for r in rows})
    # fixed effects: the trial's source cohort (its clean revision: a code change landing mid-study) and its GPU x
    # hour block (drift), so neither reads as a switch effect; switch effects are within cohort and block
    block = lambda r: f"{r['host']}@{int((r.get('t') or 0) // BLOCK_S)}"
    cohorts = sorted({str(r.get("rev")) for r in rows})
    blocks = sorted({block(r) for r in rows})
    fe = [("rev", c) for c in cohorts[1:]] + [("block", b) for b in blocks[1:]]
    X = np.array([[1.0] + [float(str(r.get("rev")) == c if kind == "rev" else block(r) == c) for kind, c in fe] +
                  [float(r["params"].get(k, default.get(k, 0)) == v) for k, v in cols] for r in rows])
    hosts = [None] * (1 + len(fe))   # (the switch columns start after the intercept and the fixed effects)
    cases = sorted({c for r in rows for c in r["cases"]})
    lines = [f"# clean1 harvest: {len(rows)} measured trials, {len(cols)} switch values; fixed effects: {len(cohorts)} source "
             f"revisions, {len(blocks)} GPU x {BLOCK_S // 60}-minute blocks", "",
             "Effect: % change of the case's speedup when the value is on (others held fixed); t = effect / SE; "
             "+ marks t >= 2.", ""]
    table = {}
    for c in cases + ["overall"]:
        y = np.array([math.log(r["cases"][c]) if c != "overall" else   # (overall: the layer's weights, sliding 46, full 16)
                      sum((46 if k.endswith("sliding") else 16) * math.log(x) for k, x in r["cases"].items()) /
                      sum(46 if k.endswith("sliding") else 16 for k in r["cases"]) for r in rows])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ beta
        dof = max(1, len(y) - X.shape[1])
        cov = np.linalg.pinv(X.T @ X) * (resid @ resid) / dof
        for j, (k, v) in enumerate(cols, start=len(hosts)):
            se = math.sqrt(max(cov[j, j], 1e-30))
            table.setdefault((k, v), {})[c] = (100 * (math.exp(beta[j]) - 1), beta[j] / se)
    lines.append("| switch | n on | overall | cases helped (t >= 2) | cases hurt (t <= -2) |")
    lines.append("|---|---|---|---|---|")
    for (k, v), eff in sorted(table.items(), key=lambda kv: -kv[1]["overall"][1]):
        n = sum(1 for r in rows if r["params"].get(k) == v)
        o = eff["overall"]
        helped = ", ".join(f"{c} {e:+.2f}% (t {t:.1f})" for c, (e, t) in eff.items() if c != "overall" and t >= 2)
        hurt = ", ".join(f"{c} {e:+.2f}% (t {t:.1f})" for c, (e, t) in eff.items() if c != "overall" and t <= -2)
        tag = " (withdrawn)" if k in WITHDRAWN else ""
        lines.append(f"| {k}={v}{tag} | {n} | {o[0]:+.2f}% (t {o[1]:.1f}){' +' if o[1] >= 2 else ''} | {helped or '-'} | {hurt or '-'} |")
    pathlib.Path(out).write_text("\n".join(lines) + "\n")
    pathlib.Path(out).with_suffix(".json").write_text(json.dumps(   # (the per-case table: the per-shape composite's input)
        {"defaults": default, "trials": len(rows), "effects": {f"{k}={v}": {c: {"pct": e, "t": t} for c, (e, t) in eff.items()}
                                                               for (k, v), eff in table.items()}}, indent=1))
    print("\n".join(lines))


def run():
    """the next harvest: its trials extracted here, fitted on a build box (the owner's rule: no analysis on the PC)"""
    import subprocess
    sys.path.insert(0, str(ROOT / "harness"))
    import vast_hosts
    d = ROOT / "receipts/harvest"
    d.mkdir(exist_ok=True)
    n = 1 + max([int(p.stem[1:]) for p in d.glob("h*.md") if p.stem[1:].isdigit()] or [0])
    extract(d / f"h{n}_trials.json")
    box = next(b for b in vast_hosts.boxes().values() if b.get("role") == "build" and b.get("ready") and not b.get("offline"))
    ssh = vast_hosts.SSH + ["-p", box["port"], f"root@{box['ip']}"]
    subprocess.run(["scp"] + vast_hosts.SSH[1:] + ["-P", box["port"], "-q", __file__, str(d / f"h{n}_trials.json"),
                    f"root@{box['ip']}:/root/"], check=True, timeout=120)
    r = subprocess.run(ssh + [f"cd /root && nice -n 19 python3 harvest.py fit h{n}_trials.json h{n}.md > /dev/null && cat h{n}.md"],
                       capture_output=True, text=True, timeout=600, check=True)
    (d / f"h{n}.md").write_text(r.stdout)
    subprocess.run(["scp"] + vast_hosts.SSH[1:] + ["-P", box["port"], "-q", f"root@{box['ip']}:/root/h{n}.json", str(d / f"h{n}.json")],
                   timeout=120)   # (the per-case table, for the per-shape composite)
    print(f"harvest h{n}: receipts/harvest/h{n}.md")


if __name__ == "__main__":
    {"run": run, "extract": extract, "fit": fit}[sys.argv[1]](*sys.argv[2:])
