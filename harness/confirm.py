"""Confirmation (owner): ONE screen of a file in the study's mode (single arm vs reviewed production, the profile's
seeds) at 400 reps, through the one flow at level 1 (harness/submit.py). Screens and confirmations are read against the
incumbent on that GPU: the mean of its CTL_N controls there nearest in time (receipts/control.jsonl; today's controls
drift, so not the whole day's) and the GPU's screen noise (the sd of its CTL_N nearest controls around their own
incumbent's mean): within noise or above overall at mean - 2 sd, ahead overall over mean + 2 sd, ahead on a case over
that case's mean + 2 sd. A screen within noise or above overall, or ahead on a case, gets a confirmation (the driver adds
study trials; a lane adds its own files). A confirmation ahead overall becomes the incumbent: incumbent.json names the
snapshot it ran (candidates/layer_cf<NNN>_<name>.py, its md5); its own run is not a control (it was selected, so it is
biased high), and the driver runs BASE fresh controls on each GPU at once; its cases ahead go to per-shape dispatch
. Before a GPU has BASE controls of the current incumbent a reading there is unclassified, never behind:
it waits and is read once they land. A lane's screen (submit.py's command line records it, LANES) gets the same reading and trigger at
collect, against the incumbent of the screen's time (INCS, the promotions); a confirmation is read against the incumbent
it was queued against, and only one against the current incumbent can promote.

    python3 harness/confirm.py add <name> <GPU host>   # candidates/layer_<name>.py
    python3 harness/confirm.py collect                  # fetch finished confirmations, print and record them

Each runs as cf<NNN>_<name> (a copy, candidates/layer_cf<NNN>_<name>.py; its bench's files /root/out/abo_cf<NNN>_<name>.*;
receipt names are never reused). Records: receipts/evidence/confirm.jsonl (one line when queued, one when collected)."""
import datetime, fcntl, hashlib, json, math, pathlib, shutil, sqlite3, statistics, subprocess, sys, time

ROOT = pathlib.Path(__file__).resolve().parent.parent
EV = ROOT / "receipts/evidence"
sys.path.insert(0, str(ROOT / "harness"))
import lane_asm, lane_report, submit, vast_hosts  # noqa: E402
REPS, CTL_N, SD = 400, 12, 2
BASE = 2   # a new incumbent's baseline on a GPU: this many fresh controls (its winning confirmation is selected, so biased high)
FIX_FLAG = 0.005   # a correctness delta adopted this far behind overall is flagged to the orchestrator and its owner
INC, CTL = ROOT / "receipts/incumbent.json", ROOT / "receipts/control.jsonl"
INCS, LANES = ROOT / "receipts/incumbents.jsonl", EV / "lane_screens.jsonl"   # (every promotion; every lane screen queued)
ADOPTED, ASSEMBLY = EV / "adopted.json", EV / "lane_assembly.jsonl"   # (the adopted lane deltas; each lane() assembly)
HOLD = EV / "promotion_hold.json"   # {"reference": an incumbent's name, "why": ...}: while it exists, a confirmation
# promotes only above that reference's level on its GPU too


def sh(host, cmd, timeout=60, stdin=None):
    port, ip = vast_hosts.load(offline=True)[0][host][:2]
    return subprocess.run(vast_hosts.SSH + ["-p", port, f"root@{ip}", cmd], input=stdin, capture_output=True, text=True,
                          timeout=timeout).stdout


def md5(path):
    return hashlib.md5(pathlib.Path(path).read_bytes()).hexdigest()


def records():
    f = EV / "confirm.jsonl"
    return [json.loads(l) for l in open(f)] if f.exists() else []


def append(rec):
    EV.mkdir(parents=True, exist_ok=True)
    with open(EV / "confirm.jsonl", "a") as f:
        f.write(json.dumps(rec, sort_keys=True) + "\n")


def incumbent():
    return json.loads(INC.read_text())


def score(cs, keys):
    """a result's speed on a set of cases: their weighted geometric mean (sliding 46, full 16, as the layer weights them)"""
    w = {k: 46 if k.endswith("sliding") else 16 for k in keys}
    return math.exp(sum(w[k] * math.log(cs[k]) for k in keys) / sum(w.values()))


def cohort(host, identity):
    """the GPU a reading belongs to: <box>g<cuda_visible>, and its physical GPU's UUID where recorded (a preempted VM can
    restart on another H200: spot1's serial changed at its 11-05 00:32 UTC restart). Controls, baselines and readings are
    compared within one cohort, so a restart onto a new GPU starts a fresh baseline"""
    h = host[:host.rindex("g") + 1] + str(identity.get("cuda_visible") or host[host.rindex("g") + 1:])
    return f"{h}:{identity['gpu_uuid']}" if identity.get("gpu_uuid") else h


def reading(name):
    """incumbent <name>'s established number (the owner, 11-05): the per-case speedups of its own 400-rep confirmation
    (the run that made it the incumbent, receipts/incumbents.jsonl's file), or None"""
    f = next((json.loads(l)["file"] for l in reversed(open(ROOT / "receipts/incumbents.jsonl").read().splitlines())
              if json.loads(l)["name"] == name), "")
    run = pathlib.Path(f).stem.removeprefix("layer_")
    return next((r["cases"] for r in reversed(records()) if r["event"] == "result" and r["run"] == run and r.get("cases")),
                None) if run.startswith("cf") else None


def noise(keys=None):
    """the screen's run-to-run sd (a property of the bench, from the controls already run: within each (GPU, incumbent)
    group, pooled), overall or on keys, and per case"""
    groups = {}
    for r in map(json.loads, open(CTL)):
        if "cases" in r:
            groups.setdefault((r["host"], r["incumbent"]), []).append(r)

    def sd(get):
        gs = [[get(r) for r in g] for g in groups.values() if len(g) > 1]
        df = sum(len(g) - 1 for g in gs)
        return math.sqrt(sum((x - statistics.mean(g)) ** 2 for g in gs for x in g) / df) if df else 0.0
    return sd((lambda r: r["geomean"]) if keys is None else (lambda r: score(r["cases"], keys))), {
        k: sd(lambda r, k=k: r["cases"].get(k, 1.0)) for k in CASE_KEYS}


def relative(res, inc, tc):
    """(11-05, option a) a screen of a targeted candidate judged within itself: its target cases' ratio to the
    incumbent's established per-case reading, divided by its untouched cases' mean ratio to theirs (the screen's GPU and
    the reference's selection offset cancel); a candidate touching every case (a bundle, a composite): against the mean of
    the incumbent's hourly controls (option b), else its reading. (verdict, corrected score, reference, sd, offset)"""
    rd, cs = reading(inc) or {}, cases(res)
    keys = [k for k in CASE_KEYS if k in cs and k in rd]
    tk = [k for k in (tc or []) if k in keys]
    un = [k for k in keys if k not in tk]
    if tk and un:
        off = score({k: cs[k] / rd[k] for k in un}, un)
        sc, ref, sd = score(cs, tk) / off, score(rd, tk), noise(tk)[0]
    else:
        tk, off = keys or list(cs), 1.0
        ctl = [r for r in map(json.loads, open(CTL)) if r["incumbent"] == inc and "cases" in r]
        sc = score(cs, tk)
        ref = statistics.mean(score(r["cases"], tk) for r in ctl) if ctl else (score(rd, tk) if rd else None)
        sd = noise(tk)[0]
    if ref is None:
        return "unclassified", sc, None, sd, off
    return ("ahead" if sc > ref + SD * sd else "behind" if sc < ref - SD * sd else "within"), sc, ref, sd, off


def controls(host, t=None, name=None, keys=None):
    """the incumbent <name> (default: the current one) as a reference, (mean, sd) overall and per case: its established
    number, its own confirmation reading, with the bench's pooled noise (the owner, 11-05: no fresh controls; a screen
    times candidate against production rep by rep, so its score is already relative); for an incumbent without one, its
    controls on <host> as before"""
    rd = reading(name or incumbent()["name"])
    if rd:
        o, per = noise(keys)
        return ((score(rd, keys) if keys else score(rd, list(rd)), o),
                {k: (rd[k], per.get(k, o)) for k in rd})
    return controls_on(host, t, name, keys)


def controls_on(host, t=None, name=None, keys=None):
    """the incumbent <name> (default: the current one) on <host> near <t> (default: now): the mean of its CTL_N controls
    there nearest <t> and the GPU's
    screen noise, the sd of the GPU's CTL_N controls nearest <t> around their own incumbent's mean (a property of the GPU
    and the screen); (mean, sd) overall and per case, or None before the incumbent has BASE controls there (no baseline).
    keys: the overall is score() on those cases only (a targeted candidate's changed cases)"""
    near = lambda rs: sorted(rs, key=lambda r: abs(r["t"] - (t or time.time())))[:CTL_N]
    every = [r for r in map(json.loads, open(CTL)) if r["host"] == host and "cases" in r]
    mine = near([r for r in every if r["incumbent"] == (name or incumbent()["name"])])
    if len(mine) < BASE:
        return None
    pool = {}
    for r in near(every):
        pool.setdefault(r["incumbent"], []).append(r)

    def dist(get):
        groups = [[get(r) for r in g] for g in pool.values() if len(g) > 1]
        df = sum(len(g) - 1 for g in groups)
        ss = sum((x - statistics.mean(g)) ** 2 for g in groups for x in g)
        return statistics.mean(get(r) for r in mine), math.sqrt(ss / df) if df else 0.0
    overall = (lambda r: r["geomean"]) if keys is None else (lambda r: score(r["cases"], keys))
    return dist(overall), {k: dist(lambda r, k=k: r["cases"][k]) for k in mine[-1]["cases"]}


def cases(res):
    return {c["case"]["load"] + " " + c["case"]["layer"]: c["speedup"] for c in res["cases"]}


CASE_KEYS = [f"{l} {y}" for l in ("decode 1", "decode 16", "decode 32", "verify 1", "verify 16", "verify 32")
             for y in ("sliding", "full")]


def ahead(d, res):
    """a bench result against the incumbent's distribution d (controls()): overall "ahead", "within" (noise), "behind"
    or "unclassified" (no baseline yet: never a loss), and the cases ahead"""
    if not d:
        return "unclassified", []
    (m, s), dc = d
    mine = cases(res)
    g = res["geomean"]
    return ("ahead" if g > m + SD * s else "within" if g >= m - SD * s else "behind",
            [k for k, (cm, cs) in dc.items() if mine[k] > cm + SD * cs])


def infeasible(md5, why):
    """the study's measured trials of these bytes become infeasible (a confirmation's full checks failed them)"""
    db = sqlite3.connect(ROOT / "receipts/abo.db", timeout=60)
    with db:
        for (tid,) in db.execute("select trial_id from trial_user_attributes where key = 'cand_md5' and value_json = ?",
                                 (json.dumps(md5),)).fetchall():
            if db.execute("select value_json from trial_user_attributes where trial_id = ? and key = 'outcome'", (tid,)).fetchone() != ('"measured"',):
                continue
            db.execute("update trial_values set value = 0.0 where trial_id = ?", (tid,))
            db.execute("delete from trial_system_attributes where trial_id = ? and key = 'constraints:feasible'", (tid,))
            db.execute("insert into trial_system_attributes (trial_id, key, value_json) values (?, 'constraints:feasible', '1.0')", (tid,))
            for k, v in (("outcome", "infeasible"), ("why", why)):
                db.execute("delete from trial_user_attributes where trial_id = ? and key = ?", (tid, k))
                db.execute("insert into trial_user_attributes (trial_id, key, value_json) values (?, ?, ?)", (tid, k, json.dumps(v)))


def incumbent_at(t):
    """the incumbent's name at time t (receipts/incumbents.jsonl: each promotion, the first one's since)"""
    rs = [json.loads(l) for l in open(INCS)]
    return max((r for r in rs if r["since"] <= t), key=lambda r: r["since"], default=rs[0])["name"]


def add(name, host, against=None, changed=None):
    """one 400-rep confirmation of candidates/layer_<name>.py on host, read against the incumbent at its collection;
    changed: the cases its code changes (a targeted candidate: read on those only)"""
    run = f"cf{len([r for r in records() if r['event'] == 'queued']):03d}_{name}"
    cand = ROOT / f"candidates/layer_{run}.py"
    shutil.copyfile(ROOT / f"candidates/layer_{name}.py", cand)
    submit.screen(run, host, 1, REPS, checks=True)
    append({"event": "queued", "run": run, "cand": f"candidates/layer_{name}.py", "cand_md5": md5(cand), "reps": REPS,
            "host": host, "against": against or incumbent()["name"], **({"changed": changed} if changed else {}),
            "t": round(time.time())})
    print(f"{run}: candidates/layer_{name}.py, {REPS} reps, level 1 on {host}", flush=True)


def collect():
    done = {r["run"] for r in records() if r["event"] != "queued"}
    for q in [r for r in records() if r["event"] == "queued" and r["run"] not in done]:
        run, host = q["run"], submit.gpu_of(q["run"], q["host"])
        if host not in vast_hosts.load(offline=True)[0]:
            append({"event": "failed", "run": run, "why": f"its host {host} left", "t": round(time.time())})
            continue
        if q.get("paired") or q.get("pair_job"):   # (the paired confirmations of 11-04 morning: withdrawn)
            continue
        if not sh(host, f"ls /root/out/abo_{run}.done 2>/dev/null").strip():
            continue
        try:
            res = json.loads(sh(host, f"cat /root/out/abo_{run}.json"))
            if "failed" in res:   # the full eager checks failed what the screen's replay checks passed: infeasible
                why = f"fails the full checks at its confirmation {run}: {res['failed']} ({res.get('case')})"
                append({"event": "failed", "run": run, "why": why, "t": round(time.time())})
                infeasible(q["cand_md5"], why)
                print(f"{run}: {why}; its trials infeasible", flush=True)
                if run.split("_", 1)[1].startswith("bundle"):   # (which member fails is found singly)
                    unbundle(run.split("_", 1)[1])
                continue
        except ValueError:
            tail = sh(host, f"grep -E 'Error|error|FAILED|Killed|Terminated' /root/out/abo_{run}.log | tail -2").strip()
            append({"event": "failed", "run": run, "why": tail[-300:], "t": round(time.time())})
            print(f"{run}: no result: {tail[-200:]}", flush=True)
            continue
        against = incumbent()["name"]   # (the incumbent at collection, not at queueing: one that changed while it ran is no
        key = cohort(host, res["identity"])   # longer the one to beat: cf268, ahead of both, was dropped at 04:29)
        d = controls(key, None, against)
        if not d:   # (no reference for that incumbent yet: read on a later collect)
            continue
        won = ahead(d, res)[1]
        # (the same policy as screens, confirm.relative: its changed cases per case against their own noise, its
        # unchanged cases cancelling the screen's offset; the hourly controls' mean only for a candidate touching every case)
        over, sc, ref, sd, off = relative(res, against, q.get("changed"))
        append({"event": "result", "run": run, "host": host, "cohort": key, "against": against, "queued_against": q.get("against"),
                "geomean": res["geomean"], "control_mean": ref, "control_sd": sd, "offset": off, "score": sc,
                "overall": over, "cases_ahead": won, "cases": cases(res), "identity": res["identity"], "t": round(time.time())})
        print(f"{run}: {res['geomean']:.4f} on {host}: score {sc:.4f} (offset {off:.4f} divided out) vs {ref and f'{ref:.4f}'} "
              f"± {sd:.4f}: {over}; cases ahead: {', '.join(won) or 'none'}", flush=True)
        nm = pathlib.Path(q["cand"]).stem.removeprefix("layer_")
        if q.get("changed"):
            append({"event": "target", "run": run, "changed": q["changed"], "score": sc, "control_mean": ref,
                    "control_sd": sd, "offset": off, "target": over, "t": round(time.time())})
        hold = json.loads(HOLD.read_text()) if HOLD.exists() else None   # (promotions held to a reference's level)
        if over == "ahead" and hold:
            dh = controls(key, None, hold["reference"])
            if not dh or res["geomean"] <= dh[0][0] + SD * dh[0][1]:
                print(f"{run}: ahead, but promotions are held ({hold['why']}): not above {hold['reference']}'s level "
                      f"on {host}", flush=True)
                append({"event": "held", "run": run, "reference": hold["reference"], "t": round(time.time())})
                over = "within"
        if over == "behind" and nm.startswith("bundle"):
            unbundle(nm)
        adopt = over == "ahead" if nm.startswith("arch5_") else over in ("ahead", "within")   # (a lane's delta: the owner's
        # stacking rule, within or ahead adopts; a study trial: ahead)
        fix = not nm.startswith("arch5_") and ASSEMBLY.exists() and any(
            r["name"] == nm and r["md5"] == q["cand_md5"] and r.get("correctness") for r in map(json.loads, open(ASSEMBLY)))
        if fix:   # (a contract-required correctness delta is adopted whatever it costs: a wrong result on the judge
            adopt = True   # costs more than any speed; one beyond FIX_FLAG is flagged so a cheaper correct form is sought)
        frz = ROOT / "receipts/evidence/adoption_freeze.json"   # (no adoption after the final file is fixed)
        if adopt and frz.exists() and time.time() >= json.loads(frz.read_text())["at"]:
            print(f"{run}: {over}, but adoption is frozen ({json.loads(frz.read_text())['why']})", flush=True)
            append({"event": "held", "run": run, "why": "adoption frozen", "t": round(time.time())})
            adopt = False
        if adopt and res["identity"]["sub_md5"] == q["cand_md5"] and against == incumbent()["name"]:   # the
            # bytes it ran (the snapshot); its own run is not a control (selected: biased high), the driver runs BASE per GPU
            old, base = incumbent(), adopted()
            if nm.startswith("arch5_"):   # a study trial carries every adopted lane delta, or it is not promoted
                n = int(nm.rsplit("_", 1)[1])
                tp = trial_point(n)
                lost = lacks(base, tp, trial_start(n), path=ROOT / f"candidates/layer_{run}.py", rev=trial_attr(n, "clean"))
                point = {"point": tp}
            else:   # a lane delta: only one lane() assembled (its switches and commit known, so the study carries them)
                rec = next((r for r in reversed([json.loads(l) for l in open(ASSEMBLY)]) if r["name"] == nm
                            and r["md5"] == q["cand_md5"]), None) if ASSEMBLY.exists() else None
                if rec is None:
                    print(f"{run}: {over}, but held: not assembled by clean_study.lane(), so the study could not carry its "
                          f"delta; assemble it there and confirm that", flush=True)
                    append({"event": "held", "run": run, "why": "not assembled by lane()", "t": round(time.time())})
                    continue
                sw, raw = rec["switches"], rec["raw"]
                point = {"point": {**rec["point"], **sw}}
                lost = lacks(base, point["point"], rec["t"], sw, rec.get("per_case"), ROOT / f"candidates/layer_{run}.py",
                             rec["clean"])
            if lost and not nm.startswith("arch5_"):   # (an adoption while it ran: the same lane on the current incumbent,
                requeue(run, nm, over, lost, q.get("changed"))   # confirmed again, no hold)
                continue
            if lost:
                print(f"{run}: {over}, but held: it lacks adopted lane deltas {lost}", flush=True)
                append({"event": "held", "run": run, "why": f"lacks adopted lane deltas {lost}", "t": round(time.time())})
                continue
            if not nm.startswith("arch5_"):
                if rec.get("per_case"):   # (a composite: its per-case overrides on the adopted ones)
                    pc = {c: dict(v) for c, v in (base.get("per_case") or {}).items()}
                    for c, v in rec["per_case"].items():
                        pc.setdefault(c, {}).update(v)
                    base["per_case"] = pc
                base["switches"].update(sw)
                base["raw"] = base["raw"] + [r for r in raw if r not in base["raw"]]
                base["commits"] = sorted(set(base["commits"]) | {rec["clean"]})
                try:   # (into clean at once: the study builds only from a clean carrying every adopted commit)
                    lane_asm.land(rec["clean"])
                except RuntimeError as e:
                    print(f"{run}: adopted, but {rec['clean']} did not land on clean: {e}", flush=True)
                base.setdefault("history", []).append({"name": nm, "run": run, "switches": sw, "raw": raw,
                                                       "per_case": rec.get("per_case"), "commit": rec["clean"],
                                                       "t": round(time.time())})
                ADOPTED.write_text(json.dumps(base, indent=1))
            if base.get("per_case"):   # (every build carries the adopted per-case overrides: lane() and the study apply them)
                point["per_case"] = base["per_case"]
            INC.write_text(json.dumps({"name": nm, "file": f"candidates/layer_{run}.py", "md5": q["cand_md5"],
                                       "since": round(time.time()), "over": old["name"], "runs": [run], **point}, indent=1))
            with open(INCS, "a") as f:
                f.write(json.dumps({"name": nm, "since": round(time.time()), "file": f"candidates/layer_{run}.py"}) + "\n")
            print(f"{nm} is the incumbent (candidates/layer_{run}.py)", flush=True)
            refresh_screens()
            if fix and res["geomean"] < m * (1 - FIX_FLAG):
                why = f"correctness delta adopted at {100 * (res['geomean'] / m - 1):+.2f}% overall: seek a cheaper correct form"
                append({"event": "flag", "run": run, "why": why, "t": round(time.time())})
                print(f"{run}: FLAG for the orchestrator and its owner: {why}", flush=True)


def adopted():
    """the adopted lane deltas every study trial carries: switches (fixed in the study), raw build flags, and the
    lane commits that must be in the clean tip"""
    return json.loads(ADOPTED.read_text()) if ADOPTED.exists() else {"switches": {}, "raw": [], "commits": []}


def lacks(base, point, t, own=(), own_pc=None, path=None, rev=None):
    """the adopted lane deltas a candidate built at <point> at time t does not carry, apart from those its own delta
    (switches <own>, per-case overrides <own_pc>) changes: lane() and the study build from adopted.json as it was then,
    which its history replays. Promoting one that lacks any drops an adopted win silently (st_3342, assembled before
    MOE_RAFN was adopted, dropped it; composite2 dropped st_3342's switches). A switch its point does not name is
    missing unless its file (<path>) shows the value built (absence cannot show what an older candidate built).
    Its source too: the revision it was assembled from (<rev>) must contain every adopted lane commit, else a candidate
    queued before a code-only fix (279d5d5, 3542d79) matches every knob yet restores the unsafe source"""
    raw, pc = set(), {}
    for h in base.get("history", []):
        if h["t"] <= t:
            raw |= {tuple(r) for r in h.get("raw", [])}
            for c, v in (h.get("per_case") or {}).items():
                pc.setdefault(c, {}).update(v)
    built = None
    lost = {}
    for k, v in base["switches"].items():
        have = point.get(k)
        if k not in point and path is not None:   # (a point recorded before lane() recorded its built point)
            if built is None:
                import clean_study
                built = clean_study.assembled_point(path.read_text())
            have = built.get(k)
        if k not in own and have != v:
            lost[k] = (v, have)
    lost.update({f"{b} {f}": "not built" for b, f in map(tuple, base["raw"]) if (b, f) not in raw})
    git = pathlib.Path(__file__).resolve().parent.parent
    lost.update({f"commit {c}": f"not in its source {rev}" for c in base["commits"] if rev is None or subprocess.run(
        ["git", "-C", str(git), "merge-base", "--is-ancestor", c, rev], capture_output=True).returncode})
    for c, sw in (base.get("per_case") or {}).items():
        lost.update({f"{c} {k}": (v, pc.get(c, {}).get(k)) for k, v in sw.items()
                     if k not in (own_pc or {}).get(c, {}) and pc.get(c, {}).get(k) != v})
    return lost


def trial_point(number):
    """a study trial's switches: its sampled parameters and the fixed ones it was built with"""
    db = sqlite3.connect(f"file:{ROOT}/receipts/abo.db?mode=ro", uri=True, timeout=60)
    sid = db.execute("select study_id from studies where study_name = 'clean1'").fetchone()[0]
    tid = db.execute("select trial_id from trials where study_id = ? and number = ?", (sid, number)).fetchone()[0]
    p = {name: json.loads(dist)["attributes"]["choices"][int(value)] for name, value, dist in
         db.execute("select param_name, param_value, distribution_json from trial_params where trial_id = ?", (tid,))}
    fx = db.execute("select value_json from trial_user_attributes where trial_id = ? and key = 'fixed'", (tid,)).fetchone()
    return {**(json.loads(fx[0]) if fx else {}), **p}


def trial_attr(number, key):
    """a study trial's user attribute (its clean revision: "clean")"""
    db = sqlite3.connect(f"file:{ROOT}/receipts/abo.db?mode=ro", uri=True, timeout=60)
    sid = db.execute("select study_id from studies where study_name = 'clean1'").fetchone()[0]
    tid = db.execute("select trial_id from trials where study_id = ? and number = ?", (sid, number)).fetchone()[0]
    v = db.execute("select value_json from trial_user_attributes where trial_id = ? and key = ?", (tid, key)).fetchone()
    return json.loads(v[0]) if v else None


def trial_start(number):
    """when a study trial was asked (it is built at once): optuna stores it in UTC, without a zone"""
    db = sqlite3.connect(f"file:{ROOT}/receipts/abo.db?mode=ro", uri=True, timeout=60)
    sid = db.execute("select study_id from studies where study_name = 'clean1'").fetchone()[0]
    s = db.execute("select datetime_start from trials where study_id = ? and number = ?", (sid, number)).fetchone()[0]
    return datetime.datetime.fromisoformat(s).replace(tzinfo=datetime.timezone.utc).timestamp()


def least_loaded():
    """the GPU for a confirmation: the one with the fewest confirmation-level jobs (0-1) with the relay or on its box
    (all went to the primary, one at a time: eleven queued on spot3 before the 07:00 freeze)"""
    try:
        st = json.loads((submit.RELAY / "state.json").read_text())
    except (OSError, ValueError):
        st = {}
    return min(vast_hosts.gpus(), key=lambda h: sum(1 for n, v in st.items() if v["gpu"] == h and v.get("level", 3) <= 1
                                                   and v["phase"] != "cancelled" and n not in _done_runs()))


def _done_runs():
    return {r["run"] for r in records() if r["event"] in ("result", "failed")}


def requeue(run, nm, over, lost, changed):
    """lane <nm>'s confirmation <run> read {over} but lacks deltas adopted since its assembly: the lane assembled again
    on the current incumbent (lane_asm) and confirmed; only a conflict with clean stops it"""
    try:
        new = lane_asm.reassemble(nm, adopted()["commits"])
    except RuntimeError as e:
        print(f"{run}: {over}, but held: {e}", flush=True)
        append({"event": "held", "run": run, "why": str(e)[-300:], "t": round(time.time())})
        return
    append({"event": "reassembled", "run": run, "lacked": sorted(map(str, lost)), "as": new, "t": round(time.time())})
    print(f"{run}: {over}; it lacked {lost}: {nm} assembled again on {incumbent()['name']} as {new}, confirmed", flush=True)
    add(new, least_loaded(), None, changed)


def target_delta(name):
    """lane <name>'s own reading as a value: the mean log ratio, in %, of its target cases' speedups to its matched
    parent's controls on the same GPU cohort (its report: lane_report.py); None without such controls"""
    rep = lane_report.OUT
    r = next((json.loads(l) for l in reversed(open(rep).read().splitlines()) if json.loads(l)["name"] == name), None) \
        if rep.exists() else None
    ds = [x["delta_pct"] for x in (r or {}).get("targets", []) if x.get("delta_pct") is not None]
    return 100 * (math.exp(sum(math.log1p(d / 100) for d in ds) / len(ds)) - 1) if ds else None


def adopted_readings():
    """{knob: (value args, its target delta)} for the lane deltas carried by the incumbent chain (the incumbents' names in
    receipts/incumbents.jsonl, a bundle's members): what an alternative of that knob must beat, each read against its
    own matched parent's controls on its own GPU (not absolute screen scores across baselines)"""
    rows = [json.loads(l) for l in open(LANES)] if LANES.exists() else []
    out = {}
    for inc in [json.loads(l)["name"] for l in open(ROOT / "receipts/incumbents.jsonl")]:
        b = next((r for r in reversed(rows) if r["event"] == "bundled" and r.get("as") == inc), None)
        for m in (b["names"] if b else [inc]):
            a = (lane_asm.record(m) or {}).get("args", [])
            for k in lane_asm.knobs(a):
                out[k] = (a, target_delta(m))
    return out


def refresh_screens():
    """after an adoption: each lane screen still waiting with the relay (not yet built) is assembled again on the new
    incumbent under a new name and queued in its place; a file already queued is never rewritten"""
    state = json.loads((submit.RELAY / "state.json").read_text())
    rows = [json.loads(l) for l in open(LANES)] if LANES.exists() else []
    seen = {r["name"] for r in rows if r["event"] == "read"}
    for q in [r for r in rows if r["event"] == "queued" and r["name"] not in seen]:
        if state.get(q["name"], {}).get("phase") != "queued":
            continue
        try:
            new = lane_asm.reassemble(q["name"], adopted()["commits"])
        except RuntimeError as e:
            print(f"lane {q['name']}: left as queued: {e}", flush=True)
            continue
        submit.cancel(q["name"])
        st = json.loads((submit.RELAY / "state.json").read_text())   # (the GPU with the fewest lane-level jobs in flight)
        host = min(vast_hosts.gpus(), key=lambda h: sum(1 for v in st.values() if v["gpu"] == h and v.get("level", 3) <= 2
                                                        and v["phase"] in ("queued", "building", "built", "shipping", "shipped")))
        submit.screen(new, host, q.get("level", 2), 200, q.get("checks", False))
        with open(LANES, "a") as f:
            f.write(json.dumps({"event": "read", "name": q["name"], "void": True, "why": f"reassembled as {new} after an adoption",
                                "t": round(time.time())}) + "\n")
            f.write(json.dumps({**{k: v for k, v in q.items() if k not in ("name", "host", "t")}, "name": new, "host": host,
                                "t": round(time.time())}) + "\n")
        print(f"lane {q['name']}: assembled again on {incumbent()['name']} as {new}, screened", flush=True)


def lanes():
    """each finished lane screen, read against the incumbent of its time on its GPU: within noise or above overall, or
    ahead on a case, gets one confirmation (as the driver does for study screens)"""
    seen = {r["name"] for r in map(json.loads, open(LANES)) if r["event"] == "read"} if LANES.exists() else set()
    for q in [r for r in map(json.loads, open(LANES)) if r["event"] == "queued" and r["name"] not in seen] if LANES.exists() else []:
        name, host = q["name"], submit.gpu_of(q["name"], q["host"])
        if host not in vast_hosts.load()[0] or not sh(host, f"ls /root/out/abo_{name}.done 2>/dev/null").strip():
            continue
        try:
            res = json.loads(sh(host, f"cat /root/out/abo_{name}.json"))
        except ValueError:
            res = {}
        inc = incumbent_at(res.get("identity", {}).get("t_start", q["t"]))
        rec = lane_asm.record(name)   # (against the incumbent it was assembled on, where that one has controls on its GPU)
        if "geomean" in res and rec and rec["incumbent"] != inc and controls(cohort(host, res["identity"]),
                                                                             res["identity"].get("t_start"), rec["incumbent"]):
            inc = rec["incumbent"]
        d = controls(cohort(host, res["identity"]), res["identity"].get("t_start"), inc) if "geomean" in res else None
        if "geomean" in res and not d:   # (no control of that incumbent there yet: read later)
            continue
        over, won = ahead(d, res) if d else ("no result", [])
        with open(LANES, "a") as f:
            f.write(json.dumps({"event": "read", "name": name, "geomean": res.get("geomean"), "against": inc, "overall": over,
                                "cases_ahead": won, "t": round(time.time())}) + "\n")
        print(f"lane {name}: {res.get('geomean')} vs {inc} on {host}: {over} overall, ahead on {', '.join(won) or 'no case'}", flush=True)
        tc = q.get("changed")   # (option a: its targets judged against its own untouched cases; a candidate touching
        rv = relative(res, inc, tc) if "geomean" in res else ("no result", None, None, 0, 1.0)   # every case: (b))
        over, tb, won = rv[0], rv[0] == "behind", [k for k in won if k in (tc or won)]
        with open(LANES, "a") as f:
            f.write(json.dumps({"event": "judged", "name": name, "verdict": rv[0], "score": rv[1], "reference": rv[2],
                                "sd": rv[3], "offset": rv[4], "t": round(time.time())}) + "\n")
        try:   # (the report: lane_report.py forwards receipts/evidence/lane_reports.jsonl)
            import lane_report
            lane_report.report(name, res, host, "behind on its targets" if tb else ("within/ahead: pending for the next bundle"
                               if over in ("within", "ahead") or set(won) & set(tc or won) else f"{over} overall"), tc)
        except Exception as e:  # noqa: BLE001
            print(f"lane {name}: no report: {e}", flush=True)
        if not tb and (over in ("within", "ahead") or set(won) & set(tc or won)):   # (confirmed in the next batch)
            with open(LANES, "a") as f:
                f.write(json.dumps({"event": "pending", "name": name, "changed": q.get("changed"), "geomean": res.get("geomean"),
                                    "over": over, "t": round(time.time())}) + "\n")


BATCH_S = 1200   # (the owner) lane deltas read within/ahead wait at most this long for a batch, or until two wait


def single(name, changed):
    """lane <name> confirmed alone, assembled again on the current incumbent first when it was assembled on another (on
    the primary GPU: a confirmation must finish between stops)"""
    rec = lane_asm.record(name)
    if rec and rec["incumbent"] != incumbent()["name"]:
        try:
            name = lane_asm.reassemble(name, adopted()["commits"])
        except RuntimeError as e:
            print(f"lane {name}: not confirmed: {e}", flush=True)
            return
    add(name, least_loaded(), None, changed)


def batch():
    """the owner: every lane delta that read within/ahead since the last batch, confirmed together as ONE candidate
    (lane_asm.bundle: their commits merged on clean, all their flags) targeted on the union of their cases, once two wait
    or the oldest has waited BATCH_S; a member left out of the bundle (a conflict, a composite) is confirmed alone"""
    rows = [json.loads(l) for l in open(LANES)] if LANES.exists() else []
    taken = {n for r in rows if r["event"] in ("bundled", "single", "closed") for n in r.get("names", [r.get("name")])}
    pending = [r for r in rows if r["event"] == "pending" and r["name"] not in taken]
    held = adopted_readings()   # (an alternative of a knob the incumbent carries stays eligible; it goes on, as a value
    def beats(r, k):   # (its target delta against its own parent above the adopted value's; unknown either way: waits)
        mine, theirs = target_delta(r["name"]), held[k][1]
        return mine is not None and theirs is not None and mine > theirs
    pending = [r for r in pending   # swap through the bundle, only if its own delta beats the adopted value's)
               if not any(k in held and set(held[k][0]) != set((lane_asm.record(r["name"]) or {}).get("args", []))
                          and not beats(r, k)
                          for k in lane_asm.knobs((lane_asm.record(r["name"]) or {}).get("args", [])))]
    if not pending or (len(pending) < 2 and time.time() - pending[0]["t"] < BATCH_S):
        return
    log = lambda e: open(LANES, "a").write(json.dumps({**e, "t": round(time.time())}) + "\n")
    if len(pending) == 1:
        log({"event": "single", "name": pending[0]["name"]})
        return single(pending[0]["name"], pending[0].get("changed"))
    k = 1 + sum(1 for r in rows if r["event"] == "bundled")
    name = f"bundle{k}_{incumbent()['name']}"
    closed = {r["name"] for r in rows if r["event"] == "closed"}   # (a withdrawal before the batch forms: left out)
    pending = [r for r in pending if r["name"] not in closed]
    if not pending:
        return
    by = {r["name"]: r for r in pending}
    try:
        inn, out = lane_asm.bundle(name, list(by), {m: by[m].get("geomean") for m in by})
    except RuntimeError as e:
        print(f"batch: {e}: confirmed singly", flush=True)
        inn, out = [], [r["name"] for r in pending]
    alt = [m for m in out if lane_asm.record(m) and not lane_asm.record(m).get("composite_of")
           and lane_asm.knobs(lane_asm.record(m)["args"]) & {k for x in inn for k in lane_asm.knobs(lane_asm.record(x)["args"])}]
    out = [m for m in out if m not in alt]   # (an alternative of a knob in the bundle stays pending with its own reading)
    if len(inn) >= 2:
        changed = sorted({c for m in inn for c in (by[m].get("changed") or [])}) or None
        log({"event": "bundled", "as": name, "names": inn, "left_out": out, "alternatives_pending": alt})
        print(f"batch: {name} = {', '.join(inn)}; confirmed on {', '.join(changed or ['every case'])}", flush=True)
        add(name, least_loaded(), None, changed)
    else:
        out = inn + out
        if alt:   # (no bundle: the best of each knob's alternatives singly, the rest still pending)
            out += [m for m in alt if not lane_asm.knobs(lane_asm.record(m)["args"]) & {k for x in out for k in lane_asm.knobs(lane_asm.record(x)["args"])}][:1]
    for m in sorted(out, key=lambda m: -(by[m].get("geomean") or 0)):
        log({"event": "single", "name": m})
        single(m, by[m].get("changed"))


def unbundle(nm):
    """a bundle that read behind or failed its checks: its members confirmed singly, the best screen first (the owner);
    of a knob's alternatives only the best, the rest stay pending"""
    rows = [json.loads(l) for l in open(LANES)] if LANES.exists() else []
    b = next((r for r in reversed(rows) if r["event"] == "bundled" and r["as"] == nm), None)
    if b is None:
        return
    by = {r["name"]: r for r in rows if r["event"] == "pending"}
    with open(LANES, "a") as f:
        f.write(json.dumps({"event": "unbundled", "as": nm, "t": round(time.time())}) + "\n")
    seen = set()
    for m in sorted(b["names"], key=lambda m: -(by.get(m, {}).get("geomean") or 0)):
        k = lane_asm.knobs((lane_asm.record(m) or {}).get("args", []))
        if k & seen:
            continue
        seen |= k
        print(f"{nm}: behind or failed; {m} confirmed alone", flush=True)
        single(m, by.get(m, {}).get("changed"))


if __name__ == "__main__":
    a = sys.argv[1:]
    with open(EV / "confirm.lock", "a") as lk:   # one at a time (two collects at once queued each lane's confirmation twice)
        fcntl.flock(lk, fcntl.LOCK_EX)
        if a[0] == "add":
            add(a[1], a[2], *a[3:4])
        else:
            collect()
            lanes()
            batch()
