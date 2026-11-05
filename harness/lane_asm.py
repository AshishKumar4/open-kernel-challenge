"""A lane candidate is (commit, flags, cases), assembled on the current incumbent whenever it is confirmed (11-05):
an adoption no longer leaves queued lane candidates held for lacking it, nor needs a rebuild by hand. A lane commit that
not built on the clean tip is merged onto it first (branch lane/<name>-clean); a conflict is the only stop (it goes back
to the lane's owner). An
adopted lane commit is merged into clean at its adoption, so the study and later lanes carry it."""
import hashlib, json, pathlib, re, subprocess, sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
ASSEMBLY = ROOT / "receipts/evidence/lane_assembly.jsonl"


def git(*a, cwd=ROOT):
    return subprocess.run(["git", "-C", str(cwd), *a], capture_output=True, text=True)


def merge_in(tree, commit, message):
    """<commit> merged into the worktree's HEAD; a conflict aborts it (it goes back to the lane's owner, or to the orchestrator for an
    ended lane: no automatic resolution). Whether it merged"""
    if not git("merge", "--no-ff", "-q", commit, "-m", message, cwd=tree).returncode:
        return True
    git("merge", "--abort", cwd=tree)
    return False


def _merge(onto, commit, name, message):
    """commit merged onto <onto> in a scratch worktree as branch <name>; its sha, or raises with the conflict"""
    tree = ROOT.parent / f"okc-merge-{re.sub(r'[^a-z0-9_.-]', '_', name.lower())}"
    git("worktree", "remove", "--force", str(tree))
    r = git("worktree", "add", "-q", "-B", name, str(tree), onto)
    if r.returncode:
        raise RuntimeError(f"worktree for {name}: {r.stderr[-300:]}")
    try:
        if not merge_in(tree, commit, message):
            raise RuntimeError(f"{commit} conflicts with {onto}")
        return git("rev-parse", "--short", "HEAD", cwd=tree).stdout.strip()
    finally:
        git("worktree", "remove", "--force", str(tree))


def on_clean(name, commit, adopted_commits=()):
    """<commit> if it already carries the clean tip, else the lane's delta merged onto the clean tip (lane/<name>-clean):
    every later source on clean (adopted lanes, fixes landed directly) comes with it"""
    if not git("merge-base", "--is-ancestor", "clean", commit).returncode:
        return commit
    clean = git("rev-parse", "--short", "clean").stdout.strip()
    return _merge("clean", commit, f"lane/{name}-clean", f"{name} ({commit}) on clean {clean}: its lane on the current incumbent")


def record(name):
    return next((r for r in reversed([json.loads(l) for l in open(ASSEMBLY)]) if r["name"] == name), None) if ASSEMBLY.exists() else None


def reassemble(name, adopted_commits):
    """lane <name>'s (commit, flags) assembled again on the current incumbent as <base>_r<k>; the new name"""
    rec = record(name)
    if rec is None:
        raise RuntimeError(f"{name}: no lane() assembly to repeat")
    base = re.sub(r"_r\d+$", "", name)
    names = {json.loads(l)["name"] for l in open(ASSEMBLY)}
    new = next(f"{base}_r{k}" for k in range(1, 1000) if f"{base}_r{k}" not in names)
    commit = on_clean(new, rec["clean"], adopted_commits)
    if rec.get("composite_of"):   # (a composite: its recorded per-case delta replayed exactly, not its args, which are none)
        plan = ROOT / f"receipts/evidence/composites/{new}.json"
        plan.parent.mkdir(parents=True, exist_ok=True)
        plan.write_text(json.dumps({"changes": rec["per_case"], "from": name}, indent=1))
        cmd = ["composite", new, commit, str(plan)]
    else:
        cmd = ["lane", new, commit, *rec["args"], *(["--correctness"] if rec.get("correctness") else [])]
    r = subprocess.run([sys.executable, str(ROOT / "harness/clean_study.py"), *cmd], cwd=ROOT, capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(f"{new}: {cmd[0]}() failed: {(r.stdout + r.stderr)[-300:]}")
    carries_nothing(new, name)
    return new


def carries_nothing(new, of):
    """raises when candidate <new> equals the current incumbent's file: its delta did not carry over. Not against
    its _parent: a source-only lane (no flags) builds both from its changed source, so they are equal by design"""
    inc = json.loads((ROOT / "receipts/incumbent.json").read_text())["file"]
    md5 = lambda f: hashlib.md5((ROOT / f).read_bytes()).hexdigest()
    if md5(f"candidates/layer_{new}.py") == md5(inc):
        raise RuntimeError(f"{new}: assembled equal to the incumbent: {of}'s delta did not carry over")


def land(commit):
    """an adopted lane commit into clean (a fast-forward when clean is its ancestor, else a merge); raises on a conflict"""
    if not git("merge-base", "--is-ancestor", commit, "clean").returncode:
        return
    old = git("rev-parse", "clean").stdout.strip()
    if not git("merge-base", "--is-ancestor", "clean", commit).returncode:
        new = git("rev-parse", commit).stdout.strip()
    else:
        new = _merge("clean", commit, f"land/{commit}", f"Merge the adopted lane commit {commit} into clean")
        new = git("rev-parse", new).stdout.strip()
    r = git("update-ref", "refs/heads/clean", new, old)
    if r.returncode:
        raise RuntimeError(f"clean moved while landing {commit}: {r.stderr[-200:]}")
    rules = ROOT / "receipts/evidence/land_with.json"   # ([{"if": [commits], "also": commit}]: a landing that brings in
    for rule in json.loads(rules.read_text()) if rules.exists() else []:   # one of "if" brings "also" too, e.g. the
        if any(not git("merge-base", "--is-ancestor", c, "clean").returncode for c in rule["if"]):   # retirement of
            land(rule["also"])                                                                       # dead code)


def knobs(args):
    """the (build, macro) or switch each lane arg sets: "moe_d32s:-DMOE_CA=9" -> ("moe_d32s", "MOE_CA"); "gle_t8=1" ->
    (None, "gle_t8"); a raw flag is its own knob"""
    out = set()
    for a in args:
        build, sep, flag = a.partition(":")
        if not sep:
            out.add((None, a.split("=")[0]))
        else:
            m = re.match(r"-D(\w+)", flag)
            out.add((build, m.group(1) if m else flag))
    return out


def bundle(name, members, score=None):
    """the lane deltas <members> as ONE candidate <name> (the owner: one confirmation for every delta that read within or
    ahead since the last adoption): their commits merged one after another onto the clean tip (lane/<name>-clean), all
    their flags. At most one value per knob (build, macro): alternatives of one knob (MOE_CA=1, 2, ... on moe_d32s) are
    not stackable, and the best screen's (score) is taken; a member that sets a knob already taken, whose commit
    conflicts, or a composite, is left out. (the members in, the members out)"""
    recs = {m: record(m) for m in members}
    taken = set()
    for m in sorted(members, key=lambda m: -((score or {}).get(m) or 0)):
        if recs[m] and not recs[m].get("composite_of"):
            k = knobs(recs[m]["args"])
            if k & taken:
                recs[m] = None   # (an alternative of a knob a better screen sets: left out, it stays pending)
            else:
                taken |= k
    tree = ROOT.parent / f"okc-merge-{name}"
    git("worktree", "remove", "--force", str(tree))
    if git("worktree", "add", "-q", "-B", f"lane/{name}-clean", str(tree), "clean").returncode:
        raise RuntimeError(f"worktree for {name}")
    inn, out = [], []
    try:
        for m, rec in recs.items():
            if rec is None or rec.get("composite_of"):
                out.append(m)
                continue
            if not git("merge-base", "--is-ancestor", rec["clean"], "HEAD", cwd=tree).returncode:
                inn.append(m)
                continue
            if merge_in(tree, rec["clean"], f"{name}: {m} ({rec['clean']})"):
                inn.append(m)
            else:
                out.append(m)
        rules = ROOT / "receipts/evidence/carry_with.json"   # ([{"if": [commits], "also": commit}]: a bundle with a member
        for rule in json.loads(rules.read_text()) if rules.exists() else []:   # built on one of "if" carries "also" too)
            if any(not git("merge-base", "--is-ancestor", c, recs[m]["clean"]).returncode for c in rule["if"] for m in inn) \
                    and git("merge-base", "--is-ancestor", rule["also"], "HEAD", cwd=tree).returncode:
                if not merge_in(tree, rule["also"], f"{name}: {rule['also']} carried ({rule.get('why', '')[:60]})"):
                    raise RuntimeError(f"{name}: carried {rule['also']} conflicts")
        commit = git("rev-parse", "--short", "HEAD", cwd=tree).stdout.strip()
    finally:
        git("worktree", "remove", "--force", str(tree))
    if len(inn) < 2:
        return inn, out
    args = list(dict.fromkeys(a for m in inn for a in recs[m]["args"]))
    r = subprocess.run([sys.executable, str(ROOT / "harness/clean_study.py"), "lane", name, commit, *args], cwd=ROOT,
                       capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(f"{name}: lane() failed: {(r.stdout + r.stderr)[-300:]}")
    carries_nothing(name, "the bundle")
    return inn, out
