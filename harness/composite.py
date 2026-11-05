"""The per-shape composite: from a harvest's per-case table (receipts/harvest/h<n>.json), for every case and study
switch, the value the harvest says helps that case (t >= 2; the largest such effect), else the switch's default where the
incumbent's value hurts that case (t <= -2), else the incumbent's value. A switch is decided only on the cases its builds
(or layer constant) serve; harvest-withdrawn switches are never proposed. Reports the switches whose build serves several
cases that want different values (those need a per-case build variant before the composite can carry them).
    python3 harness/composite.py <harvest json> [out.json]"""
import json, pathlib, sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
CASE = {"decode 1 sliding": "d1s", "decode 1 full": "d1f", "verify 1 sliding": "v1s", "verify 1 full": "v1f",
        "decode 16 sliding": "d16s", "decode 16 full": "d16f", "decode 32 sliding": "d32s", "decode 32 full": "d32f",
        "verify 16 sliding": "v16s", "verify 16 full": "v16f", "verify 32 sliding": "v32s", "verify 32 full": "v32f"}
# the cases each build serves (the incumbent's dispatch trace, receipts/evidence/dbt/dbt_h1_selection.json, and MOE_SWITCHES)
BUILD_CASES = {"attention": list(CASE.values()),
               "sk2": ["d1s", "d1f", "d16s", "d16f", "d32s", "d32f", "v1s", "v1f"], "sk2_narrow": ["v1s", "v1f"],
               "sk2_gle": ["d1s", "d1f", "v1s", "v1f"], "rq128": ["v16s", "v16f"], "rq256": ["v32s", "v32f"],
               **{f"moe_{c}": [c] for c in ("d1s", "d1f", "v1s", "v1f", "d16s", "d16f", "d32s", "d32f")},
               "moe_big": ["v16s", "v16f", "v32s", "v32f"]}
MOE_CLASSES = {"MOE_TINY": ["moe_d1s", "moe_d1f"], "MOE_TINY2": ["moe_v1s", "moe_v1f"], "MOE_MID": ["moe_d16s", "moe_d16f"],
               "MOE_WGS": ["moe_d32s", "moe_d32f", "moe_big"], "MOE_DF": ["moe_v1s", "moe_v1f", "moe_d16s", "moe_d16f"]}
CONST_CASES = {"nsplit16": ["v16s"], "v32s": ["v32s"], "sdef": ["d1s", "d1f"],   # switches that reach only some of their
               "vunroll": ["v16f", "v32f"], "stages": ["d16f", "d32f"], "mid16": ["d16s"]}   # build's cases: the cases their
# code path runs on (ATT_FLAT_VERIFY_UNROLL is the L8 flat verify only, ATT_FLAT_DECODE_STAGES the flat
# decode at 16 and 32 full, the n_splits rows their own case, SDEF the t1 builds, MID16 moe_d16s)
TRACE = ROOT / "receipts/evidence/dbt/inc_readd_selection.json"   # the incumbent's actual per-case launches (trace_dispatch)


def dispatch_values(trace=TRACE):
    """the incumbent's per-case values of the switches its dispatch decides per case, from its trace
    (never an assumption): sk2c7 = the sk2 families' c7, rqc7 = the rq family's c7, p05 = the p05 of the launches it
    applies to"""
    t = json.loads(pathlib.Path(trace).read_text())
    out = {"sk2c7": {}, "rqc7": {}, "p05": {}}
    for c in t["cases"]:
        sk = {l["c7"] for l in c["launches"] if l["family"].startswith("sk2")}
        rq = {l["c7"] for l in c["launches"] if l["family"].startswith("rq")}
        pp = ({l["p05"] for l in c["launches"] if l.get("p05_applies")} or   # (where no launch takes p05, the value its sk2
              {l["p05"] for l in c["launches"] if l["family"] == "sk2" and l.get("p05") is not None})   # builds carry)
        for k, v in (("sk2c7", sk), ("rqc7", rq), ("p05", pp)):
            if len(v) == 1:
                out[k][c["case"]] = v.pop()
    return out, t["md5"]


DISPATCH, DISPATCH_MD5 = dispatch_values()


def study():
    """the study's DIMS and APPLY, read from clean_study.py's text (no optuna, no harness imports)"""
    src = (ROOT / "harness/clean_study.py").read_text()
    ns = {"__name__": "x"}
    for start, end in (("DIMS = {", "DIMS = {"), ("SK2 = (", "APPLY = {")):
        i = src.index(start)
        exec(compile(src[i:src.index("\n}\n", src.index(end)) + 3], start, "exec"), ns)
    return ns["DIMS"], ns["APPLY"]


def builds_of(k, apply):
    where = apply[k][0]
    out = []
    for w in where:
        out += MOE_CLASSES.get(w, [w])
    return out


def withdrawn_switches():
    """the switches the harvest never proposes (harness/harvest.py WITHDRAWN: e.g. tanhsilu, which broke the accuracy rule)"""
    src = (ROOT / "harness/harvest.py").read_text()
    line = next(l for l in src.splitlines() if l.startswith("WITHDRAWN = "))
    return set(eval(line.split("=", 1)[1].split("#")[0]))


def compose(harvest, inc_point, withdrawn=None):
    withdrawn = withdrawn_switches() if withdrawn is None else withdrawn
    dims, apply = study()
    eff = harvest["effects"]
    table, conflicts, prov = {c: {} for c in CASE.values()}, [], {}
    for k, vals in dims.items():
        if k in withdrawn:
            continue
        cases = CONST_CASES.get(k) or sorted({c for b in builds_of(k, apply) for c in BUILD_CASES.get(b, [])})
        dflt = vals[0]
        for name, c in CASE.items():
            if c not in cases:
                continue
            inc = DISPATCH.get(k, {}).get(c, inc_point.get(k, dflt))
            helps = [(eff[f"{k}={v}"][name]["pct"], v, eff[f"{k}={v}"][name]["t"]) for v in vals[1:]
                     if f"{k}={v}" in eff and eff[f"{k}={v}"][name]["t"] >= 2]
            if helps:
                pct, v, t = max(helps)
                table[c][k] = v
                prov[(c, k)] = f"{k}={v} helps {pct:+.2f}% (t {t:.1f})"
            elif inc != dflt and f"{k}={inc}" in eff and eff[f"{k}={inc}"][name]["t"] <= -2:
                table[c][k] = dflt
                prov[(c, k)] = f"{k}={inc} hurts {eff[f'{k}={inc}'][name]['pct']:+.2f}% (t {eff[f'{k}={inc}'][name]['t']:.1f}): default"
            else:
                table[c][k] = inc
        if k in CONST_CASES:   # (only its own path's cases: no shared-build conflict beyond them)
            want = {c: table[c][k] for c in CONST_CASES[k] if k in table[c]}
            if len(set(map(str, want.values()))) > 1:
                conflicts.append({"switch": k, "build": "path", "per_case": want})
            continue
        for b in builds_of(k, apply):
            want = {c: table[c][k] for c in BUILD_CASES.get(b, []) if k in table[c]}
            if len(set(map(str, want.values()))) > 1:
                conflicts.append({"switch": k, "build": b, "per_case": want})
    return table, conflicts, prov


if __name__ == "__main__":
    h = json.load(open(sys.argv[1]))
    inc = json.loads((ROOT / "receipts/incumbent.json").read_text())
    point = inc.get("point") or {}
    table, conflicts, prov = compose(h, point)
    dims, _ = study()
    base = lambda c, k: DISPATCH.get(k, {}).get(c, point.get(k, dims[k][0]))   # (the incumbent's value on that case)
    changed = {c: {k: v for k, v in sw.items() if v != base(c, k)} for c, sw in table.items()}
    if not inc["md5"].startswith(DISPATCH_MD5[:8]):
        sys.exit(f"the dispatch trace is of {DISPATCH_MD5[:8]}, the incumbent is {inc['md5'][:8]}: trace it first")
    out = {"harvest": sys.argv[1], "incumbent": inc["name"], "incumbent_md5": inc["md5"], "trace": str(TRACE),
           "dispatch": DISPATCH, "point": point, "per_case": table, "changes": changed,
           "why": {f"{c} {k}": w for (c, k), w in prov.items()}, "shared_build_conflicts": conflicts}
    if len(sys.argv) > 2:
        pathlib.Path(sys.argv[2]).write_text(json.dumps(out, indent=1))
    for c, ch in changed.items():
        print(c, {k: f"{base(c, k)}->{v}" for k, v in ch.items()})
    print("shared-build conflicts:")
    for x in conflicts:
        print(" ", x["switch"], x["build"], x["per_case"])
