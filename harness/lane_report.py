"""A lane screen's reading as its owner needs it: name, md5, the GPU, geomean against its matched parent's controls (the
incumbent it was assembled on, same GPU cohort), the per-case speedups on its target cases against those controls'
per-case means, and the verdict. confirm.lanes() writes one per reading to receipts/evidence/lane_reports.jsonl; the
agent forwarding them sends each to the lane's owner.
    python3 harness/lane_report.py <lane name>"""
import json, pathlib, statistics, sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "harness"))
import confirm, lane_asm  # noqa: E402

OUT = ROOT / "receipts/evidence/lane_reports.jsonl"


def report(name, res, host, verdict, changed):
    """the reading of lane <name> (its result JSON res, read on host) against the incumbent it was assembled on"""
    rec = lane_asm.record(name) or {}
    inc = rec.get("incumbent") or confirm.incumbent()["name"]
    key = confirm.cohort(confirm.submit.gpu_of(name, host), res["identity"])   # (where the relay ran it)
    rd = confirm.reading(inc) or {}   # (the reference: the incumbent's established number, its own confirmation reading,
    ctl = [rd] if rd else []          # with the screen's own offset on its untouched cases divided out)
    off = confirm.relative(res, inc, changed)[4] if rd else 1.0
    mine = {f"{c['case']['load']} {c['case']['layer']}": c for c in res["cases"]}
    rows = []
    for k in (changed or sorted(mine)):
        c = mine.get(k)
        if c is None:
            continue
        m = rd.get(k)
        rows.append({"case": k, "speedup": round(c["speedup"], 4), "us": round(c["ours_us"], 3),
                     "parent_speedup": m and round(m, 4), "delta_pct": m and round(100 * (c["speedup"] / off / m - 1), 2),
                     "parent_us": m and round(c["base_us"] / m, 3)})
    g = [confirm.score(rd, list(rd))] if rd else []
    out = {"name": name, "md5": res["identity"].get("sub_md5"), "gpu": key, "against": inc, "n_controls": len(ctl),   # (1: its confirmation reading)
           "geomean": round(res["geomean"], 4), "parent_geomean": g and round(statistics.mean(g), 4),
           "verdict": verdict, "targets": rows, "args": rec.get("args"), "commit": rec.get("clean"), "offset": round(off, 5)}
    with open(OUT, "a") as f:
        f.write(json.dumps(out) + "\n")
    return out


def text(r):
    t = "; ".join(f"{x['case']} {x['speedup']} vs {x['parent_speedup']} ({x['delta_pct']:+.2f} %, {x['us']} us vs ~{x['parent_us']})"
                  if x["parent_speedup"] else f"{x['case']} {x['speedup']}" for x in r["targets"])
    return (f"{r['name']} ({(r['md5'] or '')[:8]}, {', '.join(r['args'] or [])}): geomean {r['geomean']} vs parent "
            f"{r['parent_geomean']} ({r['against']}, {r['gpu'][:14]}; untouched-case offset {r.get('offset', 1):.4f} divided out): "
        f"{r['verdict']}. Targets: {t}")


if __name__ == "__main__":
    for l in open(OUT) if OUT.exists() else []:
        r = json.loads(l)
        if r["name"] == sys.argv[1]:
            print(text(r))
