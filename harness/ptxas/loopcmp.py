# loops >= 100 instructions: start/end pc, size, 128 B i-cache lines spanned, start offset within its line, and the
# convergence/sync/branch instructions inside (BSSY, BSYNC, WARPSYNC, BAR, SYNCS waits, BRA, BRA.DIV, HMMA)
import re, sys, collections
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from ctl import decode

for f in sys.argv[1:]:
    print("==", f)
    for fn, ins in decode(f).items():
        pcs = {i["pc"]: k for k, i in enumerate(ins)}
        for k, i in enumerate(ins):
            m = re.search(r"BRA(?:\.\S+)? (0x[0-9a-f]+)", i["t"])
            if not (m and "U.ANY" not in i["t"] and int(m.group(1), 16) <= i["pc"] and int(m.group(1), 16) in pcs):
                continue
            a = pcs[int(m.group(1), 16)]
            body = ins[a:k + 1]
            if len(body) < 100:
                continue
            c = collections.Counter()
            for x in body:
                t = re.sub(r"^@!?U?P\w+ ", "", x["t"])
                op = t.split()[0]
                for key in ("BSSY", "BSYNC", "WARPSYNC", "BAR", "SYNCS.PHASECHK", "BRA.DIV", "HMMA", "NANOSLEEP", "LDG", "LDS"):
                    if op.startswith(key):
                        c[key] += 1
                if op == "BRA":
                    c["BRA"] += 1
            s, e = ins[a]["pc"], ins[k]["pc"] + 16
            lines = (e - 1) // 128 - s // 128 + 1
            print(f"  {s:#07x}-{e:#07x} n={len(body):5d} lines={lines:4d} start%128={s % 128:3d} cycles={sum(x['stall'] for x in body):6d} "
                  + " ".join(f"{k2}:{v}" for k2, v in sorted(c.items())))
