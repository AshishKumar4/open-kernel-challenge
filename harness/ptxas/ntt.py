# per HGMMA-shape stage loops of a cubin: instructions, static cycles, R2UR, LDL/STL inside
import re, sys
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from ctl import decode

rows = []
for fn, ins in decode(sys.argv[1]).items():
    pcs = {i["pc"]: k for k, i in enumerate(ins)}
    for k, i in enumerate(ins):
        m = re.search(r"BRA(?:\.\S+)? (0x[0-9a-f]+)", i["t"])
        if not (m and "U.ANY" not in i["t"] and int(m.group(1), 16) <= i["pc"] and int(m.group(1), 16) in pcs):
            continue
        a = pcs[int(m.group(1), 16)]
        body = ins[a:k + 1]
        hg = [x["t"] for x in body if "HGMMA" in x["t"]]
        if not hg or len(body) > 1500:
            continue
        n = sorted({int(re.search(r"HGMMA\.64x(\d+)x16", h).group(1)) for h in hg})
        rows.append((n[0], len(body), sum(x["stall"] for x in body), sum("R2UR" in x["t"] for x in body),
                     sum(re.search(r"\b(LDL|STL)", x["t"]) is not None for x in body)))
for r in sorted(rows):
    print("  N=%-3d n=%d cyc=%d r2ur=%d ldst=%d" % r)
