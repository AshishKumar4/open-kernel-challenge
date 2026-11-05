# innermost loops that issue HGMMA: size, HGMMA count, static issue cycles (sum of stall counts) and the
# instruction mix per HGMMA, to compare two WGS-style kernels' stage loops
import re, sys, collections
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from ctl import decode

for f in sys.argv[1:]:
    for fn, ins in decode(f).items():
        pcs = {i["pc"]: k for k, i in enumerate(ins)}
        loops = []
        for k, i in enumerate(ins):
            m = re.search(r"BRA(?:\.\S+)? (0x[0-9a-f]+)", i["t"])
            if m and int(m.group(1), 16) <= i["pc"] and int(m.group(1), 16) in pcs:
                a = pcs[int(m.group(1), 16)]
                body = ins[a:k + 1]
                hg = sum("HMMA" in x["t"] for x in body)
                if hg:
                    loops.append((len(body), a, k, hg, body))
        loops.sort()
        seen = []
        for n, a, k, hg, body in loops:
            if any(a <= sa and sk <= k for sa, sk in seen):   # keep innermost only
                continue
            seen.append((a, k))
            ops = collections.Counter(re.sub(r"^@!?U?P\w+ ", "", x["t"]).split()[0].split(".")[0] for x in body)
            cyc = sum(x["stall"] for x in body)
            top = ", ".join(f"{o}:{c}" for o, c in ops.most_common(12))
            print(f"{f.split('/')[-3] if '/x/' in f else f}: loop {ins[a]['pc']:#x}-{ins[k]['pc']:#x} n={n} HMMA={hg} "
                  f"static_cycles={cyc} ({cyc / hg:.1f}/HMMA)  {top}")
