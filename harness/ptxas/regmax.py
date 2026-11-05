# setmaxnreg regions: USETMAXREG sites, and the highest vector register touched in HGMMA stage loops (consumer
# budget actually used) and anywhere in the function
import re, sys
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from ctl import decode

for fn, ins in decode(sys.argv[1]).items():
    smr = [(hex(i["pc"]), " ".join(i["t"].split()[:3])) for i in ins if "USETMAXREG" in i["t"]]
    allr = max((int(r) for i in ins for r in re.findall(r"\bR(\d+)\b", i["t"])), default=-1)
    pcs = {i["pc"]: k for k, i in enumerate(ins)}
    loopmax = -1
    for k, i in enumerate(ins):
        m = re.search(r"BRA(?:\.\S+)? (0x[0-9a-f]+)", i["t"])
        if m and "U.ANY" not in i["t"] and int(m.group(1), 16) <= i["pc"] and int(m.group(1), 16) in pcs:
            body = ins[pcs[int(m.group(1), 16)]:k + 1]
            if any("HGMMA" in x["t"] for x in body) and len(body) < 1500:
                loopmax = max([loopmax] + [int(r) for x in body for r in re.findall(r"\bR(\d+)\b", x["t"])])
    print(f"USETMAXREG: {smr}  max R in HGMMA stage loops: R{loopmax}  max R anywhere: R{allr}")
