# signature of "kernel-long uniform kept in a vector register": an R2UR inside a loop whose source register is not
# written anywhere in that loop (a loop-invariant value converted every iteration); per module, innermost loops
import re, sys
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from ctl import decode


def dst(t):
    t = re.sub(r"^@!?U?P\w+ ", "", t)
    op, _, rest = t.partition(" ")
    if op.startswith(("ST", "BRA", "EXIT", "SYNCS", "UBLK", "UTMA", "HGMMA", "BAR", "RED", "WARP")):
        return set()
    m = re.match(r"\s*(R\d+)", rest)
    if not m:
        return set()
    r = int(m.group(1)[1:])
    w = 2 if re.search(r"\.64|WIDE|\.128", op) else 1
    w = 4 if ".128" in op else w
    return {f"R{r + i}" for i in range(w)}


for f in sys.argv[1:]:
    out = []
    for fn, ins in decode(f).items():
        pcs = {i["pc"]: k for k, i in enumerate(ins)}
        spans = []
        for k, i in enumerate(ins):
            m = re.search(r"BRA(?:\.\S+)? (0x[0-9a-f]+)", i["t"])
            if m and "BRA.U.ANY" not in i["t"] and int(m.group(1), 16) <= i["pc"] and int(m.group(1), 16) in pcs:
                spans.append((pcs[int(m.group(1), 16)], k))
        for a, k in spans:
            if any(a < b <= c < k for b, c in spans):
                continue
            body = ins[a:k + 1]
            written = set().union(*[dst(x["t"]) for x in body])
            inv = [x["t"] for x in body if re.match(r"(?:@!?U?P\w+ )?R2UR(?:\.\w+)? UR\d+, (R\d+)", x["t"])
                   and re.match(r"(?:@!?U?P\w+ )?R2UR(?:\.\w+)? UR\d+, (R\d+)", x["t"]).group(1) not in written]
            if inv:
                users = sorted({re.sub(r"^@!?U?P\w+ ", "", x["t"]).split()[0] for x in body
                                if re.search(r"HGMMA|UTMA|UBLK|SYNCS|LDS|STS|ULDC", x["t"]) and "UR" in x["t"]})[:6]
                out.append((len(inv), len(body), hex(ins[a]["pc"]), users))
    out.sort(reverse=True)
    print(f.split("/")[-2], out[:4])
