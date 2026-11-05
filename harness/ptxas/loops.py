# list SASS loops (backward branches): head pc, tail pc, body size, and an opcode digest of the body
import sys, re, hashlib
def parse(f):
    ins = []
    for l in open(f):
        m = re.match(r"\s+/\*([0-9a-f]+)\*/\s+(.*?);", l)
        if m: ins.append((int(m.group(1), 16), m.group(2).strip()))
    return ins
def loops(f):
    ins = parse(f); out = []
    for i, (pc, t) in enumerate(ins):
        m = re.search(r"\bBRA\b.*?(0x[0-9a-f]+)", t)
        if m and int(m.group(1), 16) <= pc:
            h = int(m.group(1), 16); body = [x for p, x in ins if h <= p <= pc]
            ops = [re.sub(r"^@!?U?P\w+\s+", "", x).split()[0] for x in body]
            regs = sorted({int(r) for x in body for r in re.findall(r"\bR(\d+)\b", x)})
            out.append((h, pc, len(body), hashlib.md5(" ".join(ops).encode()).hexdigest()[:6], hashlib.md5(" ".join(body).encode()).hexdigest()[:6], max(regs) if regs else -1))
    return out
for f in sys.argv[1:]:
    L = loops(f); print(f, len(L), "loops")
    for x in L: print("  head %5x tail %5x n=%4d ops=%s full=%s maxR=%d" % x)
