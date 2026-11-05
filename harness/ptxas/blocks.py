# a loop's top-level conditional blocks (forward branches whose skipped range lies inside the loop, outermost only):
# size, static cycles, HMMA inside; and the always-executed remainder
#   python3 blocks.py file.cubin loop_start_pc
import re, sys
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from ctl import decode

f, p0 = sys.argv[1], int(sys.argv[2], 16)
for fn, ins in decode(f).items():
    pcs = {i["pc"]: k for k, i in enumerate(ins)}
    if p0 not in pcs:
        continue
    a = pcs[p0]
    k = next(j for j in range(a, len(ins)) if re.search(r"BRA(?:\.\S+)? (0x[0-9a-f]+)", ins[j]["t"])
             and int(re.search(r"BRA(?:\.\S+)? (0x[0-9a-f]+)", ins[j]["t"]).group(1), 16) == p0 and "U.ANY" not in ins[j]["t"])
    blocks = []
    for j in range(a, k):
        m = re.match(r"(@!?P\d+ )BRA (0x[0-9a-f]+)", ins[j]["t"])
        if m:
            t = int(m.group(2), 16)
            if ins[j]["pc"] < t <= ins[k]["pc"] and t in pcs:
                blocks.append((j + 1, pcs[t], m.group(1).strip()))
    outer = [b for b in blocks if not any(c[0] <= b[0] and b[1] <= c[1] and c != b for c in blocks)]
    skipped = set()
    for s, e, pred in outer:
        body = ins[s:e]
        skipped.update(range(s, e))
        print(f"  block {ins[s]['pc']:#x}-{ins[e - 1]['pc']:#x} guard {pred:6s} n={len(body):4d} cycles={sum(x['stall'] for x in body):5d} "
              f"HMMA={sum('HMMA' in x['t'] for x in body)}")
    rest = [ins[j] for j in range(a, k + 1) if j not in skipped]
    print(f"  always-executed remainder: n={len(rest)} cycles={sum(x['stall'] for x in rest)} HMMA={sum('HMMA' in x['t'] for x in rest)} (loop total {k - a + 1})")
