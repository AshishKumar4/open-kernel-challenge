# instruction-class mix of two loops (by start pc) in one cubin, side by side, with the difference; full opcode
# (with modifiers) so MOV vs IMAD.MOV vs PRMT etc. separate
#   python3 mix.py file.cubin pcA pcB
import re, sys, collections
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from ctl import decode

f, pa, pb = sys.argv[1], int(sys.argv[2], 16), int(sys.argv[3], 16)
for fn, ins in decode(f).items():
    pcs = {i["pc"]: k for k, i in enumerate(ins)}

    def body(p):
        a = pcs[p]
        k = next(j for j in range(a, len(ins)) if re.search(r"BRA(?:\.\S+)? (0x[0-9a-f]+)", ins[j]["t"])
                 and int(re.search(r"BRA(?:\.\S+)? (0x[0-9a-f]+)", ins[j]["t"]).group(1), 16) == p
                 and "U.ANY" not in ins[j]["t"])
        return ins[a:k + 1]

    A, B = body(pa), body(pb)
    ca = collections.Counter(re.sub(r"^@!?U?P\w+ ", "", x["t"]).split()[0] for x in A)
    cb = collections.Counter(re.sub(r"^@!?U?P\w+ ", "", x["t"]).split()[0] for x in B)
    print(f"A {pa:#x}: n={len(A)} cycles={sum(x['stall'] for x in A)}   B {pb:#x}: n={len(B)} cycles={sum(x['stall'] for x in B)}")
    rows = sorted(set(ca) | set(cb), key=lambda o: -(cb[o] - ca[o]))
    for o in rows:
        if ca[o] or cb[o]:
            print(f"  {o:34s} A={ca[o]:4d} B={cb[o]:4d} d={cb[o] - ca[o]:+4d}")
