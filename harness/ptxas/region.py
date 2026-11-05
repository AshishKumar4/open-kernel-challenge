# instructions attributed to a source line range (any level of the inline chain) in an nvdisasm -gi listing whose
# instructions carry control bits from the matching cubin: count, stall sum, opcode mix, LDL/STL, loads, and the
# pc span; usage: region.py li.gi li.cubin mod.cu lo hi
import re, sys, collections
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from ctl import decode

gi, cubin, lo, hi = sys.argv[1], sys.argv[2], int(sys.argv[4]), int(sys.argv[5])
ctl = {}
for fn, ins in decode(cubin).items():
    for i in ins:
        ctl[i["pc"]] = i
chain, fresh, sel = [], False, []
for l in open(gi):
    if "//##" in l:
        nums = [int(x) for x in re.findall(r"line (\d+)", l)]
        chain = nums if not fresh else chain + nums
        fresh = True
        continue
    m = re.match(r"\s+/\*([0-9a-f]{4,})\*/\s+(.*?);", l)
    if not m:
        continue
    fresh = False
    pc = int(m.group(1), 16)
    if any(lo <= n <= hi for n in chain) and pc in ctl:
        sel.append(ctl[pc])
ops = collections.Counter(re.sub(r"^@!?U?P\w+ ", "", i["t"]).split()[0] for i in sel)
print(f"n={len(sel)} stall_sum={sum(i['stall'] for i in sel)} pc {min(i['pc'] for i in sel):#x}..{max(i['pc'] for i in sel):#x}")
print("loads:", {k: v for k, v in ops.items() if re.match(r"(LDG|LDS|LDC|LDL|STL|ATOM|RED|SHFL)", k)})
print("top:", ", ".join(f"{k}:{v}" for k, v in ops.most_common(16)))
waits = sum(1 for i in sel if i["wait"])
print(f"instructions waiting on a scoreboard: {waits}; NOP: {ops.get('NOP', 0)}; BAR: {sum(v for k, v in ops.items() if k.startswith('BAR'))}")
# loops whose head lies in the region: size, stalls, BSSY/WARPSYNC/collectives inside
selpc = {i["pc"] for i in sel}
for fn, ins in decode(cubin).items():
    pcs = {i["pc"]: k for k, i in enumerate(ins)}
    for k, i in enumerate(ins):
        m = re.search(r"BRA(?:\.\S+)? (0x[0-9a-f]+)", i["t"])
        if m and "U.ANY" not in i["t"] and int(m.group(1), 16) <= i["pc"] and int(m.group(1), 16) in selpc:
            body = ins[pcs[int(m.group(1), 16)]:k + 1]
            if len(body) > 400:
                continue
            c = collections.Counter(re.sub(r"^@!?U?P\w+ ", "", x["t"]).split()[0] for x in body)
            print(f"  loop {body[0]['pc']:#x}-{i['pc']:#x} n={len(body)} stalls={sum(x['stall'] for x in body)} "
                  f"BSSY={c['BSSY']} WARPSYNC={c['WARPSYNC.COLLECTIVE'] + c['WARPSYNC']} SHFL={sum(v for kk, v in c.items() if kk.startswith('SHFL'))} "
                  f"LDS={sum(v for kk, v in c.items() if kk.startswith('LDS'))} LDG={sum(v for kk, v in c.items() if kk.startswith('LDG'))} BAR={sum(v for kk, v in c.items() if kk.startswith('BAR'))}")
