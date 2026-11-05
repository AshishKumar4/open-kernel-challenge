"""Natural loops of each kernel in a cuobjdump -sass listing, by the control-flow graph (a back edge is a branch to a block
that dominates it), not by the address span between a branch and its target: per function, each loop's header, latch,
instruction count, HMMA/HGMMA count, plus the function's BRA.DIV count (out-of-line divergent-collective paths)."""
import re, sys

INS = re.compile(r"/\*([0-9a-f]{4,})\*/\s+(.*?)\s*;")
FN = re.compile(r"Function : (\S+)")


def functions(path):
    fns, cur = {}, None
    for line in open(path):
        m = FN.search(line)
        if m:
            cur = fns.setdefault(m.group(1), [])
            continue
        m = INS.search(line)
        if m and cur is not None:
            cur.append((int(m.group(1), 16), m.group(2)))
    return fns


def target(text):
    m = re.search(r"\bBRA(?:\.\S+)?\s+(?:!?U?P\w+,\s*|UR\d+,\s*)?(0x[0-9a-f]+)", text)
    return int(m.group(1), 16) if m else None


def loops(ins):
    pcs = [p for p, _ in ins]
    at = {p: i for i, p in enumerate(pcs)}
    leaders = {pcs[0]}
    for i, (p, t) in enumerate(ins):
        tg = target(t)
        if tg is not None and tg in at:
            leaders.add(tg)
        if (tg is not None or re.search(r"\b(EXIT|RET|BRX|JMX)\b", t)) and i + 1 < len(ins):
            leaders.add(pcs[i + 1])
    starts = sorted(leaders)
    block = {}
    for k, s in enumerate(starts):
        e = starts[k + 1] if k + 1 < len(starts) else pcs[-1] + 16
        block[s] = [i for i in range(at[s], len(ins)) if pcs[i] < e]
    succ = {s: set() for s in starts}
    for k, s in enumerate(starts):
        p, t = ins[block[s][-1]]
        tg = target(t)
        unconditional = tg is not None and not t.startswith("@") and not re.search(r"\bBRA\.DIV\b", t)
        if tg is not None and tg in block:
            succ[s].add(tg)
        if not unconditional and not re.search(r"^(EXIT|RET|BRX|JMX)\b", t) and k + 1 < len(starts):
            succ[s].add(starts[k + 1])
    pred = {s: set() for s in starts}
    for s, ss in succ.items():
        for x in ss:
            pred[x].add(s)
    reach, stack = {starts[0]}, [starts[0]]
    while stack:
        for x in succ[stack.pop()]:
            if x not in reach:
                reach.add(x)
                stack.append(x)
    dom = {s: set(reach) for s in reach}
    dom[starts[0]] = {starts[0]}
    changed = True
    while changed:
        changed = False
        for s in starts:
            if s == starts[0] or s not in reach:
                continue
            ps = [dom[p] for p in pred[s] if p in reach]
            new = {s} | (set.intersection(*ps) if ps else set())
            if new != dom[s]:
                dom[s], changed = new, True
    out = []
    for s in reach:
        for h in succ[s]:
            if h in dom[s]:   # back edge s -> h
                body, stack = {h, s}, [s]
                while stack:
                    for p in pred[stack.pop()]:
                        if p not in body and p in reach:
                            body.add(p)
                            stack.append(p)
                idx = [i for b in body for i in block[b]]
                mma = sum(1 for i in idx if re.search(r"\bH[G]?MMA\b|\bHGMMA", ins[i][1]))
                out.append((h, ins[block[s][-1]][0], len(idx), mma))
    return sorted(out), sum(1 for _, t in ins if "BRA.DIV" in t)


for f in sys.argv[1:]:
    for fn, ins in sorted(functions(f).items()):
        ls, div = loops(ins)
        name = re.sub(r".*attn_kernelILi(\d+)ELb(\d)ELi(\d+)E.*", r"L\1 F\2 M\3", fn) if "attn_kernel" in fn else fn[-40:]
        print(f"{name}: BRA.DIV {div}; loops {len(ls)}: " + " ".join(f"[{h:#x}<-{l:#x} n={n} mma={m}]" for h, l, n, m in ls))
