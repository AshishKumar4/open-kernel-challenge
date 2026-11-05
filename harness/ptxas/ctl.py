# decode Hopper SASS control bits from cuobjdump -sass output: per instruction (pc, text, stall, yield, wbar, rbar, wait, reuse)
import re, sys, subprocess
def decode(cubin):
    out = subprocess.run(["cuobjdump", "-sass", cubin], capture_output=True, text=True).stdout
    funcs, cur, pend = {}, None, None
    for l in out.splitlines():
        m = re.match(r"\s*Function : (\S+)", l)
        if m: cur = funcs.setdefault(m.group(1), []); continue
        m = re.match(r"\s+/\*([0-9a-f]{4,})\*/\s+(.*?);\s*/\* (0x[0-9a-f]+) \*/", l)
        if m: pend = (int(m.group(1), 16), " ".join(m.group(2).split()), int(m.group(3), 16)); continue
        m = re.match(r"\s+/\* (0x[0-9a-f]+) \*/", l)
        if m and pend and cur is not None:
            hi = int(m.group(1), 16)
            w = (hi >> 52) & 0x3F
            cur.append(dict(pc=pend[0], t=pend[1], stall=(hi >> 41) & 0xF, yld=(hi >> 45) & 1, wbar=(hi >> 46) & 7,
                            rbar=(hi >> 49) & 7, wait=w, reuse=(hi >> 58) & 0xF))
            pend = None
    return funcs
def fmt(i):
    sb = lambda b: "-" if b == 7 else str(b)
    return "S%02d %s W%s R%s w%s r%x  %s" % (i["stall"], "Y" if i["yld"] else ".", sb(i["wbar"]), sb(i["rbar"]),
        "".join(str(k) for k in range(6) if i["wait"] >> k & 1) or "-", i["reuse"], i["t"])
if __name__ == "__main__":
    for f, ins in decode(sys.argv[1]).items():
        print("==", f)
        for i in ins: print("  %05x %s" % (i["pc"], fmt(i)))
