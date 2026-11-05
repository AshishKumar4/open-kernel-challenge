# waterfall (BRA.U.ANY) loops in SASS: the uniform-datapath instruction each serializes, and whether R2URs sit inside
# (= ptxas could not prove its operands warp-uniform)
import re, sys

for f in sys.argv[1:]:
    ins, fn = [], None
    for l in open(f):
        m = re.match(r"\s*Function : (\S+)", l)
        if m:
            fn = m.group(1)
        m = re.match(r"\s+/\*([0-9a-f]+)\*/\s+(.*?);", l)
        if m:
            ins.append((int(m.group(1), 16), " ".join(m.group(2).split()), fn))
    for k, (pc, t, fn) in enumerate(ins):
        m = re.search(r"BRA\.U\.ANY (0x[0-9a-f]+)", t)
        if not m:
            continue
        head = int(m.group(1), 16)
        body = [x for x in ins[:k + 1] if head <= x[0] <= pc and x[2] == fn]
        r2ur = sum("R2UR" in x[1] for x in body)
        ops = [re.sub(r"^@!?U?P\w+ ", "", x[1]).split()[0] for x in body]
        core = [o for o in ops if o not in ("ELECT", "R2UR", "PLOP3.LUT", "BRA.U.ANY", "IMAD.MOV.U32", "MOV")]
        print(f"{f}  fn={fn[:40]}  {head:#x}-{pc:#x} n={len(body)} R2UR_inside={r2ur} {'BAD' if r2ur else 'ok'}  {' '.join(core)[:80]}")
