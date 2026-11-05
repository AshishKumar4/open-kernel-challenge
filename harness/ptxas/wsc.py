# per function: WARPSYNC.COLLECTIVE emulation sites before the first EXIT (inline: run on the hot path) and the BRA.DIV
# count (out-of-line divergent-collective paths)
#   python3 wsc.py <mod.sass> ...
import re, sys
for f in sys.argv[1:]:
    fns, cur = {}, None
    for l in open(f):
        m = re.search(r"Function : (\S+)", l)
        if m: cur = fns.setdefault(m.group(1), [])
        m = re.match(r"\s+/\*([0-9a-f]{4,})\*/\s+(.*?)\s*;", l)
        if m and cur is not None: cur.append((int(m.group(1), 16), " ".join(m.group(2).split())))
    for fn, ins in fns.items():
        ex = [pc for pc, t in ins if re.search(r"\bEXIT\b", t) and not t.startswith("@")]
        last = max(ex) if ex else 0
        first_exit = min(ex) if ex else 0
        w = [pc for pc, t in ins if "WARPSYNC.COLLECTIVE" in t]
        inl = [hex(pc) for pc in w if pc < first_exit]
        print(f"{f.split('/')[-2]} {fn[-50:]} exits={[hex(e) for e in ex][:4]} wsc={len(w)} before_first_exit={inl[:12]} bradiv={sum('BRA.DIV' in t for _, t in ins)}")
