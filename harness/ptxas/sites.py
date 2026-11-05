# BAD waterfall loops (R2UR inside the BRA.U.ANY loop) -> the source call sites (innermost "inlined at" line), from
# nvdisasm -gi output; prints function, call-site line and its text, the serialized instruction, count
import re, sys, collections

dis, src = sys.argv[1], open(sys.argv[2]).read().split("\n")
fn, site, ins = None, None, []
for l in open(dis):
    m = re.match(r"\s*\.text\.(\S+):", l)
    if m:
        fn = m.group(1)
    m = re.search(r'//## File ".*?", line (\d+)(?: inlined at ".*?", line (\d+))?', l)
    if m:
        site = int(m.group(2) or m.group(1))
        continue
    m = re.match(r"\s+/\*([0-9a-f]{4,})\*/\s+(.*?);", l)
    if m:
        ins.append((" ".join(m.group(2).split()), fn, site))
out = collections.Counter()
for k, (t, f, s) in enumerate(ins):
    if "BRA.U.ANY" not in t:
        continue
    j = k
    while j > 0 and "ELECT" not in ins[j][0] and k - j < 16:
        j -= 1
    body = ins[j:k + 1]
    if not any("R2UR" in x[0] for x in body):
        continue
    core = next((x for x in body if re.search(r"UBLK|UTMA", x[0])), body[-2])
    out[(re.sub(r"^_Z\d+", "", f)[:28], core[2], core[0].split()[0])] += 1
for (f, s, op), n in sorted(out.items(), key=lambda x: (x[0][1] or 0)):
    print(f"line {s}: {op} x{n} [{f}]  | {src[s - 1].strip()[:110] if s else '?'}")
