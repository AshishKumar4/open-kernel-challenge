# BSSY (and STL/LDL) inside a pc range of an nvdisasm -gi listing, with the source line chain (innermost + inlined-at)
#   python3 bssy.py li.gi mod.cu lo hi
import re, sys

lo, hi = int(sys.argv[3], 16), int(sys.argv[4], 16)
src = open(sys.argv[2]).read().split("\n")
chain = None
fresh = False
for l in open(sys.argv[1]):
    m = re.search(r'//## File ".*?", line (\d+)(?: inlined at ".*?", line (\d+))?', l)
    if m:
        if not fresh:
            chain = (int(m.group(1)), int(m.group(2)) if m.group(2) else None)
            fresh = True
        continue
    fresh = False
    m = re.match(r"\s+/\*([0-9a-f]{4,})\*/\s+(.*?);", l)
    if not m:
        continue
    pc = int(m.group(1), 16)
    if lo <= pc <= hi and re.search(r"\b(BSSY|STL|LDL|BRA.DIV|WARPSYNC)\b", m.group(2)):
        ln, at = chain or (0, None)
        print(f"{pc:#07x} {' '.join(m.group(2).split())[:40]:40s} L{ln}: {src[ln - 1].strip()[:80] if ln else ''}"
              + (f"  [in L{at}: {src[at - 1].strip()[:50]}]" if at else ""))
