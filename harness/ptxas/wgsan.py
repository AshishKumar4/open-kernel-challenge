# WGS-build health: ptxas -v spill bytes per kernel, local-memory traffic (STL/LDL) and how much of it sits inside
# loops (backward-branch ranges), and the issue-cycle distance from the R2UR feeding each HGMMA descriptor
#   python3 wgsan.py <dir with mod.ptx/mod.sass> ...
import re, subprocess, sys, collections
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from ctl import decode

for d in sys.argv[1:]:
    v = subprocess.run(["/usr/local/cuda/bin/ptxas", "-arch=sm_90a", "-v", f"{d}/mod.ptx", "-o", "/tmp/gp_wgs.cubin"],
                       capture_output=True, text=True).stderr
    spills = re.findall(r"(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads", v)
    out = collections.Counter()
    for fn, ins in decode("/tmp/gp_wgs.cubin").items():
        loops = []
        for i in ins:
            m = re.search(r"BRA(?:\.\S+)? (0x[0-9a-f]+)", i["t"])
            if m and int(m.group(1), 16) <= i["pc"]:
                loops.append((int(m.group(1), 16), i["pc"]))
        inloop = lambda pc: any(h <= pc <= t for h, t in loops)
        for k, i in enumerate(ins):
            op = re.sub(r"^@!?U?P\w+ ", "", i["t"]).split()[0]
            if op.startswith(("STL", "LDL")):
                out[op.split(".")[0] + ("_loop" if inloop(i["pc"]) else "_flat")] += 1
            if op.startswith("HGMMA"):
                urs = set(re.findall(r"gdesc\[UR(\d+)\]", i["t"]))
                urs |= {str(int(u) + 1) for u in list(urs)}
                dist = 0
                for j in range(k - 1, max(-1, k - 30), -1):
                    dist += ins[j]["stall"]
                    m = re.match(r"(?:@!?U?P\w+ )?R2UR(?:\.\w+)? UR(\d+)", ins[j]["t"])
                    if m and m.group(1) in urs:
                        out["hgmma_r2ur<=15" if dist <= 15 else "hgmma_r2ur>15"] += 1
                        break
                else:
                    out["hgmma_no_r2ur"] += 1
    print(d, "spill(frame,st,ld)=", sorted(set(spills)), dict(out))
