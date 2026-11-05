# build every module each scored case runs, with its exact flags (BUILD_VARIANTS and per-case overrides resolved by
# the layer's own _case_builds / _build_spec), into <out>/<module>/; per module: C7xxx, count, bad/total waterfalls, REG,
# spill, hash; needs elect.py beside it (main's harness/ptxas) and tvm_ffi importable
#   J=12 python3 dumpx.py <assembled layer.py> <out>   (also writes <out>/cases.json: modules per scored case)
import importlib.util, os, sys, re, subprocess, hashlib, json, concurrent.futures as cf
os.environ["TVM_FFI_CUDA_ARCH_LIST"] = "9.0a"
cand, out = sys.argv[1], sys.argv[2]
spec = importlib.util.spec_from_file_location("L", cand); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
plans = m._case_builds()
SH = {"d1": (1, 1), "d16": (16, 1), "d32": (32, 1), "v1": (8, 8), "v16": (128, 8), "v32": (256, 8)}
cases = {}
for c, (t, l) in SH.items():
    for s in (True, False):
        sh = (t, l, s)
        fam = plans.get(sh) or m._shape_families(sh)
        cases[c + ("s" if s else "f")] = sorted(set(fam.values()))
mods = sorted({x for v in cases.values() for x in v})
json.dump(cases, open(f"{out}/cases.json", "w"), indent=1)
WRAP = ["-Xcompiler", "-fPIC", "-std=c++17", "-O2", "-gencode=arch=compute_90a,code=sm_90a"]
import tvm_ffi
INC = ["-I" + os.path.join(os.path.dirname(tvm_ffi.__file__), "include")]
H = os.path.dirname(os.path.abspath(__file__))
def build(name):
    resolved, (src, fns, flags) = m._build_spec(name)
    d = f"{out}/{name}"; os.makedirs(d, exist_ok=True)
    open(f"{d}/mod.cu", "w").write(src); open(f"{d}/flags.txt", "w").write(" ".join(flags))
    r = subprocess.run(["nice", "-n", "19", "nvcc"] + WRAP + flags + INC + ["-c", "mod.cu", "-o", "mod.o", "--keep", "--keep-dir", ".", "-Xptxas", "-v"], cwd=d, capture_output=True, text=True)
    if r.returncode: return name, "FAIL " + r.stderr[-300:]
    sass = subprocess.run(["cuobjdump", "-sass", f"{d}/mod.o"], capture_output=True, text=True).stdout
    open(f"{d}/mod.sass", "w").write(sass)
    cub = [f for f in os.listdir(d) if f.endswith(".cubin")]
    if cub: os.rename(f"{d}/{cub[0]}", f"{d}/mod.cubin")
    ins = [re.sub(r"/\*.*?\*/", "", l).strip() for l in sass.splitlines() if re.match(r"\s+/\*[0-9a-f]{4,}\*/", l)]
    el = subprocess.run(["python3", f"{H}/elect.py", f"{d}/mod.sass"], capture_output=True, text=True).stdout
    open(f"{d}/elect.txt", "w").write(el)
    codes = sorted(set(re.findall(r"\((C7\d{3})\)", r.stderr)))
    regs = re.findall(r"Used (\d+) registers", r.stderr); spill = re.findall(r"(\d+) bytes spill stores", r.stderr)
    open(f"{d}/ptxas_v.txt", "w").write(r.stderr)
    return name, f"{' '.join(codes)} n={len(ins)} bad={el.count('BAD')}/{len(el.splitlines())} REG={sorted(set(regs))} SPILL={sorted(set(spill))} {hashlib.md5(chr(10).join(ins).encode()).hexdigest()[:8]}"
with cf.ThreadPoolExecutor(int(os.environ.get("J", "8"))) as ex:
    for n, r in ex.map(build, mods): print(f"{n:22s} {r}", flush=True)
for c, v in cases.items(): print(c, v)
