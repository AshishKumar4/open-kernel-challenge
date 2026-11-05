# build every module of a candidate .py with its exact per-build flags (the judge's tvm_ffi wrapper flags + the
# candidate's own), into <out>/<build>/; then per build: SASS count, bad/total waterfall loops, REG, hash
#   python3 buildx.py <candidate.py> <out> [build-prefix ...]
import ast, json, os, re, subprocess, sys, hashlib, concurrent.futures as cf

cand, out = sys.argv[1], sys.argv[2]
only = sys.argv[3:]
src = open(cand).read()
strs = {n.targets[0].id: ast.literal_eval(n.value) for n in ast.parse(src).body
        if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id.endswith("_CU")
        and isinstance(n.value, ast.Constant)}
a = src.index("MOE_SWITCHES = {")
ns = dict(strs)
# top-level int constants the switch table may reference (MOE_SDEF ...)
for n in ast.parse(src).body:
    if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and isinstance(n.value, ast.Constant) and isinstance(n.value.value, int):
        ns[n.targets[0].id] = n.value.value
exec(src[a:src.index("_MODULES = {}", a)], ns)
# BUILDS edits later in the file (appended study switches): top-level statements that mention BUILDS
_end = src.index("_MODULES = {}", a)
for n in ast.parse(src).body:
    seg = ast.get_source_segment(src, n) or ""
    if n.col_offset == 0 and src.find(seg) > _end and "BUILDS" in seg and not isinstance(n, (ast.FunctionDef, ast.ClassDef, ast.Import, ast.ImportFrom)):
        exec(seg, ns)
WRAP = ["-Xcompiler", "-fPIC", "-std=c++17", "-O2", "-gencode=arch=compute_90a,code=sm_90a"]
INC = ["-I" + os.environ.get("TVM_FFI_INCLUDE", "/root/gp/pylib/tvm_ffi/include")]


def build(name):
    s, _, flags = ns["BUILDS"][name]
    d = f"{out}/{name}"
    os.makedirs(d, exist_ok=True)
    open(f"{d}/mod.cu", "w").write(s)
    open(f"{d}/flags.txt", "w").write(" ".join(flags))
    r = subprocess.run(["taskset", "-c", "24-31", "nice", "nvcc"] + WRAP + flags + INC +
                       ["-c", "mod.cu", "-o", "mod.o", "--keep", "--keep-dir", "."], cwd=d, capture_output=True, text=True)
    if r.returncode:
        return name, "BUILD FAILED " + r.stderr[-300:]
    sass = subprocess.run(["cuobjdump", "-sass", f"{d}/mod.o"], capture_output=True, text=True).stdout
    open(f"{d}/mod.sass", "w").write(sass)
    ins = [re.sub(r"/\*.*?\*/", "", l).strip() for l in sass.splitlines() if re.match(r"\s+/\*[0-9a-f]{4,}\*/", l)]
    el = subprocess.run(["python3", os.path.join(os.path.dirname(os.path.abspath(__file__)), "elect.py"), f"{d}/mod.sass"], capture_output=True, text=True).stdout
    res = subprocess.run(["cuobjdump", "-res-usage", f"{d}/mod.o"], capture_output=True, text=True).stdout
    regs = sorted({int(x) for x in re.findall(r"REG:(\d+)", res)})
    spill = sum(int(x) for x in re.findall(r"STACK:(\d+)", res))
    v = subprocess.run(["ptxas", "-arch=sm_90a", "-v", f"{d}/mod.ptx", "-o", "/dev/null"] +
                       [f.split()[-1] for f in flags if False], capture_output=True, text=True).stderr
    codes = sorted(set(re.findall(r"\((C7\d{3})\)", v + r.stderr)))
    return name, (f"{' '.join(codes) + ' ' if codes else ''}n={len(ins)} bad={el.count('BAD')}/{len(el.splitlines())} REG={regs if len(regs) > 1 else regs[0]} "
                  f"STACK={spill} {hashlib.md5(chr(10).join(ins).encode()).hexdigest()[:8]}")


names = [n for n in ns["BUILDS"] if not only or any(n.startswith(p) for p in only)]
with cf.ThreadPoolExecutor(8) as ex:
    for n, r in sorted(ex.map(build, names)):
        print(f"{n:14s} {r}")
