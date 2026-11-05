"""Build every CUDA extension a candidate's prepare() would build, on the CPU only (no GPU used), into
$TORCH_EXTENSIONS_DIR, so a later bench process on the GPU host loads cached builds instead of compiling.
   cpu_prebuild.py candidates/layer_<name>.py
Every builder is called with the static inputs prepare() gives it, so each ends built or failed (any exception: the
compiler's, the candidate's assertion, or a missing .so); exit status 0 only when all built, else 2 (never published).
The choices prepare() makes from the weights before building (PRMT scale decode, SC6) are the ones the reference
weights give (every scored and judge run uses them)."""
import importlib.util
import fcntl
import json
import os
import sys
import threading
import time

import torch
import torch.utils.cpp_extension as cx

_spec = importlib.util.spec_from_file_location("tvm_publish", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                                                                           "tvm_publish.py"))
_tp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_tp)
_tp.install()   # (tvm-ffi modules: built privately, published by one rename; harness/tvm_publish.py)

if os.environ.get("OKC_BUILD_FOR") == "h200":   # a build box without an H200: builders name and target the H200's arch
    class _H200:
        major, minor, multi_processor_count, name, total_memory = 9, 0, 132, "NVIDIA H200", 150754820096
    torch.cuda.get_device_capability = lambda *a, **k: (9, 0)
    torch.cuda.get_device_properties = lambda *a, **k: _H200()
t0 = time.time()
built = []
_li = cx.load_inline
TRACE = os.environ.get("OKC_TRACE_BUILDS")   # (freeze.py's dead-code pass: each build's sources and flags, nothing compiled)


def _trace(kind):
    def rec(*a, **k):
        name, pos = (k.pop("name"), a) if "name" in k else (a[0], a[1:])
        args = dict(zip(("cpp_sources", "cuda_sources", "functions"), pos), **k)
        with open(TRACE, "a") as f:
            f.write(json.dumps({"kind": kind, "name": name, **{x: v for x, v in args.items() if isinstance(v, (str, list, tuple))}}) + "\n")
        built.append((name, 0.0))
        import unittest.mock
        return unittest.mock.MagicMock()
    return rec


_name_locks, _name_locks_guard = {}, threading.Lock()


def _rec(name, *a, **k):
    if os.environ.get("OKC_USING"):   # every extension the trial needs, as it is reached (a build box's cache keeps these)
        with open(os.environ["OKC_USING"], "a") as f:
            f.write(name + "\n")
    with _name_locks_guard:
        lock = _name_locks.setdefault(name, threading.Lock())
    with lock:   # one load of a name at a time in this process : torch bumps its versioner before taking the
        t = time.time()   # build's file lock, so a second thread could skip the build and import a .so not yet written
        r = _li(name, *a, **k)
    built.append((name, round(time.time() - t, 1)))
    return r


cx.load_inline = _trace("torch") if TRACE else _rec   # (before the candidate loads: its import-time builders get it)
_li = _li if not TRACE else cx.load_inline
path = sys.argv[1]
spec = importlib.util.spec_from_file_location("cand", path)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
for nm in dir(m):
    if getattr(m, nm) is _li:
        setattr(m, nm, _trace("torch") if TRACE else _rec)
if TRACE and hasattr(m, "_tvm_load_inline"):
    m._tvm_load_inline = _trace("tvm")
if hasattr(m, "_MA_SFX_IF_PRMT"):
    m._MA_SFX = m._MA_SFX_IF_PRMT if getattr(m, "_MA_PRMT_SCALES", False) else ()
if hasattr(m, "_MA_SC6W"):
    m._MA_SC6W = bool(getattr(m, "SC6_WGS", False) and m._MA_SFX and m._wgs_on())
jobs = []
if hasattr(m, "_stream_extension"):   # the streaming MoE build: the main engine and the narrow one (defines only, no weights)
    jobs.append(("stream", m._stream_extension, ()))
    if getattr(m, "STREAM_NARROW_DEFINES", None) is not None:
        jobs.append(("stream narrow", m._stream_extension, (None, m.STREAM_NARROW_DEFINES)))
for nm in ("_att7", "_moe", "_moe_tiny", "_moe_tiny2", "_moe_mid", "bfp__get_ext", "_graft_glue_module", "_graft_b2_module",
           "_sl_cluster_module", "_de_ext", "_router_fp32_ext", "_gle_ext", "_gle_b1x_module", "_flushext", "_pfside_ext"):
    if hasattr(m, nm):
        jobs.append((nm, getattr(m, nm), ()))
if hasattr(m, "_moe"):
    jobs.append(("_moe(32)", m._moe, (32,)))
if hasattr(m, "_CHEN") and "_rqext" in m._CHEN:   # prepare's rq extension (_rqstart): its variants are static tables
    C = m._CHEN
    jobs.append(("rq", C["_rqext"], ([C["RQV"][C["RQI"][nm]] for nm in C["RQPROD"]],)))
    if "_flushext" in C:   # the calibration's L2 flush kernel
        jobs.append(("chen flush", C["_flushext"], ()))
if hasattr(m, "_mf_cases") and getattr(m, "_MA_SFX", ()):   # prepare's A/B barrier twins (_mf_start_base_builds)
    _off = ("-DOKC_SC6OFF=1",) if (getattr(m, "SC6_WGS", False) and m._wgs_on()) else ()
    for _i, (_, _defs) in enumerate(m._mf_cases(m._MA_SFX)):
        jobs.append((f"mf twin {_i}", m._ma_ext, (tuple(_defs) + ("-DOKC_MFIN=0",) + _off + tuple(m._MA_SFX),)))
if not TRACE and hasattr(m, "_build") and isinstance(getattr(m, "BUILDS", None), dict):   # the clean layer: each of its
    import hashlib                                                                          # tvm-ffi modules, as prepare's

    def _prefix(b):   # (the module's cache dir prefix, as the layer's _build names it: muse_<family>_<md5(src + flags)>)
        if hasattr(m, "_build_spec"):
            resolved, (src, _, flags) = m._build_spec(b)
            return f"muse_{m._build_family(resolved)}_{hashlib.md5((src + repr(flags)).encode()).hexdigest()[:12]}"
        src, _, flags = m.BUILDS[b]
        return f"muse_{b}_{hashlib.md5((src + repr(flags)).encode()).hexdigest()[:12]}"

    def _clean(b):   # (counted with the rest: the module's name; its cache dir's prefix for box_ship, OKC_TVM_USING)
        t = time.time()
        m._build(b)
        built.append((f"muse_{b}", round(time.time() - t, 1)))
        if os.environ.get("OKC_TVM_USING"):
            with open(os.environ["OKC_TVM_USING"], "a") as f:
                f.write(_prefix(b) + "\n")
    if hasattr(m, "_case_builds"):   # (prepare's per-case variants: _case_builds adds them to BUILD_VARIANTS from
        m._case_builds()             # CASE_OVERRIDES at prepare, so at import they are not there yet)
    _seen = {}   # (_start_builds builds BUILDS and BUILD_VARIANTS, the per-case variants too: every one is built here and
    for _b in (*m.BUILDS, *getattr(m, "BUILD_VARIANTS", {})):   # shipped, else prepare compiles it on the GPU box inside
        _seen.setdefault(_prefix(_b), _b)                       # its exclusive section, the GPU idle (15:19))
    jobs += [(f"clean {b}", _clean, (b,)) for b in _seen.values()]   # tvm-ffi modules, into ~/.cache/tvm-ffi
for _nm, _fn, _a in getattr(m, "PREBUILD_JOBS", ()):   # a candidate's own prepare-only builders (build metadata: freeze.py
    jobs.append((_nm, getattr(m, _fn) if isinstance(_fn, str) else _fn, tuple(_a)))   # drops it from a submission)
failed = []


SLOTS = int(os.environ.get("OKC_COMPILE_SLOTS", "0"))   # box-wide compiles at once (box_build.sh: the box's threads)


def slot():
    """one of the box's SLOTS compile slots (a file lock each, shared by every build on the box): held for one job"""
    while True:
        for i in range(SLOTS):
            f = open(f"/root/out/cslot.{i}.lock", "a")
            try:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return f
            except BlockingIOError:
                f.close()
        time.sleep(0.2)


def run(nm, f, a):
    lk = slot() if SLOTS else None
    try:
        f(*a)
    except Exception as e:  # noqa: BLE001
        # every builder gets the static inputs prepare would pass, so any exception is a failed build: the compiler's
        # (with its own output), the candidate's (an assertion), or a missing .so (another extension failed)
        failed.append(f"{nm}: {type(e).__name__}: {str(e)[:300]}\n{str(e)[-3000:] if len(str(e)) > 300 else ''}")
    finally:
        if lk:
            lk.close()


ths = [threading.Thread(target=run, args=j) for j in jobs]
for th in ths:
    th.start()
for th in ths:
    th.join()
print(f"prebuilt {len(built)} extensions in {time.time() - t0:.0f} s: {built}")
for e in failed:
    print("FAILED:", e)
sys.exit(2 if failed else 0)
