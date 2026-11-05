"""Correctness tests and judge-style timing for the five kernel problems, run inside the challenge image.

    python harness/okc_harness.py test  <problem> [--cases scored|tests|nearby|all] [--file submission.py]
    python harness/okc_harness.py bench <problem> [--cases ...] [--reps 40]
    python harness/okc_harness.py once  <problem> --case "batch=1,ctx=80000,..."   (one eager call; for ncu)

problem: proj | moe | attn | layer | prefill. Cases come from the problem's TASK.md (tests) and the scored
shapes recorded in profiles/*.json (benchmarks[*].case). Timing follows the judge: each shape is
captured as a CUDA graph and replayed after reading a buffer twice the L2 size (cold, clean L2), alternating with SGLang's kernel
(baseline.py) on the same inputs; the speedup is the median of paired ratios after subtracting the
replay cost of an empty graph.
"""
import argparse
import importlib.util
import json
import math
import mmap
import os
import pathlib
import re
import shutil
import statistics
import sys
import time

import torch
import torch.utils._pytree

# OKC_GPU_LOCK=<file>: two benches share one GPU and never overlap a timed section. Each holds the lock shared for its
# whole run and exclusive only while it times (gpu_timing below: both prepares, which time their own plan picks, and
# every timed replay): nothing else touches the GPU then, since every other GPU user holds the lock shared (another
# bench) or exclusive for its whole run (a job that does not know the lock, vqg.sh). flock converts by dropping and
# re-taking, so two benches converting at once cannot deadlock.
import faulthandler, fcntl, signal
faulthandler.register(signal.SIGUSR1, all_threads=True)   # the runner's time limit sends it first: the log shows where it hung
_GPU_LOCK = None
_MODE = None   # how this process holds the GPU lock: "SH", "EX" while it times (gpu_timing), None without it
if os.environ.get("OKC_GPU_LOCK"):
    _GPU_LOCK = open(os.environ["OKC_GPU_LOCK"], "a")
    fcntl.flock(_GPU_LOCK, fcntl.LOCK_SH)
    _MODE = "SH"


class gpu_timing:
    """exclusive use of the GPU for a timed section (with OKC_GPU_LOCK), shared again after"""

    def __enter__(self):   # (body: the section's own seconds, inside the lock; wait: seconds spent taking the lock
        self.wait, self.body = 0.0, 0.0   # before it and taking the shared lock back after: reported apart)
        if _GPU_LOCK:
            torch.cuda.synchronize()
            t = time.time()
            fcntl.flock(_GPU_LOCK, fcntl.LOCK_EX)
            globals()["_MODE"] = "EX"
            self.wait = time.time() - t
            if self.wait > 0.1:
                print(f"  [gpu lock] {self.wait:.1f} s waiting to time", flush=True)
            self.nice = os.getpriority(os.PRIO_PROCESS, 0)
            try:   # the launcher thread outranks everything else on the box while it times (root in the container)
                os.setpriority(os.PRIO_PROCESS, 0, -10)
            except OSError:
                pass
        self.t_in = time.time()
        return self

    def __exit__(self, *exc):
        if _GPU_LOCK:
            torch.cuda.synchronize()
        self.body = time.time() - self.t_in
        if _GPU_LOCK:
            try:
                os.setpriority(os.PRIO_PROCESS, 0, self.nice)
            except OSError:
                pass
            # flock is not fair: re-taking the shared lock at once could beat a bench waiting for exclusive time, which
            # then waited out this bench's whole run (52-83 s waits, 120 s kills); let a waiter in first
            t = time.time()
            fcntl.flock(_GPU_LOCK, fcntl.LOCK_UN)
            time.sleep(0.05)
            fcntl.flock(_GPU_LOCK, fcntl.LOCK_SH)
            self.wait += time.time() - t
            globals()["_MODE"] = "SH"


import hashlib
import threading
import torch.utils.cpp_extension as _cx
_cx_load_inline = _cx.load_inline
_LOADED = {}   # extension name -> the hash of the build this process loaded under it


def _named_load_inline(name, *a, **k):
    """the name the candidate asks for (stable: the build box ships it, gpu1 compiles a runtime variant once and reuses
    it), unless this process already loaded another build under that name (a persistent bench: production, an earlier
    candidate): then that build's own name, not torch's in-process name_v1 compiled into the first one's directory"""
    kk = sorted((x, v) for x, v in k.items() if x not in ("verbose", "build_directory", "keep_intermediates"))
    h = hashlib.md5(repr((a, kk)).encode()).hexdigest()[:10]
    return _cx_load_inline(name if _LOADED.setdefault(name, h) == h else f"{name}_{h}", *a, **k)


_cx.load_inline = _named_load_inline   # (before any candidate loads: its own load_inline imports get this one)


def _no_gpu_lock(f):
    """CPU-only extension work (a compile, loading a built .so) holds no GPU lock: the lock is released, then the mode
    held before is taken again (the GPU work queued so far finishes first). Only the main thread does this: a builder on
    another thread (prepare's barrier twins) must not drop the lock the main thread's timing holds."""
    def g(*a, **k):
        mode = _MODE if _GPU_LOCK and threading.current_thread() is threading.main_thread() else None
        if mode:
            torch.cuda.synchronize()
            fcntl.flock(_GPU_LOCK, fcntl.LOCK_UN)
        try:
            return f(*a, **k)
        finally:
            if mode:
                fcntl.flock(_GPU_LOCK, fcntl.LOCK_EX if mode == "EX" else fcntl.LOCK_SH)
    return g


_cx._jit_compile = _no_gpu_lock(_cx._jit_compile)
try:
    import tvm_ffi.cpp as _tvm_cpp
    _tvm_cpp.load_inline = _no_gpu_lock(_tvm_cpp.load_inline)
    _spec = importlib.util.spec_from_file_location("tvm_publish", pathlib.Path(__file__).with_name("tvm_publish.py"))
    _tp = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_tp)
    _tp.install()   # (each module built in a private directory, then published by one rename: no shared build dir)
except ImportError:
    pass

if os.environ.get("OKC_BUILD_LOG"):   # the extensions this process compiles (cache misses): one line each, name and seconds
    import torch.utils.cpp_extension as _cx
    _cx_build = _cx._write_ninja_file_and_build_library

    def _logged_build(name, *a, **k):
        # torch runs ninja for every extension a process loads; only a rewritten .so is a compile (ninja's no-op is not)
        so = os.path.join(k.get("build_directory") or a[6], f"{name}.so")
        m0 = os.path.getmtime(so) if os.path.exists(so) else None
        t = time.time()
        _cx_build(name, *a, **k)
        if m0 is None or os.path.getmtime(so) != m0:
            with open(os.environ["OKC_BUILD_LOG"], "a") as f:
                f.write(f"{name} {time.time() - t:.1f}\n")
    _cx._write_ninja_file_and_build_library = _logged_build

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
DIRS = {"proj": "muse_bf16_proj", "moe": "muse_moe_w4a16", "attn": "muse_attention_decode",
        "layer": "muse_decode_layer", "prefill": "muse_prefill_layer"}
HAS_WEIGHTS = {"proj": True, "moe": True, "attn": False, "layer": True, "prefill": True}
LAYERS = ("layer", "prefill")  # whole-layer problems: case fields from the weights, outputs collected


def load_module(path: pathlib.Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def parse_value(v: str):
    for cast in (int, float):
        try:
            return cast(v)
        except ValueError:
            pass
    return v


def task_tests(pdir: pathlib.Path):
    text = (pdir / "TASK.md").read_text()
    block = text.split("must pass these", 1)[1].split("```", 2)[1]
    cases = []
    for line in block.strip().splitlines():
        kv = {}
        for part in line.split(","):
            k, v = part.strip().split(" ", 1)
            kv[k] = parse_value(v.strip())
        cases.append(kv)
    return cases


def scored_cases(pdir: pathlib.Path):
    for f in sorted((pdir / "profiles").glob("*.json")):
        d = json.loads(f.read_text())
        out = []
        for b in d["benchmarks"]:
            c = dict(b["case"])
            c.pop("repeats", None)
            out.append(c)
        return out
    raise FileNotFoundError(f"no profile json in {pdir}/profiles")


def nearby_cases(pdir: pathlib.Path):
    """Shapes near the scored ones (the organizers' reruns also time nearby and held-out shapes): other token
    counts, batches and contexts, each derived from a scored case with its other fields kept."""
    base = scored_cases(pdir)
    out, seen = [], set()

    def add(c):
        key = tuple(sorted((k, str(v)) for k, v in c.items() if k not in ("seed", "weight")))
        if key not in seen:
            seen.add(key)
            out.append(c)

    for c in base:
        if "m" in c:  # dense projections
            for m in (2, 4, 12, 24, 48, 64, 96, 192):
                add({**c, "m": m, "load": f"decode {m}"})
        elif "bs" in c:  # experts
            for bs in (2, 4, 12, 24, 48, 64, 96, 192):
                add({**c, "bs": bs, "load": f"decode {bs}"})
        elif "batch" in c:  # attention, layer
            for b in (2, 4, 8, 24):
                q_len = c.get("q_len", 1)
                n = {**c, "batch": b, "load": f"{'verify' if q_len > 1 else 'decode'} {b}"}
                if b > 1:
                    n["spread"] = c.get("spread", "agentic")
                add(n)
            for ctx in (8000, 30000, 120000):
                add({**c, "ctx": ctx})
    return out


def parse_case(s: str):
    return {k: parse_value(v) for k, v in (p.split("=", 1) for p in s.split(","))}



INPUT_FLOOR = 50 << 30   # (the input cache's filesystem keeps this much free: /dev/shm is the box's RAM)

_PINNED = {}   # an input cache file -> its host mapping, every storage registered with CUDA (this process, once)


def _from_pinned(path):
    """a cached input on the GPU: the file mapped on the host once per process and its storages registered with CUDA
    (cudaHostRegister: page-locked, so the copy runs at DMA speed, ~55 GB/s against ~6.5 GB/s from the pageable mapping
    torch.load(map_location="cuda") copies from: a decode-32 case's 20 inputs, 14 GB, took 2.1 s, 0.3 s this way after
    the first load; 15 s of every bench), then copied per storage, so tensors sharing one keep sharing it"""
    h = _PINNED.get(path)
    if h is None:
        # (a shared, read-only mapping: the box's bench servers pin the same page-cache pages, one copy per box; a private
        # mapping pinned for writing made each server its own copy, 86 GiB on spot1, so only one runner fitted)
        torch.serialization.set_default_mmap_options(mmap.MAP_SHARED)
        try:
            h = torch.load(path, mmap=True, map_location="cpu", weights_only=False)
        finally:
            torch.serialization.set_default_mmap_options(mmap.MAP_PRIVATE)
        leaves = [t for t in torch.utils._pytree.tree_leaves(h) if isinstance(t, torch.Tensor)]
        for st in {t.untyped_storage().data_ptr(): t.untyped_storage() for t in leaves if t.untyped_storage().nbytes()}.values():
            r = int(torch.cuda.cudart().cudaHostRegister(st.data_ptr(), st.nbytes(), 8))   # (cudaHostRegisterReadOnly)
            if r:
                raise RuntimeError(f"cudaHostRegister failed ({r}) on {path}")
        _PINNED[path] = h
    gpu = {}

    def mv(t):
        st = t.untyped_storage()
        if not st.nbytes():
            return torch.empty(t.size(), dtype=t.dtype, device="cuda")
        if st.data_ptr() not in gpu:
            gpu[st.data_ptr()] = torch.UntypedStorage(st.nbytes(), device="cuda")
            gpu[st.data_ptr()].copy_(st, non_blocking=True)
        return torch.empty(0, dtype=t.dtype, device="cuda").set_(gpu[st.data_ptr()], t.storage_offset(), t.size(), t.stride())

    out = torch.utils._pytree.tree_map(lambda x: mv(x) if isinstance(x, torch.Tensor) else x, h)
    torch.cuda.current_stream().synchronize()
    return out


class Problem:
    def __init__(self, name: str, submission: pathlib.Path, against: pathlib.Path | None = None, keep=None, tag=""):
        """keep (benchd): a Problem whose reference, weights and prepared base side this one reuses; only the submission
        is loaded (as module sub_<name><tag>) and prepared."""
        self.name = name
        self.dir = ROOT / DIRS[name]
        sys.path.insert(0, str(self.dir))
        self.memo = keep.memo if keep else None
        if keep:
            self.ref, self.base, self.base_label, self.weights, self.base_state = (
                keep.ref, keep.base, keep.base_label, keep.weights, keep.base_state)
            self.files = (submission, keep.files[1])
        else:
            self.ref = load_module(self.dir / "reference.py", f"ref_{name}")
            # The side we time against: SGLang's kernel, or another submission (paired A/B of two candidates).
            self.base = load_module(against or self.dir / "baseline.py", f"base_{name}")
            self.base_label = against.stem if against else "sglang"
            self.files = (submission, against or self.dir / "baseline.py")
            self.weights = self.base_state = None
        t0 = time.time()
        torch.cuda.reset_peak_memory_stats()
        self.rest = torch.cuda.memory_allocated()   # (a bench's peak memory is read above what was resident before it)
        self.sub = load_module(submission, f"sub_{name}{tag}") if submission else None
        self.sub_state = None
        if HAS_WEIGHTS[name] and not submission:   # (benchd's kept side: reference, weights and base only)
            self.weights = self.ref.generate_weights(self.ref.WEIGHTS_SEED)
            with gpu_timing():
                self.base_state = self.base.prepare({k: v.clone() for k, v in self.weights.items()})
        elif HAS_WEIGHTS[name]:
            if not keep:
                self.weights = self.ref.generate_weights(self.ref.WEIGHTS_SEED)
            # prepare() times plans itself (_tune, autotune, A/B picks): exclusive, so a second bench on the GPU cannot
            # bias its picks (compile-ahead leaves it loading cached extensions, not compiling)
            with gpu_timing():
                self.sub_state = self.sub.prepare({k: v.clone() for k, v in self.weights.items()})
                torch.cuda.synchronize()
                self.prepare_s = time.time() - t0
                print(f"[{name}] import + prepare {self.prepare_s:.1f} s", flush=True)
                if not keep:
                    self.base_state = self.base.prepare({k: v.clone() for k, v in self.weights.items()})
        else:
            print(f"[{name}] import {time.time() - t0:.1f} s", flush=True)

    def inputs(self, case: dict, seed: int):
        """one fresh input, generated on the GPU from its seed. With OKC_INPUT_CACHE (a box's bench servers: one host
        directory they share, /dev/shm/okc_inputs) it is kept there the first time, written to a temporary name and
        renamed, and later calls map it and copy it to the GPU: the same tensors, strides and aliasing (torch.save), no
        regeneration; a cached file that does not load is made again. Nothing is written while the directory's filesystem has
        under INPUT_FLOOR bytes free."""
        d = os.environ.get("OKC_INPUT_CACHE")
        if d:
            key = (sorted(case.items()), seed, hashlib.md5((self.dir / "reference.py").read_bytes()).hexdigest())
            path = pathlib.Path(d) / (hashlib.md5(repr(key).encode()).hexdigest() + ".pt")
            if path.exists():
                try:
                    return _from_pinned(path)
                except Exception as e:   # noqa: BLE001  (a file a VM stop cut short: made again below)
                    print(f"input cache: {path.name} unreadable ({str(e)[:80]}): generated again", flush=True)
                    path.unlink(missing_ok=True)
        kw = {k: v for k, v in case.items() if k not in ("load", "weight", "seed")}
        if self.name in LAYERS:
            kw.update(self.ref.case_fields({**kw, "seed": seed}, self.weights))
        data = self.ref.generate_input(**kw, seed=seed)
        if d and shutil.disk_usage(d).free > INPUT_FLOOR:
            tmp = path.with_suffix(f".{os.getpid()}.tmp")
            with open(tmp, "wb") as f:   # (on disk before its name: a preempted VM kept renamed but unwritten files)
                torch.save(data, f)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)
        return data

    def call(self, fn_mod, state, data):
        return fn_mod.kernel_fn(data, state) if HAS_WEIGHTS[self.name] else fn_mod.kernel_fn(data)

    def check(self, data, out, key=None, clean=None):
        """key (the input's identity: case and seed) and clean (that input as generated, untouched by any call): with a
        memo (benchd), the reference's work is computed once per input, from clean, and reused by every later check"""
        if key is not None and self.memo is not None and self.name in LAYERS:
            return self._memo_check(data, out, key, clean)
        if self.name == "attn" or self.name in LAYERS:
            raw = self.ref.collect(data, out)
            return self.ref.check_implementation(data, raw, *(() if self.name == "attn" else (self.weights,)))
        return self.ref.check_implementation(data, out, self.weights)

    def _memo_check(self, data, out, key, clean):
        """check_implementation itself, with its reference work memoized per input (benchd): the reference front reads
        only the original inputs, taken from clean (a submission that wrote outside its new slots cannot skew the
        reference later candidates are checked against), and the back depends on the front and the submission's expert
        ids: kept per front with the ids it was computed for, recomputed when a check brings other ids (the reference
        with exactly the submission's ids, as without the memo). The memo holds only the reference's own results
        (attention, router and MoE outputs), in host memory: no input tensors, no GPU memory."""
        ref, front0, back0 = self.ref, self.ref.ref_front, self.ref.ref_back
        fronts = {}   # id of a front handed to check_implementation -> its memo key

        def front(d, w, query_dtype=torch.bfloat16):
            k = (key, str(query_dtype))
            if k not in self.memo:
                self.memo[k] = _to(front0(clean, w, query_dtype), "cpu")
            f = _to(self.memo[k], "cuda")
            fronts[id(f)] = k
            return f

        def back(f, w, ids):
            k, i = (fronts[id(f)], "back"), ids.cpu().numpy().tobytes()
            if self.memo.get(k, (None,))[0] != i:
                self.memo[k] = (i, _to(back0(f, w, ids), "cpu"))
            return _to(self.memo[k][1], "cuda")

        ref.ref_front, ref.ref_back = front, back
        try:
            raw = ref.collect(data, out)
            return ref.check_implementation(data, raw, self.weights)
        finally:
            ref.ref_front, ref.ref_back = front0, back0


def _to(x, device):
    """a reference result (tensors in dicts / tuples) on device"""
    if isinstance(x, torch.Tensor):
        return x.to(device)
    if isinstance(x, dict):
        return {k: _to(v, device) for k, v in x.items()}
    if isinstance(x, tuple):
        return tuple(_to(v, device) for v in x)
    return x


def clone_inputs(data):
    return {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in data.items()}


def run_tests(prob: Problem, cases, seed0=1234):
    ok_all = True
    for i, case in enumerate(cases):
        data = prob.inputs(case, case.get("seed", seed0 + i))
        try:
            out = prob.call(prob.sub, prob.sub_state, clone_inputs(data) if prob.name not in ("attn", "layer") else data)
            torch.cuda.synchronize()
            ok, msg = prob.check(data, out)
        except Exception as e:  # report and continue with the next case
            ok, msg = False, f"{type(e).__name__}: {e}"
        ok_all &= bool(ok)
        print(f"{'PASS' if ok else 'FAIL'} {case}: {msg}", flush=True)
    return ok_all


def l2_flusher():
    """Cold L2 as the judge makes it: read (not write) a buffer twice the L2 size, so no dirty lines are
    written back during the timed call."""
    size = 2 * torch.cuda.get_device_properties(0).L2_cache_size
    buf = torch.ones(size // 4, dtype=torch.float32, device="cuda")
    out = torch.empty((), dtype=torch.float32, device="cuda")
    flush = lambda: torch.sum(buf, dim=0, out=out)   # noqa: E731
    flush()   # (its first use here, before time_graph's gate holds the stream: lazily loading the reduction kernel
    torch.cuda.synchronize()   # synchronizes, and under the held stream it deadlocked; standalone problems, whose
    return flush               # prepare never ran a sum, hung at their first flush, Nov 4)


def capture(fn):
    fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    torch.cuda.synchronize()
    return g


class _Gate:
    """Holds the stream until the host has queued a whole timed rep: cuStreamWaitValue32 on a pinned host word, then the
    flush, the start event, the graph and the end event are enqueued, then the word is released. The GPU then runs the
    rep back to back: a descheduled launcher can no longer land inside the timed interval, whatever the host load."""

    def __init__(self):
        import ctypes
        self.cu = ctypes.CDLL("libcuda.so.1")
        self.word = torch.zeros(1, dtype=torch.int32, pin_memory=True)
        self.view = self.word.numpy()
        dptr = ctypes.c_uint64()
        assert self.cu.cuMemHostGetDevicePointer_v2(ctypes.byref(dptr), ctypes.c_void_p(self.word.data_ptr()), 0) == 0
        self.dptr, self.gen = dptr.value, 0
        self.wait = self.cu.cuStreamWaitValue32_v2
        self.wait.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint32, ctypes.c_uint32]

    def hold(self):
        self.gen += 1   # CU_STREAM_WAIT_VALUE_GEQ (0): the stream waits until the word reaches this rep's number
        assert self.wait(ctypes_stream(), self.dptr, self.gen, 0) == 0

    def release(self):
        self.view[0] = self.gen


def ctypes_stream():
    import ctypes
    return ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)


_GATE = None


def time_graph(g, flush, reps):
    global _GATE
    _GATE = _GATE or _Gate()
    times = []
    for _ in range(reps):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        _GATE.hold()
        try:   # a raise after hold() must not leave the stream (and the GPU lock with it) blocked
            flush()   # the judge's cold L2: a read just before the start event
            s.record()
            g.replay()
            e.record()
        finally:
            _GATE.release()
        e.synchronize()
        times.append(s.elapsed_time(e) * 1000.0)
    return times


def run_bench(prob: Problem, cases, reps=40, seed0=777, out_json=None, fresh=1, cache=None):
    """Judge-style timing (TASK.md "How a run is timed"): per shape, `fresh` inputs (the case seed + k; every timed
    call gets new inputs, repetition r copies input r % fresh into each side's captured input buffers, outside the
    timing); one CUDA graph per side; each replay after a cold-L2 read; the two sides take turns on the same input, the
    order swapped every repetition; the speedup is the median of paired ratios, the time SGLang's (or --against's)
    median over it; the judge's fixed replay cost is subtracted. Every input's outputs are checked first: any failure
    ends the run with no score (exit 3).
    cache (benchd, a dict kept across runs in one process): the flush buffer and the fixed replay cost are made by the
    first run and reused; with cache["inputs"] also, per shape, the fresh inputs and the base side's captured graph. That
    keeps ~100 GB of inputs resident, too much beside a second runner, so benchd leaves it off; the checks' reference work
    is memoized by input identity regardless (Problem.memo)."""
    # OKC_SEED_BLOCK=b (confirmations): every case seed + 1000 b, a block of inputs no screen has seen (b = 0: the
    # profile's seeds; a case's fresh inputs are seed + 0..19, so blocks never overlap)
    block = int(os.environ.get("OKC_SEED_BLOCK", "0"))
    t_start = time.time()
    flush = cache.setdefault("flush", l2_flusher()) if cache is not None else l2_flusher()
    # the judge's fixed replay cost (its replay_overhead): a one-node and a two-node graph of a trivial kernel,
    # overhead = 2 x one - two (each after the cold-L2 read), the cost of a replay less its nodes' own time
    if cache is not None and "fixed" in cache:
        fixed, t1, t2 = cache["fixed"]
    else:
        tiny = torch.zeros(1, device="cuda")
        one = capture(lambda: tiny.add_(1))
        two = capture(lambda: (tiny.add_(1), tiny.add_(1)))
        with gpu_timing():
            t1 = statistics.median(time_graph(one, flush, 40))
            t2 = statistics.median(time_graph(two, flush, 40))
        fixed = 2 * t1 - t2
        if cache is not None:
            cache["fixed"] = (fixed, t1, t2)
    print(f"fixed replay cost {fixed:.2f} us (one node {t1:.2f}, two nodes {t2:.2f}); {fresh} fresh input(s) per shape", flush=True)
    loads, rows = {}, []
    # OKC_BENCH_NOCHECK=1: skip the per-input reference checks (the other half of a role-swapped pair already ran them
    # on the same inputs); the graph-replay checks stay
    nocheck = os.environ.get("OKC_BENCH_NOCHECK") == "1"
    ncheck = int(os.environ.get("OKC_BENCH_NCHECK", fresh))   # inputs checked per shape (screens: a few)
    for i, case in enumerate(cases):
        seed = case.get("seed", seed0 + i) + 1000 * block
        ident = repr(sorted(case.items()))   # (with seed + k, an input's identity: the memo's key)
        tc0 = time.time()
        kept = cache.setdefault("cases", {}).get(i) if cache is not None and cache.get("inputs") else None
        datas = kept[0] if kept else [prob.inputs(case, seed + k) for k in range(fresh)]
        tc1 = time.time()
        checks = []
        for k, data in enumerate([] if nocheck else datas[:ncheck]):
            for label, mod, st in (("sub", prob.sub, prob.sub_state),):   # the --against side is a known-correct build
                dc = clone_inputs(data)
                o = prob.call(mod, st, dc)
                torch.cuda.synchronize()
                ok, msg = prob.check(dc, o, key=(ident, seed + k), clean=data)
                if k == 0 or not ok:
                    checks.append(f"{label} {'PASS' if ok else 'FAIL'} {msg}")
                if not ok:  # ineligible: no score at all, a nonzero exit, FAILED in the results
                    print(f"  check: {' | '.join(checks)}", flush=True)
                    print(f"FAILED: {label} outputs fail the reference check on {case} (input {k})", flush=True)
                    if out_json:
                        pathlib.Path(out_json).write_text(json.dumps({"failed": f"{label}: {msg}", "case": case}, indent=1))
                    sys.exit(3)
        print(f"  check ({fresh} inputs): " + " | ".join(checks), flush=True)
        same = all(isinstance(v, torch.Tensor) == isinstance(datas[0][key], torch.Tensor) and
                   (not isinstance(v, torch.Tensor) or (v.shape == datas[0][key].shape and v.dtype == datas[0][key].dtype))
                   for d in datas for key, v in d.items())
        if not same:
            print("  (input shapes vary with the seed: one input for this shape)", flush=True)
            datas = datas[:1]
        tc2 = time.time()
        d_sub = clone_inputs(datas[0])
        held = {}
        gs = capture(lambda: held.__setitem__("o", prob.call(prob.sub, prob.sub_state, d_sub)))
        if kept:
            d_base, gb = kept[1], kept[2]
        else:
            d_base = clone_inputs(datas[0])
            gb = capture(lambda: prob.call(prob.base, prob.base_state, d_base))
            if cache is not None and cache.get("inputs"):
                cache["cases"][i] = (datas, d_base, gb)

        def load(k):  # fresh input k into both sides' captured buffers (the outputs they wrote are inputs only here)
            for dst in (d_sub, d_base):
                for key, v in datas[k].items():
                    if isinstance(v, torch.Tensor):
                        dst[key].copy_(v)
            torch.cuda.synchronize()

        # the outputs of graph replays (the judge's execution mode) are checked too, before timing: 4 of the fresh inputs on a
        # screen, every one of them with the full checks (controls, confirmations, anything promoted or submitted: some
        # failures appear only under the graph, v6m 143)
        for k in range(min(4, len(datas)) if nocheck else len(datas)):
            load(k)
            gs.replay()
            torch.cuda.synchronize()
            ok, msg = prob.check(d_sub, held["o"], key=(ident, seed + k), clean=datas[k])
            if not ok:
                print(f"FAILED: sub graph-replay outputs fail the reference check on {case} (input {k}): {msg}", flush=True)
                if out_json:
                    pathlib.Path(out_json).write_text(json.dumps({"failed": f"graph replay: {msg}", "case": case}, indent=1))
                sys.exit(3)
        # TASK.md: the two sides take turns on the same input, the order swapped; a pair's ratio is the geometric mean
        # of its two orders, the shape's speedup the median over pairs
        ratios, ts, tb = [], [], []
        tc3 = time.time()
        with gpu_timing() as gt:
            for r in range(reps):
                load((r // 2) % len(datas))
                res = {}
                for g in ((gs, gb) if r % 2 == 0 else (gb, gs)):
                    res[id(g)] = time_graph(g, flush, 1)[0] - fixed
                ts.append(res[id(gs)])
                tb.append(res[id(gb)])
                ratios.append(res[id(gb)] / max(res[id(gs)], 1e-3))
        pairs = [math.sqrt(ratios[2 * j] * ratios[2 * j + 1]) for j in range(len(ratios) // 2)]
        sp = statistics.median(pairs)
        ratios = pairs
        mb = statistics.median(tb)
        q = sorted(ratios)
        lo, hi = q[int(0.25 * len(q))], q[int(0.75 * len(q)) - 1]
        print(f"{case}: {prob.base_label} {mb:8.1f} us  ours {mb / sp:8.1f} us  speedup {sp:.4f}x  (IQR {lo:.4f}-{hi:.4f})", flush=True)
        print(f"  wall s: inputs {tc1 - tc0:.1f}  checks {tc2 - tc1:.1f}  capture+replay checks {tc3 - tc2:.1f}  timed {gt.body:.1f}  "
              f"lock wait {gt.wait:.1f}", flush=True)   # (timed: inside the exclusive lock only; its waits apart)
        rows.append({"case": case, "base_us": mb, "ours_us": mb / sp, "speedup": sp, "iqr": [lo, hi], "checks": checks,
                     "pairs": [round(x, 5) for x in ratios]})
        load = case.get("load", f"case{i}")
        w = case.get("weight", 1)
        loads.setdefault(load, [0.0, 0.0])
        loads[load][0] += w * mb
        loads[load][1] += w * mb / sp
        del gs, gb, datas, d_sub, d_base, held
        torch.cuda.empty_cache()   # (cached shapes keep theirs in cache)
    sps = [b / s for b, s in loads.values()]
    geo = math.exp(sum(math.log(x) for x in sps) / len(sps))
    for (load, (b, s)), x in zip(loads.items(), sps):
        print(f"  load {load:10s} {prob.base_label} {b:10.1f} us  ours {s:10.1f} us  {x:.3f}x")
    print(f"geomean speedup {geo:.4f}x", flush=True)
    if out_json:
        import hashlib, socket
        ident = {"sub": str(prob.files[0]), "sub_md5": hashlib.md5(pathlib.Path(prob.files[0]).read_bytes()).hexdigest(),
                 "against": str(prob.files[1]), "against_md5": hashlib.md5(pathlib.Path(prob.files[1]).read_bytes()).hexdigest(),
                 "seed_block": block, "reps": reps, "fresh": fresh, "server": "benchd" if cache is not None else "process",
                 "host": socket.gethostname(),
                 "gpu": torch.cuda.get_device_name(), "cuda_visible": os.environ.get("CUDA_VISIBLE_DEVICES"),
                 "gpu_uuid": str(torch.cuda.get_device_properties(torch.cuda.current_device()).uuid),   # (the physical H200)
                 "t_start": round(t_start), "t_end": round(time.time()),
                 "eager_checks": 0 if nocheck else min(ncheck, fresh),
                 "replay_checks": min(4, fresh) if nocheck else fresh,
                 "prepare_s": round(getattr(prob, "prepare_s", 0), 1),   # the candidate's import + prepare (the judge caps it)
                 "peak_gb": round((torch.cuda.max_memory_allocated() - prob.rest) / 2**30, 2)}   # above the bench's start
        pathlib.Path(out_json).write_text(json.dumps({"geomean": geo, "cases": rows, "identity": ident}, indent=1))
    return geo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["test", "bench", "once"])
    ap.add_argument("problem", choices=list(DIRS))
    ap.add_argument("--cases", default="all")
    ap.add_argument("--case", default=None)
    ap.add_argument("--file", default=None)
    ap.add_argument("--against", default=None, help="bench against this submission instead of SGLang's kernel")
    ap.add_argument("--reps", type=int, default=40)
    ap.add_argument("--limit", type=int, default=0, help="only the first N cases")
    ap.add_argument("--json", default=None, help="bench: per-case results to this file")
    ap.add_argument("--fresh", type=int, default=20, help="bench: fresh inputs per shape (repetition r uses r % N)")
    args = ap.parse_args()
    pdir = ROOT / DIRS[args.problem]
    prob = Problem(args.problem, pathlib.Path(args.file) if args.file else pdir / "submission.py",
                   pathlib.Path(args.against) if args.against else None)
    if args.case:
        cases = [parse_case(args.case)]
    elif args.cases == "tests":
        cases = task_tests(pdir)
    elif args.cases == "scored":
        cases = scored_cases(pdir)
    elif args.cases == "nearby":
        cases = nearby_cases(pdir)
    else:
        cases = task_tests(pdir) + scored_cases(pdir)
    if args.limit:
        cases = cases[: args.limit]
    if args.mode == "test":
        sys.exit(0 if run_tests(prob, cases) else 1)
    if args.mode == "bench":
        run_bench(prob, scored_cases(pdir) if args.cases == "all" and not args.case else cases, args.reps, out_json=args.json, fresh=args.fresh)
    if args.mode == "once":  # the second call runs inside the NVTX range "okc_case" (ncu --nvtx-include okc_case/)
        data = prob.inputs(cases[0], 1)
        prob.call(prob.sub, prob.sub_state, data)
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_push("okc_case")
        prob.call(prob.sub, prob.sub_state, data)
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_pop()


if __name__ == "__main__":
    main()
