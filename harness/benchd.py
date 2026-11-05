"""Persistent study bench, one per GPU runner slot. The reference, the weights and production's prepared state (and the
flush buffer and fixed replay cost) are made once per process; each request then benches one candidate exactly as
`okc_harness.py bench layer --against reviewed/muse_decode_layer.py` does (the same inputs, checks, graph-replay
checks, judge-style paired timing, log and json) without a process start, torch and CUDA init, the reference's weights
or production's import and prepare. The shapes' inputs are made per request as in a fresh bench: kept, they hold
~100 GB, too much beside the other runner. The checks' reference outputs are kept, per input (Problem.memo). The
server exits after MAX_RUNS requests or IDLE_S idle seconds; the next request starts a fresh one. A request's time
limit counts from the moment the server takes it: the server's own start (torch, the reference, production) is not the
candidate's time.
    python3 harness/benchd.py <socket> candidates/layer_<name>.py <out.json> <reps> <log> <limit s>   (a request)"""
import contextlib, gc, json, os, pathlib, signal, socket, subprocess, sys, time

ROOT = pathlib.Path(__file__).resolve().parent.parent
MAX_RUNS, IDLE_S, START_S = 25, 1800, 600
AGAINST = ROOT / "reviewed/muse_decode_layer.py"


def serve(path):
    # the GPU lock is held (shared) per request and for the start only: an idle server holding it would block every other bench's
    # exclusive timing on this GPU (one box's second GPU stalled 16 min on that)
    lockpath = os.environ.pop("OKC_GPU_LOCK", None)
    import fcntl
    import torch
    import okc_harness as H
    pathlib.Path(path + ".pid").write_text(str(os.getpid()))   # the client kills this server when its request times out

    @contextlib.contextmanager
    def gpu_shared():
        if lockpath:
            H._GPU_LOCK = open(lockpath, "a")
            fcntl.flock(H._GPU_LOCK, fcntl.LOCK_SH)
            H._MODE = "SH"
        try:
            yield
        finally:
            if H._GPU_LOCK:
                H._GPU_LOCK.close()   # (releases the shared lock)
                H._GPU_LOCK, H._MODE = None, None

    with gpu_shared():   # production, the reference and the weights, once, before the socket exists (a request waits)
        keep = H.Problem("layer", None, AGAINST)
        keep.memo = {}
    srv = socket.socket(socket.AF_UNIX)
    srv.bind(path)
    srv.listen(1)
    srv.settimeout(IDLE_S)
    cache, runs = {}, 0
    code = {f: os.path.getmtime(f) for f in (H.__file__, __file__)}   # (the harness it runs: a deployed change takes
    while runs < MAX_RUNS:                                                # effect at the next request, a fresh server)
        try:
            conn, _ = srv.accept()
        except socket.timeout:
            break
        with conn:
            req = json.loads(conn.makefile().readline())
            log = open(req["log"], "a", buffering=1)   # written as it goes: a killed request leaves its log
            rc, dead = 0, False
            with gpu_shared(), contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                try:
                    os.environ["OKC_BUILD_LOG"] = req["build_log"]   # (the harness's build hook reads it per build)
                    os.environ["OKC_BENCH_NOCHECK"] = "1" if req.get("nocheck") else "0"   # (screens: the replay checks only)
                    prob = H.Problem("layer", pathlib.Path(req["file"]), None, keep=keep, tag=f"_{runs}")
                    H.run_bench(prob, H.scored_cases(ROOT / H.DIRS["layer"]), req["reps"], out_json=req["json"],
                                fresh=20, cache=cache)
                except SystemExit as e:   # a failed check (exit 3), as the bench's own process would end
                    rc = e.code if isinstance(e.code, int) else 1
                except Exception as e:  # noqa: BLE001
                    import traceback
                    traceback.print_exc()
                    rc = 1
                    dead = "CUDA" in str(e) or "cuda" in type(e).__module__ or "AcceleratorError" in type(e).__name__
                finally:
                    prob = None
                    sys.modules.pop(f"sub_layer_{runs}", None)
                    gc.collect()
                    try:   # a sticky CUDA error (illegal address, device assert) shows here if not before
                        torch.cuda.synchronize()
                        torch.cuda.empty_cache()
                    except Exception:  # noqa: BLE001
                        dead = True
            runs += 1
            print(f"benchd: reference memo {len(keep.memo)} entries, {nbytes(keep.memo) / 2**30:.1f} GB (host)", file=log)
            log.close()
            conn.sendall((json.dumps({"rc": rc}) + "\n").encode())
            if dead or any(os.path.getmtime(f) != t for f, t in code.items()):   # (the context is gone, or the harness
                break                                                              # changed: the next request, a fresh one)
    srv.close()
    os.unlink(path)


def nbytes(x):
    if hasattr(x, "element_size"):
        return x.numel() * x.element_size()
    if isinstance(x, dict):
        return sum(nbytes(v) for v in x.values())
    return sum(nbytes(v) for v in x) if isinstance(x, tuple) else 0


def request(path, file, out_json, reps, log, limit):
    """bench one candidate on the slot's server (started if there is none); its log goes to <log> as it runs. At <limit>
    seconds from the server taking it, or killed (the runner's backstop), the request kills the server with it (a stack
    dump first, in the server's log): a hung candidate must not hang the slot."""
    def kill_server(*_):
        with contextlib.suppress(OSError, ValueError):
            pid = int(pathlib.Path(path + ".pid").read_text())
            os.kill(pid, signal.SIGUSR1)   # (okc_harness's faulthandler)
            time.sleep(1)
            os.kill(pid, 9)
        with contextlib.suppress(OSError):
            os.unlink(path)
        os._exit(124)
    signal.signal(signal.SIGTERM, kill_server)
    signal.signal(signal.SIGALRM, kill_server)
    t0, started = time.time(), False
    while True:
        try:
            c = socket.socket(socket.AF_UNIX)
            c.connect(path)
            break
        except OSError as e:
            if isinstance(e, ConnectionRefusedError):   # a dead server's socket
                os.unlink(path)
            if not os.path.exists(path) and not started:   # no server for this slot: start one (it holds the GPU lock as any bench)
                subprocess.Popen([sys.executable, __file__, "--serve", path], start_new_session=True,
                                 stdout=open(path + ".log", "a"), stderr=subprocess.STDOUT)
                started = True
            if time.time() - t0 > START_S:
                sys.exit(f"benchd: no server at {path} after {START_S} s")
            time.sleep(2)
    signal.alarm(limit)
    with c:
        c.sendall((json.dumps({"file": file, "json": out_json, "reps": reps, "log": log,
                               "nocheck": os.environ.get("OKC_BENCH_NOCHECK") == "1",
                               "build_log": os.environ.get("OKC_BUILD_LOG", os.devnull)}) + "\n").encode())
        line = c.makefile().readline()
    if not line:   # the server died mid-request (a crash): the slot starts a fresh one next time
        with contextlib.suppress(OSError):
            os.unlink(path)
        sys.exit(1)
    sys.exit(json.loads(line)["rc"])


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT / "harness"))
    if sys.argv[1] == "--serve":
        serve(sys.argv[2])
    else:
        request(sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5], int(sys.argv[6]))
