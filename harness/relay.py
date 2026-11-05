"""The build relay, on the PC: every GPU job's one flow (harness/submit.py). It holds the build queue and every job's state;
the boxes cannot reach the PC, so it makes every move:
  queued    submitted (submit.py appends "<name> <GPU host> <level>" to QUEUE and writes its GPU job line to JOBS);
  building  on a build box (box_build.sh: the whole machine, nice 19), CONC[box] builds at once;
  built     done there (the box keeps it in its cache);
  shipped   the build box copied it straight to its GPU box and queued its job there (box_ship.sh: every copy must
            succeed, else it is tried again), once that GPU has fewer than PREFETCH jobs waiting (level 0: at once); the
            build box keeps its copy until the GPU box shows the job (confirmed), then deletes it (box_gc.sh). A failed
            build (cpu_prebuild exit != 0) is never benched: its compiler error goes to the GPU box as the job's result
            (infeasible), through the driver's usual result path.
Builds start and ships go by level, then submission time (the GPU's runner takes its queue the same way). Each pass polls
every box, ships, and only then starts builds (one ssh each), so a finished build waits for the next poll, not for the
pass's build starts. A build box that goes away sends its jobs back to queued; a paused one (not ready) starts no build
but finishes and ships what it has; a GPU box that goes away leaves its unshipped jobs here (the driver re-asks the ones
that were shipped).
    python3 harness/relay.py          (runs until killed; the state is in STATE, the depths in DEPTHS)"""
import concurrent.futures, contextlib, fcntl, json, pathlib, subprocess, sys, threading, time

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "harness"))
import vast_hosts  # noqa: E402

D = vast_hosts.STATE / "relay"
QUEUE, JOBS, STATE, DEPTHS = D / "queue.txt", D / "jobs", D / "state.json", D / "depths.json"
PREFETCH = 5
ORPHAN_S = 1200   # a stopped spot GPU's jobs on its queue this recently (confirmed) go to a GPU still up (run twice at worst)
RECEIPTS = [(ROOT / "receipts/abo.jsonl", "name"), (ROOT / "receipts/control.jsonl", "name"),
            (ROOT / "receipts/evidence/confirm.jsonl", "run"), (ROOT / "receipts/evidence/lane_screens.jsonl", "name")]


def receipted():
    """the job names with a completion receipt here (a study result, a control, a confirmation's result or failure, a
    lane screen's reading)"""
    names = set()
    for f, k in RECEIPTS:
        if f.exists():
            for l in open(f):
                try:
                    d = json.loads(l)
                except ValueError:
                    continue
                if d.get("event", "result") in ("result", "failed", "read", "held", "reassembled") and d.get(k):
                    names.add(d[k])
    return names


LANE_EXTRA = 2   # builds of level 0-2 jobs (lanes, gates, confirmations) a box takes beyond CONC: they never wait out
# a box full of study trials
CONC = {}   # builds at once per box (owner): 6 on every box (>= 24 threads; compilers capped at threads, niced), the default
# below; a box's registry "conc" overrides it (gpu1 building for itself: 2; build1, 64 GB: 4, where build1 at 61 GB ran out at 4)
CPU = {}   # build box -> (time, its cgroup's CPU time used, us) at the last pass
BOXES = {}   # build box -> its CPU use (cores) since the last pass, quota and free memory: in DEPTHS


(D / "cm").mkdir(exist_ok=True)
SSH = vast_hosts.SSH + ["-o", "ControlMaster=auto", "-o", f"ControlPath={D}/cm/%C", "-o", "ControlPersist=600"]   # (a
# kept connection per box for the pass's short polls: a new ssh through Vast's proxies costs 4-14 s, and a pass makes
# several in turn, so passes took about a minute and queued trials waited that long for idle builders, 20:17-20:27 UTC
# Nov 4. Only the polls: commands that leave a process running (ships, builds) or run long (GC) held a kept
# connection's sessions open and stalled every command after them, so they keep their own)


def sh(port, ip, cmd, timeout=60, stdin=None, keep=False):
    r = subprocess.run((SSH if keep else vast_hosts.SSH) + ["-p", port, f"root@{ip}", cmd], input=stdin,
                       capture_output=True, text=True, timeout=timeout)
    return r.stdout


def start(d, n, g, tar):
    """the job's candidate and GPU job line (a tar stream) to build box d in one ssh, and its build started there"""
    own = f"OKC_COMPILE_SLOTS={d['slots']} OKC_BUILD_CPUS={d['build_cpus']} " if d.get("builds") else ""   # ("cpus" is the GPUs' list)
    return subprocess.run(vast_hosts.SSH + ["-p", d["port"], f"root@{d['ip']}",   # (its own connection: a build left
                                                                                 # running kept a kept one's session open)
                          f"i=/root/in_{n} && rm -rf $i && mkdir -p $i && tar -xf - -C $i && mv $i/candidates/layer_{n}.py /root/okc/candidates/ "
                          f"&& mv $i/jobs/{n}.txt /root/out/job_{n}.txt && rm -f /root/out/build_{n}.sh && {{ [ ! -f $i/jobs/{n}.build ] || "
                          f"mv $i/jobs/{n}.build /root/out/build_{n}.sh; }} && rm -rf /root/lanes/{n} && {{ [ ! -d $i/lanes/{n} ] || "
                          f"{{ mkdir -p /root/lanes && mv $i/lanes/{n} /root/lanes/{n}; }}; }} "
                          f"&& rm -rf $i /root/out/built_{n}.rc && ({own}setsid bash "
                          f"/root/okc/harness/h200/box_build.sh {n} {g} > /dev/null 2>&1 < /dev/null &)"],   # (only the build in the
                          # background: a backgrounded chain would read /dev/null, not the tar stream)
                          input=tar, capture_output=True, timeout=120).returncode == 0


def order(kv):
    return kv[1].get("level", 3), kv[1].get("t0", kv[1]["t"])


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


D.mkdir(parents=True, exist_ok=True)
JOBS.mkdir(exist_ok=True)
QUEUE.touch()

def write(path, obj):
    """a state file the driver reads unlocked: written to a temporary name, then renamed over it"""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj))
    tmp.replace(path)

state = json.loads(STATE.read_text()) if STATE.exists() else {}
_gc, GCX = None, concurrent.futures.ThreadPoolExecutor(8)
while True:
    t_pass = time.time()
    reg = vast_hosts.boxes()
    hosts, GPUS, EXTS, _, drain = vast_hosts.load()   # (one snapshot per pass: a box the guard flags mid-pass is gone next pass)
    builders = {b: d for b, d in reg.items() if (d.get("role") == "build" or d.get("builds")) and not d.get("offline")}
    done_cache = {}

    def done_for(b):
        """the jobs of box b with a completion receipt: here (receipted), or a .done among its outputs where they went"""
        if b not in done_cache:
            names, to = receipted(), reg.get(b, {}).get("moved_to")
            if to and f"{to}g0" in hosts:
                try:
                    names |= {f[4:-5] for f in sh(*hosts[f"{to}g0"][:2], "ls /root/out | grep '^abo_.*[.]done$'", timeout=120).split()}
                except Exception:  # noqa: BLE001  (unreachable this pass: nothing moves)
                    names = None
            done_cache[b] = names
        return done_cache[b] if done_cache[b] is not None else {n for n in state}
    for n, s in state.items():   # a spot GPU stopped (preempted, or by its controller, harness/h200/nebius.py): its jobs
        b = s["gpu"].rsplit("g", 1)[0]   # in flight go to the least busy GPU still up, rebuilt there and run again; with
        d = reg.get(b, {})               # none, they wait queued for it (the controller empties its queue files when it
        # comes back, so nothing runs twice). A draining spot GPU (the controller moving to a cheaper one) runs what it
        # has, and its queued jobs go to a GPU that takes work
        # (a job delivered to a stopped box goes too when it has no completion receipt: delivered within ORPHAN_S (it has
        # likely not run, and lane screens starve on a box that keeps being preempted), or, whatever its age, when the box
        # left for good (drained, or its outputs handed to "moved_to"): no candidate is lost; at worst one runs twice)
        on_box = (s["phase"] == "confirmed" and d.get("offline")
                  and (time.time() - s["t"] < ORPHAN_S or d.get("moved_to") or d.get("drain")) and n not in done_for(b))
        if d.get("spot") and (d.get("offline") and (s["phase"] in ("queued", "building", "built", "shipping", "shipped") or on_box)
                              or d.get("drain") and s["phase"] == "queued"):
            up = [h for h in hosts if h != s["gpu"] and h not in drain]
            if on_box and not up:
                continue
            if up:
                to = min(up, key=lambda h: sum(1 for v in state.values() if v["gpu"] == h and v["phase"] in ("queued", "building", "built", "shipping", "shipped")))
                log(f"{n}: {s['gpu']} {'went offline' if d.get('offline') else 'drains'} ({s['phase']}): requeued on {to}")
                s.update(gpu=to, phase="queued", box=None, t=time.time())
            elif s["phase"] != "queued":
                log(f"{n}: {s['gpu']} went offline ({s['phase']}): queued again for it")
                s.update(phase="queued", box=None, t=time.time())
    for b, d in reg.items():   # a spot GPU back after a stop ("back", harness/h200/nebius.py): the jobs it had (confirmed)
        h = f"{b}g0"           # and never finished died with its runner and its emptied queue: queued again
        lost = [n for n, s in state.items() if s["phase"] == "confirmed" and s["gpu"].rsplit("g", 1)[0] == b
                and s["t"] < d.get("back", 0) and s.get("back") != d["back"]] if d.get("spot") and h in hosts else []
        if lost:
            try:
                done = set(sh(*hosts[h][:2], "ls /root/out | grep '^abo_.*[.]done$'", timeout=120).split())
            except Exception:  # noqa: BLE001
                continue
            for n in lost:
                state[n]["back"] = d["back"]
                if f"abo_{n}.done" not in done:
                    log(f"{n}: lost on {state[n]['gpu']} by its stop: queued again")
                    state[n].update(phase="queued", box=None, t=time.time())
    # (a GPU box with "builds" builds for its own GPUs: conc, slots, cpus from the registry; no copy, no GC of its live caches;
    # a paused box, not ready, starts no build but finishes and ships what it has)
    # 1. new submissions
    with open(D / "queue.lock", "a") as lk:   # submit.py appends under the same lock: the new submissions are in the state
        fcntl.flock(lk, fcntl.LOCK_EX)         # file before the queue file is cleared (a relay stopped here loses none)
        lines = [l.split() for l in QUEUE.read_text().splitlines() if l.strip()]
        cancel = [l.strip() for l in (D / "cancel.txt").read_text().splitlines() if l.strip()] if (D / "cancel.txt").exists() else []
        for name, host, *level in lines:
            state.setdefault(name, {"gpu": host, "phase": "queued", "t": time.time(), "t0": time.time(), "level": int((level or [3])[0])})
        if lines:
            write(STATE, state)
        QUEUE.write_text("")
        (D / "cancel.txt").write_text("")
    for name in cancel:   # the driver voided its option: never built or shipped; a build in progress is killed
        s = state.get(name)
        if s and s["phase"] == "building" and s.get("box") in builders:
            with contextlib.suppress(Exception):
                bd = builders[s["box"]]
                # the build's whole process group (its compilers too), and only once it is gone may GC take its dirs
                sh(bd["port"], bd["ip"], f"p=$(cat /root/out/pgid_{name}.txt 2>/dev/null) && {{ kill -TERM -- -$p 2>/dev/null; "
                                         f"for i in $(seq 30); do kill -0 -- -$p 2>/dev/null || break; sleep 1; done; "
                                         f"kill -KILL -- -$p 2>/dev/null; }}; true", timeout=90)
        if s:
            s.update(phase="cancelled", t=time.time())
            log(f"{name} cancelled (voided option)")
    # 2. build boxes: finished builds and ships, and each one's free build slots (filled in 6)
    free, polls, refused = {}, {}, set()
    for b, d in builders.items():
        mine = [n for n, s in state.items() if s.get("box") == b and s["phase"] == "building"]
        going = [n for n, s in state.items() if s.get("box") == b and s["phase"] == "shipping"]
        # the CPU quota, the CPU time used so far (us) and free memory (GB), cgroup v2 or v1
        polls[b] = (
                     "if [ -f /sys/fs/cgroup/cpu.max ]; then q=$(cat /sys/fs/cgroup/cpu.max); "
                     "u=$(awk '/usage_usec/ {print $2}' /sys/fs/cgroup/cpu.stat); "
                     "f=$(( ( $(cat /sys/fs/cgroup/memory.max | sed s/max/999999999999999/) - $(cat /sys/fs/cgroup/memory.current) ) >> 30 )); "
                     "else q=\"$(cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us) $(cat /sys/fs/cgroup/cpu/cpu.cfs_period_us)\"; "
                     "u=$(( $(cat /sys/fs/cgroup/cpuacct/cpuacct.usage) / 1000 )); "
                     "f=$(( ( $(cat /sys/fs/cgroup/memory/memory.limit_in_bytes) - $(cat /sys/fs/cgroup/memory/memory.usage_in_bytes) ) >> 30 )); fi; "
                     "a=${q%% *}; b=${q##* }; [ \"$a\" != max ] && [ \"$a\" -gt 0 ] 2>/dev/null && echo $(( a / b )) || nproc; "
                     "echo $u; m=$(awk '/MemAvailable/ {print int($2 / 1048576)}' /proc/meminfo); echo $(( f < m ? f : m )); "
                     + " ".join(f"[ -f /root/out/built_{n}.rc ] && echo {n} $(cat /root/out/built_{n}.rc);" for n in mine)
                     # (a build gone without its exit status, after two minutes: lost, built again. Its start never
                     # reached the box, or a GC with an older keep list took its files: stuck "building" for hours)
                     + " ".join(f"[ -f /root/out/built_{n}.rc ] || {{ p=$(cat /root/out/pgid_{n}.txt 2>/dev/null); "
                                f"[ -n \"$p\" ] && kill -0 -- -$p 2>/dev/null; }} || echo {n} lost;"
                                for n in mine if time.time() - state[n]["t"] > 120)
                     + " ".join(f"grep -q 'shipped {n}' /root/out/ship_{n}.out 2>/dev/null && echo {n} shipped;"
                                f"{{ p=$(cat /root/out/shippid_{n}.txt 2>/dev/null); [ -n \"$p\" ] && kill -0 $p 2>/dev/null; }} || "   # (its own PID: a pattern
                                f"grep -q 'shipped {n}' /root/out/ship_{n}.out 2>/dev/null || {{ if grep -q 'no job line' "   # matched this
                                f"/root/out/ship_{n}.out 2>/dev/null; then echo {n} lost; elif grep -q 'Permission denied' "   # shell too)
                                f"/root/out/ship_{n}.out 2>/dev/null; then echo {n} denied; else echo {n} unshipped; fi; }};"
                                for n in going) + " true")
    with concurrent.futures.ThreadPoolExecutor(max(1, len(polls))) as ex:   # (every box at once: a pass waits for the slowest)
        futs = {b: ex.submit(sh, builders[b]["port"], builders[b]["ip"], c, keep=True) for b, c in polls.items()}
    for b, d in builders.items():
        try:
            out = futs[b].result()
        except Exception:  # noqa: BLE001
            continue
        o = out.split("\n")
        if len(o) < 3:
            continue
        quota, used_us, free_gb = int(o[0]), int(o[1]), int(o[2])
        last = CPU.get(b)   # (the box's CPU use since the last pass, logged)
        CPU[b] = (time.time(), used_us)
        busy = (used_us - last[1]) / 1e6 / (time.time() - last[0]) if last and time.time() > last[0] else 0.0
        starts = (CONC.get(b, d.get("conc", 6)) - sum(1 for x in state.values() if x.get("box") == b and x["phase"] == "building")
                  if d.get("ready") else 0)
        BOXES[b] = {"cpu": round(busy, 1), "quota": quota, "free_gb": free_gb}
        for l in o[3:]:
            if not l.strip():
                continue
            n, what = l.split()
            if what == "shipped":
                state[n].update(phase="shipped", t=time.time())
                log(f"{n} shipped from {b} to {state[n]['gpu']}")
            elif what == "lost":   # its job line or its build is gone from the build box: built again from the relay's copy
                # (a missing shippid file once read as PID 0, "kill -0 0", always alive: shipping stuck for hours)
                state[n].update(phase="queued", box=None, t=time.time())
                log(f"{n}: its job line was lost on {b}: queued to build again")
            elif what in ("unshipped", "denied"):   # the copy failed: ship it again (refused: the box's key reinstalled first)
                state[n].update(phase="built", t=time.time())
                if what == "denied":
                    refused.add((b, state[n]["gpu"]))
            else:
                state[n].update(phase="built", rc=int(what), t=time.time())
        free[b] = starts
    for b, h in refused:   # the GPU box refused the build box's transfer key: Vast rewrites authorized_keys to the
        with contextlib.suppress(Exception):   # account's keys when they change (02:29 and 16:09 UTC Nov 4). Only our own
            key = sh(builders[b]["port"], builders[b]["ip"], "cat /root/.ssh/xfer.pub").strip()   # build box's key is
            if key.startswith("ssh-") and "'" not in key and "\n" not in key:                    # appended, if absent;
                sh(*hosts[h][:2], f"cd /root/.ssh && grep -qxF '{key}' authorized_keys || {{ "     # the file is never
                                  f"[ -z \"$(tail -c1 authorized_keys)\" ] || echo >> authorized_keys; "  # rewritten, so no
                                  f"echo '{key}' >> authorized_keys; }}")                          # key is ever dropped
                log(f"{b}: its transfer key was refused by {h}: installed there again")
    # 3. a build box gone: its trials back to the queue
    for n, s in state.items():   # (paused, not ready, is not gone; a ship in transit from it never lands)
        if s["phase"] in ("building", "built", "shipping") and s.get("box") not in builders:
            s.update(phase="queued", box=None, t=time.time())
            log(f"{n}: build box {s.get('box')} gone, queued again")
    # 4. ship built jobs to their GPU while it has fewer than PREFETCH jobs waiting (level 0 at once)
    with concurrent.futures.ThreadPoolExecutor(max(1, len(hosts))) as ex:
        counts = {h: ex.submit(sh, port, ip, f"grep -o '^run p[0-9]' {qf} | cut -c6; true", keep=True) for h, (port, ip, qf, _) in hosts.items()}
    waiting, levels = {}, {}   # jobs waiting on each GPU, and their levels (a job counts only those at its level or above it)
    for h, f in counts.items():
        with contextlib.suppress(Exception):
            levels[h] = [int(x) for x in f.result().split() if x.isdigit()]
            waiting[h] = len(levels[h])
    ships = []
    for n, s in sorted(state.items(), key=order):
        if s["phase"] != "built" or s["box"] not in builders or s["gpu"] not in waiting:
            continue
        h = s["gpu"]
        lv = s.get("level", 3)   # (a lane job or confirmation never waits behind study trials it outranks)
        qn = lambda x: sum(1 for y in levels.get(x, []) if y <= lv)   # (jobs waiting there at its level or above)
        if n.startswith("arch5_c1_") and "_ctl_" not in n:   # a study trial goes to whichever GPU of its box frees first (its
            # modules are in the box's shared tvm-ffi cache, not a GPU's extension dir), not the one it was built for (# 56 built for g0 while g1's queue sat empty)
            h2 = min((x for x in waiting if x[:x.rindex("g")] == h[:h.rindex("g")]), key=lambda x: (qn(x), x != h))
            if h2 != h:   # (box_ship queues it there, its job line's extension dir rewritten: its 5th argument)
                s.setdefault("built_for", h)
                s["gpu"] = h = h2
        if lv > 0 and qn(h) >= PREFETCH:
            continue
        port, ip = hosts[h][:2]
        bd, g = builders[s["box"]], GPUS[h]
        if s.get("rc"):   # the build failed (the candidate's): its result, never a bench (it would compile on the GPU)
            (D / "fail").mkdir(exist_ok=True)   # its whole build log, kept on the PC (the build box deletes it)
            subprocess.run(["scp"] + vast_hosts.SSH[1:] + ["-P", bd["port"], "-q", f"root@{bd['ip']}:/root/out/prebuild_{n}.log",
                            str(D / "fail" / f"{n}.log")], timeout=120)
            tail = sh(bd["port"], bd["ip"], f"grep -E '^FAILED' /root/out/prebuild_{n}.log | cut -c1-120; "
                                             f"grep -m6 -E 'error:|fatal|No such file|Killed|signal|ninja: error' /root/out/prebuild_{n}.log | cut -c1-300")
            sh(port, ip, f"cat > /root/out/abo_{n}.log && touch /root/out/abo_{n}.done",
               stdin=(f"errors detected in the compilation (prebuild exit {s['rc']} on {s['box']}):\n{tail}\n" if s["rc"] == 2 else
                      f"build did not finish (prebuild exit {s['rc']} on {s['box']}: timed out or killed, ours):\n{tail}\n"))
            # (cpu_prebuild exits 2 on a failed build; 124 is box_build's timeout, 137 / 143 a kill)
            s.update(phase="shipped", t=time.time())
            log(f"{n}: build failed (exit {s['rc']}), reported as its result")
            continue
        # in the background on the build box (ships run side by side; their starts too, below)
        to = "local" if (bd["port"], bd["ip"]) == (port, ip) else ip   # (built on the GPU box itself)
        gb = GPUS.get(s.get("built_for", h), g)   # (the GPU its build's extension dir is for; queued on g)
        ships.append((n, bd, f"rm -f /root/out/ship_{n}.out; setsid bash /root/okc/harness/h200/box_ship.sh {n} {gb} {port} {to} {g} "
                             f"> /root/out/ship_{n}.out 2>&1 < /dev/null & echo $! > /root/out/shippid_{n}.txt"))
        s.update(phase="shipping", t=time.time())
        waiting[h] += 1
        levels.setdefault(h, []).append(lv)
    with concurrent.futures.ThreadPoolExecutor(max(1, len(ships))) as ex:   # (a box slow to answer: on a later pass)
        for (n, _, _), f in [(x, ex.submit(sh, x[1]["port"], x[1]["ip"], x[2])) for x in ships]:
            try:
                f.result()
            except (subprocess.SubprocessError, OSError):
                state[n].update(phase="built", t=time.time())
    depths = {h: {"ready_on_gpu": w,
                  "built": sum(1 for s in state.values() if s["gpu"] == h and s["phase"] == "built"),
                  "building": sum(1 for s in state.values() if s["gpu"] == h and s["phase"] == "building"),
                  "queued": sum(1 for s in state.values() if s["gpu"] == h and s["phase"] == "queued")} for h, w in waiting.items()}
    # 5. confirmed: the GPU box has the job queued, running or done; then, each minute, each build box keeps only what its
    # unconfirmed trials need (box_gc.sh: nothing stays on a build box once its GPU box has it)
    for h, (port, ip, qf, _) in hosts.items():
        ship = [n for n, s in state.items() if s["gpu"] == h and s["phase"] == "shipped"]
        if not ship:
            continue
        try:
            have = set(sh(port, ip, "; ".join(f"grep -q 'abo_{n}[.]' {qf} || [ -f /root/out/abo_{n}.done ] || pgrep -f 'layer_{n}[.]py' "
                                             f"> /dev/null && echo {n}" for n in ship) + "; true", keep=True).split())
        except Exception:  # noqa: BLE001
            continue
        for n in have:
            state[n].update(phase="confirmed", t=time.time())
    if time.time() - globals().get("_gc_t", 0) > 60 and not (_gc and _gc.is_alive()):   # (in the background: a pass never
        _gc_t = time.time()                                                               # waits on a box's du or GC)
        gc = {b: [n for n, s in state.items() if s.get("box") == b and s["phase"] in ("building", "built", "shipping", "shipped")]
              for b, d in builders.items() if not d.get("builds")}   # (a GPU box's ext dirs are its benches' live caches)
        _gc = threading.Thread(target=lambda jobs: [f.result() for f in jobs if not f.exception()], daemon=True, args=(
            [GCX.submit(sh, builders[b]["port"], builders[b]["ip"], "bash /root/okc/harness/h200/box_gc.sh " + " ".join(keep), 120)
             for b, keep in gc.items()] + [GCX.submit(sh, port, ip, "bash /root/okc/harness/h200/gpu_gc.sh", 600)
                                           for port, ip in {(p, i) for p, i, *_ in hosts.values()}],))
        _gc.start()
    # 6. new builds in the boxes' free slots, by level, then submission time. A study trial (arch5_*, not a control) is
    # built for whichever GPU of its box has the least work ahead of it (waiting, built, building), chosen now, not at
    # dispatch: it builds into that GPU's extension dir (a copied dir recompiles: torch rewrites its build.ninja paths);
    # controls, confirmations and lane jobs stay on the GPU they were submitted for (their readings are that GPU's)
    ahead = {h: waiting.get(h, 0) + sum(1 for x in state.values() if x["gpu"] == h and x["phase"] in ("built", "building"))
             for h in hosts}
    starts = []
    for n, s in sorted(state.items(), key=order):
        if s["phase"] != "queued" or s["gpu"] not in hosts:
            continue
        lv = s.get("level", 3)
        own = s["gpu"].rsplit("g", 1)[0]   # (a GPU box that builds builds its own jobs, and only those: a build box elsewhere
        # ships to it slowly (build1 -> Japan), and it ships to others slowly (gpu4 built spot1's controls, 03:37 UTC))
        mine = [x for x in free if (x == own) if builders[x].get("builds")] if reg.get(own, {}).get("builds") else \
               [x for x in free if not builders[x].get("builds")]
        b = max(mine, key=free.get, default=None)
        if b is None or free[b] <= (-LANE_EXTRA if lv <= 2 else 0):   # (a lane, gate or confirmation job, level 0-2,
            continue   # starts over a box's study builds: up to LANE_EXTRA more at once; a job whose own builder is
            #            full waits, the next one may build elsewhere: gpu4's full slots stopped build1 building spot1's, 04:40)
        lane = "env -u OKC_GPU_LOCK" in (JOBS / f"{n}.txt").read_text() if (JOBS / f"{n}.txt").exists() else False
        if lane:   # a lane run job (it times nothing against controls): the GPU of its box with less level-0/1 work ahead
            urgent = {x: sum(1 for lv in levels.get(x, []) if lv <= 1) + sum(
                1 for y in state.values() if y["gpu"] == x and y.get("level", 3) <= 1 and y["phase"] in ("built", "building", "shipping"))
                for x in hosts if x[:x.rindex("g")] == s["gpu"][:s["gpu"].rindex("g")]}
            h = min(urgent, key=lambda x: (urgent[x], x != s["gpu"]))
            if h != s["gpu"]:
                log(f"{n}: lane job asked for {s['gpu']}, routed to {h} (level-0/1 work ahead {urgent})")
                s["gpu"] = h
        elif n.startswith("arch5_") and "_ctl_" not in n:   # (study trials: least work ahead; controls and screens stay)
            h = min((x for x in ahead if x[:x.rindex("g")] == s["gpu"][:s["gpu"].rindex("g")]), key=ahead.get)
            if h != s["gpu"]:   # (its job line names the GPU's extension dir)
                job = JOBS / f"{n}.txt"
                job.write_text(job.read_text().replace(f"TORCH_EXTENSIONS_DIR={EXTS[s['gpu']]} ",
                                                       f"TORCH_EXTENSIONS_DIR={EXTS[h]} "))
                log(f"{n}: dispatched for {s['gpu']}, built for {h}")
                s["gpu"] = h
            ahead[h] += 1
        extra = [x for x in (f"jobs/{n}.build", f"lanes/{n}") if (D / x).exists()]   # (a lane job's build step and files)
        tar = subprocess.run(["tar", "-cf", "-", "-C", str(ROOT), f"candidates/layer_{n}.py", "-C", str(D), f"jobs/{n}.txt", *extra],
                             capture_output=True, timeout=60)
        if tar.returncode:   # (submitted without its candidate or job line: never buildable)
            s.update(phase="cancelled", t=time.time())
            log(f"{n} cancelled: {tar.stderr.decode().strip()[-200:]}")
            continue
        starts.append((n, b, tar.stdout))
        s.update(phase="building", box=b, t=time.time())
        free[b] -= 1
    with concurrent.futures.ThreadPoolExecutor(max(1, len(starts))) as ex:   # (every start at once: one ssh each)
        futs = [(n, b, ex.submit(start, builders[b], n, GPUS[state[n]["gpu"]], t)) for n, b, t in starts]
    for n, b, f in futs:
        try:
            ok = f.result()
        except (subprocess.SubprocessError, OSError):
            ok = False
        if ok:
            log(f"{n} building on {b} for {state[n]['gpu']} (level {state[n].get('level', 3)})")
        else:   # (a box slow to answer: its build starts on a later pass)
            state[n].update(phase="queued", box=None, t=time.time())
    # 7. forget confirmed jobs after a day; write the state and the depths
    state = {n: s for n, s in state.items() if not (s["phase"] in ("confirmed", "cancelled") and time.time() - s["t"] > 86400
                                                    and not reg.get(s["gpu"].rsplit("g", 1)[0], {}).get("offline"))}
    write(STATE, state)
    write(DEPTHS, {"t": time.time(), **depths, "boxes": BOXES, "pass_s": round(time.time() - t_pass, 1)})
    time.sleep(1)
