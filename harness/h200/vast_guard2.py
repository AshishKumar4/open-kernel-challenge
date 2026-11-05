"""vast.ai guard and H200 SXM watch for the okc- instances (every instance whose label starts with okc-).
Every 30 s (backing off while Vast rate-limits) it reads the instances and the account's credit; every 2 minutes it copies
each instance's /root/out into receipts/vast/<label>/ and reports the sweep's demand (see demand(); information only).
The watch (plan) searches the H200 SXM offers every pass and logs each in-limit one when first seen. It adds a GPU
only while the GPU-job backlog is real (more than the running GPUs finished in the last hour): the cheapest per GPU (verified or not) within
the owner's limits and run-rate cap, on demand or at the holding bid, booked at once, with harness/h200/onboard.py run
on it. A booking counts once onboarding proved its GPUs (a matmul on each; the registry marks it proven); one not proven
in 20 minutes is destroyed. A candidate's result never drains or retires a box. A box beyond the base one drains when
the backlog stays near zero and is retired once idle. Each decision is logged.

Below the credit floor it retires every instance. Retiring one instance:
- stop its writers (queue runners, jobs, compile-ahead builds, and the GPU telemetry, whose csv would otherwise change
  under the dry run forever) and verify they are gone;
- empty its queues;
- export /root/out, requiring rsync rc 0;
- verify with a dry run that nothing is left to transfer;
- only then destroy it, confirmed gone from `show instances`.
It fails closed: an instance whose export is not verified is kept and retried every 60 s ("EXPORT UNVERIFIED, instance
kept" in the log).
    python3 vast_guard2.py <floor_usd>           the guard and the watch
    python3 vast_guard2.py retire <instance id>  retire one instance now"""
import json, os, pathlib, re, sqlite3, subprocess, sys, time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import vast_hosts  # noqa: E402

V = "vastai"   # the vast.ai CLI, logged in
SSH = vast_hosts.SSH[1:]
BANNER = ("Welcome to vast.ai", "Have fun!")   # printed on stderr by every vast ssh session
WRITERS = "[v]qg?[.]sh|[t]imeout 1[25]00|[s]wapbench|[o]kc_harness|[h]arness/h200|[a]stra_|[n]vidia-smi --query"   # bracketed: no self-match
JOBS = "[o]kc_harness|[s]wapbench|[e]atstp|[c]pu_prebuild"   # a box with none of these and empty queues is idle
IMAGE = "lmsysorg/sglang:dev-muse-spark-open"
# some hosts start the container with modes sshd refuses ("bad ownership or modes for file /root/.ssh/authorized_keys",
# on one Romania host): the onstart logs them and re-applies root ownership and strict modes for 15 minutes
ONSTART = ("(ls -lan / /root /root/.ssh > /proc/1/fd/1 2>&1; for i in $(seq 180); do chown root:root / /root /root/.ssh "
           "/root/.ssh/authorized_keys; chmod 755 /; chmod 700 /root /root/.ssh; chmod 600 /root/.ssh/authorized_keys; sleep 5; "
           "done) > /dev/null 2>&1 &")
OKC = str(pathlib.Path(__file__).resolve().parent.parent.parent)
STATE = f"{OKC}/receipts/vast/watch_state.json"   # bookings, arrivals seen, the backlog's low mark
ONBOARD_LOG = str(vast_hosts.STATE / "onboard.log")

# The owner's rules, H200 SXM only (Vast's "H200"; the NVL is listed as "H200 NVL"), ranked by price per GPU only:
CAP = 10.0                       # the okc- run rate, $/h, at most (the owner: $8-10/h; no minimum count)
MAX_GPUS = 3                     # running GPUs, at most
BASE = "spot1"                     # not drained on a low backlog (the owner: stay at gpu1's 2 GPUs for now)
GROW_H = 1.0                     # grow while the GPU-job backlog (queued + running + prebuilding + enqueued, every box)
                                 # exceeds this many hours of the last hour's completions
LOW_H, LOW_S = 0.25, 3600        # drain a box beyond BASE once the backlog stays under LOW_H hours of work for LOW_S
LIMIT = {1: (3.50, 2.50), 2: (6.20, 5.00)}   # GPUs -> (on-demand $/h below, bid $/h below); larger boxes need approval
HOLD_UNDER = 0.10                # a bid that holds: this far per GPU under its limit ($2.40 / GPU) ...
BID_ROOM = 0.50                  # ... placed only where the market's minimum bid is at least this far under it
STORAGE = 0.11                   # $/h of a new instance's 120 GB disk (offers are priced with the default disk)
READY_S = 1200                   # a booking proven (its first study trial measured) within this, or destroyed


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def ssh(port, ip, cmd):
    return subprocess.run(["ssh", *SSH, "-p", port, f"root@{ip}", cmd], capture_output=True, text=True, timeout=120)


def noise(err):
    return "\n".join(l for l in err.splitlines() if l.strip() and not l.startswith(BANNER))


def rsync(port, ip, dst, *extra):
    return subprocess.run(["rsync", "-a", *extra, "--timeout=60", "-e", "ssh " + " ".join(SSH) + f" -p {port}",
                           f"root@{ip}:/root/out/", dst], capture_output=True, text=True, timeout=600)


def instances():
    """{id: dict(port, ip, label, gpus, cost ($/h), base (GPU $/h), bid, running, machine)} of every okc- instance (port
    None until its ssh is mapped)"""
    r = subprocess.run([V, "show", "instances", "--raw"], capture_output=True, text=True, timeout=60)
    return {str(i["id"]): dict(port=((i.get("ports") or {}).get("22/tcp") or [{}])[0].get("HostPort"), ip=i.get("public_ipaddr"),
                               label=i["label"][4:], gpus=i["num_gpus"], cost=i.get("dph_total") or 0.0,
                               base=i.get("dph_base") or 0.0, bid=bool(i.get("is_bid")), running=i.get("actual_status") == "running",
                               machine=i.get("machine_id"))
            for i in json.loads(r.stdout) if (i.get("label") or "").startswith("okc-")}


def stop_writers(port, ip):
    """raises unless the runner and every job are verified gone and every queue is empty"""
    for pat in WRITERS.split("|"):
        ssh(port, ip, f"pkill -f '{pat}'")   # rc not trusted (1 = nothing to kill): verified below
    ssh(port, ip, "for q in /root/queue*.txt; do : > $q; done")
    time.sleep(5)
    r = ssh(port, ip, f"pgrep -f '{WRITERS}'")
    if r.returncode != 1 or r.stdout.strip() or noise(r.stderr):
        raise RuntimeError(f"writers not verified stopped: rc={r.returncode} {(r.stdout + noise(r.stderr)).strip()[:150]}")
    r = ssh(port, ip, "! find /root -maxdepth 1 -name 'queue*.txt' -size +0 | grep -q .")
    if r.returncode != 0 or noise(r.stderr):
        raise RuntimeError(f"queue not verified empty: rc={r.returncode} {noise(r.stderr)[:150]}")


def final_export(port, ip, dst):
    """raises unless /root/out copied with rc 0 and a dry run then lists nothing left to transfer"""
    ssh(port, ip, "mkdir -p /root/out")   # a box that never ran a job
    r = rsync(port, ip, dst)
    if r.returncode != 0:
        raise RuntimeError(f"rsync rc={r.returncode} {noise(r.stderr)[:200]}")
    r = rsync(port, ip, dst, "-n", "--itemize-changes")
    left = [l for l in r.stdout.splitlines() if l.strip()]
    if r.returncode != 0 or left:
        raise RuntimeError(f"verify rc={r.returncode}, {len(left)} items left: {left[:3]}")


def exported(port, ip, label):
    """one attempt: writers stopped and the export verified"""
    dst = f"{OKC}/receipts/vast/{label}/"
    os.makedirs(dst, exist_ok=True)
    try:
        stop_writers(port, ip)
        final_export(port, ip, dst)
    except Exception as e:  # noqa: BLE001
        log(f"{label} EXPORT UNVERIFIED, instance kept: {e}"[:300])
        return False
    log(f"{label} export verified ({dst})")
    return True


def destroy(iid, label):
    """destroy, retried until `show instances` no longer lists it"""
    while True:
        try:   # a hung or failed destroy retries like any other failure
            subprocess.run([V, "destroy", "instance", iid, "-y"], capture_output=True, timeout=120)
        except Exception as e:  # noqa: BLE001
            log(f"{label} destroy failed: {e!r}"[:160])
        time.sleep(20)
        try:
            if iid not in instances():
                log(f"{label} destroyed (confirmed)")
                return
        except Exception as e:  # noqa: BLE001
            log(f"{label} show instances failed: {e!r}"[:160])
        log(f"{label} still listed: retrying destroy")


def ran_work(label):
    """has this box ever run a job: any synced output besides its telemetry, or a job line in a runner log"""
    d = f"{OKC}/receipts/vast/{label}"
    if not os.path.isdir(d):
        return False
    for f in os.listdir(d):
        if f.startswith("vq") and f.endswith(".log"):
            if "== gpu" in open(os.path.join(d, f)).read():
                return True
        elif f not in ("util.csv",) and not f.startswith("runners"):
            return True
    return False


def hold_bid(n):
    return round(n * (LIMIT[n][1] / n - HOLD_UNDER), 2)


def vast(*args):
    """a Vast CLI call's parsed json; RateLimited when Vast throttles the API"""
    r = subprocess.run([V, *args, "--raw"], capture_output=True, text=True, timeout=60)
    if "429" in r.stdout[:300] + r.stderr[:300] or "Too Many Requests" in r.stdout[:300] + r.stderr:
        raise RateLimited(r.stderr[:120])
    return json.loads(r.stdout)


class RateLimited(Exception):
    pass


def demand(hosts, state, now):
    """(completed, backlog, ideas), GPU jobs alike (study trials and lane jobs):
    completed: jobs the GPU runners finished in the last hour (their exit lines in /root/out/vq*.log, every ready box);
    backlog: jobs not yet finished: queued on every ready box (abo_ included), running there, study trials in flight but
    not yet queued (compile-ahead) and enqueued study trials (Optuna WAITING) not yet dispatched;
    ideas: arrivals in the last hour of new dimensions and enqueued candidates (information, not a scaling input)."""
    done = queued = running = abo_seen = 0
    for h in hosts.values():
        if not (h.get("ready") and h["running"] and h["port"]):
            continue
        r = ssh(h["port"], h["ip"], "date -u +%s; cat /root/queue*.txt 2>/dev/null | grep -c .; "
                                    "pgrep -af 'timeout 1[25]00 bash' | grep -v pgrep; echo ==; tail -n 400 -q /root/out/vq*.log")
        head, _, logs = r.stdout.partition("==\n")
        lines = head.splitlines()
        if len(lines) < 2:
            continue
        box_now, queued_here = int(lines[0]), int(lines[1])
        jobs = lines[2:]
        queued += queued_here
        running += len(jobs)
        abo_seen += sum(1 for l in jobs if "abo_" in l)
        day = time.strftime("%Y-%m-%d ", time.gmtime(box_now))
        for m in re.finditer(r"^   gpu\d+ exit \S+ (\d\d:\d\d:\d\d)$", logs, re.M):
            t = time.mktime(time.strptime(day + m.group(1), "%Y-%m-%d %H:%M:%S")) - time.timezone
            t -= 86400 if t > box_now + 60 else 0   # yesterday's
            done += box_now - 3600 <= t <= box_now
        r = ssh(h["port"], h["ip"], "cat /root/queue*.txt 2>/dev/null | grep -c abo_")
        abo_seen += int((r.stdout.split() or ["0"])[0])
    db = sqlite3.connect(f"file:{OKC}/receipts/abo.db?mode=ro", uri=True)
    states = dict(db.execute("select t.state, count(*) from trials t join studies s on t.study_id = s.study_id "
                             "where s.study_name = 'clean1' group by t.state").fetchall())
    waiting = {str(n) for (n,) in db.execute("select t.number from trials t join studies s on t.study_id = s.study_id "
                                             "where s.study_name = 'clean1' and t.state = 'WAITING'")}
    db.close()
    prebuilding = max(0, states.get("RUNNING", 0) - abo_seen)   # study trials in flight, not yet in a queue
    backlog = queued + running + prebuilding + states.get("WAITING", 0)
    src = open(f"{OKC}/harness/clean_study.py").read()
    dims = set(re.findall(r'"(\w+)": \[', src[src.index("DIMS = {"):src.index("\n}\n", src.index("DIMS = {"))]))
    for kind, items in (("candidate", waiting), ("dimension", dims)):
        new = items - set(state["seen"].get(kind, []))
        if state["seen"].get(kind) is not None:   # the first pass records what is there, not arrivals
            state["arrivals"] += [[now, kind] for _ in new]
        state["seen"][kind] = sorted(set(state["seen"].get(kind, [])) | items)
    state["arrivals"] = [a for a in state["arrivals"] if a[0] >= now - 3600]
    ideas = {k: sum(1 for a in state["arrivals"] if a[1] == k) for k in ("candidate", "dimension")}
    state["backlogs"] = [b for b in state.get("backlogs", []) if b[0] >= now - 3600] + [[now, backlog]]
    return done, backlog, ideas


def plan(hosts, state, rates, now):
    """Every pass: search the H200 SXM offers and log each in-limit one when first seen. Add the cheapest per GPU (within
    CAP, up to MAX_GPUS running) only while the GPU-job backlog is real: over GROW_H hours of the last hour's completions. Drain a box beyond the base one once the backlog stays under LOW_H hours of work for
    LOW_S. A booking counts once its first study trial is measured and is destroyed if not proven in READY_S."""
    reg = vast_hosts.boxes()
    for iid, t in list(state["booked"].items()):   # bookings: proven (counted), or destroyed after READY_S
        h = hosts.get(iid)
        if h and reg.get(h["label"], {}).get("proven"):
            log(f"watch: {h['label']} ({iid}) proven (a matmul on every GPU)")
            del state["booked"][iid]
        elif h is None or now - t > READY_S:
            if h:
                log(f"watch: {h['label']} ({iid}) not proven in {READY_S // 60} min "
                    f"(onboarding did not prove its GPUs): destroying")
                subprocess.run(["pkill", "-f", f"onboard.py {iid}"], capture_output=True)
                if ran_work(h["label"]) and not (h["port"] and exported(h["port"], h["ip"], h["label"])):
                    log(f"{h['label']} EXPORT UNVERIFIED, instance kept: it has run work")
                    del state["booked"][iid]
                    continue
                destroy(iid, h["label"])
                reg.pop(h["label"], None)
                vast_hosts.REGISTRY.write_text(json.dumps(reg, indent=1))
            del state["booked"][iid]
    live = {iid: h for iid, h in hosts.items() if h["running"]}   # a pre-empted (stopped) box holds no GPU
    gpus, cost = sum(h["gpus"] for h in live.values()), sum(h["cost"] for h in live.values())
    found = []
    for o in vast("search", "offers", "gpu_name=H200 rentable=true verified=any", "-o", "dph_total"):
        n, mb = o["num_gpus"], o.get("min_bid") or 1e9
        if o["gpu_name"] != "H200" or n not in LIMIT:
            continue
        where = f"offer {o['id']} {n}x, {o.get('geolocation')}, machine {o.get('machine_id')}"
        if o["dph_total"] < LIMIT[n][0]:
            found.append((o["dph_base"] + STORAGE, n, o["id"], [], f"{where}: on-demand ${o['dph_base'] + STORAGE:.2f}/h"))
        if mb <= hold_bid(n) - BID_ROOM:
            found.append((hold_bid(n) + STORAGE, n, o["id"], ["--bid_price", f"{hold_bid(n):.2f}"],
                          f"{where}: bid ${hold_bid(n):.2f}/h (market minimum ${mb:.2f}) + storage"))
    for f in found:
        if f[2] not in state["seen_offers"]:
            state["seen_offers"].append(f[2])
            log(f"watch: in-limit offer seen: {f[4]}, ${f[0] / f[1]:.2f}/GPU/h")
    if not rates:
        return
    done, backlog, _ = rates
    state["low_since"] = (state.get("low_since") or now) if backlog < LOW_H * max(done, 1) else None
    extra = {b: d for b, d in reg.items() if d.get("ready") and not d.get("drain") and b != BASE
             and d.get("role") != "build"}   # (a build box is not GPU capacity: never drained on the GPU backlog)
    if state["low_since"] and now - state["low_since"] >= LOW_S and extra and not state["booked"]:
        dear = max(extra, key=lambda b: next((h["cost"] / h["gpus"] for h in live.values() if h["label"] == b), 0))
        reg[dear]["drain"] = True
        vast_hosts.REGISTRY.write_text(json.dumps(reg, indent=1))
        log(f"watch: GPU-job backlog under {LOW_H:g} h of work for {LOW_S // 60} min: draining {dear} (retired once idle)")
        state["low_since"] = now
    # a stall is never demand: grow only while measured trials are completing (one in the last 25 min, and no run of
    # trials without a result)
    recs = [json.loads(l) for l in open(f"{OKC}/receipts/abo.jsonl")][-6:]
    meas = [json.loads(l).get("t", 0) for l in open(f"{OKC}/receipts/abo.jsonl") if '"status": "measured"' in l]
    healthy = bool(meas) and now - max(meas) < 1500 and not all(r.get("status") == "no-result" for r in recs)
    grow = healthy and backlog > GROW_H * max(done, 1)   # more GPU work pending than the running GPUs finish in an hour
    fit = [f for f in found if gpus + f[1] <= MAX_GPUS and cost + f[0] <= CAP]
    if not grow or state["booked"] or not fit or now - state.get("asked", 0) < 3600:
        return
    price, n, oid, args, what = min(fit, key=lambda f: f[0] / f[1])
    state["asked"] = now   # (no booking here: another GPU only on the orchestrator's approval; logged once an hour)
    log(f"watch: GROWTH WARRANTED, NOT BOOKED (needs the orchestrator's approval): {what}, ${price / n:.2f}/GPU/h (GPU-job backlog "
        f"{backlog} > {done} completed in the last hour; {gpus} GPUs running at ${cost:.2f}/h, cap ${CAP:.2f})")


def retire_idle(hosts):
    """a draining box with empty queues and no job: verified export, destroyed, out of the registry"""
    reg = vast_hosts.boxes()
    for b, d in list(reg.items()):
        iid = str(d.get("id"))
        if not d.get("drain") or iid not in hosts:
            continue
        if ssh(d["port"], d["ip"], f"pgrep -f '{JOBS}' || find /root -maxdepth 1 -name 'queue*.txt' -size +0 | grep -q .").returncode == 0:
            continue
        if exported(d["port"], d["ip"], b):
            destroy(iid, b)
            reg.pop(b)
            vast_hosts.REGISTRY.write_text(json.dumps(reg, indent=1))


if sys.argv[1] == "retire":
    h = {iid: d for iid, d in instances().items()}[sys.argv[2]]
    while ran_work(h["label"]) and not (h["port"] and exported(h["port"], h["ip"], h["label"])):   # no work: nothing to export
        time.sleep(60)
    destroy(sys.argv[2], h["label"])
    sys.exit(0)
floor = float(sys.argv[1])
hosts, pause, k, rates = {}, 30, 0, None
state = {"booked": {}, "seen": {}, "arrivals": [], "backlogs": [], "seen_offers": [], "low_since": None}
if os.path.exists(STATE):
    state.update({k_: v for k_, v in json.load(open(STATE)).items() if k_ in state})
while True:
    now = time.time()
    try:
        hosts = {iid: h for iid, h in instances().items()}
        c = float(vast("show", "user").get("credit"))
        pause = 30
    except RateLimited as e:
        pause = min(300, pause * 2)
        log(f"Vast rate-limits the API ({e}): polling every {pause} s")
        time.sleep(pause)
        continue
    except Exception as e:  # noqa: BLE001
        log(f"show instances / credit failed {e!r}"[:160])
        time.sleep(pause)
        continue
    if c < floor:
        log(f"credit ${c:.2f} below ${floor}: stopping writers, verified export, then destruction (per instance)")
        pending = dict(hosts)
        while pending:   # round-robin: an instance whose export does not verify holds up no other
            for iid, h in list(pending.items()):
                if not ran_work(h["label"]) or h["port"] and exported(h["port"], h["ip"], h["label"]):   # no work: no export
                    destroy(iid, h["label"])
                    del pending[iid]
                elif not h["port"]:
                    log(f"{h['label']} EXPORT UNVERIFIED, instance kept: no ssh port mapped, and it has run work")
            if pending:
                time.sleep(60)
        log("all instances exported (verified) and destroyed")
        break
    reg = vast_hosts.boxes()
    for h in hosts.values():
        h["ready"] = reg.get(h["label"], {}).get("ready", False)
    changed = False   # the study dispatches only to running boxes
    for h in hosts.values():
        d = reg.get(h["label"])
        if d is not None and d.get("offline", False) != (not h["running"]):
            d["offline"] = not h["running"]
            changed = True
            log(f"{h['label']} {'offline: no new trials' if d['offline'] else 'running again: back in the study'}")
    if changed:
        vast_hosts.REGISTRY.write_text(json.dumps(reg, indent=1))
    if k % 4 == 0:   # every 2 minutes: receipts, demand
        for iid, h in hosts.items():
            if not h["port"]:
                continue
            dst = f"{OKC}/receipts/vast/{h['label']}/"
            os.makedirs(dst, exist_ok=True)
            try:
                r = rsync(h["port"], h["ip"], dst)
                if r.returncode not in (0, 23):   # 23: no /root/out yet
                    log(f"{h['label']} rsync rc={r.returncode} {noise(r.stderr)[:150]}")
            except Exception as e:  # noqa: BLE001
                log(f"{h['label']} rsync failed: {e!r}"[:200])
        try:
            rates = demand(hosts, state, now)
        except Exception as e:  # noqa: BLE001
            log(f"demand failed {e!r}"[:160])
            rates = None
        if k % 20 == 0 and rates:
            log(f"credit ${c:.2f}; {sum(h['gpus'] for h in hosts.values())} GPUs at ${sum(h['cost'] for h in hosts.values()):.2f}/h; "
                f"last hour: {rates[0]} GPU jobs completed; backlog {rates[1]} GPU jobs; ideas arrived: {rates[2]}")
        try:
            retire_idle(hosts)
        except Exception as e:  # noqa: BLE001
            log(f"retire_idle failed {e!r}"[:160])
    try:
        plan(hosts, state, rates, now)
    except RateLimited as e:
        pause = min(300, pause * 2)
        log(f"Vast rate-limits the API ({e}): polling every {pause} s")
    except Exception as e:  # noqa: BLE001
        log(f"watch failed {e!r}"[:160])
    json.dump(state, open(STATE, "w"))
    k += 1
    time.sleep(pause)
