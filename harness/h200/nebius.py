"""The GPU box: a preemptible Nebius H200 in the cheapest region, under the owner's price cap (CAP_USD per GPU-hour).

The cap is set provider-side: in each region's project a pricing policy with max CAP_USD exists before any VM, and the
VMs run under it, so Nebius itself never bills above it (above it the policy blocks and the VM is preempted).

    python3 harness/h200/nebius.py watch         # each minute: a preempted box back; the GPU moved to a cheaper region
    python3 harness/h200/nebius.py up <region>   # a GPU box there (created, or its stopped VM started), set up, fed
    python3 harness/h200/nebius.py setup         # after a CAP_USD change: every region's pricing policy updated
    python3 harness/h200/nebius.py status        # prices, the state file, the Nebius boxes of the registry

One box feeds the study (the owner), a second only while the lane backlog is long (BACKLOG_N, BACKLOG_S); each in its
own region: "spot1" in eu-north1, each new region's box the next "nb<n>" (registry: "provider": "nebius", "spot",
"benchd"). With fewer fed under the cap than wanted, watch runs `up` in the cheapest region without one; when a region
stays MOVE_USD per hour cheaper than the dearest fed one for MOVE_S (the owner: as cheap as
possible), it runs `up` there; once that box is ready the dearest beyond the wanted number drains (it takes no new work, and the relay moves
its queued jobs to the new GPU) and, once it is idle (no job of it with the relay, its queue empty, no job running), its
outputs go to the fed box, readers follow them there (registry "moved_to": submit.gpu_of), and it is stopped with its disk
kept: coming back to its region later is a start, not a new setup. A new box is another H200: the study's per-GPU
cohorts start on it from the incumbent's baseline (clean_study.py ctl_level). One `up` at a time (UP_LOCK)."""
import fcntl, json, os, pathlib, statistics, subprocess, sys, time, urllib.request

CAP_USD = 2.50   # the owner's cap, USD per GPU-hour: the one edit for an owner change (then `setup` updates the policies)
MOVE_USD, MOVE_S = 0.50, 1800   # (the owner) another region this much cheaper per hour for this long: move there
PER_REGION = 2   # preemptible VMs a region's quota allows: a slot "<region>/2" is the second box there
RECOVER_S = 600   # the work a preemption costs (its recovery, the cut jobs): price per productive hour =
#                   price * (1 + RECOVER_S / the region's median run between stops)
CONFIRM_S = 600   # a box whose median run between preemptions is shorter than a confirmation is not the primary
BACKLOG_N, BACKLOG_S = 8, 900   # (the owner) one GPU; a second only while at least this many lane and confirmation jobs
#                                 (levels 0-2) wait with the relay for this long, drained once none have waited as long
PLATFORM, PRESET = "gpu-h200-sxm", "1gpu-16vcpu-200gb"
REGIONS = ("eu-north1", "eu-west1", "us-central1")   # the H200 regions in the public spot feed
POLICY = "okcspot250"
DISK_GIB = 320   # the full input cache (158 GB), /opt/sglang (22), modules, ext dirs, OS: ~245 GB used
D = pathlib.Path(os.environ.get("OKC_STATE", "~/.okc")).expanduser()
STATE = D / "nebius.json"   # {"tenant", "regions": {region: {"project", "subnet", "policy_id", "instance_id", "box"}},
#                              "cheaper": {region: since}, "up": {"region", "t"}}
UP_LOCK = D / "nebius_up.lock"   # held by an `up` for its whole run
SETUP_TRIES = 8   # an `up` gives up after this many cut setups (preemptions of the new box or of the fed one)
KEYS = D / "nebius_keys.txt"   # the public keys the boxes' root takes, on the host (port 22) and in the container (2222)
RELAY_STATE = D / "relay/state.json"
IN_FLIGHT = ("queued", "building", "built", "shipping", "shipped")
CLI = os.environ.get("NEBIUS_CLI", str(D / "nebius/bin/nebius"))
PROFILE = os.environ.get("NEBIUS_PROFILE", "okc")
H200 = pathlib.Path(__file__).resolve().parent
ROOT = H200.parent.parent
sys.path.insert(0, str(ROOT / "harness"))
import vast_hosts  # noqa: E402


def log(msg):
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def prices():
    """{region: USD per GPU-hour} for the H200 from Nebius's public spot feed (no account needed)"""
    rq = urllib.request.Request("https://api.nebius.cloud/billing/v1alpha1/spot-price?pageSize=1000",
                                headers={"User-Agent": "curl/8"})   # (Python's default agent is refused, 403)
    with urllib.request.urlopen(rq, timeout=30) as r:
        items = json.load(r)["items"]
    return {i["region"]: float(i["price"]["value"]) for i in items
            if (i.get("computeInstance") or {}).get("platform") == PLATFORM and i["region"] in REGIONS}


def nebius(*args, parse=True):
    """the nebius CLI under the owner's profile; its JSON output"""
    r = subprocess.run([CLI, "--profile", PROFILE, *args, "--format", "json"], capture_output=True, text=True, timeout=600)
    if r.returncode:
        raise RuntimeError(f"nebius {' '.join(args[:3])}: {r.stderr.strip()[:300]}")
    return json.loads(r.stdout) if parse and r.stdout.strip() else r.stdout


def locked(path, update):
    """the JSON file at path, update(it) applied and written under its lock (watch and an `up` both write it); it"""
    with open(path.with_suffix(".lock"), "a") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        d = json.loads(path.read_text()) if path.exists() else {}
        if update:
            update(d)
            path.with_suffix(".tmp").write_text(json.dumps(d, indent=1))
            os.replace(path.with_suffix(".tmp"), path)
        return d


def state(update=None):
    return locked(STATE, update)


def registry(update):
    locked(vast_hosts.REGISTRY, update)


def policy(project):
    """the project's pricing policy (max CAP_USD), created or brought to CAP_USD; its id"""
    try:
        pol = nebius("billing", "pricing-policy", "get-by-name", "--name", POLICY, "--parent-id", project)
    except RuntimeError:
        pol = nebius("billing", "pricing-policy", "create", "--parent-id", project, "--name", POLICY,
                     "--compute-instance-spec-v1-platform", PLATFORM, "--pricing-max-price-v1-max-price", f"{CAP_USD:.3f}")
        log(f"{project}: pricing policy {POLICY} created, max ${CAP_USD:.3f} per GPU-hour")
    else:
        if float(pol["spec"]["pricing"]["max_price_v1"]["max_price"]) != CAP_USD:
            nebius("billing", "pricing-policy", "update", pol["metadata"]["id"], "--pricing-max-price-v1-max-price", f"{CAP_USD:.3f}")
            log(f"{project}: pricing policy {POLICY} updated to max ${CAP_USD:.3f}")
    return (pol.get("metadata") or {}).get("id") or pol.get("resource_id")


def base(slot):
    """the region of a slot ("eu-north1" or its second box's "eu-north1/2")"""
    return slot.split("/")[0]


def region(st, r):
    """slot r's entry, its ids found once: the tenant's project in its region, its subnet, its pricing policy, its box's
    name"""
    e = st["regions"].setdefault(r, {})
    if not e.get("project"):
        e["project"] = next(p["metadata"]["id"] for p in nebius("iam", "v2", "project", "list", "--parent-id", st["tenant"])["items"]
                            if p["spec"].get("region") == base(r))
    if not e.get("subnet"):
        e["subnet"] = nebius("vpc", "subnet", "list", "--parent-id", e["project"])["items"][0]["metadata"]["id"]
    if not e.get("policy_id"):
        e["policy_id"] = policy(e["project"])
    if not e.get("box"):
        e["box"] = f"nb{1 + max((int(v['box'][2:]) for v in st['regions'].values() if v.get('box')), default=0)}"
    return e


def region_image(e):
    """the slot's region's boot image (made from a set-up box's disk: the container image pulled, modules, scored inputs,
    so a new box there pulls nothing), or None"""
    st = state()
    return next((v["image_id"] for k, v in st["regions"].items() if base(k) == base(next(
        k2 for k2, v2 in st["regions"].items() if v2 is e or v2.get("box") == e.get("box"))) and v.get("image_id")), None)


def create(e):
    """the box's VM under its region's pricing policy: preemptible (stopped on preemption), a static IP, root's keys"""
    keys = "".join(f"      - {k}\n" for k in KEYS.read_text().splitlines() if k.startswith("ssh-"))
    user_data = ("#cloud-config\ndisable_root: false\nssh_pwauth: false\nusers:\n  - default\n  - name: okc\n"
                 "    sudo: ALL=(ALL) NOPASSWD:ALL\n    shell: /bin/bash\n    ssh_authorized_keys:\n" + keys +
                 "write_files:\n  - path: /etc/ssh/sshd_config.d/00-okc.conf\n    content: |\n"
                 "      PermitRootLogin prohibit-password\n"
                 "runcmd:\n  - mkdir -p /root/.ssh\n  - cp /home/okc/.ssh/authorized_keys /root/.ssh/authorized_keys\n"
                 "  - chmod 700 /root/.ssh\n  - chmod 600 /root/.ssh/authorized_keys\n  - systemctl restart ssh\n")
    name = f"okc-{e['box']}"
    out = nebius("compute", "instance", "create", "--parent-id", e["project"], "--name", name,
                 "--resources-platform", PLATFORM, "--resources-preset", PRESET,
                 "--boot-disk-managed-disk-name", f"{name}-boot", "--boot-disk-managed-disk-type", "network_ssd",
                 "--boot-disk-managed-disk-size-gibibytes", str(DISK_GIB),
                 *(["--boot-disk-managed-disk-source-image-id", img] if (img := region_image(e)) else
                   ["--boot-disk-managed-disk-source-image-family-image-family", "ubuntu24.04-cuda13.0"]),
                 "--boot-disk-attach-mode", "READ_WRITE",
                 "--network-interfaces", json.dumps([{"name": "eth0", "ip_address": {}, "public_ip_address": {"static": True},
                                                       "subnet_id": e["subnet"]}]),
                 "--cloud-init-user-data", user_data,
                 "--recovery-policy", "fail", "--preemptible-on-preemption", "stop",
                 "--spot-pricing-policy-id", e["policy_id"])
    return out.get("resource_id") or out["metadata"]["id"]


def vm(e):
    """(state, public ip) of the box's VM"""
    s = nebius("compute", "instance", "get", "--id", e["instance_id"])["status"]
    nic = (s.get("network_interfaces") or [{}])[0]
    return s["state"], (nic.get("public_ip_address") or {}).get("address", "").split("/")[0]


def ssh(ip, cmd, port="22", timeout=600):
    return subprocess.run(vast_hosts.SSH + ["-p", port, f"root@{ip}", cmd], capture_output=True, text=True, timeout=timeout)


def scp(ip, files, dest, port="22"):
    subprocess.run(["scp", "-q", *vast_hosts.SSH[1:], "-P", port, *map(str, files), f"root@{ip}:{dest}"], check=True, timeout=600)


def ship_bundle(ip):
    """main's committed harness on the box (bundle.sh: the code every box runs), into the container's /root/okc"""
    subprocess.run(["bash", str(H200 / "bundle.sh")], cwd=ROOT, check=True, stdout=subprocess.DEVNULL, timeout=600)
    scp(ip, [D / "okc_box.tgz"], "/root/", port="2222")
    ssh(ip, "mkdir -p /root/okc && tar xzf /root/okc_box.tgz -C /root/okc", port="2222").check_returncode()


def recover(ip):
    """the container, the current bundle, its runner and the scored inputs back on a started VM (nebius_recover.sh);
    whether it answered"""
    scp(ip, [H200 / "nebius_recover.sh"], "/root/")
    if "recovered" not in ssh(ip, "docker start okc > /dev/null && docker exec okc bash -c 'mkdir -p /run/sshd; "
                                  "pgrep -x sshd > /dev/null || /usr/sbin/sshd -p 2222 -o PermitRootLogin=prohibit-password "
                                  "-o PasswordAuthentication=no' && echo recovered").stdout:
        return False
    ship_bundle(ip)   # (before the runner and the input generator start from it: a box keeps no harness of its own)
    return "recovered" in ssh(ip, "bash /root/nebius_recover.sh").stdout


def idle(box, ip):
    """no job of the box with the relay (not yet on it), none in its queues, none running (a runner's last log line is
    its job's start until the job exits: harness/h200/vqg.sh)"""
    s = json.loads(RELAY_STATE.read_text()) if RELAY_STATE.exists() else {}
    if any(v["gpu"].rsplit("g", 1)[0] == box and v["phase"] in IN_FLIGHT for v in s.values()):
        return False
    return ssh(ip, "! grep -qs . /root/queue*.txt && ! tail -qn1 /root/out/vq*.log 2>/dev/null | grep -q '^== gpu'",
               port="2222", timeout=60).returncode == 0


def pull(ip, src, opts, *dirs):
    """each of <dirs> (container paths) copied from the box at src to the box at ip, box to box (streamed through this
    machine it ran at 0.25 MB/s): ip's own key, added to src's keys, and rsync there"""
    key = ssh(ip, "[ -f /root/.ssh/seed ] || ssh-keygen -q -t ed25519 -N '' -C seed -f /root/.ssh/seed; cat /root/.ssh/seed.pub",
              port="2222").stdout.strip().splitlines()[-1]
    ssh(src, f"grep -qxF '{key}' /root/.ssh/authorized_keys || echo '{key}' >> /root/.ssh/authorized_keys", port="2222").check_returncode()
    e = "-e 'ssh -i /root/.ssh/seed -p 2222 -o StrictHostKeyChecking=accept-new'"
    ssh(ip, " && ".join(f"mkdir -p {d} && rsync -a --partial {opts} {e} root@{src}:{d} {d}" for d in dirs), port="2222",
        timeout=3600).check_returncode()


def hand_over(ip, to):
    """the box's outputs (/root/out: results, logs, lane outputs) to the fed box at <to>, none of its files replaced"""
    pull(to, ip, "--ignore-existing --exclude util.csv", "/root/out/")


def wait(what, check, limit, every=20):
    t0 = time.time()
    while not check():
        if time.time() - t0 > limit:
            raise RuntimeError(f"{what}: not after {limit // 60} min")
        time.sleep(every)


def set_up(e):
    """the box's VM running, the container (nebius_up.sh), the fed box's modules and extension dir, the bundle, the
    runner and the scored inputs (nebius_recover.sh); its ip. Every step rerunnable"""
    b = e["box"]

    def running():
        s, _ = vm(e)
        if s == "STOPPED":
            nebius("compute", "instance", "start", "--id", e["instance_id"], parse=False)
        return s == "RUNNING"
    wait(f"{b}: VM running", running, 1800)
    ip = vm(e)[1]
    registry(lambda reg: reg.update({b: {**reg.get(b, {}), "port": "2222", "ip": ip, "cpus": ["7-15"], "ready": False,
                                         "drain": False, "proven": False, "offline": False, "spot": True,
                                         "provider": "nebius", "benchd": True, "instance": e["instance_id"]}}))
    def reachable():
        if vm(e)[0] != "RUNNING":   # (preempted while waiting: the retry starts it again)
            raise RuntimeError("stopped before its ssh answered")
        return ssh(ip, "true", timeout=60).returncode == 0
    wait(f"{b}: ssh at {ip}", reachable, 900)
    log(f"{b}: VM running at {ip}; the container")
    scp(ip, [KEYS, H200 / "nebius_up.sh"], "/root/")
    out = ssh(ip, "bash /root/nebius_up.sh", timeout=3600)
    if "container up" not in out.stdout:
        raise RuntimeError(f"{b}: nebius_up.sh: {(out.stdout + out.stderr)[-400:]}")
    log(f"{b}: {out.stdout.strip().splitlines()[-1]}")
    fed = [x for x, d in vast_hosts.boxes().items() if d.get("provider") == "nebius" and x != b and d.get("ready") and not d.get("drain")]
    has = ssh(ip, "ls /root/.cache/tvm-ffi 2>/dev/null | head -1", port="2222").stdout.strip()
    if fed and not has:   # (the fed box's production modules and extension dir: a study bench has 120 s and compiles nothing;
        #                    a box coming back has its own)
        wait(f"{b}: {fed[0]} online to copy from", lambda: not vast_hosts.boxes()[fed[0]].get("offline"), 900)
        src = vast_hosts.boxes()[fed[0]]["ip"]
        pull(ip, src, "--exclude '*.lock' --exclude .build_locks --exclude .private --exclude .incoming --exclude .trash",
             "/root/.cache/tvm-ffi/", "/root/ext0/")
        log(f"{b}: modules and ext0 from {fed[0]}")
    if not recover(ip):
        raise RuntimeError(f"{b}: nebius_recover.sh did not answer")
    # (a fresh disk: fed once its scored inputs exist, or its first benches make them inline and time out (spot3, 02:45
    # UTC); a box back from its disk has them: no wait)
    wait(f"{b}: the scored inputs", lambda: reachable() and ssh(ip, "[ -f /root/okc_inputs/.complete ]", port="2222",
                                                                timeout=60).returncode == 0, 1800)

    return ip


def up(r):
    """a GPU box in region r, end to end: its VM (created, or the stopped one started), the container (nebius_up.sh),
    the fed box's modules and extension dir, the bundle, the runner and the scored inputs (nebius_recover.sh); then it
    is ready and every other Nebius box drains. Rerunnable; one at a time (UP_LOCK, else it exits)"""
    lk = open(UP_LOCK, "a")
    try:
        fcntl.flock(lk, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        sys.exit(f"up {r}: another up runs (nebius.json: up)")
    st = state(lambda st: (region(st, r), st.update(up={"region": r, "t": round(time.time())})))
    e = st["regions"][r]
    b = e["box"]
    if not e.get("instance_id"):
        e["instance_id"] = create(e)
        state(lambda st: st["regions"][r].update(instance_id=e["instance_id"]))
        log(f"{b}: VM {e['instance_id']} created in {r}")

    for attempt in range(1, SETUP_TRIES + 1):
        try:
            ip = set_up(e)
            break
        except Exception as ex:   # (a preemption of it or of the fed box it copies from cuts a step: every step reruns,
            if attempt == SETUP_TRIES:   # a stopped VM is started again)
                raise
            log(f"{b}: setup attempt {attempt} cut ({str(ex)[-150:]}); again in 60 s")
            time.sleep(60)

    st = state()
    region_of, want = {v["box"]: k for k, v in st["regions"].items() if v.get("box")}, st.get("want", 1)
    p = prices()

    def feed(reg):
        reg[b].update(ready=True, proven=True, offline=False, ip=ip)
        reg[b].pop("moved_to", None)
        fed = [x for x, d in reg.items() if x != b and d.get("provider") == "nebius" and d.get("ready") and not d.get("drain")]
        for x in sorted(fed, key=lambda x: p.get(base(region_of.get(x, "")), 1e9), reverse=True)[:max(0, len(fed) + 1 - want)]:
            reg[x]["drain"] = True   # (beyond the wanted number of fed boxes, the dearest drain)
    registry(feed)
    state(lambda s: s.setdefault("since", {}).update({b: round(time.time())}))
    log(f"{b}: ready in {r}, fed; Nebius boxes fed: {[x for x, d in vast_hosts.boxes().items() if d.get('provider') == 'nebius' and d.get('ready') and not d.get('drain')]}")


def backlog(st):
    """the wanted number of fed boxes (the owner: one; two while lane and confirmation jobs pile up), kept in the state"""
    s = json.loads(RELAY_STATE.read_text()) if RELAY_STATE.exists() else {}
    n, now = sum(1 for v in s.values() if int(v.get("level", 3)) <= 2 and v["phase"] in IN_FLIGHT), time.time()

    def upd(x):
        x["long_t"] = (x.get("long_t") or now) if n >= BACKLOG_N else None
        x["none_t"] = (x.get("none_t") or now) if n == 0 else None
        if x.get("long_t") and now - x["long_t"] >= BACKLOG_S:
            x["want"] = 2
        elif x.get("none_t") and now - x["none_t"] >= BACKLOG_S:
            x["want"] = 1
    return state(upd).get("want", 1)


def run_ended(b):
    """a fed box stopped: its run's length kept (the last 9), for the primary's choice"""
    def upd(s):
        t0 = s.setdefault("since", {}).pop(b, None)
        if t0:
            s.setdefault("runs", {}).setdefault(b, []).append(round(time.time() - t0))
            s["runs"][b] = s["runs"][b][-9:]
    state(upd)


def median_run(st, b):
    """box b's median run between stops, its current run counted (a box with none ended yet is not taken for the best:
    spot3, fed at 02:44 UTC, was primary over spot1 and spot1 drained)"""
    now_run = time.time() - st.get("since", {}).get(b, time.time())
    return statistics.median((st.get("runs", {}).get(b) or []) + [now_run])


def choose_primary(st, reg):
    """the fed box with the longest median run between preemptions, among those whose median is at least a
    confirmation (CONFIRM_S; a box with no ended run yet counts as long): confirmations go there first (vast_hosts.gpus)"""
    fed = [x for x, d in reg.items() if d.get("provider") == "nebius" and d.get("ready") and not d.get("drain")]
    med = {x: median_run(st, x) for x in fed}
    ok = [x for x in fed if med[x] >= CONFIRM_S] or fed
    p = max(ok, key=med.get) if ok else None
    if any(bool(reg[x].get("primary")) != (x == p) for x in fed):
        registry(lambda reg: [reg[x].update(primary=(x == p)) for x in fed])
        log(f"primary: {p} (median runs between stops: {', '.join(f'{x} {med[x] / 60:.0f} min' for x in fed)})")


def tend(e, d, reg, r):
    """one ready box: the fed one started again after a preemption and recovered once it runs (its jobs lost with the
    stop are queued again by the relay: "back"); a draining one, once idle, hands its outputs to the fed box and is
    stopped (its disk kept); a draining one a preemption stopped first stays stopped: the relay queues its jobs
    without a completion receipt on a GPU that is up"""
    b = e["box"]
    s, ip = vm(e)
    if d.get("drain"):
        fed = [x for x, v in reg.items() if v.get("provider") == "nebius" and v.get("ready") and not v.get("drain")]
        if d.get("moved_to") or not fed:
            return
        if s == "STOPPED":   # (stays stopped: the relay queues its jobs without a completion receipt on a GPU up)
            registry(lambda reg: reg[b].update(offline=True, moved_to=fed[0]))
            log(f"{b}: stopped while draining (preempted): left for good, its disk kept")
        elif s == "RUNNING" and d.get("offline"):
            nebius("compute", "instance", "stop", "--id", e["instance_id"], parse=False)
            registry(lambda reg: reg[b].update(moved_to=fed[0]))
            log(f"{b}: draining and offline: stopped, its disk kept")
        elif s == "RUNNING" and idle(b, ip):
            hand_over(ip, reg[fed[0]]["ip"])
            registry(lambda reg: reg[b].update(offline=True, moved_to=fed[0]))
            nebius("compute", "instance", "stop", "--id", e["instance_id"], parse=False)
            log(f"{b}: drained; its outputs on {fed[0]}; stopped, its disk kept")
    elif s == "STOPPED":
        if not d.get("offline"):
            log(f"{b} STOPPED (preempted): offline, starting it again")
            registry(lambda reg: reg[b].update(offline=True, ip=ip))
            run_ended(b)
        nebius("compute", "instance", "start", "--id", e["instance_id"], parse=False)
    elif s == "RUNNING" and d.get("offline") and recover(ip):
        registry(lambda reg: reg[b].update(offline=False, ip=ip, back=time.time()))
        state(lambda s: s.setdefault("since", {}).update({b: round(time.time())}))
        log(f"{b} back at {ip}: online")


def up_running():
    with open(UP_LOCK, "a") as lk:
        try:
            fcntl.flock(lk, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        return False


def move(st, reg):
    """the owner: the wanted number of fed boxes (backlog()), as cheap as possible. Fewer fed under the cap (a fed region
    above it: the policy stops its VM): `up` at once in the cheapest region under the cap without one; more: the dearest
    drains. Else a region MOVE_USD per hour cheaper than the dearest fed one for MOVE_S: `up` there (its
    feed drains the dearest). A failed `up` starts its region's clock again"""
    u = st.get("up")
    while True:   # (each exited `up` reaped: its zombie would hold nothing, but it is our child)
        try:
            if not os.waitpid(-1, os.WNOHANG)[0]:
                break
        except ChildProcessError:
            break
    if up_running():
        return
    if u:
        ok = reg.get(st["regions"][u["region"]].get("box"), {}).get("ready")
        log(f"up {u['region']}: {'done' if ok else 'failed'} (nebius_up_{u['region']}.log)")
        state(lambda s: (s.pop("up"), s.setdefault("cheaper", {}).pop(u["region"], None)))
        return
    fed = [r for r, e in st["regions"].items() if reg.get(e.get("box"), {}).get("ready") and not reg[e["box"]].get("drain")]
    raw, want = prices(), backlog(st)
    runs = {}
    for r, e in st["regions"].items():   # (a region's runs between stops: all its boxes')
        runs.setdefault(base(r), []).extend(st.get("runs", {}).get(e.get("box"), []))
    # (by price per productive hour, not price alone: a region preempted every few minutes produces nothing)
    p = {s: raw[base(s)] * (1 + RECOVER_S / statistics.median(runs[base(s)])) if runs.get(base(s)) else raw[base(s)]
         for s in [b for b in raw] + [f"{b}/{k}" for b in raw for k in range(2, PER_REGION + 1)]}
    free = sorted((s for s in p if s not in fed and raw[base(s)] <= CAP_USD), key=p.get)
    now = time.time()
    if len([r for r in fed if raw.get(base(r), 0) <= CAP_USD]) < want:
        if free:
            launch(free[0], f"{len(fed)} fed of {want}: {free[0]} ${raw[base(free[0])]:.2f} (${p[free[0]]:.2f} per productive hour)")
        return
    if len(fed) > want:   # (the backlog cleared: the dearest drains, tend stops it once idle; of equal price, the one with
        #                    the shorter median run between stops)
        x = st["regions"][max(fed, key=lambda r: (p.get(r, 0), -median_run(st, st["regions"][r]["box"])))]["box"]
        registry(lambda reg: reg[x].update(drain=True))
        log(f"{len(fed)} fed, {want} wanted: {x} drains")
        return
    a = max(fed, key=lambda r: p.get(r, 0))

    def mark(s):
        c = s.setdefault("cheaper", {})
        for r in p:
            if r in free and p[r] <= p.get(a, 0) - MOVE_USD:
                c.setdefault(r, now)
            else:
                c.pop(r, None)
    due = [r for r, t in state(mark)["cheaper"].items() if now - t >= MOVE_S]
    if due:
        r = min(due, key=p.get)
        launch(r, f"{r} ${p[r]:.2f} vs {a} ${p[a]:.2f} per productive hour for {MOVE_S // 60} min")


def launch(r, why):
    log(f"{why}: up {r}")
    subprocess.Popen([sys.executable, __file__, "up", r], cwd=ROOT, start_new_session=True,
                     stdout=open(D / f"nebius_up_{r}.log", "a"), stderr=subprocess.STDOUT)


def watch():
    while True:
        try:
            st, reg = state(), vast_hosts.boxes()
            for r, e in st["regions"].items():
                if e.get("instance_id") and reg.get(e["box"], {}).get("ready"):   # (a box being set up is its up's)
                    tend(e, reg[e["box"]], reg, r)
            choose_primary(state(), vast_hosts.boxes())
            move(state(), vast_hosts.boxes())
        except Exception as e:   # (logged and retried next minute)
            log(f"pass failed: {e}")
        time.sleep(60)


if __name__ == "__main__":
    cmd = (sys.argv[1:2] or ["status"])[0]
    if cmd == "watch":
        watch()
    elif cmd == "up":
        up(sys.argv[2])
    elif cmd == "setup":
        state(lambda st: [e.update(policy_id=policy(e["project"])) for e in st["regions"].values()])
    elif cmd == "status":
        print(json.dumps({"cap_usd": CAP_USD, "prices": prices(), "state": state(),
                          "boxes": {b: d for b, d in vast_hosts.boxes().items() if d.get("provider") == "nebius"}}, indent=1))
