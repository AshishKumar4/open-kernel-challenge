"""A rented Vast box as a spot GPU box for a fixed time (the owner: gpu3, 2x H200 on an interruptible bid, ~4 hours):
each minute, an outbid or stopped instance is marked offline (the relay moves its jobs, as for Nebius: relay.py "spot")
and, once running again, set up again (harness/h200/vast_setup.sh: bundle, queues, runners) and marked online with its
"back" time (the relay queues what the stop lost again). At the deadline its outputs come home
(receipts/vast/<label>_out) and go to the fed box <to> (registry "moved_to": readers find them there), then it is
destroyed.
    python3 harness/h200/vast_spot.py <label> <instance id> <deadline, epoch s> <to: a fed box>"""
import json, os, pathlib, subprocess, sys, time

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "harness"))
import vast_hosts  # noqa: E402

V = "vastai"   # the vast.ai CLI, logged in
label, iid, deadline, to = sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4]


def log(msg):
    print(f"{time.strftime('%H:%M:%S')} {label}: {msg}", flush=True)


def registry(update):
    sys.path.insert(0, str(ROOT / "harness/h200"))
    import nebius   # (its locked read-modify-write of the registry)
    nebius.registry(update)


def status():
    r = subprocess.run([V, "show", "instance", iid, "--raw"], capture_output=True, text=True, timeout=60)
    return json.loads(r.stdout).get("actual_status") if r.returncode == 0 and r.stdout.strip() else None


def ssh(box, cmd, timeout=600):
    d = vast_hosts.boxes()[box]
    return subprocess.run(vast_hosts.SSH + ["-p", d["port"], f"root@{d['ip']}", cmd], capture_output=True, text=True,
                          timeout=timeout)


registry(lambda reg: reg[label].update(spot=True))
while time.time() < deadline:
    try:
        s, d = status(), vast_hosts.boxes().get(label, {})
        if d.get("ready") and s and s != "running" and not d.get("offline"):
            registry(lambda reg: reg[label].update(offline=True))
            log(f"{s}: offline, the relay moves its jobs")
        elif s == "running" and d.get("offline"):
            r = subprocess.run(["bash", str(ROOT / "harness/h200/vast_setup.sh"), label], capture_output=True, text=True,
                               timeout=900, cwd=ROOT)
            if r.returncode == 0:
                registry(lambda reg: reg[label].update(offline=False, back=time.time()))
                log("running again: set up, online")
    except Exception as e:   # (logged and retried next minute)
        log(f"pass failed: {e}")
    time.sleep(60)
log("deadline: its outputs home and to the fed box, then destroyed")
registry(lambda reg: reg[label].update(drain=True))
time.sleep(300)   # (what it runs now finishes; its queued jobs go to the fed box: the relay, drain)
home = ROOT / f"receipts/vast/{label}_out"
home.mkdir(parents=True, exist_ok=True)
d, t = vast_hosts.boxes()[label], vast_hosts.boxes()[to]
e = " ".join(vast_hosts.SSH)
subprocess.run(["rsync", "-a", "--exclude", "util.csv", "-e", f"{e} -p {d['port']}", f"root@{d['ip']}:/root/out/", f"{home}/"],
               timeout=3600)
subprocess.run(["rsync", "-a", "--ignore-existing", "-e", f"{e} -p {t['port']}", f"{home}/", f"root@{t['ip']}:/root/out/"],
               timeout=3600)
registry(lambda reg: reg[label].update(offline=True, moved_to=to))
r = subprocess.run([V, "destroy", "instance", iid], capture_output=True, text=True, timeout=120)
log(f"destroyed ({r.stdout.strip()[-80:]}); its outputs in {home} and on {to}")
