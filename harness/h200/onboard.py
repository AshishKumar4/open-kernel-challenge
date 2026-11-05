"""Add (or re-add) a vast.ai H200 SXM instance to the study, end to end, with no manual step:
  1. wait until it runs with its ssh port mapped (a missing port: one reboot);
  2. wait for ssh (sshd refusing the key's modes, a Romania host fault: one recreate with the guard's mode-fix onstart);
  3. probe the box (GPUs, CPU quota, each GPU's NUMA node) and pick each GPU's compile-ahead cores off its node, under
     the quota;
  4. register it (receipts/vast/boxes.json, not ready) and label it okc-<label> for the guard;
  5. set it up (harness/h200/vast_setup.sh: bundle, extension dirs, queues, runners, telemetry);
  6. prebuild production's extensions for every GPU (a study bench has 120 s and compiles nothing);
  7. check every GPU (python, torch, a matmul), generate the scored inputs on its disk (/root/okc_inputs, kept), and mark
     the box ready and proven: the study dispatches to it from its next round. A candidate's result never qualifies, drains or retires a box.
Rerunnable: every step is idempotent, and a ready box is left as it is. Exit status 0 once ready, else 1 (the guard
destroys an instance not proven within 20 minutes).
    python3 harness/h200/onboard.py <instance id>"""
import json, pathlib, re, subprocess, sys, time

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "harness"))
import vast_hosts  # noqa: E402

V = "vastai"   # the vast.ai CLI, logged in
IMAGE = "lmsysorg/sglang:dev-muse-spark-open"
iid = sys.argv[1]


def log(msg):
    print(time.strftime("%H:%M:%S"), f"onboard {iid}:", msg, flush=True)


def info():
    r = subprocess.run([V, "show", "instance", iid, "--raw"], capture_output=True, text=True, timeout=60)
    return json.loads(r.stdout)


def sh(port, ip, cmd, timeout=120):
    return subprocess.run(vast_hosts.SSH + ["-p", port, f"root@{ip}", cmd], capture_output=True, text=True, timeout=timeout)


def register(label, **kw):
    reg = vast_hosts.boxes()
    reg[label] = {**reg.get(label, {}), **kw}
    vast_hosts.REGISTRY.write_text(json.dumps(reg, indent=1))


def until(what, test, limit, every=10):
    t0 = time.time()
    while time.time() - t0 < limit:
        r = test()
        if r:
            return r
        time.sleep(every)
    log(f"{what}: not within {limit // 60} min")
    return None


i = info()
label = i["label"][4:] if (i.get("label") or "").startswith("okc-") else f"v{iid}"
if not (i.get("label") or "").startswith("okc-"):
    subprocess.run([V, "label", "instance", iid, f"okc-{label}"], capture_output=True, timeout=60)
if vast_hosts.boxes().get(label, {}).get("ready"):
    log(f"{label} is ready already")
    sys.exit(0)
port_of = lambda d: ((d.get("ports") or {}).get("22/tcp") or [{}])[0].get("HostPort")
up = lambda: (lambda d: d if d.get("actual_status") == "running" and port_of(d) else None)(info())
d = until("running with its ssh port mapped", up, 300)
if not d:
    subprocess.run([V, "reboot", "instance", iid], capture_output=True, timeout=60)
    log("no ssh port mapped: rebooted")
    d = until("running with its ssh port mapped after the reboot", up, 300)
    if not d:
        sys.exit(1)
port, ip = port_of(d), d["public_ipaddr"]
ok = lambda: sh(port, ip, "true", 40).returncode == 0
if not until("ssh", ok, 180):
    sys.path.insert(0, str(ROOT / "harness/h200"))
    onstart = re.search(r'^ONSTART = (\(.*?\))\n', (ROOT / "harness/h200/vast_guard2.py").read_text(), re.S | re.M).group(1)
    subprocess.run([V, "update", "instance", iid, "--image", IMAGE, "--onstart", eval(onstart)], capture_output=True, timeout=60)
    subprocess.run([V, "recycle", "instance", iid], capture_output=True, timeout=60)   # (a reboot keeps the container: its
    # onstart never ran and sshd kept refusing, gpu4 03:00 UTC; a recycle makes it again from the cached image in ~30 s)
    log("ssh refused: recreated with the mode-fix onstart")
    d = until("running with its ssh port mapped after the recreate", up, 300)
    if not d or not until("ssh after the recreate", lambda: sh(port_of(d), d["public_ipaddr"], "true", 40).returncode == 0, 180):
        sys.exit(1)
    port, ip = port_of(d), d["public_ipaddr"]
# each GPU's compile-ahead cores: a contiguous range off its NUMA node, all GPUs' ranges together within half the quota
probe = sh(port, ip, "cat /sys/fs/cgroup/cpu.max; nproc --all; nvidia-smi topo -m | grep '^GPU[0-9]'").stdout.split("\n")
quota = int(probe[0].split()[0]) // int(probe[0].split()[1]) if probe[0].split()[0] != "max" else int(probe[1])
ncpu = int(probe[1])
aff = [re.findall(r"(\d+)-(\d+)", l)[-2:] for l in probe[2:] if l.startswith("GPU")]
per = max(4, min(15, quota * 2 // 3 // max(1, len(aff))))
cpus, taken = [], set()
for a in aff:
    near = {c for lo, hi in a for c in range(int(lo), int(hi) + 1)}
    free = [c for c in range(ncpu) if c not in near and c not in taken] or [c for c in range(ncpu) if c not in taken]
    pick = free[:per]
    taken |= set(pick)
    cpus.append(f"{pick[0]}-{pick[-1]}" if pick == list(range(pick[0], pick[-1] + 1)) else ",".join(map(str, pick)))
log(f"{label}: {len(aff)} GPU(s), quota {quota} CPUs, compile-ahead cores {cpus}")
register(label, id=int(iid), port=port, ip=ip, cpus=cpus, ready=False, drain=False)
held = ROOT / "receipts/vast/image_machines.json"   # (the machine holds our image now: harness/h200/vast_offers.py books such first)
m = json.loads(held.read_text()) if held.exists() else {}
m.setdefault(str(i.get("machine_id")), {"t": round(time.time()), "image": IMAGE, "where": f"{i.get('geolocation', '')}, {label}"})
held.write_text(json.dumps(m, indent=1))
r = subprocess.run(["bash", str(ROOT / "harness/h200/vast_setup.sh"), label], capture_output=True, text=True, timeout=900)
if not until("runners on every GPU", lambda: all(f"vqg.sh {g} " in sh(port, ip, "pgrep -af 'vq[g].sh'").stdout
                                                 for g in range(len(cpus))), 120, 5):   # (runners.sh starts them in the background)
    log(f"setup: no runners: {(r.stdout + r.stderr)[-300:]}")
    sys.exit(1)
# production's extensions into every GPU's extension dir before the first trial: a study bench has 120 s and compiles
# nothing (the candidate's side is prebuilt per trial; production is the same every time)
r = sh(port, ip, " ; ".join(f"cd /root/okc && PYTHONHASHSEED=0 TORCH_EXTENSIONS_DIR=/root/ext{g} TORCH_CUDA_ARCH_LIST=9.0a MAX_JOBS=8 "
                            f"CUDA_VISIBLE_DEVICES={g} nice -n 19 taskset -c {c} timeout 1500 python3 harness/h200/cpu_prebuild.py "
                            f"reviewed/muse_decode_layer.py > /root/out/prebuild_production_{g}.log 2>&1; echo rc$?"
                            for g, c in enumerate(cpus)), timeout=3200)
if r.stdout.count("rc0") != len(cpus):
    log(f"production prebuild failed: {r.stdout.strip()[-200:]}")
    sys.exit(1)
# the proof: python, torch and a matmul on every GPU (a candidate's result never qualifies or retires a box)
r = sh(port, ip, "cd /root/okc && python3 -c \"import torch\nfor g in range(torch.cuda.device_count()):\n"
                 "    a = torch.randn(4096, 4096, device=f'cuda:{g}')\n    print(g, float((a @ a).abs().sum()) > 0)\"", timeout=300)
if r.stdout.count("True") != len(cpus):
    log(f"GPU check failed: {(r.stdout + r.stderr)[-300:]}")
    sys.exit(1)
# the scored inputs on its disk before it is fed (a fresh box's first benches would make them inline and time out); on
# GPU 0, while production's extensions are already built
r = sh(port, ip, "cd /root/okc && mkdir -p /root/okc_inputs && { [ -f /root/okc_inputs/.complete ] || { OKC_INPUT_CACHE=/root/okc_inputs "
                 "CUDA_VISIBLE_DEVICES=0 timeout 1800 python3 harness/h200/gen_inputs.py > /root/out/gen_inputs.log 2>&1 "
                 "&& touch /root/okc_inputs/.complete; }; } && echo inputs-ok", timeout=1900)
if "inputs-ok" not in r.stdout:
    log(f"scored inputs failed: {(r.stdout + r.stderr)[-300:]}")
    sys.exit(1)
register(label, ready=True, proven=True, proven_t=round(time.time()))
log(f"{label} ready: every GPU ran a matmul; the study dispatches to it from its next round")
