"""The vast.ai boxes that the clean study (clean_study.py), the relay, confirmations (confirm.py) and the guard use, read
from the registry that harness/h200/onboard.py writes (receipts/vast/boxes.json):
    label: {"id": vast instance id, "port": ssh port, "ip": ip, "cpus": [compile-ahead cores of each GPU], "ready": bool,
            "drain": bool, "role": "build" for a build box, "offline": bool and "spot": bool for a preemptible GPU box,
            "benchd": bool for a GPU whose benches go to the slot's persistent bench server (harness/benchd.py)}
A box takes jobs once its one-trial smoke passed (ready), and none once it drains (its jobs in flight finish, then the
guard retires it). Each GPU has its own queue, lock and extension dir under /root/{queue,ext}<gpu>, popped by its
runner (harness/h200/vqg.sh <gpu>). The relay's build boxes compile each job's modules (harness/h200/box_build.sh); a
GPU box that builds for itself does so on a core range per GPU, well under the box's CPU quota (a throttled cgroup
would stall the bench process too) and off the GPU's NUMA node."""
import json, os, pathlib

REGISTRY = pathlib.Path(__file__).resolve().parent.parent / "receipts/vast/boxes.json"
STATE = pathlib.Path(os.environ.get("OKC_STATE", "~/.okc")).expanduser()   # the PC-side state: the relay's queue, logs, the boxes' known_hosts
SSH = ["ssh", *(["-i", os.path.expanduser(os.environ["OKC_SSH_KEY"])] if os.environ.get("OKC_SSH_KEY") else []),
       "-o", f"UserKnownHostsFile={STATE / 'known_hosts'}",
       "-o", "StrictHostKeyChecking=accept-new", "-o", "ConnectTimeout=20", "-o", "IdentitiesOnly=yes"]


def boxes():
    return json.loads(REGISTRY.read_text()) if REGISTRY.exists() else {}


def load(offline=False):
    """(HOSTS, GPU, EXT, CPUS, DRAIN) over the ready boxes, one entry per GPU named <label>g<gpu>:
    HOSTS: (ssh port, ip, queue, lock); GPU: index; EXT: extension dir; CPUS: compile-ahead cores; DRAIN: GPUs that
    take no new trials. offline: the boxes the guard saw stopped too (a spot GPU's jobs wait for it in the relay), a
    drained one at the box it handed its outputs to ("moved_to": its results are read there, harness/h200/nebius.py)"""
    reg = boxes()
    hosts, gpu, ext, cpus, drain = {}, {}, {}, {}, set()
    for b, d in reg.items():
        if not d.get("ready") or (d.get("offline") and not offline) or d.get("role") == "build":   # (offline: the guard saw it stopped)
            continue
        at = reg.get(d.get("moved_to"), d)
        for g, c in enumerate(d["cpus"]):
            h = f"{b}g{g}"
            hosts[h] = (at["port"], at["ip"], f"/root/queue{g}.txt", f"/root/queue{g}.lock")
            gpu[h], ext[h], cpus[h] = g, f"/root/ext{g}", c   # one extension dir per GPU (a shared one hung benches)
            if d.get("drain"):
                drain.add(h)
    return hosts, gpu, ext, cpus, drain


HOSTS, GPU, EXT, CPUS, DRAIN = load()



def gpus():
    """the GPUs that take new work (the study's targets, a lane's default): ready and not draining; a spot GPU offline
    too (the relay holds its jobs until it is back, or moves them to another GPU up). The primary first (registry
    "primary", harness/h200/nebius.py: the box whose runs between preemptions are long enough for a confirmation)"""
    hosts, _, _, _, drain = load(offline=True)
    reg = boxes()
    return sorted((h for h in hosts if h not in drain), key=lambda h: (not reg[h.rsplit("g", 1)[0]].get("primary"), h))

LIMIT = 170   # a bench's time limit (s; benchd: from the server taking it)


def bench_job(name, ext, host, reps, level, checks):
    """the GPU queue's line benching candidates/layer_<name>.py against reviewed production, single arm, <reps> reps, at
    priority <level> (harness/submit.py): /root/out/abo_<name>.json, .log, .built (each extension it still compiled, name
    and seconds) and .done. checks: the 20 eager reference checks per shape before the replay checks (controls,
    confirmations, anything frozen or submitted); without, a screen's replay checks of the timed path only (owner)"""
    run = f"run p{level} cd /root/okc && " + ("" if checks else "export OKC_BENCH_NOCHECK=1 && ")
    if boxes().get(host.rsplit("g", 1)[0], {}).get("benchd"):
        return (f"{run}: > /root/out/abo_{name}.built && OKC_BUILD_LOG=/root/out/abo_{name}.built PYTHONHASHSEED=0 "
                f"TORCH_EXTENSIONS_DIR={ext} timeout -k 10 900 python3 harness/benchd.py /root/benchd$OKC_SLOT.sock "
                f"candidates/layer_{name}.py /root/out/abo_{name}.json {reps} /root/out/abo_{name}.log {LIMIT}; rc=$?; "
                f"touch /root/out/abo_{name}.done; exit $rc")
    return (f"{run}: > /root/out/abo_{name}.built && OKC_BUILD_LOG=/root/out/abo_{name}.built "
            f"PYTHONHASHSEED=0 TORCH_EXTENSIONS_DIR={ext} timeout -s USR1 -k 5 {LIMIT} python3 harness/okc_harness.py bench layer --file candidates/layer_{name}.py "
            f"--against reviewed/muse_decode_layer.py --reps {reps} --json /root/out/abo_{name}.json > /root/out/abo_{name}.log 2>&1; "
            f"rc=$?; touch /root/out/abo_{name}.done; exit $rc")


def builder(box):
    """(ssh port, ip) of the ready build box serving GPU box <box> (registry role "build", "for": <box>), or None"""
    return next(((d["port"], d["ip"]) for d in boxes().values()
                 if d.get("role") == "build" and d.get("for") == box and d.get("ready") and not d.get("offline")), None)
