"""Cold prepare (required before any `okc check`): the judge compiles every module inside prepare(), with no build
cache, under a 300 s cap. This benches a file on a GPU box the same way: an empty torch extension dir, triton and tvm-ffi
caches and XDG cache, ccache off, 20 reps (only prepare's time matters), through the one flow at level 1 under the lane
limit. A file passes with prepare at most LIMIT; receipts/evidence/cold_prepare.jsonl records each reading by md5
(harness/judge_check.py requires a passing one for the exact bytes it checks).
    python3 harness/cold_prepare.py queue <name> [GPU host]     candidates/layer_<name>.py
    python3 harness/cold_prepare.py read <name> [GPU host]      prints and records the reading once the job is done"""
import hashlib, json, pathlib, re, shutil, subprocess, sys, time

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "harness"))
import submit, vast_hosts  # noqa: E402
CAP = LIMIT = 300   # the judge's prepare cap: a file passes below it (no stricter veto of our own)
RECEIPTS = ROOT / "receipts/evidence/cold_prepare.jsonl"


def queue(name, host=None):
    host = host or vast_hosts.gpus()[0]
    shutil.copyfile(ROOT / f"candidates/layer_{name}.py", ROOT / f"candidates/layer_cold_{name}.py")   # (the job ships its own
    # copy: candidates/layer_<name>.py itself reaches the GPU box only with a job of that name, cold_dbt_h1 found none there)
    cmd = (f"D=$(mktemp -d /root/cold_{name}.XXXX) && mkdir -p $D/ext $D/triton $D/tvm $D/xdg && "
           f"CCACHE_DISABLE=1 TORCH_EXTENSIONS_DIR=$D/ext TRITON_CACHE_DIR=$D/triton TVM_FFI_CACHE_DIR=$D/tvm XDG_CACHE_HOME=$D/xdg "
           f"python3 harness/okc_harness.py bench layer --file candidates/layer_cold_{name}.py --against reviewed/muse_decode_layer.py "
           f"--reps 20 --json /root/out/cold_{name}.json > /root/out/cold_{name}.log 2>&1; rc=$?; rm -rf $D; exit $rc")
    submit.run(f"cold_{name}", host, 1, cmd)   # (its own job name: the relay keeps one job per name, and the
    # candidate's screen has its name)


def read(name, host=None):
    port, ip = vast_hosts.load()[0][submit.gpu_of(f"cold_{name}", host or vast_hosts.gpus()[0])][:2]
    out = subprocess.run(vast_hosts.SSH + ["-p", port, f"root@{ip}", f"ls /root/out/abo_cold_{name}.done > /dev/null && "
                         f"cat /root/out/cold_{name}.json 2>/dev/null; echo; grep -E 'import \\+ prepare|Killed|FAILED|Error' "
                         f"/root/out/cold_{name}.log | tail -3"], capture_output=True, text=True, timeout=60).stdout
    if not out.strip():
        sys.exit(f"{name}: not done yet")
    i = out.find("{")
    res = json.loads(out[i:out.rindex("}") + 1]) if i >= 0 else {}
    m = re.search(r"import \+ prepare ([0-9.]+) s", out)   # (the bench may end at the lane limit after prepare: its first
    prep = res.get("identity", {}).get("prepare_s") or (float(m.group(1)) if m else None)   # timed runs are cold as well)
    md5 = hashlib.md5((ROOT / f"candidates/layer_{name}.py").read_bytes()).hexdigest()
    rec = {"name": name, "md5": md5, "host": host, "prepare_s": prep, "pass": prep is not None and prep < LIMIT,
           "checks": "pass" if res.get("cases") else res.get("failed") or "no result", "t": round(time.time())}
    with open(RECEIPTS, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(f"{name} ({md5[:8]}): cold prepare {prep} s (limit {LIMIT}, judge cap {CAP}): {'PASS' if rec['pass'] else 'FAIL'}; "
          f"{out[out.rindex('}') + 1:].strip() if i >= 0 else out.strip()}")


def passed(md5):
    """a passing cold-prepare reading of these bytes exists"""
    return RECEIPTS.exists() and any(r["md5"] == md5 and r["pass"] for r in map(json.loads, open(RECEIPTS)))


if __name__ == "__main__":
    {"queue": queue, "read": read}[sys.argv[1]](*sys.argv[2:])
