"""A file about to be submitted, validated on one GPU (freeze, size/scan, cold, ordered gates, 12x200 soak,
freshness), the GPU work queued at level 0 so it runs ahead of everything there:
  freeze      harness/freeze.py + harness/noassert.py: candidates/layer_<name>_final.py, the bytes to submit;
  size/scan   the final as a JSON string under the judge's request cap (judge_check.MAX_JSON), and harness/import_check.py;
  cold        the final benched with every cache empty (harness/cold_prepare.py's job): prepare under the judge's 300 s;
  gates       okc_harness test layer --cases tests on the final (the 13 task tests), and --cases nearby (nearby shapes);
  soak        12 jobs, one per scored case, 200 reps on fresh inputs with the full checks (harness/soak.py);
  freshness   the final against its source: the frozen bytes import and carry the same build plan (freeze.py checks the
              module's final string state against the original's).
    python3 harness/submit_validate.py queue <name> <GPU host>     then   python3 harness/submit_validate.py read <name>"""
import hashlib, json, pathlib, subprocess, sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "harness"))
import submit, vast_hosts  # noqa: E402

MAX_JSON = int(0.95 * 2**20)
md5 = lambda p: hashlib.md5(p.read_bytes()).hexdigest()
cmd, name = sys.argv[1], sys.argv[2]
cand, frozen, final = (ROOT / f"candidates/layer_{name}{s}.py" for s in ("", "_frozen", "_final"))
fin = f"{name}_final"
if cmd == "queue":
    host = sys.argv[3]
    for step in (["python3", "harness/freeze.py", str(cand), str(frozen)], ["python3", "harness/noassert.py", str(frozen), str(final)]):
        r = subprocess.run(step, cwd=ROOT, capture_output=True, text=True)
        if r.returncode:
            sys.exit(f"{name}: {step[1]} failed: {(r.stdout + r.stderr)[-400:]}")
    size = len(json.dumps(final.read_text()))
    if size > MAX_JSON:
        sys.exit(f"{name}: final {size} B as JSON, over the cap {MAX_JSON}")
    r = subprocess.run(["python3", "harness/import_check.py", str(final)], cwd=ROOT, capture_output=True, text=True)
    if r.returncode:
        sys.exit(f"{name}: import check failed: {(r.stdout + r.stderr)[-400:]}")
    import confirm   # (freshness: the final carries every adopted lane delta and is the candidate's own bytes, frozen)
    rec = confirm.lane_asm.record(name) or {}
    lacks = confirm.lacks(confirm.adopted(), rec["point"], rec["t"], set(rec.get("switches", {})), rec.get("per_case"),
                          final, rec["clean"]) if rec.get("point") else None
    print(f"{name}: final {final.name} md5 {md5(final)} (candidate {md5(cand)}), {size} B as JSON, imports clean; "
          f"freshness: lacks {lacks if lacks is not None else 'n/a (not a lane() assembly: the incumbent file itself)'}")
    lane = vast_hosts.STATE / "val" / fin   # (the final travels with each job as a lane file:
    (lane / "candidates").mkdir(parents=True, exist_ok=True)              # the relay prebuilds it and ships both)
    (lane / "candidates" / f"layer_{fin}.py").write_bytes(final.read_bytes())
    f = f"candidates/layer_{fin}.py"
    env = "export PYTHONPATH=/root/okc/harness PYTHONHASHSEED=0;"
    cold = (f"{env} D=$(mktemp -d /root/cold_{fin}.XXXX) && mkdir -p $D/ext $D/triton $D/tvm $D/xdg && "
            f"CCACHE_DISABLE=1 TORCH_EXTENSIONS_DIR=$D/ext TRITON_CACHE_DIR=$D/triton TVM_FFI_CACHE_DIR=$D/tvm XDG_CACHE_HOME=$D/xdg "
            f"python3 /root/okc/harness/okc_harness.py bench layer --file {f} --against /root/okc/reviewed/muse_decode_layer.py "
            f"--reps 20 --json /root/out/val_{fin}_cold.json > /root/out/val_{fin}_cold.log 2>&1; rc=$?; rm -rf $D; exit $rc")
    submit.run(f"val_{fin}_cold", host, 0, cold, files=str(lane))
    submit.run(f"val_{fin}_gates", host, 0, f"{env} python3 /root/okc/harness/okc_harness.py test layer --cases tests --file {f} "
                                             f"> /root/out/val_{fin}_gates.txt 2>&1", files=str(lane))
    submit.run(f"val_{fin}_nearby", host, 0, f"{env} python3 /root/okc/harness/okc_harness.py test layer --cases nearby --file {f} "
                                              f"> /root/out/val_{fin}_nearby.txt 2>&1", files=str(lane))
    for i in range(12):
        n = f"val_{fin}_soak{i:02d}"
        submit.run(n, host, 0, f"{env} timeout 300 python3 /root/okc/harness/soak.py {f} --reps 200 --fresh 20 --cases {i} "
                               f"--json /root/out/{n}.json > /root/out/{n}.txt 2>&1", files=str(lane))
    print(f"{name}: cold, gates and 12 soak jobs queued on {host} at level 0")
elif cmd == "read":
    host = submit.gpu_of(f"val_{fin}_cold", sys.argv[3] if len(sys.argv) > 3 else vast_hosts.gpus()[0])
    port, ip = vast_hosts.load(offline=True)[0][host][:2]
    sh = lambda c: subprocess.run(vast_hosts.SSH + ["-p", port, f"root@{ip}", c], capture_output=True, text=True, timeout=120).stdout
    print(sh(f"cd /root/out; for f in val_{fin}_cold.log; do grep -E 'prepare|import \\+ prepare|PASS|FAIL' $f | tail -3; done; "
             f"echo '-- gates'; grep -cE '^PASS' val_{fin}_gates.txt; grep -E '^FAIL' val_{fin}_gates.txt | head -3; "
             f"echo '-- soak'; for i in $(seq -w 0 11); do f=val_{fin}_soak$i.txt; [ -f $f ] && echo \"$i $(tail -1 $f | cut -c1-110)\"; done"))
