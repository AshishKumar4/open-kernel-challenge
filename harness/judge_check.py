"""One unscored judge check of a stacked candidate with a substantial confirmed win: confirmed paired and role-swapped on
a fresh seed block, clearly ahead of the incumbent (confirm.py collect calls this at JUDGE_MIN). The owner's rules: checks
are spent sample-efficiently (small single gains are confirmed on our GPUs only, and stacked first), RESERVE of the
decode layer's 20 daily checks stay unused, a candidate is checked once, and scored runs stay off until a 2 %+ win.
The candidate's final source as a submission would carry it: frozen (harness/freeze.py), asserts dropped
(harness/noassert.py), its request within the judge's cap (the source travels as a JSON string; 922,081 B went
through, 2,740,477 B was refused "request too large": the cap is taken as 1 MiB, the file kept under MAX_JSON); then
`okc check muse_decode_layer <file>` (it waits for the result). Each check is recorded in
receipts/evidence/judge_checks.jsonl (candidate and final md5, job, the check's output tail).
    python3 harness/judge_check.py <name>      (candidates/layer_<name>.py; the final source: candidates/layer_<name>_final.py)"""
import hashlib, json, pathlib, re, subprocess, sys, time

ROOT = pathlib.Path(__file__).resolve().parent.parent
OKC = str(pathlib.Path.home() / ".okc/bin/okc")
name = sys.argv[1]
cand, frozen, final = (ROOT / f"candidates/layer_{name}{s}.py" for s in ("", "_frozen", "_final"))
RESERVE = 3
MAX_JSON = int(0.95 * 2**20)   # (the file as a JSON string, bytes)
md5 = lambda p: hashlib.md5(p.read_bytes()).hexdigest()
LOG = ROOT / "receipts/evidence/judge_checks.jsonl"
if LOG.exists() and any(json.loads(l)["cand_md5"] == md5(cand) for l in LOG.read_text().splitlines() if l.strip()):
    sys.exit(f"{cand.name}: checked already")
left = re.search(r"muse_decode_layer.*?(\d+) of \d+ checks", subprocess.run([OKC, "problems"], capture_output=True, text=True).stdout)
if not left or int(left.group(1)) <= RESERVE:
    sys.exit(f"{left.group(1) if left else 'unknown'} judge checks left today: the reserve of {RESERVE} stays unused")
for step in (["python3", "harness/freeze.py", str(cand), str(frozen)], ["python3", "harness/noassert.py", str(frozen), str(final)]):
    r = subprocess.run(step, cwd=ROOT, capture_output=True, text=True)
    if r.returncode:
        sys.exit(f"{step[1]} failed: {(r.stdout + r.stderr)[-400:]}")
if len(json.dumps(final.read_text())) > MAX_JSON:
    sys.exit(f"{final.name}: {len(json.dumps(final.read_text()))} bytes as a JSON string, over {MAX_JSON}")
import cold_prepare  # noqa: E402
if not cold_prepare.passed(hashlib.md5(final.read_bytes()).hexdigest()):   # (the judge's 300 s prepare cap, measured cold first)
    sys.exit(f"{final.name}: no passing cold prepare; run python3 harness/cold_prepare.py queue {final.stem.removeprefix('layer_')}")
r = subprocess.run([OKC, "check", "muse_decode_layer", str(final)], capture_output=True, text=True, timeout=7200)
out = r.stdout + r.stderr
job = (re.findall(r"job ([0-9a-f]{16})", out) or [None])[0]
rec = {"name": name, "cand_md5": md5(cand), "final_md5": md5(final), "job": job, "rc": r.returncode,
       "tail": out.strip().splitlines()[-12:], "t": round(time.time())}
with open(LOG, "a") as f:
    f.write(json.dumps(rec) + "\n")
print(out[-1500:])
