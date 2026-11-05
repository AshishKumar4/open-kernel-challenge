"""The clean study (the owner's rebase order): the clean codebase's compile-time switches, every point of those no
adopted lane delta fixes (confirmed losers excluded) screened once per distinct built file.

A trial is the clean tip assembled (clean/assemble.py at the `clean` branch) with its switches applied: each switch adds
nvcc flags to the BUILDS entries it belongs to, by a block written right after BUILDS (before any build starts), so the
module names (md5 of source and flags) follow; a switch with a host side also sets its layer text (CONSTS). Dimensions
are added here as the lanes land them (DIMS: name -> values, the first the clean default; APPLY: name -> (BUILDS keys or
a MOE_SWITCHES predicate, flags)). Trials go through the relay (harness/submit.py), are screened single-arm against
reviewed production (200 reps, replay checks), recorded in receipts/abo.db (study clean1) and receipts/abo.jsonl, and
read against the incumbent's controls on their GPU (harness/confirm.py): within noise or ahead overall, or ahead on a
case, gets one 400-rep confirmation, which promotes when ahead.

python3 harness/clean_study.py <ready target>            (runs until killed)
python3 harness/clean_study.py lane <name> <commit> k=v   (a lane's candidate at the incumbent's switches)"""
import ast
import fcntl
import hashlib
import itertools
import json
import math
import os
import pathlib
import random
import re
import statistics
import subprocess
import sys
import time


import optuna

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "harness"))
import confirm  # noqa: E402
import identity  # noqa: E402
import submit  # noqa: E402
import vast_hosts  # noqa: E402

ARGS = sys.argv[1:]
SSH = vast_hosts.SSH
REPS = 200   # a screen's repetitions (its time limit: vast_hosts.LIMIT)
MODE, SEEDS = f"screen: single arm vs reviewed production, {REPS} reps", "profile"
CS = {}   # the per-case TPE studies (case_studies), set in main
CASE_STORAGE = optuna.storages.RDBStorage(f"sqlite:///{ROOT}/receipts/abo_cases.db", engine_kwargs={"connect_args": {"timeout": 60}})


def _fast_sqlite(conn, _):   # (the case studies' own file, WAL and no fsync per commit: an ask on the shared abo.db spent
    c = conn.cursor()        # 0.76 s of 1.3 in 38 commits, 3 s live, x12 cases a proposal: 0.02 s here)
    c.execute("PRAGMA journal_mode=WAL")
    c.execute("PRAGMA synchronous=NORMAL")
    c.close()


import sqlalchemy  # noqa: E402
sqlalchemy.event.listen(CASE_STORAGE.engine, "connect", _fast_sqlite)
CASE_STORAGE.engine.dispose()
STORAGE = optuna.storages.RDBStorage(f"sqlite:///{ROOT}/receipts/abo.db", engine_kwargs={"connect_args": {"timeout": 60}})
RELAY = pathlib.Path(os.environ.get("OKC_STATE", "~/.okc")).expanduser() / "relay"   # harness/relay.py: the build queue and every job's state
AGAINST_MD5 = identity.file_md5(ROOT / "reviewed/muse_decode_layer.py")[:12]
inflight = {}   # name -> (study, trial, params, params, host, dispatched at); a control: study None
log = open(ROOT / "receipts/abo.jsonl", "a")
CTL_EVERY, CTL_FIRST = 3600, 2   # an incumbent's control per GPU each hour; its first CTL_FIRST at once (its baseline;
#                                  2 per new GPU or UUID, at the lanes' level, interleaved with their screens)
CTL, incumbent = confirm.CTL, confirm.incumbent
UNREAD = ROOT / "receipts/unread.json"   # screens read before their incumbent had a baseline (kept across restarts)
ctl_t, ctl_of, unread = {}, {}, json.loads(UNREAD.read_text()) if UNREAD.exists() else {}
md5_of = {}   # each trial in flight -> its built file's md5
gpu_now = {}   # each GPU -> the cohort (confirm.cohort) of its latest reading
REQUEUE = ROOT / "receipts/requeue.json"   # [{"of": trial, "draw": its case_draw}]: trials to run again (a kill by our
# scheduling must not lose a draw's intent; it is re-expressed on the current baseline and linked by requeue_of)


def sh(host, cmd, timeout=60):
    port, ip = vast_hosts.load(offline=True)[0][host][:2]
    try:
        return subprocess.run(SSH + ["-p", port, f"root@{ip}", cmd], capture_output=True, text=True, timeout=timeout).stdout
    except subprocess.TimeoutExpired:   # (a spot box stopped mid-call: nothing read, asked again next round)
        return ""


def dispatch(name, host, lvl=4, checks=False):
    """the relay builds it and ships it to host's queue (checks: the 20 eager reference checks too, as controls get)"""
    submit.screen(name, host, lvl, REPS, checks)
    return host


def result(host, name):
    """None while running, else (outcome, bench json or the failure's reason)"""
    out = sh(host, f"[ -f /root/out/abo_{name}.done ] && (cat /root/out/abo_{name}.json 2>/dev/null; echo; echo ==LOG; "
                   f"grep -m1 ' error: ' /root/out/abo_{name}.log; grep -m1 'Ptx assembly aborted' /root/out/abo_{name}.log; "
                   f"grep -E 'Error|error|FAILED|Killed|Terminated|no config' /root/out/abo_{name}.log | tail -2)")
    if not out.strip():
        return None
    js, _, lg = out.partition("==LOG")
    try:
        j = json.loads(js)
    except ValueError:
        j = {}
    lines = lg.strip().splitlines()
    last = (next((l for l in lines if " error: " in l), None) or (lines[-1] if lines else "no error in the log (timeout or killed)"))[:200]
    if "geomean" in j:
        return "measured", j
    if "failed" in j:
        return "infeasible", "check: " + j["failed"][:200]
    if re.search(r"build did not finish|internal compiler error|Segmentation fault|\bKilled\b|signal 9", lg):   # the box's
        return "no-result", "ours: " + last
    if re.search(r"errors detected in the compilation|Ptx assembly aborted|no config|AssertionError", lg):
        return "infeasible", "compile: " + last
    return "no-result", last if lines else f"killed at the bench's {vast_hosts.LIMIT} s limit (a hang) or by the box"


def ident(name, host=None):
    f = ROOT / f"candidates/layer_{name}.py"
    return {"run": name, "cand_md5": identity.file_md5(f) if f.exists() else None, "against_md5": AGAINST_MD5,
            "mode": MODE, "seeds": SEEDS, "host": host}


def case_studies(fp):
    """one TPE study per scored case over that case's switches (the owner: TPE learns each case's values from that case's
    own outcome), per baseline fingerprint fp (an adoption starts fresh ones): its trials are the per-case draws
    of the main study's trials, told the case's matched delta (tell_cases)"""
    return {c: optuna.create_study(study_name=f"{STUDY_NAME}_{fp[:8]}_{c}", storage=CASE_STORAGE, direction="maximize",
                                   load_if_exists=True, sampler=optuna.samplers.TPESampler(
                                       n_startup_trials=8, multivariate=True, constant_liar=True, seed=None))
            for c in CASES}


def tpe_draw(cs, space, bl, lost):
    """each case's switches proposed by its case study (one categorical per switch: its values in the case's space); a
    case's proposal that is a setting tried and lost there (percase_losers) is told FAIL and asked again. ({case: values},
    {case: its case study's trial number})"""
    d, nums = {}, {}
    for c, sw in space.items():
        if not sw:
            continue
        dists = {k: optuna.distributions.CategoricalDistribution([json.dumps(v) for v in vs]) for k, vs in sw.items()}
        for _ in range(20):
            tc = cs[c].ask(dists)
            d[c] = {k: json.loads(v) for k, v in tc.params.items()}
            if (c, json.dumps({k: v for k, v in d[c].items() if v != bl[c][k]}, sort_keys=True)) not in lost:
                break
            cs[c].tell(tc.number, state=optuna.trial.TrialState.FAIL)
        nums[c] = [cs[c].study_name, tc.number]
    return d, nums


def tell_cases(cs, trial_attrs, outcome, cases, cohort=None):
    """the main trial's per-case draws told to their case studies: measured -> the case's matched delta, its log speedup
    less the log of the incumbent it was drawn on there, that incumbent's established number (its own confirmation
    reading: the owner, 11-05; raw speedups across baselines teach TPE drift); none, or not measured: FAIL"""
    inc = trial_attrs.get("incumbent")
    rd = confirm.reading(inc) if outcome == "measured" and inc else None   # (its established number, the owner)
    ctl = [rd] if rd else []
    for c, sn_n in (trial_attrs.get("case_trials") or {}).items():
        sn, n = sn_n if isinstance(sn_n, list) else (f"{STUDY_NAME}_{c}", sn_n)
        try:
            st = cs[c] if c in cs and cs[c].study_name == sn else optuna.load_study(
                study_name=sn, storage=CASE_STORAGE if sn.count("_") >= 2 else STORAGE)
            base = [x[CASES[c]] for x in ctl if CASES[c] in x]
            if cases and CASES[c] in cases and base:
                st.tell(n, math.log(cases[CASES[c]]) - math.log(statistics.mean(base)))
            else:
                st.tell(n, state=optuna.trial.TrialState.FAIL)
        except (RuntimeError, ValueError, KeyError):   # (told already: a restart's re-read)
            pass


def finish(study, t, name, p, outcome, value, idn, why=None, extra=None):
    """tell the study and append the record: measured -> value; infeasible -> constraint (0.0, never read); no-result -> FAIL"""
    st = study._storage
    for k, v in {**idn, "outcome": outcome, **({"why": why} if why else {}), **(extra or {})}.items():
        st.set_trial_user_attr(t._trial_id, k, v)
    if outcome == "no-result":
        study.tell(t.number, state=optuna.trial.TrialState.FAIL)
    else:
        optuna.trial.Trial(study, t._trial_id).set_constraint("feasible", 0.0 if outcome == "measured" else 1.0)
        study.tell(t.number, value if outcome == "measured" else 0.0)
    tell_cases(CS, study._storage.get_trial(t._trial_id).user_attrs, outcome, (extra or {}).get("cases"), idn.get("host"))
    rec = {"study": study.study_name, "trial": t.number, "name": name, "params": p, "status": outcome, **idn, "t": round(time.time())}
    rec.update({"geomean": value} if outcome == "measured" else {"result": why})
    log.write(json.dumps({**rec, **(extra or {})}) + "\n")
    log.flush()


def alive(host, name):
    """the job will still report: with the relay (a ship in transit included), queued on the GPU box, running, or done"""
    try:
        if json.loads((RELAY / "state.json").read_text()).get(name, {}).get("phase") in ("queued", "building", "built", "shipping"):
            return True
    except (OSError, ValueError):
        pass
    return bool(sh(host, f"grep -l abo_{name}[.] /root/queue*.txt 2>/dev/null; ls /root/out/abo_{name}.done 2>/dev/null; "
                         f"pgrep -f 'layer_{name}[.]py' 2>/dev/null | head -1").strip())


def lost(study, t, why):
    """a trial whose bench will never report: FAIL with the reason (its cases' configurations stay unscreened, so they
    are drawn again)"""
    study._storage.set_trial_user_attr(t._trial_id, "outcome", "no-result")
    study._storage.set_trial_user_attr(t._trial_id, "why", why)
    study.tell(t.number, state=optuna.trial.TrialState.FAIL)
    tell_cases(CS, study._storage.get_trial(t._trial_id).user_attrs, "no-result", None)
    print(f"{study.study_name} #{t.number}: {why}; FAIL", flush=True)


def ctl_level(h):
    """the incumbent's control due on GPU h: only the hourly one, on one GPU an hour across all (the owner's Nov 3 rule,
    11-05: no per-promotion or per-GPU baseline controls; a screen is relative already), at the lowest level 4; or None"""
    if vast_hosts.boxes().get(h.rsplit("g", 1)[0], {}).get("offline") or h != vast_hosts.gpus()[0]:
        return None
    last = max((r["t"] for r in map(json.loads, open(CTL))), default=0)
    return 4 if time.time() - max(last, max(ctl_t.values(), default=0)) >= CTL_EVERY else None


def triage(name, host, r):
    """a measured screen against the incumbent's controls on its GPU: a confirmation, nothing, or (no baseline yet)
    unclassified, read again each round"""
    over, won = confirm.ahead(confirm.controls(host), r)
    if over == "unclassified":
        unread[name] = (host, r)
        UNREAD.write_text(json.dumps(unread))
        print(f"{name} unclassified: no baseline of {incumbent()['name']} on {host} yet", flush=True)
    elif over == "ahead":   # one 400-rep confirmation on this GPU, ahead of every routine job: only a trial that screens
        # ahead of the incumbent overall (the owner: "leaderboard top only"; no per-case or within-noise trigger)
        print(f"{name} screens ahead overall (cases ahead: {', '.join(won) or 'none'}): confirmation", flush=True)
        subprocess.run([sys.executable, str(ROOT / "harness/confirm.py"), "add", name, host], cwd=ROOT)


STUDY_NAME = "clean1"
READY = int(ARGS[0]) if ARGS and ARGS[0].isdigit() else 8

# the switches: values (the first is the clean default) and how each value applies
DIMS = {
    "p05": [0, 1],           # SK2_P05: retimed sk2 ring depths (port list p05)
    "sk2bf16p": [0, 1],      # SK2_BF16P: sk2 k-warp partials as BF16
    "dfpoll": [0, 1],        # MOE_DFPOLL: DF phase-B poll as packed arithmetic
    "spoll2": [0, 1],        # MOE_SPOLL2: retimed phase-B poll sleeps, untuned DF classes
    "wsblk_t2": [0, 1],      # MOE_WSBLK=2 on TINY2 (port list kwsblk tiny2_2)
    "wsblk_m1": [0, 1],      # MOE_WSBLK=1 on MID (port list kwsblk mid_1)
    "ts2": [0, 1],           # MOE_TANHSILU=2: phase-A SiLU via tanh.approx, its x < -8 tail exact (value 1 broke
                             # the BF16 accuracy rule there and is withdrawn: receipts/eat/eat_silu.txt)
    "bf16red": [0, 1],       # MOE_BF16RED: small MoE cross-warp partials as BF16
    "sk2c7": [0, 1],         # SK2_C7: the sk2 7-op codec layout (the layer constant goes with the flag)
    "rqc7": [0, 1],          # RQ_C7: rq's aligned exponent window, as SK2_C7
    "stages": [3, 4],        # ATT_FLAT_DECODE_STAGES: the flat decode's ring depth (2 quarantined)
    "vunroll": [2, 1, 4],    # ATT_FLAT_VERIFY_UNROLL: the flat verify's key-loop unroll
    "nsplit16": [11, 8, 9],  # ATTENTION_TUNED[(16, 8, True, 1)] n_splits
    "v32s": [4, 5, 6, 8],    # ATTENTION_TUNED[(32, 8, True, 1)] n_splits
    "sdef": [0, 1],          # MOE_SDEF: the t1 MoE builds' deferred NVFP4 block scales (a layer constant only)
    "brel": [0, 1],          # MOE_BREL: the skinny grid barrier's arrive as atom.add.release.gpu, its two fence.sc.gpu gone
    "mid16": [0, 1],         # MOE_MID16 on moe_d16s: one <=16-token MoE job per expert (d16f stays off: m16i -0.6%)
    "rul_moes": [5, 3, 7],      # ptxas --register-usage-level on MoE skinny (TINY / TINY2 / MID) (5 is its default)
    "dlc_moes": ["none", "cg", "ca"],   # ptxas --def-load-cache on MoE skinny (TINY / TINY2 / MID) (none: its default)
    "rul_moew": [5, 3, 7],      # ptxas --register-usage-level on MoE WGS (d32 and big) (5 is its default)
    "dlc_moew": ["none", "cg", "ca"],   # ptxas --def-load-cache on MoE WGS (d32 and big) (none: its default)
    "rul_att": [5, 3, 7],      # ptxas --register-usage-level on attention (5 is its default)
    "dlc_att": ["none", "cg", "ca"],   # ptxas --def-load-cache on attention (none: its default)
    "rul_sk2": [5, 3, 7],      # ptxas --register-usage-level on sk2 / sk2_narrow / sk2_gle (5 is its default)
    "dlc_sk2": ["none", "cg", "ca"],   # ptxas --def-load-cache on sk2 / sk2_narrow / sk2_gle (none: its default)
    "rul_rq": [5, 3, 7],      # ptxas --register-usage-level on rq128 / rq256 (5 is its default)
    "dlc_rq": ["none", "cg", "ca"],   # ptxas --def-load-cache on rq128 / rq256 (none: its default)
    "sc6s": [0, 1],          # MOE_SC6S: skinny unit copies' scales as three six-bit planes (a layer constant:
                             # prepare packs the planes and adds -DMOE_SC6S=1 to the skinny builds itself)
    "gle_t8": [0, 1],        # GLE_T8: the verify-1 shapes on the T = 1 chain, boundary 1 in o's epilogue and the GLE pooler
                             # on the 16 u/v rows (a layer constant: it selects sk2_gle_v1g itself)
    "sk2_ce": [0, 15],       # SK2_CE: codec E in sk2, a bit mask 1 d1, 2 v1, 4 d16, 8 d32 (a layer constant:
                             # prepare packs the streams and selects the -DSK2_CE=1 builds itself)
    "rq_pool": [0, 1, 2],    # RQ_POOL: the T=128 pooler through rq128 and the router as its own FP32 GEMM (a
                             # layer constant, prepare only)
    "rq_o": [0, 1],          # RQ_O: the T=128 o projection through rq128 12-bit (a layer constant)
    # per-shape constants of the clean source, each default the value the incumbent builds (its raw flags), so a
    # draw at the default writes nothing
    "ca_d32s": [2, 9, 5, 12],   # MOE_CA on moe_d32s: the WGS phase-A cost table (raw: 9, then 2)
    "ca_d32f": [7, 2, 9, 11],   # MOE_CA on moe_d32f (raw: 7)
    "ca_big": [0, 2, 7, 9],     # MOE_CA on moe_big (v16 / v32; the source table)
    "brev": [0, 1],             # MOE_BREV: d32 phase-B jobs in reverse order (d32f -0.2..-1.0 us, d32s +0.4..0.9: per case)
    "rqst128": [5, 4, 6, 8],    # RQ_STAGES on rq128: ring slots (28 KB each, 8 fit)
    "rqst256": [5, 4, 3],       # RQ_STAGES on rq256 (44 KB each, 5 fit)
    "rq_wpre": [0, 1],          # RQ_WPRE: the W stream's first slots issued before the PDL wait
    "xslot_t1": [2, 1, 0],      # MOE_XSLOT on moe_d1s / moe_d1f (raw: 2)
    "xslot_v1f": [0, 1],        # MOE_XSLOT on moe_v1f (1: the 4th ring slot from the last phase-A unit)
    "elect": [0, 1],            # SK2_ELECT where sk2 runs without it (d16s / d32 have it: sk2_d16_32)
    "pdle": [0, 1],             # MOE_PDLE: the MoE's PDL trigger at entry (global -0.4%: per case)
}
CONSTS = {   # name -> the layer text a value sets (the host side of a switch; {} is the value; the first value's is the clean text)
    "sk2c7": "\nSK2_C7 = {}",
    "rqc7": "\nRQ_C7 = {}",
    "nsplit16": "\n    (16, 8, True, 1): dict(n_splits={},",
    "v32s": "\n    (32, 8, True, 1): dict(n_splits={},",
    "sdef": "\nMOE_SDEF = {}",
    "sc6s": "\nMOE_SC6S = {}",
    "gle_t8": "\nGLE_T8 = {}",
    "sk2_ce": "\nSK2_CE = {}",
    "rq_pool": "\nRQ_POOL = {}",
    "rq_o": "\nRQ_O = {}",
}
SK2 = ("sk2",)
SK2_ALL = ("sk2", "sk2_narrow", "sk2_gle")
SMALL = ("MOE_TINY", "MOE_TINY2", "MOE_MID")
APPLY = {   # name -> (where: BUILDS keys, or the MOE_SWITCHES a moe_ class must have one of, flags)
    "p05": (SK2, ["-DSK2_P05=1"]),
    "sk2bf16p": (SK2_ALL, ["-DSK2_BF16P=1"]),
    "dfpoll": (("MOE_DF",), ["-DMOE_DFPOLL=1"]),
    "spoll2": (("MOE_DF",), ["-DMOE_SPOLL2=1"]),
    "wsblk_t2": (("MOE_TINY2",), ["-DMOE_WSBLK=2"]),
    "wsblk_m1": (("MOE_MID",), ["-DMOE_WSBLK=1"]),
    "ts2": (SMALL, ["-DMOE_TANHSILU=2"]),
    "bf16red": (SMALL, ["-DMOE_BF16RED=1"]),
    "sk2c7": (SK2_ALL, ["-DSK2_C7=1"]),
    "brel": (SMALL, ["-DMOE_BREL=1"]),
    "mid16": (("moe_d16s",), ["-DMOE_MID16=1"]),
    "rqc7": (("rq128", "rq256"), ["-DRQ_C7=1"]),
    "stages": (("attention",), ["-DATT_FLAT_DECODE_STAGES={}"]),
    "vunroll": (("attention",), ["-DATT_FLAT_VERIFY_UNROLL={}"]),
    "rul_moes": (SMALL, ["-Xptxas", "--register-usage-level={}"]),
    "dlc_moes": (SMALL, ["-Xptxas", "--def-load-cache={}"]),
    "rul_moew": (("MOE_WGS",), ["-Xptxas", "--register-usage-level={}"]),
    "dlc_moew": (("MOE_WGS",), ["-Xptxas", "--def-load-cache={}"]),
    "rul_att": (("attention",), ["-Xptxas", "--register-usage-level={}"]),
    "dlc_att": (("attention",), ["-Xptxas", "--def-load-cache={}"]),
    "rul_sk2": (SK2_ALL, ["-Xptxas", "--register-usage-level={}"]),
    "dlc_sk2": (SK2_ALL, ["-Xptxas", "--def-load-cache={}"]),
    "rul_rq": (("rq128", "rq256"), ["-Xptxas", "--register-usage-level={}"]),
    "dlc_rq": (("rq128", "rq256"), ["-Xptxas", "--def-load-cache={}"]),
    "ca_d32s": (("moe_d32s",), ["-DMOE_CA={}"]),
    "ca_d32f": (("moe_d32f",), ["-DMOE_CA={}"]),
    "ca_big": (("moe_big",), ["-DMOE_CA={}"]),
    "brev": (("MOE_B4FIN",), ["-DMOE_BREV={}"]),
    "rqst128": (("rq128",), ["-DRQ_STAGES={}"]),
    "rqst256": (("rq256",), ["-DRQ_STAGES={}"]),
    "rq_wpre": (("rq128", "rq256"), ["-DRQ_WPRE={}"]),
    "xslot_t1": (("moe_d1s", "moe_d1f"), ["-DMOE_XSLOT={}"]),
    "xslot_v1f": (("moe_v1f",), ["-DMOE_XSLOT={}"]),
    "elect": (SK2, ["-DSK2_ELECT={}"]),
    "pdle": (("MOE_TINY", "MOE_TINY2", "MOE_MID", "MOE_WGS"), ["-DMOE_PDLE={}"]),
}
SCOPE = {"elect": {"d1s", "d1f", "v1s", "v1f", "d16f"}}   # switches the study draws only on these cases (elsewhere a
#                                                          build variant already sets them)
HARVEST_EVERY = 25   # completed trials between harvests (harness/harvest.py)
WITHDRAWN = {"tanhsilu": 1}   # values out of the study: their trials still out are recorded, never confirmed
INC_POINT = {'p05': 1, 'sk2bf16p': 1, 'dfpoll': 0, 'spoll2': 0, 'wsblk_t2': 0, 'wsblk_m1': 0, 'ts2': 1, 'bf16red': 0, 'sk2c7': 0, 'rqc7': 0, 'stages': 3, 'vunroll': 2, 'nsplit16': 11, 'v32s': 4, 'sdef': 0, 'rul_moes': 5, 'dlc_moes': 'none', 'rul_moew': 7, 'dlc_moew': 'ca', 'rul_att': 5, 'dlc_att': 'ca', 'rul_sk2': 7, 'dlc_sk2': 'cg', 'rul_rq': 7, 'dlc_rq': 'none', 'mid16': 1}   # hk_m16j's switches (arch5_c1_1784 + mid16): the incumbent's point when it
# is a lane candidate (a promoted study trial's are its own parameters: inc_point())
ANCHOR = "\n_MODULES = {}\n"   # BUILDS ends right before it


def clean_tip(at="clean", tree=ROOT.parent / "okc-clean-study"):
    """the clean branch's (or a lane commit's, in its own tree) assembled file and its commit"""
    rev = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", at], capture_output=True, text=True).stdout.strip()
    if not tree.exists():
        subprocess.run(["git", "-C", str(ROOT), "worktree", "add", "-q", "--detach", str(tree), at], check=True)
    subprocess.run(["git", "-C", str(tree), "checkout", "-q", "--detach", at], check=True)
    sys.path.insert(0, str(tree / "clean"))
    import importlib
    import assemble
    importlib.reload(assemble)
    return assemble.assemble(), rev


def assembled_point(text):
    """the switches a file make_clean assembled was built with, read back from its text: each switch's flags (the lines
    it wrote end in its name; a MoE switch's flags are three lines below) or its layer text (CONSTS); a switch with
    neither is at its clean default"""
    head = "\n# the study's switches for this trial\n"
    lines = text.split(head, 1)[1].split(ANCHOR, 1)[0].splitlines() if head in text else []
    p = {}
    for k, vs in DIMS.items():
        if k not in APPLY:
            p[k] = next((v for v in vs if CONSTS[k].format(v) in text), None)
            continue
        at = next((i for i, l in enumerate(lines) if l.endswith(f"# {k}")), None)
        if at is None:
            p[k] = vs[0]
            continue
        line = lines[at + 3] if lines[at].startswith("for _c") else lines[at].rsplit("   #", 1)[0]
        flags = ast.literal_eval(line[line.rindex("+ ") + 2:].rstrip(")"))
        p[k] = next((v for v in vs[1:] if [f.format(v) for f in APPLY[k][1]] == flags), None)   # (written only off the default)
    return p


def make_clean(name, p, text, raw=()):
    """candidates/layer_<name>.py: the clean tip with the trial's switches (and a lane's raw (build, flag) pairs); True
    when written"""
    lines = [f'BUILDS[{b!r}] = (BUILDS[{b!r}][0], BUILDS[{b!r}][1], BUILDS[{b!r}][2] + [{f!r}])   # raw' for b, f in raw]
    for k, v in sorted(p.items()):
        if v == DIMS[k][0] or k not in APPLY:
            continue
        where, flags = APPLY[k]
        flags = [f.format(v) for f in flags]   # (a many-valued switch's flag carries its value)
        if where[0].startswith("MOE_"):
            lines.append(f'for _c, _sw in MOE_SWITCHES.items():   # {k}\n'
                         f'    if set(_sw) & set({sorted(where)!r}):\n'   # (sorted: a set's repr follows the process's hash seed)
                         f'        _s, _f, _x = BUILDS["moe_" + _c]\n'
                         f'        BUILDS["moe_" + _c] = (_s, _f, _x + {flags!r})')
        else:
            for b in where:
                lines.append(f'BUILDS[{b!r}] = (BUILDS[{b!r}][0], BUILDS[{b!r}][1], BUILDS[{b!r}][2] + {flags!r})   # {k}')
    for k, form in CONSTS.items():
        if p.get(k, DIMS[k][0]) != DIMS[k][0]:
            assert text.count(form.format(DIMS[k][0])) == 1, f"{k}: its layer text moved"
            text = text.replace(form.format(DIMS[k][0]), form.format(p[k]), 1)
    if lines:
        assert text.count(ANCHOR) == 1, "the BUILDS anchor moved"
        text = text.replace(ANCHOR, "\n# the study's switches for this trial\n" + "\n".join(lines) + "\n" + ANCHOR, 1)
    (ROOT / f"candidates/layer_{name}.py").write_text(text)
    return True


def main():
    global _LOCK   # (one study driver at a time: two restarted together both proposed and dispatched, 05:01 UTC)
    _LOCK = open(pathlib.Path(os.environ.get("OKC_STATE", "~/.okc")).expanduser() / "clean_study.lock", "a")
    try:
        fcntl.flock(_LOCK, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        sys.exit("another clean_study driver runs")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(study_name=STUDY_NAME, storage=STORAGE, direction="maximize", load_if_exists=True,
                                sampler=optuna.samplers.TPESampler(n_startup_trials=8, multivariate=True, group=True,
                                                                   constant_liar=True, seed=None))
    global CS
    text, rev = clean_tip()
    print(f"clean study {STUDY_NAME}: clean {rev}, {len(DIMS)} switches, ready {READY}", flush=True)
    hosts = vast_hosts.load(offline=True)[0]   # (each box asked once, not each trial on each GPU: 64 trials took minutes)
    snap, relay = {}, {}
    try:
        relay = json.loads((RELAY / "state.json").read_text())
    except (OSError, ValueError):
        pass
    for h in hosts:
        if hosts[h][:2] not in snap:
            out = sh(h, "grep -oh 'abo_[A-Za-z0-9_]*[.]' /root/queue*.txt 2>/dev/null; ls /root/out 2>/dev/null | "
                        "grep -E '^abo_.*[.]done$'; pgrep -af 'layer_[A-Za-z0-9_]*[.]py' 2>/dev/null | "
                        "grep -o 'layer_[A-Za-z0-9_]*[.]py'")
            snap[hosts[h][:2]] = {re.sub(r"^(abo_|layer_)|[.](done|py)?$", "", x) for x in out.split()}
    up = lambda h, n: relay.get(n, {}).get("phase") in ("queued", "building", "built", "shipping") and relay[n].get("gpu") == h \
        or n in snap.get(hosts[h][:2], set())
    for t in study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.RUNNING,)):   # (a restart: adopt what is out)
        name = f"arch5_c1_{t.number:04d}"
        host = next((h for h in hosts if up(h, name)), None)
        if host:
            inflight[name] = (study, t, dict(t.params), dict(t.params), host, time.time())
            md5_of[name] = hashlib.md5((ROOT / f"candidates/layer_{name}.py").read_bytes()).hexdigest()
            print(f"adopted {name} on {host}", flush=True)
        else:   # never dispatched or its host gone: FAIL (no phantom running trial for the constant liar), run again
            lost(study, t, "not on any host at the driver's start")
    recorded = {json.loads(l)["name"] for l in open(CTL)}
    md5s = {hashlib.md5((ROOT / json.loads(l)["file"]).read_bytes()).hexdigest(): json.loads(l)["name"] for l in open(confirm.INCS)
            if (ROOT / json.loads(l)["file"]).exists()}
    for f in sorted(ROOT.glob(f"candidates/layer_{STUDY_NAME}_ctl_*.py")):   # and its controls still out (else run again,
        name = f.stem.removeprefix("layer_")                                  # and their results never recorded)
        host = name.split("_ctl_")[1].rsplit("_", 2)[0]
        md5 = hashlib.md5(f.read_bytes()).hexdigest()
        if name not in recorded and md5 in md5s and host in hosts and up(host, name):
            inflight[name] = (None, None, {}, {}, host, time.time())
            ctl_of[name] = (md5s[md5], md5)
            print(f"adopted control {name} ({md5s[md5]}) on {host}", flush=True)
    rounds, space_state, space, g, bl, fp = 0, None, {}, {}, {}, None
    harvested = sum(1 for t in study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,)))
    while True:
        rounds += 1
        if rounds % 20 == 0:   # (the clean tip moves as the lanes land: reassemble)
            text, rev = clean_tip()
        for name, (s, t, p6, p, host, t0) in list(inflight.items()):
            host = submit.gpu_of(name, host)
            r = result(host, name)
            if r is None:
                continue
            del inflight[name]
            md5_of.pop(name, None)
            outcome, r = r
            if outcome == "measured":
                gpu_now[host] = host = confirm.cohort(host, r["identity"])
            if s is None:   # an incumbent control
                inc, want = ctl_of.get(name, (incumbent()["name"], incumbent()["md5"]))
                if outcome == "measured" and r["identity"]["sub_md5"] == want:
                    with open(CTL, "a") as f:
                        f.write(json.dumps({"name": name, "incumbent": inc, "host": host, "geomean": r["geomean"],
                                            "cases": confirm.cases(r), "t": round(time.time())}) + "\n")
                    print(f"{name} control {r['geomean']:.4f} on {host}", flush=True)
                continue
            idn = dict(ident(name, host), clean=t.user_attrs.get("clean"))
            if outcome == "measured":
                finish(study, t, name, p, "measured", r["geomean"], idn, extra={"cases": confirm.cases(r)})
                print(f"{name} {r['geomean']:.4f} on {host} {p}", flush=True)
                if not any(p.get(k) == v for k, v in WITHDRAWN.items()) and not t.user_attrs.get("case_draw"):
                    triage(name, host, r)   # (a per-case draw is data for its cases' winners: percase_confirm)
            else:
                finish(study, t, name, p, outcome, None, idn, why=r)
                print(f"{name} {outcome}: {str(r)[:120]}", flush=True)
        for name, (host, r) in list(unread.items()):
            if confirm.controls(host):
                del unread[name]
                UNREAD.write_text(json.dumps(unread))
                if not withdrawn(name):
                    triage(name, host, r)
        measured = sum(1 for t in study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,)))
        if measured - harvested >= HARVEST_EVERY:   # the harvest: each switch's effect per case, every 25 trials
            harvested = measured
            subprocess.Popen([sys.executable, str(ROOT / "harness/harvest.py"), "run"], cwd=ROOT, start_new_session=True,
                             stdout=open(ROOT / "receipts/harvest/harvest.log", "a"), stderr=subprocess.STDOUT)
        if rounds % 3 == 0:   # confirmations: collect (a verdict, a new incumbent; a substantial win's judge check)
            subprocess.run([sys.executable, str(ROOT / "harness/confirm.py"), "collect"], cwd=ROOT)
        # the incumbent's controls (its baseline first), then trials up to READY in flight
        for target in vast_hosts.gpus():
            lvl = ctl_level(target)
            if lvl is None:
                continue
            inc = incumbent()
            src = (ROOT / inc["file"]).read_bytes()
            name = f"{STUDY_NAME}_ctl_{target}_{int(time.time())}_{len(ctl_of)}"
            (ROOT / f"candidates/layer_{name}.py").write_bytes(src)
            inflight[name] = (None, None, {}, {}, dispatch(name, target, lvl, checks=True), time.time())
            ctl_of[name] = (inc["name"], inc["md5"])
            ctl_t[target] = time.time()
        missing = [c for c in confirm.adopted()["commits"]
                   if subprocess.run(["git", "-C", str(ROOT), "merge-base", "--is-ancestor", c, rev]).returncode]
        if missing:   # (an adoption just landed on clean: the tip is re-read at once, not at the next 20-round refresh)
            text, rev = clean_tip()
            missing = [c for c in missing
                       if subprocess.run(["git", "-C", str(ROOT), "merge-base", "--is-ancestor", c, rev]).returncode]
        if missing:   # (an adopted lane commit not in the clean tip: no new trials until it lands, else promotions drop it)
            print(f"no new trials: adopted lane commits {missing} are not in clean {rev}", flush=True)
        base = confirm.adopted()   # (every adopted lane delta, fixed switches, raw flags, commits in the tip)
        f = ROOT / f"candidates/layer_{STUDY_NAME}_point.py"
        if (rev, json.dumps(base, sort_keys=True)) != space_state:   # (the tip or an adoption moved: the space afresh)
            space_state = (rev, json.dumps(base, sort_keys=True))
            g = {**inc_point(), **base["switches"]}   # (the global point: the incumbent's; per-case draws go on top)
            make_clean(f.stem.removeprefix("layer_"), g, text, [tuple(r) for r in base["raw"]])
            space, bl = case_space(f.read_text(), base), baselines(g, base)
            fp = hashlib.md5(f.read_bytes()).hexdigest()   # (the baseline's bytes: fixed switches, raw flags, clean source)
            CS = case_studies(fp)   # (a new baseline: fresh case studies)
            print(f"per-case space at clean {rev}: " + ", ".join(f"{c} {sorted(sw)}" for c, sw in space.items()), flush=True)
        if rounds % 10 == 0 and not missing:
            percase_confirm(study, space, bl, fp)
        tries, seen_md5s, losers = 0, measured_md5s(study), percase_losers()   # (once a round: each reads every trial)
        again = json.loads(REQUEUE.read_text()) if REQUEUE.exists() else []   # (trials that never measured: their draws
        while not missing and any(space.values()) and sum(1 for v in inflight.values() if v[0] is not None) < READY:
            tries += 1                                                        # re-expressed on the current baseline first)
            if tries > 3 * READY:   # (draws keep building files already measured: the space is covered for now)
                break
            of = again.pop(0) if again else None
            d, nums = (reexpress(of["draw"], space, bl), {}) if of else tpe_draw(CS, space, bl, losers)
            if of:
                REQUEUE.write_text(json.dumps(again))
            pc = {c: dict(v) for c, v in (base.get("per_case") or {}).items()}   # (the adopted overrides, then the draw's:
            for c, x in d.items():                                                 # any value off the global point, or
                over = {k: v for k, v in x.items() if v != g[k] or k in pc.get(c, {})}   # an adopted override's switch)
                if over:
                    pc.setdefault(c, {}).update(over)
            make_clean(f.stem.removeprefix("layer_"), g, text, [tuple(r) for r in base["raw"]])
            apply_per_case(f, pc)
            md5 = hashlib.md5(f.read_bytes()).hexdigest()
            seen = seen_md5s.get(md5) or [f"in flight as {n}" for n, x in md5_of.items() if x == md5]
            if seen:   # (a file already measured is never screened again; the incumbent's controls measure its own)
                print(f"draw {d}: builds {md5[:8]}, measured already {seen[:4]}: not screened again", flush=True)
                tell_cases(CS, {"case_trials": nums}, "duplicate", None)
                continue
            t = study.ask()   # (no parameters: its per-case draw, from the case studies, is its record)
            for k, v in {"fixed": g, "clean": rev, "case_draw": d, "baseline": fp, "case_trials": nums,
                         "incumbent": incumbent()["name"],
                         **({"requeue_of": of["of"]} if of else {})}.items():
                study._storage.set_trial_user_attr(t._trial_id, k, v)
            name = f"arch5_c1_{t.number:04d}"   # (arch5_: the relay balances it between the GPUs)
            f.rename(ROOT / f"candidates/layer_{name}.py")
            md5_of[name] = md5
            target = min(vast_hosts.gpus(), key=lambda h: sum(1 for v in inflight.values() if v[4] == h))
            inflight[name] = (study, t, g, g, dispatch(name, target, 4), time.time())
            print(f"{name} -> {target} (level 4)" + (f" again for trial {of['of']}" if of else "") + f" per case {d}", flush=True)
        time.sleep(20)


CASES = {f"{m}{b}{l}": f"{'decode' if m == 'd' else 'verify'} {b} {'sliding' if l == 's' else 'full'}"
         for m in "dv" for b in (1, 16, 32) for l in "sf"}
CODE = {"sdef", "mid16"}   # switches that select a lane's code (MOE_SDEF, MOE_MID16): fixed per case, as raw flags are
PER_CASE_N, PER_CASE_T = 6, 3.0   # a switch's value clearly ahead on a case: this many screens of it and of the
#                                   incumbent's value there, and Welch t on the case's log speedup at least this


def case_space(text, base):
    """per scored case: each switch that executes there (the per-case framework, clean/assemble.apply_case_switches,
    raises on one that does not) -> all its values. Adopted switches too (a global winner may lose on one shape),
    their adopted value the case's baseline (baselines); a lane's source or flag delta stays fixed (raw flags, and
    the switches that select lane code: CODE). A global loser may still win on one shape (the owner); a setting tried
    and lost on a case is excluded as a whole (percase_losers)"""
    asm, space = sys.modules["assemble"], {}
    for c in CASES:
        space[c] = {}
        for k, vs in DIMS.items():
            if k in CODE or c not in SCOPE.get(k, {c}):
                continue
            try:
                asm.apply_case_switches(text, {c: {k: vs[1]}}, DIMS, APPLY, set(WITHDRAWN))
            except ValueError:
                continue
            space[c][k] = vs
    return space


def baselines(g, base):
    """each case's baseline: the global point, under the adopted per-case overrides (composites' confirmed winners)"""
    return {c: {**g, **(base.get("per_case") or {}).get(c, {})} for c in CASES}


def percase_losers():
    """(case, setting) a per-case confirmation tried and lost on the current incumbent: each of its changed cases read
    against its own control there (the incumbent it was read against, on its GPU, at its time), and only a case behind
    on its own reading marked, as its exact setting (the plan's switches on that case): a losing case never erases a
    winning one, nor a losing switch its companion. A loss prunes only on the base it was measured on (the incumbent's
    bytes it was read against): after an adoption the same setting interacts differently. A cancelled
    confirmation has no result and marks nothing"""
    tried = ROOT / "receipts/evidence/percase_tried.jsonl"
    plans = {json.loads(l)["name"]: json.loads(l)["plan"] for l in open(tried)} if tried.exists() else {}
    recs = confirm.records()
    queued = {r["run"]: r for r in recs if r["event"] == "queued"}
    md5 = {json.loads(l)["name"]: hashlib.md5((ROOT / json.loads(l)["file"]).read_bytes()).hexdigest()
           for l in open(confirm.INCS) if (ROOT / json.loads(l)["file"]).exists()}
    lost = set()
    for r in recs:
        nm = r["run"].split("_", 1)[1] if r["event"] == "result" else None
        if nm not in plans or md5.get(r["against"]) != incumbent()["md5"]:
            continue
        key = r.get("cohort") or confirm.cohort(r.get("host") or queued[r["run"]]["host"], r["identity"])   # (before 11-05:
        d = confirm.controls(key, r["t"], r["against"])                                                    # where queued)
        for c, sw in plans[nm].items() if d else ():
            m, sd = d[1][CASES[c]]
            if r["cases"][CASES[c]] < m - confirm.SD * sd:
                lost.add((c, json.dumps(sw, sort_keys=True)))
    return lost


def reexpress(old, space, bl):
    """a recorded per-case draw on the current space: each case's switches at the recorded value where it drew one (and
    the value still exists), else at the case's baseline"""
    return {c: {k: old.get(c, {}).get(k, bl[c][k]) if old.get(c, {}).get(k, bl[c][k]) in vs else bl[c][k]
                for k, vs in sw.items()} for c, sw in space.items()}


def draw(space, bl, lost):
    """each case's switches, each value drawn uniformly and independently (so on a case one switch's values meet the
    others' values at random, and its comparison there is not confounded by them); a case's setting (its values off
    the case's baseline, bl) tried and lost there (percase_losers) is drawn again"""
    d = {}
    for c, sw in space.items():
        for _ in range(100):
            d[c] = {k: random.choice(vs) for k, vs in sw.items()}
            if (c, json.dumps({k: v for k, v in d[c].items() if v != bl[c][k]}, sort_keys=True)) not in lost:
                break
    return d


def case_winners(study, space, bl, fp):
    """per case and switch, from that case's own screens (not main effects over global configurations, which misled
    the d16 composite): the value clearly ahead of the case's baseline value there (bl: the adopted one; PER_CASE_N
    screens of each, Welch t >= PER_CASE_T on the case's log speedup), the best such -> {case: {switch: value}}; only
    screens drawn on the current baseline (fp: the md5 of the global point assembled with the raw flags from the clean
    source, so an adoption of a switch, a raw flag (HCRED) or a source change (279d5d5) each starts a new cohort), and
    on one GPU (gpu1g0, gpu1g1 and spot1g0 readings of one file never pool); a switch's best value over the GPUs'
    cohorts, each read within its own"""
    by = {}
    for t in study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,)):
        if t.user_attrs.get("case_draw") and t.user_attrs.get("outcome") == "measured" and t.user_attrs.get("baseline") == fp:
            for c, x in t.user_attrs["case_draw"].items():
                for k, v in x.items():
                    by.setdefault((t.user_attrs.get("host"), c, k), {}).setdefault(json.dumps(v), []).append(
                        math.log(t.user_attrs["cases"][CASES[c]]))
    plan, best_of = {}, {}
    for (h, c, k), values in sorted(by.items(), key=lambda kv: (str(kv[0][0]), kv[0][1], kv[0][2])):
        inc = values.get(json.dumps(bl[c][k]), [])
        best = None
        for key, xs in values.items():
            v = json.loads(key)
            if v == bl[c][k] or v not in space.get(c, {}).get(k, ()) or min(len(xs), len(inc)) < PER_CASE_N:
                continue
            diff = statistics.mean(xs) - statistics.mean(inc)
            t = diff / math.sqrt(statistics.variance(xs) / len(xs) + statistics.variance(inc) / len(inc))
            if t >= PER_CASE_T and (best is None or diff > best[0]):
                best = (diff, t, v, len(xs), len(inc))
        if best and best[0] > best_of.get((c, k), (-1,))[0]:
            best_of[(c, k)] = best
            plan.setdefault(c, {})[k] = best[2]
            print(f"{c}: {k} {best[2]} ahead of the baseline's {bl[c][k]} there on {h} by {100 * best[0]:+.2f}% "
                  f"(t {best[1]:.1f}, n {best[3]} / {best[4]})", flush=True)
    lost = percase_losers()
    for c in [c for c in plan if (c, json.dumps(plan[c], sort_keys=True)) in lost]:   # (tried there and lost)
        print(f"{c}: {plan.pop(c)} was tried there and lost: not proposed again", flush=True)
    return plan


def percase_confirm(study, space, bl, fp):
    """the cases' clear winners (case_winners), assembled through lane() (composite: recorded, so an adoption carries
    them) and confirmed targeted on those cases; one such confirmation at a time, each plan once"""
    plan = case_winners(study, space, bl, fp)
    tried = ROOT / "receipts/evidence/percase_tried.jsonl"
    done = [json.loads(l)["plan"] for l in open(tried)] if tried.exists() else []
    out = {r["run"] for r in confirm.records() if r["event"] != "queued"}
    if not plan or plan in done or any(r["event"] == "queued" and r["run"].split("_", 1)[1].startswith("pc")
                                      and r["run"] not in out for r in confirm.records()):
        return
    name = f"pc{len(done) + 1}"
    path = ROOT / f"receipts/harvest/{name}_plan.json"
    path.write_text(json.dumps({"changes": plan, "from": "case_winners", "t": round(time.time())}, indent=1))
    r = subprocess.run([sys.executable, str(ROOT / "harness/clean_study.py"), "composite", name, "clean", str(path)],
                       cwd=ROOT, capture_output=True, text=True)
    if r.returncode:
        print(f"{name}: assembly failed: {(r.stdout + r.stderr)[-300:]}", flush=True)
        return
    with open(tried, "a") as fh:
        fh.write(json.dumps({"name": name, "plan": plan, "t": round(time.time())}) + "\n")
    target = vast_hosts.gpus()[0]   # (the primary GPU: a confirmation must finish between stops)
    confirm.add(name, target, None, changed=[CASES[c] for c in plan])
    print(f"{name}: the cases' winners {plan} confirmed targeted on {target}", flush=True)


def measured_md5s(study):
    """each built file already measured -> its results: the study's screens, and each incumbent's controls"""
    m = {}
    for t in study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,)):
        if t.user_attrs.get("outcome") == "measured" and t.user_attrs.get("cand_md5"):
            m.setdefault(t.user_attrs["cand_md5"], []).append(round(t.value, 4))
    inc = {json.loads(l)["name"]: hashlib.md5((ROOT / json.loads(l)["file"]).read_bytes()).hexdigest()
           for l in open(confirm.INCS) if (ROOT / json.loads(l)["file"]).exists()}
    for l in open(CTL):
        r = json.loads(l)
        if r["incumbent"] in inc:
            m.setdefault(inc[r["incumbent"]], []).append(round(r["geomean"], 4))
    return m


def withdrawn(name):
    """a study trial (arch5_c1_<n>) whose switches include a withdrawn value"""
    if not name.startswith("arch5_c1_"):
        return False
    t = optuna.load_study(study_name=STUDY_NAME, storage=STORAGE).trials[int(name.rsplit("_", 1)[1])]
    return any(t.params.get(k) == v for k, v in WITHDRAWN.items())


def inc_point(name=None):
    """an incumbent's switches (default: the current one's): a promoted study trial's parameters; a lane candidate's,
    its incumbent.json "point", else the point of the incumbent it was adopted over ("point_of", confirm.py), else
    INC_POINT (kept by hand); a switch added since, at its default"""
    return {**{k: vs[0] for k, vs in DIMS.items()}, **_inc_point(name)}


def _inc_point(name=None):
    inc = incumbent()
    name = name or inc["name"]
    if name.startswith("arch5_c1_"):
        t = optuna.load_study(study_name=STUDY_NAME, storage=STORAGE).trials[int(name.rsplit("_", 1)[1])]
        return {k: v for k, v in {**t.user_attrs.get("fixed", {}), **t.params}.items() if k in DIMS}
    if name == inc["name"] and "point" in inc:
        return inc["point"]
    if name == inc["name"] and inc.get("point_of"):
        return _inc_point(inc["point_of"])
    return INC_POINT


def lane(name, at, args, per_case=True, correctness=False):
    """a lane's candidate: commit <at> assembled at the incumbent's switches plus <args>: switch=value, or build:flag for
    a flag no switch has (a lane commit's own macro); its md5 and the same point without them (its parent).
    correctness: a contract-required fix, adopted at its confirmation whatever it reads (confirm.py)"""
    missing = [c for c in confirm.adopted()["commits"] if subprocess.run(["git", "-C", str(ROOT), "merge-base", "--is-ancestor",
                                                                             c, at]).returncode]
    if missing:   # (a lane commit without an adopted delta's source would carry its flags but not its code)
        raise SystemExit(f"{at} lacks adopted lane commits {missing}: rebase it on the clean tip")
    text, rev = clean_tip(at, ROOT.parent / "okc-lane-assemble")   # (not the running study's tree)
    p = {k: vs[0] for k, vs in DIMS.items()}
    p.update({k: type(DIMS[k][0])(v) for k, v in inc_point().items()})
    base = confirm.adopted()   # (the adopted lane deltas: their switches are in the incumbent's point, their raw flags here)
    p.update(base["switches"])
    built = dict(p)   # (the point both are built at: an incumbent can lack an adopted switch that lane() carries)
    carried = [tuple(r) for r in base["raw"]]
    make_clean(name + "_parent", p, text, carried)
    raw = [tuple(a.split(":", 1)) for a in args if ":" in a]
    p.update({k: type(DIMS[k][0])(v) for k, v in (a.split("=", 1) for a in args if ":" not in a)})
    make_clean(name, p, text, carried + [r for r in raw if r not in carried])
    if per_case:   # (the adopted composites' per-case overrides on both: the lane's delta is its own)
        for x in (name + "_parent", name):
            apply_per_case(ROOT / f"candidates/layer_{x}.py", base.get("per_case"))
    rec = {"name": name, "clean": rev, "incumbent": incumbent()["name"], "point": built, "args": args,
           "switches": {k: type(DIMS[k][0])(v) for k, v in (a.split("=", 1) for a in args if ":" not in a)},
           "raw": raw, "t": round(time.time()), **({"correctness": True} if correctness else {}),
           "md5": hashlib.md5((ROOT / f"candidates/layer_{name}.py").read_bytes()).hexdigest(),
           "parent_md5": hashlib.md5((ROOT / f"candidates/layer_{name}_parent.py").read_bytes()).hexdigest()}
    with open(ROOT / "receipts/evidence/lane_assembly.jsonl", "a") as f:   # (what a lane candidate was assembled on, kept: an
        f.write(json.dumps(rec) + "\n")                                    # incumbent can change between asking and assembly)
    print(json.dumps(rec))


def apply_per_case(path, per_case):
    """the per-case overrides (adopted composites', or a plan's) on an assembled layer, through the clean tree's own
    assemble.apply_case_switches; a tree without it cannot carry them (raise: never a silent drop)"""
    if not per_case:
        return
    asm = sys.modules.get("assemble")
    if asm is None or not hasattr(asm, "apply_case_switches"):
        raise RuntimeError(f"per-case overrides {per_case} need clean/assemble.apply_case_switches in this tree")
    path.write_text(asm.apply_case_switches(path.read_text(), per_case, DIMS, APPLY, set(WITHDRAWN)))


def composite(name, at, plan, drop=()):
    """the per-shape composite: commit <at> assembled at the incumbent's point (lane(), recorded), then each case's
    switch overrides from a composite plan (harness/composite.py: its "changes") applied by the clean tree's own
    assemble.apply_case_switches (the seam); drop: "case:switch" pairs held out (awaiting a decision)"""
    lane(name, at, [], per_case=False)
    per_case = {c: {k: v for k, v in sw.items() if f"{c}:{k}" not in drop}
                for c, sw in json.loads(pathlib.Path(plan).read_text())["changes"].items()}
    per_case = {c: sw for c, sw in per_case.items() if sw}
    merged = {c: dict(v) for c, v in (confirm.adopted().get("per_case") or {}).items()}   # (the adopted ones, the plan's
    for c, v in per_case.items():                                                       # on top)
        merged.setdefault(c, {}).update(v)
    f = ROOT / f"candidates/layer_{name}.py"
    apply_per_case(f, merged)
    base = [json.loads(l) for l in open(ROOT / "receipts/evidence/lane_assembly.jsonl") if json.loads(l)["name"] == name][-1]
    rec = {"name": name, "clean": base["clean"], "incumbent": base["incumbent"], "point": base["point"], "args": [],
           "switches": {}, "raw": base["raw"], "composite_of": str(plan), "per_case": per_case, "dropped": sorted(drop),
           "md5": hashlib.md5(f.read_bytes()).hexdigest(), "t": round(time.time())}   # (adoptable: confirm.py reads these)
    with open(ROOT / "receipts/evidence/lane_assembly.jsonl", "a") as g:
        g.write(json.dumps(rec) + "\n")
    print(json.dumps(rec))


if __name__ == "__main__":
    if ARGS[:1] == ["composite"]:   # python3 harness/clean_study.py composite <name> <commit> <plan.json> [case:switch ...]
        composite(ARGS[1], ARGS[2], ARGS[3], tuple(ARGS[4:]))
    elif ARGS[:1] == ["lane"]:   # python3 harness/clean_study.py lane <name> <commit> [--correctness] [switch=value | build:flag ...]
        lane(ARGS[1], ARGS[2], [a for a in ARGS[3:] if a != "--correctness"], correctness="--correctness" in ARGS[3:])
    else:
        main()
