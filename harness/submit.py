"""Every GPU job enters one flow (owner): submit -> the relay's build queue -> a build box -> ship -> its GPU's queue -> a
runner (a screen goes to the slot's bench server). Its priority level is set here and kept end to end: the relay starts
builds and ships by level, then age; the GPU's runner takes the lowest level first, then the oldest.
    0  a new incumbent's first controls (its baseline: every reading against it waits for them)
    1  confirmations and per-shape work
    2  lane seeds and lane jobs (timelines, nsys, compile-only)
    3  sweep entries
    4  TPE draws and the hourly controls (bounded flows above: nothing here waits for ever)

    python3 harness/submit.py <name> <GPU host> <level> screen [reps]       candidates/layer_<name>.py, single arm vs
                                                                           reviewed production (default 200 reps; a
                                                                           screen: the replay checks only)
    python3 harness/submit.py <name> <GPU host> <level> run [--files <dir>] [--build <cmd>] <command...>
                                                                           a lane job once candidates/layer_<name>.py is
                                                                           built there (compile-only: true), under LANE_S;
                                                                           --files: a dir it carries (/root/lanes/<name>/),
                                                                           --build: its compile step there, on a build box
The GPU box keeps /root/out/abo_<name>.* (a screen's .json, .log, .built; every job's .done); a run job's output is the
runner's /root/out/vrun_<time>_<slot>.txt unless its command writes its own."""
import fcntl, json, os, pathlib, shlex, shutil, sys, time

ROOT = pathlib.Path(__file__).resolve().parent.parent
RELAY = pathlib.Path(os.environ.get("OKC_STATE", "~/.okc")).expanduser() / "relay"   # harness/relay.py: the build queue and every job's state
sys.path.insert(0, str(ROOT / "harness"))
import vast_hosts  # noqa: E402
LANE_S = 480   # a lane job's limit (s): its slowest standalone step, ~306 s, x 1.5


def gpu_of(name, host):
    """the GPU job <name> runs on: the relay moves a stopped or draining spot GPU's jobs (relay.py), so it runs where
    the relay sent it, not necessarily on <host>, the GPU it was queued for"""
    try:
        return json.loads((RELAY / "state.json").read_text()).get(name, {}).get("gpu") or host
    except (OSError, ValueError):
        return host


def submit(name, host, level, job):
    """queue candidates/layer_<name>.py with its GPU job line for <host> at <level>; a name the relay already has is
    refused (it keeps one job per name and would drop this one, while the old job's .done and result stay on its box:
    xmc1, 11-05, never ran behind an 8.6 h old gpu1 job of that name)"""
    try:
        if name in json.loads((RELAY / "state.json").read_text()):
            raise RuntimeError(f"{name}: the relay has a job of that name already; use another")
    except (OSError, ValueError):
        pass
    (RELAY / "jobs" / f"{name}.txt").write_text(job + "\n")
    with open(RELAY / "queue.lock", "a") as lk, open(RELAY / "queue.txt", "a") as f:   # the relay reads and empties it under it
        fcntl.flock(lk, fcntl.LOCK_EX)
        f.write(f"{name} {host} {level}\n")


def cancel(name):
    """void job <name> with the relay: never built or shipped (a build in progress is killed)"""
    with open(RELAY / "queue.lock", "a") as lk, open(RELAY / "cancel.txt", "a") as f:
        fcntl.flock(lk, fcntl.LOCK_EX)
        f.write(name + "\n")


def screen(name, host, level, reps, checks=False):
    """checks: the 20 eager reference checks too (controls, confirmations, anything frozen or submitted)"""
    submit(name, host, level, vast_hosts.bench_job(name, vast_hosts.load(offline=True)[2][host], host, reps, level, checks))


def run(name, host, level, command, files=None, build=None):
    """a lane's job: it holds its GPU's lock exclusively for its whole run (a job that times, profiles or loads the GPU
    by itself, nsys or a cuBLASLt search, must never overlap a bench's timed section: eat_lt_o_256 ran beside
    dbt_clean_stages2_s's stalled timing, 00:22-00:25); a bench inside it then takes no lock of its own. With no
    candidates/layer_<name>.py, a placeholder that builds nothing is written (a lane's command brings its own files: a
    real layer as the placeholder built and shipped its whole module set, and slowed every other ship)"""
    f = ROOT / f"candidates/layer_{name}.py"
    if not f.exists():
        f.write_text('"""a lane run job\'s placeholder: no builds"""\n')
    # files: a lane directory that travels with the job (to /root/lanes/<name>/ on the build box, then the GPU box); build:
    # its compile step, run in that directory on the build box (never under a GPU lock); the command then runs there too
    lane = RELAY / "lanes" / name
    if files:
        shutil.rmtree(lane, ignore_errors=True)
        shutil.copytree(files, lane, symlinks=True)
    if build:
        (RELAY / "jobs" / f"{name}.build").write_text(build + "\n")
    if files or build:
        command = f"cd /root/lanes/{name} && {command}"
    submit(name, host, level, f"run p{level} cd /root/okc && : abo_{name}. && flock \"$OKC_GPU_LOCK\" env -u OKC_GPU_LOCK "
                              f"timeout -s USR1 -k 10 {LANE_S} bash -c {shlex.quote(command)}; rc=$?; "
                              f"touch /root/out/abo_{name}.done; exit $rc")


if __name__ == "__main__":
    name, host, level, kind, *rest = sys.argv[1:]
    if kind == "screen":   # (a lane's screen: confirm.py reads it at collect, as the driver reads its study screens)
        screen(name, host, int(level), int(rest[0]) if rest else 200)
        with open(ROOT / "receipts/evidence/lane_screens.jsonl", "a") as f:
            f.write(json.dumps({"event": "queued", "name": name, "host": host, "level": int(level), "t": round(time.time())}) + "\n")
    else:   # run [--files <dir>] [--build <compile command>] <command...>: the build step runs on a build box, in the dir
        opts = {}
        while rest and rest[0] in ("--files", "--build"):
            opts[rest[0][2:]] = rest[1]
            rest = rest[2:]
        run(name, host, int(level), " ".join(rest), **opts)
    print(f"{name}: {kind} on {host} at level {level}")
