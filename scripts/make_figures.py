"""Draws every figure in figures/: the hand-built SVG diagrams (diagrams.py) and the charts from the CSVs in data/,
all in the style defined in svgkit.py.

    uv run --with matplotlib python scripts/make_figures.py
"""
import csv
import datetime as dt
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import diagrams  # noqa: E402
from svgkit import COLORS, MUTED, mpl_style  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA, OUT = ROOT / "data", ROOT / "figures"
CLOSE = dt.datetime(2026, 11, 5, 7, 59)
BLUE, RED, LIGHT = COLORS["blue"], COLORS["red"], "#A9C4F3"
mpl_style()


def rows(name):
    with open(DATA / name) as f:
        return list(csv.DictReader(f))


def when(s):
    return dt.datetime.fromisoformat(s)


def climb():
    hist = rows("rank_history.csv")
    runs = [r for r in rows("board_runs.csv") if r["speedup"]]
    fig, (top, bot) = plt.subplots(2, 1, figsize=(9, 6.2), sharex=True, gridspec_kw={"height_ratios": [3, 2]})

    t = [when(r["time_utc"]) for r in hist]
    ours = [float(r["our_board_speedup"]) for r in hist]
    top.step(t + [CLOSE], ours + [ours[-1]], where="post", color=BLUE, lw=2, label="our best run on the board")
    top.scatter([when(r["time_utc"]) for r in runs], [float(r["speedup"]) for r in runs], s=14, color=BLUE,
                alpha=0.45, zorder=3, label="each scored run")
    leader = [(when(r["time_utc"]), float(r["board_top_speedup"])) for r in hist if r["board_top_speedup"]]
    top.plot(*zip(*leader), "--o", color=MUTED, ms=4, lw=1, label="board #1")
    top.set_ylabel("speedup over SGLang")
    top.set_ylim(1.0, 2.7)
    top.legend(loc="lower right", frameon=False)
    top.set_title("Decode-layer board: our score and rank, Nov 1 to the close (UTC)")

    rank = [int(r["decode_board_rank"]) for r in hist]
    bot.step(t + [CLOSE], rank + [rank[-1]], where="post", color=RED, lw=2)
    bot.scatter(t, rank, s=12, color=RED, zorder=3)
    bot.set_yscale("log")
    bot.invert_yaxis()
    bot.set_yticks([1000, 300, 150, 100, 78, 50])
    bot.get_yaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"#{int(v)}"))
    bot.set_ylabel("rank")
    for a in (top, bot):
        a.axvline(CLOSE, color=MUTED, lw=0.8, ls=":")
    bot.annotate("track closes", (CLOSE, 60), xytext=(-70, 0), textcoords="offset points", fontsize=8, color=MUTED)
    for x, y, label in [(when("2026-11-01T22:52"), 147, "ported base, #147"), (when("2026-11-03T04:53"), 73, "#73"),
                        (when("2026-11-04T19:14"), 117, "#117"), (when("2026-11-05T08:20"), 78, "#78")]:
        bot.annotate(label, (x, y), xytext=(4, 6), textcoords="offset points", fontsize=8)
    bot.xaxis.set_major_formatter(mdates.DateFormatter("%b %d %H:%M"))
    bot.xaxis.set_major_locator(mdates.HourLocator(byhour=[0, 12]))
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(OUT / "climb.svg")


CASE_NAMES = {"d1s": "decode 1\nsliding", "d1f": "decode 1\nfull", "d16s": "decode 16\nsliding", "d16f": "decode 16\nfull",
              "d32s": "decode 32\nsliding", "d32f": "decode 32\nfull", "v1s": "verify 1\nsliding", "v1f": "verify 1\nfull",
              "v16s": "verify 16\nsliding", "v16f": "verify 16\nfull", "v32s": "verify 32\nsliding",
              "v32f": "verify 32\nfull"}


def per_case():
    cases = rows("per_case_final.csv")
    x = range(len(cases))
    names = [CASE_NAMES[c["case"]] for c in cases]
    fig, (ax, bx) = plt.subplots(2, 1, figsize=(10, 6.4), sharex=True, gridspec_kw={"height_ratios": [3, 2]})
    paired = [float(c["paired_vs_production_pct"]) for c in cases]
    cold = [float(c["cold_validation_pct"]) for c in cases]
    colors = [BLUE if c["case"].endswith("s") else LIGHT for c in cases]
    ax.bar(x, paired, 0.62, color=colors)
    ax.scatter(x, cold, marker="D", s=22, color=COLORS["ink"], zorder=3, label="single cold validation pass")
    for i, (v, c) in enumerate(zip(paired, cold)):
        ax.annotate(f"+{v:.1f}%", (i, max(v, c)), xytext=(0, 6), textcoords="offset points", ha="center", fontsize=8)
    ax.set_ylabel("faster than the Nov 1 file (%)")
    ax.set_ylim(0, 10)
    ax.set_title("Per case: the final file against our first scored ported file")
    ax.legend(loc="upper right", fontsize=8.5)
    ax.text(0.01, 0.93, "bars: 400-rep paired ratio on one GPU (dark = sliding, weight 46; light = full, weight 16)",
            transform=ax.transAxes, fontsize=8.5, color=MUTED)
    ratio = [float(c["burst_mean_us"]) / float(c["judge_floor_us"]) for c in cases]
    bx.bar(x, ratio, 0.62, color=[COLORS["red"] if r > 1.75 else "#E8A0AA" for r in ratio])
    for i, r in enumerate(ratio):
        bx.annotate(f"{r:.2f}", (i, r), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8)
    bx.axhline(1, color=MUTED, lw=0.8)
    bx.set_ylim(0, 2.5)
    bx.set_ylabel("our time / judge floor")
    bx.text(0.01, 0.86, "mean of the final 11 judge runs over the judge's speed-of-light bound (no launch costs in the bound)",
            transform=bx.transAxes, fontsize=8.5, color=MUTED)
    bx.set_xticks(list(x), names, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "per_case.svg")


def ladder():
    pts = rows("paired_ladder.csv")
    t = [dt.datetime.fromisoformat(p["time_utc"]) for p in pts]
    v = [100 * (float(p["geomean_vs_production"]) - 1) for p in pts]
    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    ax.step(t, v, where="post", color=BLUE, lw=2)
    ax.scatter(t, v, color=BLUE, s=16, zorder=3)
    ax.axhspan(v[-1] - 0.25, v[-1] + 0.25, color=COLORS["teal"], alpha=0.12, lw=0)
    ax.text(t[0], v[-1] + 0.3, "incumbent's own screen spread, 2 sd", fontsize=8, color=COLORS["teal"])
    for k, off in ((2, (-6, 12)), (6, (-6, 12)), (10, (-8, -4)), (11, (-8, 10))):
        ax.annotate(pts[k]["file"], (t[k], v[k]), xytext=off, textcoords="offset points", ha="right", fontsize=8,
                    color=MUTED)
    ax.text(0.01, 0.04, "400-rep confirmations; rows after Nov 4 23:58 ran on a different provider's H200s",
            transform=ax.transAxes, fontsize=8, color=MUTED)
    ax.set_ylabel("whole layer, % faster than the Nov 1 file")
    ax.set_title("The incumbent's paired score on our pipeline, Nov 3 to the close")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d %H:%M"))
    ax.xaxis.set_major_locator(mdates.HourLocator(byhour=[0, 12]))
    ax.set_ylim(0, 5)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(OUT / "ladder_paired.svg")


def gap():
    g = rows("gap_ledger.csv")
    slots = [("moe", "experts (MoE)", COLORS["red"]), ("attn", "attention", COLORS["violet"]),
             ("qkv", "qkv", BLUE), ("pool_router", "pooler + router", "#6E9BEA"), ("unpool", "unpooler", "#A9C4F3"),
             ("o", "o", "#C9D9F7"), ("route", "route", COLORS["amber"]), ("bounds", "boundaries", "#EBC57A")]
    fig, ax = plt.subplots(figsize=(10, 4.6))
    x = range(len(g))
    bottom = [0.0] * len(g)
    for key, label, c in slots:
        vals = [max(0.0, float(r[key])) for r in g]
        ax.bar(x, vals, 0.62, bottom=bottom, color=c, label=label)
        bottom = [b + v for b, v in zip(bottom, vals)]
    for i, r in enumerate(g):
        ax.annotate(f"{float(r['weighted_pct']):.2f}%", (i, bottom[i]), xytext=(0, 3), textcoords="offset points",
                    ha="center", fontsize=8, color=COLORS["ink"])
    ax.set_xticks(list(x), [CASE_NAMES[r["case"]] for r in g], fontsize=8)
    ax.set_ylabel("µs above the model floor")
    ax.set_title("Where the remaining time sat on Nov 4: µs above floor, and its share of the score")
    ax.legend(ncol=4, fontsize=8, loc="upper left")
    ax.set_ylim(0, 125)
    fig.tight_layout()
    fig.savefig(OUT / "gap.svg")


def burst():
    runs = sorted((r for r in rows("board_runs.csv") if "final burst" in r["note"]), key=lambda r: float(r["speedup"]))
    fig, ax = plt.subplots(figsize=(8, 4.2))
    y = range(len(runs))
    ax.hlines(len(runs) - 1, 2.307, 2.377, color=BLUE, alpha=0.35, lw=6, label="judge's 95% interval for the best run")
    colors = [RED if "validated" in r["note"] else BLUE for r in runs]
    ax.scatter([float(r["speedup"]) for r in runs], y, color=colors, zorder=3)
    labels = [r["file"].replace("layer_", "").replace("_final", "") for r in runs]
    ax.set_yticks(list(y), labels, fontsize=8)
    ax.set_xlabel("speedup over SGLang (official scored run)")
    ax.set_title("The last 11 scored runs, 07:29 to 07:39 UTC on Nov 5")
    ax.annotate("final validated file", (2.323, labels.index("cf1084_rlxpoll1_r5")), xytext=(8, -3),
                textcoords="offset points", fontsize=8, color=RED)
    ax.legend(loc="lower right", frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "final_burst.svg")



def hbm():
    pts = rows("hbm_ring_bench.csv")
    fig, ax = plt.subplots(figsize=(9, 4.6))
    series = [("132", "8192", "132 CTAs, 8 KB slots", COLORS["amber"]), ("132", "16384", "132 CTAs, 16 KB slots", BLUE),
              ("132", "36864", "132 CTAs, 36 KB slots", COLORS["violet"]), ("264", "8192", "264 CTAs, 8 KB slots", COLORS["teal"])]
    for ctas, slot, label, c in series:
        xs = [(int(p["inflight_per_sm_kb"]), float(p["tbps"])) for p in pts if p["ctas"] == ctas and p["slot_bytes"] == slot]
        ax.plot(*zip(*xs), "-o", color=c, ms=4, lw=1.8, label=label)
    for y, lab, c in ((4.8, "4.8 nominal", MUTED), (4.61, "4.61 measured plateau", RED), (3.80, "3.80 paged gather, 128 B rows", MUTED)):
        ax.axhline(y, color=c, lw=0.9, ls="--")
        ax.text(198, y + 0.04, lab, ha="right", fontsize=8.5, color=c)
    ax.axvline(48, color=MUTED, lw=0.8, ls=":")
    ax.text(50, 2.2, "knee: about 48 KB in flight per SM", fontsize=8.5, color=MUTED)
    ax.axvline(108, color=MUTED, lw=0.8, ls=":")
    ax.text(110, 2.6, "small-MoE ring: 3 x 36 KB", fontsize=8.5, color=MUTED)
    ax.set_xscale("log", base=2)
    ax.set_xticks([8, 16, 32, 48, 64, 96, 128, 192], ["8", "16", "32", "48", "64", "96", "128", "192"])
    ax.set_xlim(7, 200)
    ax.set_ylim(0.8, 5.1)
    ax.set_xlabel("bytes in flight per SM (KB)")
    ax.set_ylabel("HBM read rate (TB/s)")
    ax.legend(loc="lower right", fontsize=8.5)
    ax.set_title("How many bytes must be in flight: TMA ring streams on an H200")
    fig.tight_layout()
    fig.savefig(OUT / "hbm_plateau.svg")


def throughput():
    hrs = rows("pipeline_throughput_hourly.csv")
    t = [dt.datetime.fromisoformat(h["hour_utc"]) for h in hrs]
    fig, (ax, gx) = plt.subplots(2, 1, figsize=(9.5, 4.8), sharex=True, gridspec_kw={"height_ratios": [5, 1]})
    parts = [("study_screens", "TPE study screens", BLUE), ("lane_screens", "agents' screens", COLORS["violet"]),
             ("confirmations_400rep", "confirmations (400 reps, counted twice)", COLORS["green"]),
             ("controls", "incumbent controls", COLORS["amber"])]
    bottom = [0.0] * len(hrs)
    for key, label, c in parts:
        mult = 2 if key.startswith("conf") else 1
        v = [mult * float(h[key]) / 60 for h in hrs]
        ax.bar(t, v, width=0.036, bottom=bottom, color=c, align="edge", label=label)
        bottom = [b + x for b, x in zip(bottom, v)]
    ax.set_ylabel("200-rep units per minute")
    ax.set_ylim(0, 16)
    ax.legend(loc="upper left", fontsize=8.5, ncol=2)
    ax.set_title("Pipeline throughput per hour, all GPUs")
    notes = [("2026-11-03T09:00", "stale build locks"), ("2026-11-03T16:00", "hashed-name incident"),
             ("2026-11-04T10:00", "paired confirmations"), ("2026-11-04T22:00", "move to Nebius")]
    for x, lab in notes:
        x = dt.datetime.fromisoformat(x)
        y = next(b for b, tt in zip(bottom, t) if tt == x)
        ax.annotate(lab, (x, y), xytext=(0, 26), textcoords="offset points", ha="center", fontsize=8, color=MUTED,
                    arrowprops={"arrowstyle": "-", "color": MUTED, "lw": 0.8})
    gpus = [("2026-11-03T04:00", 3), ("2026-11-03T05:22", 0), ("2026-11-03T05:36", 3), ("2026-11-03T06:25", 2), ("2026-11-03T10:20", 3), ("2026-11-03T11:44", 2),
            ("2026-11-04T22:13", 1), ("2026-11-05T01:33", 2), ("2026-11-05T03:10", 4), ("2026-11-05T07:08", 4)]
    gt = [dt.datetime.fromisoformat(a) for a, _ in gpus]
    gx.step(gt, [n for _, n in gpus], where="post", color=COLORS["teal"], lw=2)
    gx.set_ylim(0, 4.6)
    gx.set_yticks([0, 2, 4])
    gx.set_ylabel("H200s", fontsize=9)
    gx.xaxis.set_major_formatter(mdates.DateFormatter("%b %d %H:%M"))
    gx.xaxis.set_major_locator(mdates.HourLocator(byhour=[0, 12]))
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(OUT / "throughput.svg")

if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    climb()
    per_case()
    burst()
    ladder()
    gap()
    hbm()
    throughput()
    for draw in (diagrams.banner, diagrams.pipeline, diagrams.layer, diagrams.funnel, diagrams.agents, diagrams.noise,
                 diagrams.waterfall, diagrams.ctlword, diagrams.lean, diagrams.predictions, diagrams.gantt,
                 diagrams.rules, diagrams.timed_rep, diagrams.providers, diagrams.dqx, diagrams.ladder,
                 diagrams.egraph, diagrams.margin, diagrams.pdl_chain, diagrams.wgs_stage):
        draw()
