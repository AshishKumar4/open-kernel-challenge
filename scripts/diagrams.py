"""Hand-built SVG diagrams for the README, drawn with svgkit (one palette, one type scale).

    python3 scripts/diagrams.py        # writes figures/*.svg
"""
import pathlib
import random

from svgkit import COLORS, INK, LINE, MUTED, PAPER, SOFT as SOFT_, SVG, TINTS

OUT = pathlib.Path(__file__).resolve().parent.parent / "figures"


def banner():
    W, H = 1280, 380
    s = SVG(W, H, bg=None, title="Open Kernel Challenge: 2.342x faster than SGLang on one H200, rank #78")
    s.raw('<defs><linearGradient id="bg" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#0E1A2B"/>'
          '<stop offset="1" stop-color="#17345A"/></linearGradient>'
          '<linearGradient id="die" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#22426E"/>'
          '<stop offset="1" stop-color="#16304F"/></linearGradient>'
          '<radialGradient id="glow" cx="0.5" cy="0.5" r="0.5"><stop offset="0" stop-color="#2F6FDE" stop-opacity="0.35"/>'
          '<stop offset="1" stop-color="#2F6FDE" stop-opacity="0"/></radialGradient></defs>')
    s.raw(f'<rect width="{W}" height="{H}" rx="18" fill="url(#bg)"/>')
    # faint grid
    for x in range(0, W, 32):
        s.raw(f'<line x1="{x}" y1="0" x2="{x}" y2="{H}" stroke="#FFFFFF" stroke-opacity="0.035"/>')
    for y in range(0, H, 32):
        s.raw(f'<line x1="0" y1="{y}" x2="{W}" y2="{y}" stroke="#FFFFFF" stroke-opacity="0.035"/>')

    s.text(56, 70, "OPEN KERNEL CHALLENGE  ·  KERNEL TRACK  ·  OCT 22 TO NOV 5, 2026", 13, "#8FB3E8", 700)
    s.text(56, 128, "Four days, one H200,", 46, "#FFFFFF", 750)
    s.text(56, 180, "and a team of AI agents", 46, "#FFFFFF", 750)
    s.text(56, 216, "The Muse Spark decode layer on an NVIDIA H200 SXM, tuned and screened by coding agents,", 15,
           "#B9C8DD")
    s.text(56, 237, "timed by the challenge's judge against SGLang.", 15, "#B9C8DD")
    stats = [("2.342×", "faster than SGLang"), ("9,659 µs", "per decode step"), ("#78", "decode-layer board"),
             ("6,722", "candidates screened")]
    x = 56
    for big, small in stats:
        s.raw(f'<rect x="{x}" y="268" width="168" height="78" rx="10" fill="#FFFFFF" fill-opacity="0.06" '
              f'stroke="#FFFFFF" stroke-opacity="0.14"/>')
        s.text(x + 16, 306, big, 28, "#FFFFFF", 750)
        s.text(x + 16, 330, small, 12.5, "#9FB4CF", 500)
        x += 182

    # the H200: package, die with 132 SMs, six HBM3e stacks
    cx, cy = 1050, 190
    s.raw(f'<circle cx="{cx}" cy="{cy}" r="190" fill="url(#glow)"/>')
    s.raw(f'<rect x="{cx - 170}" y="{cy - 140}" width="340" height="280" rx="14" fill="#0B1626" stroke="#3A5C8A" '
          'stroke-width="1.5"/>')
    for side in (-1, 1):
        for i in range(3):
            hx = cx + side * 132 - 22
            hy = cy - 120 + i * 82
            s.raw(f'<rect x="{hx}" y="{hy}" width="44" height="74" rx="5" fill="#1F9AA8" fill-opacity="0.22" '
                  'stroke="#3CC4D2" stroke-width="1.2"/>')
            for k in range(5):
                s.raw(f'<line x1="{hx + 6}" y1="{hy + 12 + k * 12}" x2="{hx + 38}" y2="{hy + 12 + k * 12}" '
                      'stroke="#3CC4D2" stroke-opacity="0.5"/>')
    s.raw(f'<rect x="{cx - 92}" y="{cy - 120}" width="184" height="240" rx="8" fill="url(#die)" stroke="#5B86C4" '
          'stroke-width="1.3"/>')
    rng = random.Random(4)
    cols, rows = 11, 12
    for r in range(rows):
        for c in range(cols):
            x0 = cx - 84 + c * 15.3
            y0 = cy - 112 + r * 19.2
            hot = rng.random()
            col = "#2F6FDE" if hot > 0.45 else "#4C8BF0" if hot > 0.2 else "#9FC1F7"
            s.raw(f'<rect x="{x0:.1f}" y="{y0:.1f}" width="12" height="15.5" rx="2" fill="{col}" '
                  f'fill-opacity="{0.55 + 0.4 * hot:.2f}"/>')
    s.text(cx, cy + 162, "H200 SXM  ·  132 SMs  ·  141 GB HBM3e  ·  4.6 TB/s measured", 12.5, "#9FB4CF", 600, "middle")
    s.save(OUT / "banner.svg")


def pipeline():
    W, H = 1240, 560
    s = SVG(W, H, title="The screening pipeline")
    s.heading(30, 40, "The screening pipeline", "Every GPU job, from any agent or the TPE study, took the same path.")
    s.frame(20, 80, 250, 330, "my PC: orchestration only")
    s.frame(290, 80, 250, 330, "rented CPU boxes")
    s.frame(560, 80, 330, 330, "rented H200 boxes")
    s.frame(910, 80, 310, 460, "promotion and the judge")

    s.node(40, 112, 210, 74, "Expert agents", ["one per kernel family"], "violet")
    s.node(40, 202, 210, 74, "TPE study", ["Optuna, per-case switches"], "violet")
    s.node(40, 300, 210, 92, "submit.py + relay", ["priority 0 (controls) to 4 (TPE)", "one job per name"], "amber")
    s.arrow([(145, 186), (145, 202)], "ink")
    s.arrow([(145, 276), (145, 300)], "ink")

    s.node(310, 160, 210, 120, "Build modules", ["~17 CUDA modules per file", "~53 for a per-case file",
                                                  "judge's exact flags", "ccache, no GPU"], "red")
    s.arrow([(250, 346), (280, 346), (280, 220), (310, 220)], "ink", "queued", 282, 335)

    s.node(580, 112, 290, 96, "Bench server per slot", ["2 runners per GPU", "inputs cached in pinned host memory",
                                                         "production prepared once"], "teal")
    s.node(580, 228, 290, 74, "okc_harness.py timing", ["CUDA graph, cold L2, paired, median"], "teal")
    s.node(580, 322, 290, 70, "Telemetry", ["1 s power / clock / util log"], "teal")
    s.arrow([(520, 200), (580, 160)], "ink", ".so files", 552, 200)
    s.arrow([(725, 208), (725, 228)], "ink")

    s.node(930, 112, 270, 74, "Screen", ["200 reps vs the incumbent's reading"], "blue")
    s.node(930, 206, 270, 74, "Confirm", ["400 reps, full checks"], "green")
    s.node(930, 300, 270, 74, "Bundle and stack", ["confirm the bundle again"], "green")
    s.node(930, 394, 270, 92, "Validate and review", ["cold prepare under 300 s, 13 tests,", "52 nearby shapes, 12 × 200 soak"],
           "green")
    s.arrow([(870, 265), (900, 265), (900, 149), (930, 149)], "ink")
    for y0, y1, lab in ((186, 206, "ahead"), (280, 300, "confirmed"), (374, 394, "")):
        s.arrow([(1065, y0), (1065, y1)], "ink", lab, 1100, y0 + 14)

    s.node(560, 450, 330, 74, "okc submit: the judge's H200", ["per-case table and per-kernel profile"], "red")
    s.arrow([(930, 470), (890, 487)], "red")
    s.arrow([(560, 487), (145, 487), (145, 392)], "red", "profiles and scores feed the next ideas", 350, 478,
            dash="6 4")
    s.save(OUT / "pipeline.svg")


def layer():
    W, H = 1240, 560
    s = SVG(W, H, title="The decode layer as a kernel chain")
    s.heading(30, 40, "The decode layer: a chain of kernels",
              "Dispatch by token count T (1, 8, 16, 32, 128, 256). Up to T = 128 the kernels are linked by PDL and "
              "prefetch weights before waiting; at 256, PDL is off.")
    rows_ = [
        [("QKV projection", ["sk2: 12-bit GEMV, T <= 32", "rq: 12-bit GEMM, T >= 128"], "blue"),
         ("Attention", ["split-KV over FP8 pages", "QK-norm, RoPE, cache write"], "violet"),
         ("O projection", ["sk2 GEMV"], "blue"),
         ("Boundary", ["residual + RMSNorm", "thread-block cluster"], "amber"),
         ("Pooler + router", ["one sk2 launch"], "blue")],
        [("Routing", ["top-8 of 256", "sigmoid + bias, 32 local"], "amber"),
         ("Experts", ["NVFP4 SwiGLU, persistent", "tiny / small / mid / WGS"], "red"),
         ("Unpooler", ["sk2 GEMV"], "blue"),
         ("Finish", ["residual + norm, outputs"], "amber")],
    ]
    w, h, gap = 214, 84, 30
    s.text(24, 100, "PDL edges: the next kernel's prologue and weight prefetch overlap the previous kernel's tail",
           11.5, MUTED, 500)
    for r, steps in enumerate(rows_):
        y = 112 + r * 124
        for i, (t, sub, c) in enumerate(steps):
            x = 24 + i * (w + gap)
            s.node(x, y, w, h, t, sub, c)
            if i:
                s.arrow([(x - gap + 2, y + h / 2), (x - 2, y + h / 2)], "ink")
    lastx = 24 + 4 * (w + gap) + w / 2
    s.arrow([(lastx, 196), (lastx, 216), (24 + w / 2, 216), (24 + w / 2, 234)], "ink")

    # share of kernel time per category, from the judge's per-kernel profile of our best run (data/time_shares.csv)
    import csv
    with open(OUT.parent / "data" / "time_shares.csv") as f:
        rows = list(csv.reader(f))
    cats, mean = rows[0][1:], [float(v) for v in rows[-1][1:]]
    by = 362
    s.text(24, by, "Where the GPU time goes in our best run (judge profile, 46 sliding + 16 full per load, mean of the 6 loads)",
           14, INK, 650)
    bx, total = 24, 1190
    for name, pct, c in zip(cats, mean, ("red", "blue", "violet", "amber")):
        ww = total * pct / 100
        s.rect(bx, by + 14, ww - 3, 34, fill=TINTS[c], stroke=COLORS[c], r=6)
        s.text(bx + 10, by + 36, f"{name}  {pct:.0f}%" if pct > 9 else f"{name.split(',')[0]} {pct:.0f}%", 12.5, INK, 600)
        bx += ww
    s.text(24, by + 70, "The MoE share runs from 37% (decode 1) to 56% (verify 32); projections from 50% (decode 1) to 23%. "
           "Profile kernel times come from a separate profiling pass.", 11.5, MUTED, 400, italic=True)

    # the weights stream
    wy = 462
    s.rect(24, wy, 1190, 70, fill=TINTS["teal"], stroke=COLORS["teal"], r=10)
    s.text(40, wy + 26, "HBM3e to SMs: the layer is a weight stream", 14, INK, 650)
    s.text(40, wy + 49, "BF16 projections read through a lossless 12-bit code; experts are NVFP4 (28.3 MB each). "
           "Measured plateau 4.58 to 4.70 TB/s with at least 48 KB in flight per SM.", 12.5, MUTED)
    s.save(OUT / "layer.svg")


def funnel():
    W, H = 1240, 360
    s = SVG(W, H, title="From candidates to a board row")
    s.heading(30, 40, "From 6,722 candidates to one board row", "Counts from the pipeline's receipts; bar width is log scale.")
    import math
    rows = [("screened (200 reps)", 6722, "blue", "by 02:35 UTC, Nov 5"),
            ("confirmations queued (400 reps)", 1090, "green", "1,031 read: 636 within noise, 364 behind, 28 ahead"),
            ("incumbent promotions", 64, "green", "Nov 3 to the close, each a confirmed, stacked change"),
            ("scored decode runs on record", 21, "red", "after the port; 11 of them in the last 10 minutes"),
            ("board row", 1, "red", "2.342×, 9,659.4 µs")]
    top, x0, maxw = 86, 300, 640
    for i, (lab, n, c, note) in enumerate(rows):
        y = top + i * 52
        w = 40 + maxw * math.log10(n + 1) / math.log10(6723)
        s.text(x0 - 14, y + 24, lab, 13.5, INK, 600, "end")
        s.rect(x0, y + 4, w, 32, fill=TINTS[c], stroke=COLORS[c], r=6)
        s.text(x0 + 12, y + 26, f"{n:,}", 15, COLORS[c], 750)
        s.text(x0 + w + 14, y + 25, note, 12.5, MUTED)
    s.save(OUT / "funnel.svg")



def agents():
    W, H = 1240, 760
    s = SVG(W, H, title="The agent system")
    s.heading(30, 40, "The agent system",
              "Mostly I talked to the orchestrator. It ran the expert lanes, spawned reviewers and audits, and owned the GPU budget.")
    s.node(470, 74, 300, 62, "Me", ["goals, rulings, caps, 'status?'"], "violet")
    s.node(420, 170, 400, 78, "Orchestrator: Claude Opus 5.5", ["briefs and relays rulings, approves diffs,",
                                                               "books GPUs, reports"], "blue")
    s.arrow([(620, 136), (620, 170)], "violet", both=True)
    s.text(632, 158, "1,085 messages from me; 112 blocking questions", 11.5, MUTED, 500)
    s.node(30, 170, 330, 78, "Advisor", ["a second model reading the orchestrator's", "messages and diffs (388 advisories)"],
           "muted")
    s.arrow([(360, 209), (420, 209)], "muted", dash="5 4")
    s.node(880, 170, 330, 78, "Audits and EAT rounds", ["reviewers read raw transcripts and", "receipts; 6 rounds, 1 audit of the orchestrator"],
           "amber")
    s.arrow([(880, 209), (820, 209)], "amber", dash="5 4")

    s.frame(20, 280, 1200, 250, "29 expert lanes, 1 to 7 at a time")
    s.arrow([(620, 248), (620, 280)], "blue", "1,321 messages", 680, 270)
    s.node(40, 316, 330, 92, "Operator lane", ["kernels on day one, then the base build,", "queue, TPE study, confirmations, releases"],
           "blue")
    s.arrow([(470, 118), (390, 118), (390, 362), (373, 362)], "violet", dash="5 4")
    s.text(30, 150, "Nov 2-3: 267 messages straight to the operator", 11.5, COLORS["violet"], 600)
    builders = [("Large-load MoE", "Astra, then Opus"), ("Small-load MoE edges", "Opus"),
                ("Projections, glue, codecs", "Opus, EAT-style"), ("Attention + per-shape", "Astra")]
    for k, (t, m) in enumerate(builders):
        s.node(392 + k * 206, 316, 196, 92, t, [m], "violet" if "Astra" in m else "blue")
    s.text(40, 440, "short-lived research lanes:", 12.5, INK, 650)
    lx = 220
    for t in ("algebra / FP4 decode search", "clean rewrite", "Lean model", "ptxas RE", "moonshot MoE"):
        lx += s.pill(lx, 444, t, "teal") + 10
    s.text(40, 490, "last night:", 12.5, INK, 650)
    for k in range(19):
        s.rect(130 + k * 26, 474, 20, 22, fill=TINTS["blue"], stroke=COLORS["blue"], r=4)
    s.text(640, 490, "19 fresh-attempt agents, ~90 min or 3 attempts each, reading the last one's handoff", 12, MUTED)

    s.node(30, 560, 370, 86, "Reviewers: GPT-6.1 Sol, ~125", ["pre-submit rules and integrity, kernel correctness,",
                                                              "audits, public-source sweeps"], "green")
    s.node(435, 560, 370, 86, "Scouts and a debugger", ["~25 Haiku scouts (research moved to Sol),", "1 debugger for failed trials"],
           "teal")
    s.node(840, 560, 370, 86, "The GPU pipeline", ["the only path to GPU time:", "queue, build boxes, H200s, screens"], "red")
    s.rect(30, 670, 1180, 70, fill=SOFT_, stroke=LINE, r=10)
    s.text(46, 694, "Shared memory in files", 13, INK, 650)
    s.text(46, 718, "RULES.md · decisions.md · judge-facts · evidence ledger · gap ledger · lever matrix · handoff notes · "
           "receipts (24 GB) · git: 2,219 commits on 188 branches", 12.5, MUTED, 500, mono=False)
    s.save(OUT / "agents.svg")


def noise():
    W, H = 1240, 330
    s = SVG(W, H, title="The size of a win against the size of the noise")
    s.heading(30, 40, "Wins were smaller than the judge's noise",
              "Widths to scale, in percent of the decode-layer score.")
    x0, scale = 860, 220  # px per percent
    s.raw(f'<line x1="{x0}" y1="80" x2="{x0}" y2="300" stroke="{LINE}" stroke-width="1.5"/>')
    for pct in (-1.5, -1, -0.5, 0.5, 1, 1.5):
        x = x0 + pct * scale
        s.raw(f'<line x1="{x}" y1="86" x2="{x}" y2="300" stroke="#EDF0F4"/>')
        s.text(x, 306, f"{pct:+.1f}%", 11, MUTED, 500, "middle")
    rows = [("judge, one run's 95% interval", -1.5, 1.5, "red", "±1.5%"),
            ("our last 11 scored runs, close variants", -0.77, 0.77, "amber", "2.306x to 2.342x"),
            ("the incumbent's own screens (2 sd)", -0.25, 0.25, "teal", "±0.1 to 0.25%"),
            ("most adopted changes", 0.1, 0.8, "green", "+0.1 to +0.8%")]
    for i, (lab, lo, hi, c, note) in enumerate(rows):
        y = 96 + i * 50
        s.rect(x0 + lo * scale, y, (hi - lo) * scale, 26, fill=TINTS[c], stroke=COLORS[c], r=5)
        s.text(30, y + 18, lab, 12.5, INK, 600)
        s.text(x0 + hi * scale + 10, y + 18, note, 12.5, COLORS[c], 700)
    s.save(OUT / "noise.svg")


def waterfall():
    W, H = 1240, 680
    s = SVG(W, H, title="One divergent phi, a waterfall loop, and 1 to 3 microseconds")
    s.heading(30, 40, "One divergent phi, a waterfall loop, and 1 to 3 µs",
              "How a never-taken trap in the MoE's tiny builds was worth about 0.3% of the score.")
    chain = [
        ("CUDA source", ["if (nF > 0) __trap();", "or __builtin_assume(nF <= 0)"], "blue"),
        ("NVVM (LLVM)", ["assume lets LLVM merge gridDim.x", "into a phi at a tid-dependent join"], "violet"),
        ("ptxas divergence", ["a phi at a divergent join is divergent,", "even if both inputs are equal"], "amber"),
        ("SASS", ["each ring refill becomes a waterfall:", "ELECT, 4x R2UR, BRA.U.ANY"], "red"),
    ]
    x, w = 30, 276
    for i, (t, sub, c) in enumerate(chain):
        s.node(x + i * (w + 22), 84, w, 86, t, sub, c, mono_sub=(i == 0))
        if i:
            s.arrow([(x + i * (w + 22) - 20, 127), (x + i * (w + 22) - 2, 127)], "ink")

    # the two loops, as SASS listings
    def listing(x0, y0, title, lines_, c):
        s.rect(x0, y0, 360, 34 + 19 * len(lines_), fill="#0E1A2B", stroke=COLORS[c], r=8, sw=1.5)
        s.text(x0 + 14, y0 + 22, title, 12.5, "#FFFFFF", 700)
        for k, (ins, bad) in enumerate(lines_):
            s.text(x0 + 14, y0 + 44 + 19 * k, ins, 12, "#FFB4BF" if bad else "#CFE0FF", 500, mono=True)

    listing(30, 200, "converged warp: 5 instructions", [("ELECT P0, URZ, PT", 0), ("UBLKCP [UR4], [UR6], UR8", 0),
                                                         ("... (stage bookkeeping)", 0), ("", 0), ("", 0), ("", 0)], "green")
    listing(410, 200, "divergent operands: 9 instructions, loops per lane", [("ELECT P0, URZ, PT", 0), ("R2UR UR4, R2", 1),
             ("R2UR UR5, R3", 1), ("R2UR UR6, R4", 1), ("R2UR UR7, R5", 1), ("UBLKCP [UR4], [UR6], UR8", 0),
             ("BRA.U.ANY  back to ELECT", 1)], "red")
    s.text(30, 380, "Schematic. The real listings come from harness/ptxas/elect.py over each build's cuobjdump output.", 11.5,
           MUTED, 400, italic=True)

    # fixes box
    s.rect(800, 200, 410, 168, fill=TINTS["green"], stroke=COLORS["green"], r=10)
    s.text(818, 226, "What keeps the operands uniform", 14, INK, 650)
    for k, t in enumerate(["warp index via __shfl_sync(~0u, tid >> 5, 0)", "spin loops written inside the asm block",
                           "issue under elect.sync from a converged warp", "an opaque nctaid read (asm volatile mov)",
                           "and the accident that started it: asm trap; blocks the merge"]):
        s.text(818, 254 + 22 * k, "• " + t, 12.5, INK)

    # measured bars: d1s, d1f, v1s, v1f per variant (layer us, 200-rep screens)
    data = [("control (trap), run 1", [61.01, 67.46, 101.94, 108.54], "muted"),
            ("A: assume only", [62.59, 69.35, 104.96, 111.24], "red"),
            ("RA: assume + opaque nctaid", [62.41, 68.94, 103.28, 108.81], "amber"),
            ("RT: trap + opaque nctaid", [61.92, 68.06, 104.03, 109.41], "blue"),
            ("raf1000: frequency hints", [63.81, 69.97, 105.19, 111.15], "violet")]
    cases = ["decode 1 sliding", "decode 1 full", "verify 1 sliding", "verify 1 full"]
    gy, gh = 420, 160
    s.text(30, gy, "Layer time per case, µs above the control's (200-rep screens, all checks pass; two identical controls "
           "differ by 0.7 to 1.4 µs)", 13, INK, 650)
    gx0, gw = 60, 280
    for ci, case in enumerate(cases):
        cx = gx0 + ci * gw
        base = data[0][1][ci]
        s.raw(f'<line x1="{cx}" y1="{gy + 20 + gh}" x2="{cx + 230}" y2="{gy + 20 + gh}" stroke="{LINE}"/>')
        for vi, (name, vals, c) in enumerate(data[1:]):
            d = vals[ci] - base
            bh = max(1, d * 45)
            bx = cx + 10 + vi * 54
            s.rect(bx, gy + 20 + gh - bh, 40, bh, fill=TINTS[c], stroke=COLORS[c], r=3)
            s.text(bx + 20, gy + 14 + gh - bh, f"+{d:.1f}", 11, COLORS[c], 700, "middle")
        s.text(cx + 115, gy + 40 + gh, case, 12, MUTED, 600, "middle")
    lx = 60
    for name, _, c in data[1:]:
        lx += s.pill(lx, gy + 72 + gh, name, c) + 10
    s.save(OUT / "waterfall.svg")


def ctlword():
    W, H = 1240, 250
    s = SVG(W, H, title="The Hopper SASS control word")
    s.heading(30, 40, "Reading ptxas's schedule from the SASS control bits",
              "Each 128-bit instruction's high word carries its scheduling decisions (decoded by harness/ptxas/ctl.py).")
    fields = [("opcode and operand bits", 0, 41, "muted"), ("stall", 41, 45, "red"), ("yield", 45, 46, "amber"),
              ("write SB", 46, 49, "blue"), ("read SB", 49, 52, "blue"), ("wait mask", 52, 58, "violet"),
              ("reuse", 58, 62, "teal"), ("", 62, 64, "muted")]
    x0, y0, scale = 30, 90, 18.2
    for name, a, b, c in fields:
        x = x0 + a * scale
        w = (b - a) * scale
        s.rect(x, y0, w - 2, 46, fill=TINTS[c], stroke=COLORS[c], r=4)
        if name and w < 30:
            s.text(x + w / 2, y0 - 8, name, 12, INK, 650, "middle")
            s.text(x + w / 2, y0 + 30, f"{a}", 11, MUTED, 500, "middle", mono=True)
        elif name:
            s.text(x + w / 2, y0 + 20, name, 12, INK, 650, "middle")
            s.text(x + w / 2, y0 + 37, f"{a}-{b - 1}" if b - a > 1 else f"{a}", 11, MUTED, 500, "middle", mono=True)
    s.text(30, 170, "The stall field tops out at 15 cycles. A longer dependency is encoded as stall 15 on the producer plus a NOP "
           "carrying the rest:", 13, INK)
    s.text(30, 196, "HMMA.16816 (stall 15), then NOP (stall 9): a 24-cycle accumulator chain.   Almost every mid-function NOP is "
           "this split, not alignment padding.", 13, INK, 600, mono=False)
    s.text(30, 226, "Dependent-issue distances read this way: 4 cycles inside the FMA or ALU group, 5 across, 6 into MUFU, "
           "10 after HFMA2.MMA, 24 on an HMMA chain.", 12.5, MUTED)
    s.save(OUT / "ctlword.svg")


def lean():
    W, H = 1240, 560
    s = SVG(W, H, title="What the Lean model covers")
    s.heading(30, 40, "The Lean model: what it covers, and how far to trust each part",
              "Green: kernel-checked proofs. Amber: calibrated against measurements. Red: known gaps.")
    s.frame(20, 76, 590, 420, "correctness")
    s.frame(630, 76, 590, 420, "timing")
    left = [("PTX ISA 9.4, chapter 8", ["the source of truth, then microbenchmarks"], "muted"),
            ("Axiomatic memory model", ["well-formedness + the five §8.10 axioms"], "green"),
            ("mbarrier and named barriers", ["phases, tx counts, parity, faults"], "green"),
            ("Protocols as transition systems", ["TMA ring, split-KV merge, warp joins"], "green"),
            ("Theorems", ["ring_safe, merge_sees, 4 bugs replayed as traces"], "green")]
    right = [("SASS control bits", ["17 builds extracted with cuobjdump"], "muted"),
             ("SMSP issue simulator", ["stalls, scoreboards, pipes, round robin"], "amber"),
             ("HBM / TMA delivery", ["latency + bytes/share, 571-cycle issuer"], "amber"),
             ("MoE end to end, floors", ["per-kernel and whole-layer bounds"], "amber"),
             ("Validation", ["28/30 rates ±10%, 45/45 latencies ±20%"], "amber")]
    for col, items, x in ((0, left, 40), (1, right, 650)):
        for i, (t, sub, c) in enumerate(items):
            y = 116 + i * 74
            s.node(x, y, 550, 60, t, sub, c)
            if i:
                s.arrow([(x + 275, y - 16), (x + 275, y - 2)], "ink")
    s.rect(40, 510, 560, 40, fill=TINTS["red"], stroke=COLORS["red"], r=8, dash="5 4")
    s.text(56, 535, "not proved: the operational ring refines the axiomatic model", 12.5, INK, 600)
    s.rect(650, 510, 560, 40, fill=TINTS["red"], stroke=COLORS["red"], r=8, dash="5 4")
    s.text(666, 535, "not modelled: synchronization under load (MoE kernels read 16% fast)", 12.5, INK, 600)
    s.pill(250, 101, "Alloy cross-check: 144 commands, 0 disagreements", "green")
    s.save(OUT / "lean.svg")


def predictions():
    rows = [("small-MoE stream is memory-bound, instruction cuts can't gain", "right", "consistent with 12+ cut-with-no-gain results"),
            ("big MoE (32+ tokens): more bytes in flight can't help (48 KB/SM plateau)", "right", "its 4th weight slot bought nothing"),
            ("cold instruction fetch is not the d1/v1 mechanism", "right", "flushed 21.57 vs warm 21.50 µs"),
            ("route() frequency-hint fix: +0.37% at d1s", "half", "+0.17% on its cases; adopted"),
            ("WGS cost-width fix at d32s: 9 to 10 µs", "half", "2.4 µs; the cause was right"),
            ("bulk-copy K/V ring for full attention: +0.6 to 1.4%", "wrong", "paged rows: 2.0 to 3.35 TB/s vs cp.async 3.80"),
            ("FP16 attention accumulation: +0.15 to 0.3%", "wrong", "64 of 68 conversions are K/V format; ~0"),
            ("cluster K-slice MoE: 43 to 47 µs at d32s", "wrong", "78 to 83 µs even with no exchange; no-go"),
            ("WIDE 96-token jobs: +0.55%", "wrong", "about -1% at verify 32 sliding")]
    W, H = 1240, 110 + 46 * len(rows)
    s = SVG(W, H, title="Model predictions against measurements")
    s.heading(30, 40, "What the timing model predicted, and what the GPU said",
              "Accurate where it models streaming; optimistic wherever synchronization under load decides.")
    colors = {"right": "green", "half": "amber", "wrong": "red"}
    labels = {"right": "right", "half": "right way, too big", "wrong": "wrong"}
    for i, (pred, verdict, outcome) in enumerate(rows):
        y = 84 + i * 46
        s.rect(20, y, 1200, 38, fill=PAPER if i % 2 else "#FAFBFC", stroke="none", r=6)
        s.text(36, y + 24, pred, 13, INK, 600)
        s.pill(640, y + 24, labels[verdict], colors[verdict])
        s.text(820, y + 24, outcome, 12.5, MUTED)
    s.save(OUT / "predictions.svg")


def _hours(t, t0="2026-10-31T20:00"):
    import datetime as dt
    return (dt.datetime.fromisoformat(t) - dt.datetime.fromisoformat(t0)).total_seconds() / 3600


def gantt():
    import csv
    with open(OUT.parent / "data" / "lanes.csv") as f:
        lanes = list(csv.DictReader(f))
    W, H = 1240, 140 + 34 * len(lanes) + 70
    s = SVG(W, H, title="Expert lanes over time")
    s.heading(30, 40, "Who was working when", "One bar per expert lane (Opus blue, Astra violet). Ticks: EAT rounds, PC "
              "restarts, pipeline outages.")
    x0, x1, span = 420, 1220, _hours("2026-11-05T08:00")
    X = lambda t: x0 + (x1 - x0) * _hours(t) / span  # noqa: E731
    top = 96
    for d, lab in (("2026-11-01T00:00", "Nov 1"), ("2026-11-02T00:00", "Nov 2"), ("2026-11-03T00:00", "Nov 3"),
                   ("2026-11-04T00:00", "Nov 4"), ("2026-11-05T00:00", "Nov 5")):
        s.raw(f'<line x1="{X(d):.1f}" y1="{top - 6}" x2="{X(d):.1f}" y2="{H - 50}" stroke="#EDF0F4"/>')
        s.text(X(d) + 4, top - 10, lab, 11.5, MUTED, 600)
    for k, ln in enumerate(lanes):
        y = top + 8 + k * 34
        c = "violet" if ln["model"] == "astra" else "blue"
        s.text(30, y + 16, ln["work"][:58], 12, INK, 500)
        if ln["work"].startswith("19 fresh"):
            t = _hours(ln["start_utc"])
            for m in range(19):
                xa = X(ln["start_utc"]) + m * (X(ln["end_utc"]) - X(ln["start_utc"])) / 19
                s.rect(xa, y + 3, 4, 18, fill=COLORS[c], stroke="none", r=1)
        else:
            s.rect(X(ln["start_utc"]), y + 3, max(4, X(ln["end_utc"]) - X(ln["start_utc"])), 18, fill=TINTS[c],
                   stroke=COLORS[c], r=4)
    yb = top + 8 + len(lanes) * 34 + 8
    marks = [("2026-11-03T03:47", "amber"), ("2026-11-03T04:56", "amber"), ("2026-11-03T14:21", "amber"),
             ("2026-11-03T18:13", "amber"), ("2026-11-04T11:20", "amber"), ("2026-11-04T19:24", "amber"),
             ("2026-11-01T00:52", "muted"), ("2026-11-01T03:17", "muted"), ("2026-11-01T14:21", "muted"),
             ("2026-11-02T18:05", "muted"), ("2026-11-04T02:34", "muted")]
    for t, c in marks:
        s.raw(f'<path d="M{X(t):.1f},{yb} l-5,10 l10,0 z" fill="{COLORS[c]}"/>')
    for a, b in (("2026-11-03T07:40", "2026-11-03T10:00"), ("2026-11-03T16:03", "2026-11-03T16:31"),
                 ("2026-11-04T14:46", "2026-11-04T16:04")):
        s.rect(X(a), yb + 14, X(b) - X(a), 8, fill=COLORS["red"], stroke="none", r=2)
    ly = yb + 46
    s.raw(f'<path d="M{x0},{ly - 8} l-5,10 l10,0 z" fill="{COLORS["amber"]}"/>')
    s.text(x0 + 10, ly, "EAT round", 11.5, MUTED)
    s.raw(f'<path d="M{x0 + 110},{ly - 8} l-5,10 l10,0 z" fill="{COLORS["muted"]}"/>')
    s.text(x0 + 120, ly, "orchestrator restart (PC out of memory)", 11.5, MUTED)
    s.rect(x0 + 380, ly - 8, 20, 8, fill=COLORS["red"], stroke="none", r=2)
    s.text(x0 + 406, ly, "pipeline outage", 11.5, MUTED)
    s.save(OUT / "gantt.svg")


def rules():
    rows = [("Oct 31", "the first lane spawned helper agents to write code", "experts write all their own code; helpers only review, scout, debug"),
            ("Nov 1", "agents ran builds on my PC; it ran out of memory three times", "my PC runs no compute"),
            ("Nov 1", "kept a 2x accuracy margin \"to be safe\"", "spend the whole accuracy budget: worst of 50+ seeds under the gate"),
            ("Nov 1", "probing one idea at a time, A/B on the judge", "build everything first, then test once; no A/B"),
            ("Nov 2", "asserts and validation in the timed path", "no asserts, bounds or validation in timed code"),
            ("Nov 2", "two providers running the same jobs", "one validation is enough"),
            ("Nov 3", "tests before every bench", "the bench's own checks are the test"),
            ("Nov 3", "pipeline 'improvements' caused a 3-hour outage", "pipeline frozen unless a critical bug"),
            ("Nov 3", "incumbent re-run again and again", "one 200-rep screen per idea, one 400-rep confirmation"),
            ("Nov 4", "a blocking question while I was away: 2.4 h idle", "when I'm away: decide, act, report how to undo"),
            ("Nov 4", "soaks, gates, side-by-sides on every incumbent", "validation only for the file being submitted"),
            ("Nov 5", "ideas dropped as 'not doable tonight'", "never estimate on human timescales (RULES.md)")]
    W, H = 1240, 120 + 40 * len(rows) + 60
    s = SVG(W, H, title="Rules tuned by failures")
    s.heading(30, 40, "Each rule was a fix for something that went wrong",
              "Left: what happened. Right: the rule that came out of it. The last five became notes/RULES.md on Nov 5.")
    for k, (d, what, rule) in enumerate(rows):
        y = 84 + k * 40
        s.text(30, y + 22, d, 12.5, MUTED, 700)
        s.rect(96, y + 4, 500, 30, fill=TINTS["red"], stroke=COLORS["red"], r=6, sw=1)
        s.text(108, y + 24, what, 12.5, INK)
        s.arrow([(600, y + 19), (632, y + 19)], "ink")
        s.rect(636, y + 4, 584, 30, fill=TINTS["green"], stroke=COLORS["green"], r=6, sw=1)
        s.text(648, y + 24, rule, 12.5, INK, 600)
    y = 84 + len(rows) * 40 + 14
    s.text(30, y + 20, "The rulebook stayed short on purpose: the challenge had no general process document, and most "
           "failures came from adding process, not from missing it.", 12.5, MUTED, 400, italic=True)
    s.save(OUT / "rules.svg")


def timed_rep():
    W, H = 1240, 330
    s = SVG(W, H, title="One timed repetition in the harness")
    s.heading(30, 40, "What 'judge-style' timing means", "One pair of repetitions on the GPU stream, as okc_harness.py "
              "enqueues it. The next pair swaps the order.")
    y, x = 96, 30
    steps = [("hold", 70, "muted", ""), ("flush L2", 120, "amber", ""),
             ("ev", 30, "ink", ""), ("graph replay: candidate", 260, "blue", ""), ("ev", 30, "ink", ""),
             ("flush L2", 120, "amber", ""), ("ev", 30, "ink", ""), ("graph replay: reference", 260, "violet", ""),
             ("ev", 30, "ink", "")]
    for t, w, c, note in steps:
        s.rect(x, y, w - 4, 44, fill=TINTS[c], stroke=COLORS[c], r=5)
        s.text(x + (w - 4) / 2, y + 27, t, 12.5 if w > 40 else 10.5, INK, 650, "middle")
        if note:
            s.text(x + (w - 4) / 2, y + 64, note, 11, MUTED, 500, "middle")
        x += w
    s.text(30, y - 10, "hold = cuStreamWaitValue32 on a pinned host word, released only after the whole pair is queued, so a descheduled "
           "launcher cannot land inside a timed interval; flush = a read of twice the L2; ev = CUDA event", 11.5, MUTED, 500)
    notes = ["20 fresh inputs per shape, copied into both sides' captured buffers; every replay starts with a cold L2",
             "fixed replay cost (2 x one-node graph minus a two-node graph, about 4.35 µs) subtracted from both sides",
             "pair ratio = geometric mean of the two orders; a shape's speedup = median over the pairs",
             "a load's time = 46 sliding + 16 full layers; the score = geometric mean of the six loads",
             "with two runners on a GPU, a bench holds the GPU lock shared and takes it exclusive only while it times"]
    for k, t in enumerate(notes):
        s.text(30, 210 + 22 * k, "• " + t, 12.5, INK)
    s.save(OUT / "timed_rep.svg")


def providers():
    W, H = 1240, 470
    s = SVG(W, H, title="GPU providers over time")
    s.heading(30, 40, "Where the GPUs came from, and what each cost", "Bars are rental periods; red ticks are Nebius "
              "preemptions.")
    x0, x1 = 250, 930
    t0, t1 = "2026-11-02T00:00", "2026-11-05T08:00"
    span = _hours(t1, t0)
    X = lambda t: x0 + (x1 - x0) * _hours(t, t0) / span  # noqa: E731
    for d, lab in (("2026-11-02T00:00", "Nov 2"), ("2026-11-03T00:00", "Nov 3"), ("2026-11-04T00:00", "Nov 4"),
                   ("2026-11-05T00:00", "Nov 5")):
        s.raw(f'<line x1="{X(d):.1f}" y1="80" x2="{X(d):.1f}" y2="330" stroke="#EDF0F4"/>')
        s.text(X(d) + 4, 86, lab, 11.5, MUTED, 600)
    rows = [("RunPod H200 SXM", [("2026-11-02T05:03", "2026-11-02T07:30")], "teal", "$5.29/h, about $12.5"),
            ("Lium H200 (container)", [("2026-11-02T10:09", "2026-11-02T20:12")], "teal", "$3.75/h, $38.52"),
            ("Modal H200 (profiles)", [("2026-11-02T07:00", "2026-11-04T12:00")], "muted", "per second, $49.52"),
            ("Vast.ai H200 SXM", [("2026-11-02T16:00", "2026-11-03T05:22"), ("2026-11-03T05:36", "2026-11-04T22:48"),
                                   ("2026-11-05T03:10", "2026-11-05T07:40")], "blue", "$3.02/GPU-h on demand, $395.64+"),
            ("Vast.ai CPU build boxes", [("2026-11-03T12:01", "2026-11-05T07:40")], "amber", "under $0.50/h combined"),
            ("Nebius H200 SXM spot", [("2026-11-04T21:26", "2026-11-05T07:40")], "green", "spot, $2.50/GPU-h cap, about $35 to $43")]
    for k, (name, spans, c, cost) in enumerate(rows):
        y = 100 + k * 38
        s.text(30, y + 18, name, 12.5, INK, 600)
        for a, b in spans:
            s.rect(X(a), y + 4, X(b) - X(a), 20, fill=TINTS[c], stroke=COLORS[c], r=4,
                   dash="4 3" if name.startswith("Modal") else None)
        s.text(945, y + 19, cost, 12, MUTED, 500)
    for t in ("2026-11-04T23:27", "2026-11-04T23:48", "2026-11-04T23:51", "2026-11-05T00:28", "2026-11-05T01:42",
              "2026-11-05T01:46", "2026-11-05T01:59", "2026-11-05T02:21", "2026-11-05T07:06", "2026-11-05T07:10"):
        s.raw(f'<line x1="{X(t):.1f}" y1="{100 + 5 * 38 + 1}" x2="{X(t):.1f}" y2="{100 + 5 * 38 + 27}" '
              f'stroke="{COLORS["red"]}" stroke-width="2"/>')
    # inset: Nebius spot availability under a cap
    bx, by = 30, 360
    s.text(bx, by, "Nebius spot H200, the week before: share of 15-minute slots at or under a price (cheapest of 3 regions)",
           12.5, INK, 650)
    for k, (cap, pct) in enumerate(((1.0, 20), (1.5, 40), (2.0, 61), (2.5, 72), (3.0, 82))):
        x = bx + k * 236
        s.rect(x, by + 14, 220 * pct / 100, 24, fill=TINTS["green"], stroke=COLORS["green"], r=4)
        s.text(x + 6, by + 31, f"up to ${cap:.2f}: {pct}%", 12, INK, 650)
    s.text(bx, by + 66, "Sub-$1 slots clustered at 00:00 to 07:00 UTC. We ran under a $2.50 cap; the spot price at the "
           "switch was $1.60 in eu-north1.", 11.5, MUTED, 400, italic=True)
    s.save(OUT / "providers.svg")


def _bits(s, x, y, lit, label, cell=16, color="blue", show_idx=False):
    """a 32-bit strip, bit 31 on the left; lit: {bit: text}"""
    s.text(x - 10, y + cell - 3, label, 12.5, INK, 650, "end", mono=True)
    for b in range(32):
        cx = x + (31 - b) * cell
        on = b in lit
        s.rect(cx, y, cell - 2, cell, fill=TINTS[color] if on else "#FAFBFC", stroke=COLORS[color] if on else LINE,
               r=2, sw=1)
        if on and lit[b]:
            s.text(cx + (cell - 2) / 2, y + cell - 4, lit[b], 9, COLORS[color], 700, "middle", mono=True)
        if show_idx and b % 8 == 7:
            s.text(cx + (cell - 2) / 2, y - 4, str(b), 9, MUTED, 500, "middle")


def dqx():
    W, H = 1240, 640
    s = SVG(W, H, title="The dqx NVFP4 decode")
    s.heading(30, 40, "dqx: eight FP4 weights to BF16 in 8 instructions, 5 of them integer-pipe ops",
              "prepare() stores each 32-bit word through an invertible GF(2) map, so the kernel's decode is a few shifts, "
              "XORs and masks.")
    P = {6: "m", 7: "e0", 8: "e1", 15: "s", 22: "m", 23: "e0", 24: "e1", 31: "s"}
    x0 = 210
    _bits(s, x0, 92, {}, "q (stored)", color="muted", show_idx=True)
    s.text(x0 + 32 * 16 + 14, 105, "one 32-bit word = 8 NVFP4 codes, in a layout chosen by search", 12, MUTED)
    views = [("R = rotr(q, 6)", "SHF.W: integer pipe (cost model)", "amber"), ("R << 2", "IMAD.SHL: FMA pipe (cost model)", "teal"),
             ("q << 4", "IMAD.SHL: FMA pipe (cost model)", "teal"), ("q << 6", "IMAD.SHL: FMA pipe (cost model)", "teal")]
    for k, (t, pipe, c) in enumerate(views):
        x = 30 + k * 300
        s.rect(x, 132, 280, 46, fill=TINTS[c], stroke=COLORS[c], r=8)
        s.text(x + 14, 152, t, 13, INK, 700, mono=True)
        s.text(x + 14, 170, pipe, 11.5, MUTED, 500)
    regs = [("r0", "M & q"), ("r1", "M & (q ^ R<<2)"), ("r2", "M & (R ^ q<<4)"), ("r3", "M & (R ^ q<<6)")]
    for k, (r, f) in enumerate(regs):
        y = 212 + k * 40
        s.rect(30, y - 2, 150, 22, fill=TINTS["amber"], stroke=COLORS["amber"], r=5)
        s.text(40, y + 14, f"LOP3  {f}", 11, INK, 600, mono=True)
        _bits(s, x0 + 40, y, P, r, color="blue")
    s.text(x0 + 40 + 32 * 16 + 14, 230, "M = 0x81C081C0: each register's 8 lit bits are two", 12, MUTED)
    s.text(x0 + 40 + 32 * 16 + 14, 248, "BF16 halves, (s, e1, e0, m) at bits 15/8/7/6 and 31/24/23/22,", 12, MUTED)
    s.text(x0 + 40 + 32 * 16 + 14, 266, "which is the E2M1 value x 2^-126 exactly. One bf16x2", 12, MUTED)
    s.text(x0 + 40 + 32 * 16 + 14, 284, "multiply by scale x 2^126 then gives the exact product.", 12, MUTED)
    s.text(x0 + 40 + 32 * 16 + 14, 314, "Some stored bits are read 3 times and the extra copies", 12, MUTED)
    s.text(x0 + 40 + 32 * 16 + 14, 332, "cancel through the XOR partner, so the layout has to be", 12, MUTED)
    s.text(x0 + 40 + 32 * 16 + 14, 350, "an XOR mix, not a permutation.", 12, MUTED)

    # forms table
    ty = 390
    s.text(30, ty, "Same 32 decoded bits, three layouts", 14, INK, 650)
    rows = [("DQ7, inherited", "7", "2", "1.0131"), ("dqm, a public solution", "7", "1", "1.0146"),
            ("dqx, found by search", "5", "3", "1.0171")]
    hdr = ("layout", "integer-pipe ops", "FMA-pipe ops", "screen vs production")
    xs = (30, 300, 470, 630)
    for k, h in enumerate(hdr):
        s.text(xs[k], ty + 28, h, 12, MUTED, 650)
    for i, row in enumerate(rows):
        y = ty + 52 + i * 30
        if i == 2:
            s.rect(22, y - 19, 760, 28, fill=TINTS["green"], stroke="none", r=5)
        for k, v in enumerate(row):
            s.text(xs[k], y, v, 13, INK, 700 if i == 2 else 500)
        bw = (float(row[3]) - 1.010) * 9000
        s.rect(720, y - 13, bw, 16, fill=COLORS["green"] if i == 2 else LINE, stroke="none", r=3)
    s.text(30, ty + 160, "200 paired, role-swapped repetitions each, same GPU, base and seeds. dqx: +0.39% over DQ7, +0.25% "
           "over dqm, +0.7 to 1.0% at the large loads.", 12, MUTED, 400, italic=True)
    s.text(30, ty + 182, "Removing it from the incumbent later cost 0.72% in a screen: the largest single piece of the "
           "incumbent at the time.", 12, MUTED, 400, italic=True)
    s.rect(820, ty - 10, 400, 180, fill=SOFT_, stroke=LINE, r=10)
    s.text(838, ty + 16, "Why only +0.4%", 13.5, INK, 650)
    for k, t in enumerate(["the integer pipe was 48% busy, the FMA", "pipe 42%: issue-throttled, but not the",
                           "72% the public author saw. Moving 2 ops", "per word to the FMA pipe relieves part",
                           "of it, and the stream is partly", "HBM-bound anyway."]):
        s.text(838, ty + 42 + 20 * k, t, 12, INK)
    s.save(OUT / "dqx.svg")


def ladder():
    rows = [("any ops that only move bits up (shifts left, masks, IMAD)", "impossible", "28 decoded bits from 25 inputs"),
            ("M & (q<<a ^ q<<b), 0 to 4 IMAD.SHL, no right mover", "0", "fp4_layout.c"),
            ("one logical right shift + up to 3 left, no chaining", "0", "fp4_layout2.c"),
            ("one logical right shift (IMAD.HI-able) + chained registers, shifts up to 31", "0", "8 min 24 s"),
            ("permutation layouts (four variants), BF16 and FP16 targets", "0", "fp4_layout4-6"),
            ("rotation + XOR pairs (+ chaining allowed) + up to 3 IMAD.SHL of q or R", "27,840", "dqx: 28 min"),
            ("7-op decodes: three views of q (two searches)", "0", "within the family"),
            ("7-op decodes, FP16 target: 1.01e9 candidates", "0", "within the family")]
    W, H = 1240, 110 + 44 * len(rows) + 90
    s = SVG(W, H, title="The impossibility ladder")
    s.heading(30, 40, "Each bound holds only inside the family searched",
              "Top to bottom: the families tried on Nov 3. The 'floor' we believed in that morning was 13 ops per 8 codes.")
    for k, (fam, res, note) in enumerate(rows):
        y = 84 + k * 44
        found = res not in ("0", "impossible")
        c = "green" if found else ("red" if res == "impossible" else "muted")
        s.rect(30, y, 760, 34, fill=TINTS[c] if found else "#FAFBFC", stroke=COLORS[c] if found else LINE, r=6)
        s.text(44, y + 22, fam, 12.5, INK, 650 if found else 500)
        s.pill(810, y + 22, res, c)
        s.text(930, y + 22, note, 12, MUTED, 500, mono=note.endswith(".c") or note.startswith("fp4"))
        if k == 5:
            s.raw(f'<line x1="20" y1="{y + 40}" x2="1220" y2="{y + 40}" stroke="{LINE}" stroke-dasharray="4 4"/>')
    y = 84 + len(rows) * 44 + 14
    s.text(30, y + 10, "The winning form uses a rotation (not a logical shift) as the right mover, XOR of two sources per mask, "
           "left shifts of the rotated word,", 12.5, INK)
    s.text(30, y + 32, "and an invertible non-permutation layout; it needs no chaining. From the public hint to the first "
           "solution: about 42 minutes.", 12.5, INK)
    s.save(OUT / "ladder.svg")


def egraph():
    W, H = 1240, 420
    s = SVG(W, H, title="One cost-model coefficient flips the e-graph's answer")
    s.heading(30, 40, "One coefficient in the cost model flipped the e-graph's answer",
              "egglog saturation of the layer's front half (33 rewrite rules), then exact extraction. Bytes streamed "
              "per token.")
    def bar(y, label, late, early, note, c):
        s.text(30, y + 20, label, 13, INK, 650)
        x = 330
        sc = 8.5
        s.rect(x, y + 4, late * sc, 26, fill=TINTS["red"], stroke=COLORS["red"], r=4)
        s.text(x + 8, y + 22, f"after attention {late} MB", 11.5, INK, 600)
        if early:
            s.rect(x + late * sc + 4, y + 4, early * sc, 26, fill=TINTS["amber"], stroke=COLORS["amber"], r=4)
            s.text(x + late * sc + 12, y + 22, f"under attention {early} MB", 11.5, INK, 600)
        s.text(30, y + 44, note, 12, MUTED)
    bar(86, "early bytes free (EARLY_W = 0)", 19.3, 58.7, "extracts the hoisted pooler/router: stream the input-only "
        "terms while attention runs", "red")
    bar(166, "early bytes charged (EARLY_W = 1)", 71.3, 0, "extracts production's own form; the back half too "
        "(50.3 MB, found by the MILP; greedy DAG extraction gave 75.5)", "blue")
    s.rect(30, 250, 1180, 140, fill=SOFT_, stroke=LINE, r=10)
    s.text(48, 276, "What the H200 said about the first answer", 13.5, INK, 650)
    s.text(48, 300, "hoisted pooler/router v1: 0.9386 of production; v2 on production's geometry: 0.9742; both 12/12 correct",
           12.5, INK)
    s.text(48, 322, "at decode-1 sliding, attention stretched from 6.8 to 11.7 µs: the 'idle' HBM under latency-bound "
           "attention is idle only until something uses it", 12.5, INK)
    s.text(48, 344, "the same contention killed a gate hoist (-5% at decode 1) and every cross-kernel prefetch we tried "
           "(-1 to -13%)", 12.5, INK)
    s.text(48, 370, "With the measured cost, the search correctly predicted that no linear rewrite of the dense algebra "
           "had a byte cut left.", 12.5, MUTED, 400, italic=True)
    s.save(OUT / "egraph.svg")


def margin():
    rows = [("b1 ctx 10 sliding", 0.179), ("b1 ctx 3000 full", 0.095), ("b8 ctx 2047 sliding", 0.064),
            ("b33 ctx 20000 full", 0.052), ("scored decode 1 sliding", 0.154), ("b3 q_len 8 full, outliers", 0.476),
            ("b2 ctx 300000 q_len 8", 0.533), ("b1 ctx 1M q_len 8 full", 0.608)]
    W, H = 1240, 120 + 30 * len(rows) + 60
    s = SVG(W, H, title="Spending the accuracy margin")
    s.heading(30, 40, "How much of the accuracy budget production used",
              "Worst of 50 seeds of the judge's input generator, error / tolerance on the residual update or the next "
              "attention input, whichever is larger (1,250 runs on an H200 with the judge's image).")
    x0, x1 = 300, 1100
    for v in (0, 0.25, 0.5, 0.75, 1.0):
        x = x0 + (x1 - x0) * v
        s.raw(f'<line x1="{x}" y1="80" x2="{x}" y2="{H - 50}" stroke="{"#D1495B" if v == 1 else "#EDF0F4"}" '
              f'stroke-width="{1.5 if v == 1 else 1}"/>')
        s.text(x, H - 34, "gate" if v == 1 else f"{v:g}", 11, MUTED, 600, "middle")
    for k, (lab, v) in enumerate(rows):
        y = 92 + k * 30
        c = "amber" if "q_len 8" in lab else "blue"
        s.text(x0 - 12, y + 14, lab, 12, INK, 500, "end")
        s.rect(x0, y + 2, (x1 - x0) * v, 16, fill=TINTS[c], stroke=COLORS[c], r=3)
        s.text(x0 + (x1 - x0) * v + 8, y + 15, f"{v:.3f}", 11.5, COLORS[c], 700)
    s.text(30, H - 12, "On these outputs decode (q_len 1) used 5 to 18% of its tolerance, verify (q_len 8) 43 to 61%. The new K/V "
           "entries sat at 0.17 to 0.40. Routing failures: 0.", 12, MUTED, 400, italic=True)
    s.save(OUT / "margin.svg")


def pdl_chain():
    import csv
    with open(OUT.parent / "data" / "pdl_d1s.csv") as f:
        ks = list(csv.DictReader(f))
    W, H = 1240, 120 + 40 * len(ks) + 110
    s = SVG(W, H, title="Decode 1 sliding as a PDL chain")
    s.heading(30, 40, "Decode 1, sliding layer: one layer in 55 µs",
              "nsys of one graph replay. Each kernel starts 1 to 5 µs before its predecessor ends (PDL). Bars: kernel "
              "start to end.")
    fam = {"proj": "blue", "attn": "violet", "moe": "red", "glue": "amber"}
    x0, x1, span = 250, 1180, 56.0
    X = lambda v: x0 + (x1 - x0) * v / span  # noqa: E731
    top = 90
    for v in range(0, 57, 10):
        s.raw(f'<line x1="{X(v):.1f}" y1="{top - 6}" x2="{X(v):.1f}" y2="{top + 40 * len(ks)}" stroke="#EDF0F4"/>')
        s.text(X(v), top - 10, f"{v} µs", 11, MUTED, 600, "middle")
    for k, kr in enumerate(ks):
        y = top + k * 40
        a, b = float(kr["start_us"]), float(kr["end_us"])
        c = fam[kr["family"]]
        s.text(x0 - 12, y + 20, kr["kernel"], 12.5, INK, 600, "end")
        s.rect(X(a), y + 6, X(b) - X(a), 22, fill=TINTS[c], stroke=COLORS[c], r=4)
        if k:
            pe = float(ks[k - 1]["end_us"])
            if pe > a:
                s.rect(X(a), y + 6, X(pe) - X(a), 22, fill=COLORS[c], stroke="none", r=4, opacity=0.25)
    y = top + 40 * len(ks) + 14
    s.raw(f'<line x1="{X(30.6):.1f}" y1="{top}" x2="{X(30.6):.1f}" y2="{y - 10}" stroke="{COLORS["teal"]}" '
          'stroke-width="2" stroke-dasharray="6 4"/>')
    s.text(X(30.6) + 6, y + 6, "30.6 µs: all 134.7 MB of the layer's bytes at 4.4 TB/s", 12, COLORS["teal"], 650)
    s.text(30, y + 34, "After the pooler ends, the route, experts, unpooler and finish take 21.2 µs for 34.6 MB that stream in "
           "7.9 µs. The rest is ramps, hand-offs and tails.", 12.5, INK)
    s.text(30, y + 56, "Shaded: the part of a kernel's lifetime that overlaps its predecessor (prologue and weight prefetch "
           "before griddepcontrol.wait).", 12, MUTED, 400, italic=True)
    s.save(OUT / "pdl_chain.svg")


def wgs_stage():
    W, H = 1240, 400
    s = SVG(W, H, title="Decode and tensor math serialize in the big MoE")
    s.heading(30, 40, "Big MoE: decode and tensor math run one after the other",
              "Clocks per 36 KB weight stage, the real stage loop on shared-memory-resident data, 132 CTAs x 4 "
              "warpgroups.")
    rows = [(8, 1503, 832, 1199), (32, 1909, 1024, 1199), (64, 2492, 2048, 1199)]
    x0, scale = 200, 0.31
    for k, (n, full, mma, alu) in enumerate(rows):
        y = 96 + k * 92
        s.text(x0 - 14, y + 18, f"{n} tokens", 13, INK, 650, "end")
        s.rect(x0, y, full * scale, 24, fill=TINTS["red"], stroke=COLORS["red"], r=4)
        s.text(x0 + full * scale + 10, y + 17, f"measured {full}", 12, COLORS["red"], 700)
        s.rect(x0, y + 30, alu * scale, 18, fill=TINTS["amber"], stroke=COLORS["amber"], r=3)
        s.text(x0 + 8, y + 43, f"decode (ALU) alone {alu}", 11, INK, 600)
        s.rect(x0 + alu * scale, y + 30, mma * scale, 18, fill=TINTS["violet"], stroke=COLORS["violet"], r=3)
        s.text(x0 + alu * scale + 8, y + 43, f"wgmma alone {mma}", 11, INK, 600)
        ov = max(alu, mma)
        s.rect(x0, y + 54, ov * scale, 12, fill="none", stroke=MUTED, r=3, dash="4 3")
        s.text(x0 + ov * scale + 8, y + 64, f"perfect overlap would be {ov}", 11, MUTED, 500)
    s.text(30, 362, "With register-A wgmma the decoding warps also issue the MMAs. Double-buffering the A fragments at 96 "
           "registers per thread", 12.5, INK)
    s.text(30, 382, "makes ptxas insert a wait after every wgmma (C7511/C7512). Three redesigns of the big MoE lost 3 to 27%.",
           12.5, INK)
    s.save(OUT / "wgs_stage.svg")

if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    banner()
    pipeline()
    layer()
    funnel()
    agents()
    noise()
    waterfall()
    ctlword()
    lean()
    predictions()
    gantt()
    rules()
    timed_rep()
    providers()
    dqx()
    ladder()
    egraph()
    margin()
    pdl_chain()
    wgs_stage()
