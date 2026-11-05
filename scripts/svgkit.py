"""The repo's visual style and a small SVG builder. Every diagram in figures/ is drawn with this module, so they share
one palette, one type scale and one set of shapes. Charts (matplotlib) take the same palette through `mpl_style()`.

Style:
  ink #1B2430 text and strokes      muted #5B6573 secondary text    line #C9D1DB rules and frames
  paper #FFFFFF                     soft #F4F6F9 panels
  blue  #2F6FDE ours / candidates   red #D1495B the judge, limits   green #2A9D6F confirmed, adopted
  amber #E0A030 queues, waiting     violet #7C5CC4 agents, people   teal #1F9AA8 GPUs, hardware
  Each color has a light tint (`tint`) for fills; strokes use the full color.
  Type: one sans stack, one mono stack; sizes 11 (notes), 13 (labels), 15 (node titles), 22+ (headings).
"""
import html

INK, MUTED, LINE, PAPER, SOFT = "#1B2430", "#5B6573", "#C9D1DB", "#FFFFFF", "#F4F6F9"
COLORS = {"blue": "#2F6FDE", "red": "#D1495B", "green": "#2A9D6F", "amber": "#E0A030", "violet": "#7C5CC4",
          "teal": "#1F9AA8", "ink": INK, "muted": MUTED}
TINTS = {"blue": "#E7EFFC", "red": "#FBEAED", "green": "#E4F4EE", "amber": "#FCF3E1", "violet": "#EFEAF9",
         "teal": "#E2F3F5", "ink": SOFT, "muted": SOFT}
SANS = "Inter, 'Segoe UI', -apple-system, Helvetica, Arial, sans-serif"
MONO = "'JetBrains Mono', 'SF Mono', Menlo, Consolas, monospace"


def esc(t):
    return html.escape(str(t), quote=True)


class SVG:
    def __init__(self, w, h, bg=PAPER, title=""):
        self.w, self.h, self.items = w, h, []
        self.head = [
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" '
            f'font-family="{SANS}" role="img" aria-label="{esc(title)}">',
            f"<title>{esc(title)}</title>",
            "<defs>",
            *(f'<marker id="arr-{k}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
              f'orient="auto-start-reverse"><path d="M0,1 L9,5 L0,9 z" fill="{c}"/></marker>' for k, c in COLORS.items()),
            '<filter id="lift" x="-10%" y="-10%" width="120%" height="140%"><feDropShadow dx="0" dy="1.5" '
            'stdDeviation="1.6" flood-color="#1B2430" flood-opacity="0.12"/></filter>',
            "</defs>",
        ]
        if bg:
            self.items.append(f'<rect width="{w}" height="{h}" fill="{bg}"/>')

    def raw(self, s):
        self.items.append(s)

    def text(self, x, y, t, size=13, color=INK, weight=400, anchor="start", mono=False, italic=False, opacity=1):
        fam = f' font-family="{MONO}"' if mono else ""
        sty = ' font-style="italic"' if italic else ""
        self.items.append(f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" font-weight="{weight}" '
                          f'text-anchor="{anchor}"{fam}{sty} opacity="{opacity}">{esc(t)}</text>')

    def lines(self, x, y, rows, size=13, color=INK, weight=400, anchor="start", gap=1.35, mono=False):
        for i, r in enumerate(rows):
            self.text(x, y + i * size * gap, r, size, color, weight, anchor, mono)

    def rect(self, x, y, w, h, fill=SOFT, stroke=LINE, r=8, sw=1.2, dash=None, shadow=False, opacity=1):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        f = ' filter="url(#lift)"' if shadow else ""
        self.items.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{fill}" stroke="{stroke}" '
                          f'stroke-width="{sw}"{d}{f} opacity="{opacity}"/>')

    def node(self, x, y, w, h, title, sub=(), color="blue", mono_sub=False, align="center"):
        """a card: tinted fill, a colored left band, a title and up to a few sub lines"""
        self.rect(x, y, w, h, fill=TINTS[color], stroke=COLORS[color], sw=1.3, shadow=True)
        self.items.append(f'<rect x="{x}" y="{y}" width="5" height="{h}" rx="2.5" fill="{COLORS[color]}"/>')
        n = 1 + len(sub)
        top = y + h / 2 - (n - 1) * 9 + 5
        cx, anchor = (x + w / 2 + 2, "middle") if align == "center" else (x + 16, "start")
        self.text(cx, top, title, 15, INK, 650, anchor)
        for i, s in enumerate(sub):
            self.text(cx, top + 19 + i * 17, s, 12, MUTED, 400, anchor, mono=mono_sub)

    def pill(self, x, y, t, color="blue", size=11.5, pad=8):
        w = len(str(t)) * size * 0.58 + 2 * pad
        self.rect(x, y - size - 3, w, size + 9, fill=TINTS[color], stroke=COLORS[color], r=(size + 9) / 2, sw=1)
        self.text(x + w / 2, y + 1, t, size, COLORS[color], 600, "middle")
        return w

    def arrow(self, pts, color="ink", label=None, lx=None, ly=None, dash=None, sw=1.6, both=False, lsize=11.5):
        d = "M" + " L".join(f"{a},{b}" for a, b in pts)
        da = f' stroke-dasharray="{dash}"' if dash else ""
        start = f' marker-start="url(#arr-{color})"' if both else ""
        self.items.append(f'<path d="{d}" fill="none" stroke="{COLORS[color]}" stroke-width="{sw}"{da}'
                          f' marker-end="url(#arr-{color})"{start} stroke-linejoin="round"/>')
        if label:
            if lx is None:
                (x0, y0), (x1, y1) = pts[len(pts) // 2 - 1], pts[len(pts) // 2]
                lx, ly = (x0 + x1) / 2, (y0 + y1) / 2 - 6
            self.text(lx, ly, label, lsize, MUTED, 500, "middle")

    def frame(self, x, y, w, h, label, color="muted"):
        self.rect(x, y, w, h, fill="none", stroke=LINE, r=12, sw=1.2, dash="5 4")
        self.text(x + 14, y + 20, label.upper(), 11, COLORS[color], 700)

    def heading(self, x, y, title, sub=None):
        self.text(x, y, title, 20, INK, 700)
        if sub:
            self.text(x, y + 22, sub, 13, MUTED)

    def save(self, path):
        with open(path, "w") as f:
            f.write("\n".join(self.head + self.items + ["</svg>"]) + "\n")


def mpl_style():
    """matplotlib settings matching the SVG style (charts are saved as SVG with text as paths)"""
    import matplotlib
    matplotlib.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Inter", "DejaVu Sans", "Arial"], "font.size": 10.5,
        "axes.edgecolor": LINE, "axes.labelcolor": INK, "axes.titlecolor": INK, "axes.titleweight": "bold",
        "axes.titlesize": 13, "axes.titlelocation": "left", "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": "#E6EAF0", "grid.linewidth": 0.8, "axes.axisbelow": True,
        "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK, "legend.frameon": False,
        "svg.fonttype": "path", "figure.facecolor": PAPER, "savefig.facecolor": PAPER,
    })
