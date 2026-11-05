"""E-graph model of one Muse decode layer (per token), saturated under linear-algebra rewrites.

Run: .venv-alg/bin/python algebra/layer_egraph.py

Every rewrite here is an identity over the reals; algebra/proofs.py proves each one (SymPy on
explicit matrices, Z3 on the scalar forms). Extraction uses a cost that separates three kinds of
work by when its inputs exist:
  pre   : needed by attention (q, k, v)          - on the critical path before attention
  early : depends only on the layer's inputs     - can run while attention runs
  late  : depends on the attention output z      - on the critical path after attention
and charges each streamed weight its bytes in the class it lands in. Constant subterms (products of
weights and norm vectors) are prepare-time and cost only their stored bytes when streamed.
"""

from __future__ import annotations

import dataclasses
import sys

from egglog import (EGraph, function, Expr, StringLike, String, f64, f64Like, i64, i64Like, method, rewrite, rule, ruleset,
                    vars_, eq, union)

EG = EGraph()


class CV(Expr):
    """A constant (weight-derived) vector."""

    def __init__(self, name: StringLike, n: i64Like) -> None: ...

    @classmethod
    def mul(cls, a: CV, b: CV) -> CV: ...


class Mat(Expr):
    """A constant matrix: a weight (bytes per element bpe x 100), or a prepare-time product."""

    def __init__(self, name: StringLike, rows: i64Like, cols: i64Like, bpe100: i64Like) -> None: ...

    @classmethod
    def mm(cls, a: Mat, b: Mat) -> Mat: ...

    @classmethod
    def diag(cls, c: CV) -> Mat: ...

    @classmethod
    def tr(cls, a: Mat) -> Mat: ...


class S(Expr):
    """A per-token scalar."""

    def __init__(self, name: StringLike) -> None: ...

    @classmethod
    def mul(cls, a: S, b: S) -> S: ...

    @classmethod
    def add(cls, a: S, b: S) -> S: ...

    @classmethod
    def inv(cls, a: S) -> S: ...

    @classmethod
    def sqrt(cls, a: S) -> S: ...


class V(Expr):
    """A per-token vector."""

    def __init__(self, name: StringLike, n: i64Like, when: i64Like) -> None: ...  # when: 0 pre, 1 early, 2 late

    @classmethod
    def mv(cls, a: Mat, v: V) -> V: ...

    @classmethod
    def had(cls, c: CV, v: V) -> V: ...

    @classmethod
    def add(cls, u: V, v: V) -> V: ...

    @classmethod
    def sc(cls, s: S, v: V) -> V: ...


@function
def dot(u: V, v: V) -> S: ...


A, B, C = vars_("A B C", Mat)
c, d = vars_("c d", CV)
u, v, w = vars_("u v w", V)
x, y, z = vars_("x y z", S)

linear = ruleset(
    # matrix-vector products are linear
    rewrite(V.mv(A, V.add(u, v))).to(V.add(V.mv(A, u), V.mv(A, v))),
    rewrite(V.add(V.mv(A, u), V.mv(A, v))).to(V.mv(A, V.add(u, v))),
    rewrite(V.mv(A, V.sc(x, v))).to(V.sc(x, V.mv(A, v))),
    rewrite(V.sc(x, V.mv(A, v))).to(V.mv(A, V.sc(x, v))),
    # constant diagonals and constant products fold into prepare-time matrices
    rewrite(V.mv(A, V.had(c, v))).to(V.mv(Mat.mm(A, Mat.diag(c)), v)),
    rewrite(V.had(c, V.mv(A, v))).to(V.mv(Mat.mm(Mat.diag(c), A), v)),
    rewrite(V.mv(A, V.mv(B, v))).to(V.mv(Mat.mm(A, B), v)),
    rewrite(Mat.mm(Mat.diag(c), Mat.diag(d))).to(Mat.diag(CV.mul(c, d))),
    rewrite(Mat.mm(A, Mat.mm(B, C))).to(Mat.mm(Mat.mm(A, B), C)),
    rewrite(Mat.mm(Mat.mm(A, B), C)).to(Mat.mm(A, Mat.mm(B, C))),
    rewrite(Mat.tr(Mat.mm(A, B))).to(Mat.mm(Mat.tr(B), Mat.tr(A))),
    rewrite(Mat.tr(Mat.tr(A))).to(A),
    rewrite(Mat.tr(Mat.diag(c))).to(Mat.diag(c)),
    # elementwise constant scaling
    rewrite(V.had(c, V.add(u, v))).to(V.add(V.had(c, u), V.had(c, v))),
    rewrite(V.add(V.had(c, u), V.had(c, v))).to(V.had(c, V.add(u, v))),
    rewrite(V.had(c, V.sc(x, v))).to(V.sc(x, V.had(c, v))),
    rewrite(V.sc(x, V.had(c, v))).to(V.had(c, V.sc(x, v))),
    rewrite(V.had(c, V.had(d, v))).to(V.had(CV.mul(c, d), v)),
    rewrite(CV.mul(c, d)).to(CV.mul(d, c)),
    rewrite(V.sc(x, V.sc(y, v))).to(V.sc(S.mul(x, y), v)),
    rewrite(V.sc(x, V.add(u, v))).to(V.add(V.sc(x, u), V.sc(x, v))),
    rewrite(V.add(u, v)).to(V.add(v, u)),
    # inner products (sums of squares are dot(v, v))
    rewrite(dot(u, v)).to(dot(v, u)),
    rewrite(dot(V.add(u, v), w)).to(S.add(dot(u, w), dot(v, w))),
    rewrite(dot(V.sc(x, u), w)).to(S.mul(x, dot(u, w))),
    rewrite(dot(V.mv(A, u), w)).to(dot(u, V.mv(Mat.tr(A), w))),
    rewrite(dot(V.had(c, u), w)).to(dot(u, V.had(c, w))),
    # scalar field
    rewrite(S.mul(x, y)).to(S.mul(y, x)),
    rewrite(S.add(x, y)).to(S.add(y, x)),
    rewrite(S.mul(x, S.mul(y, z))).to(S.mul(S.mul(x, y), z)),
    rewrite(S.mul(S.mul(x, y), z)).to(S.mul(x, S.mul(y, z))),
    rewrite(S.add(x, S.add(y, z))).to(S.add(S.add(x, y), z)),
    rewrite(S.mul(x, S.add(y, z))).to(S.add(S.mul(x, y), S.mul(x, z))),
)


# ---------------------------------------------------------------- the layer, front half (per token)
H, DQ, DE, NE = 8192, 1024, 4096, 256
s = V("s", H, 1)            # residual stream (input, FP32)
xin = V("x", H, 1)          # attention input (input, BF16)
zatt = V("z_att", DQ, 2)    # rms_head(attention out): exists only after attention
Wg = Mat("Wgate", DQ, H, 150)      # the qkv_gate's gate rows (12-bit packed BF16)
O = Mat("O", H, DQ, 150)
P = Mat("P", DE, H, 150)
R = Mat("R", NE, H, 400)           # FP32 router
alpha1, beta1, w1 = CV("alpha1", H), CV("beta1", H), CV("1+w_preffn", H)
inv_n = S("1/8192")
eps_a, eps_s = S("1e-8"), S("1e-5")

g = V.mv(Wg, xin)
zz = V("z", DQ, 2)          # z = z_att * sigmoid(g): the nonlinear gate is a leaf for the algebra
a = V.mv(O, zz)
ra = S.sqrt(S.add(S.mul(inv_n, dot(a, a)), eps_a))
s1 = V.add(V.had(alpha1, s), V.sc(S.inv(ra), V.had(beta1, a)))
rs1sq = S.add(S.mul(inv_n, dot(s1, s1)), eps_s)
hun = V.had(w1, s1)                      # h * rs1 (rs1 is a positive per-token scalar)
logits_un = V.mv(R, hun)                 # logits * rs1
pool_un = V.mv(P, hun)                   # pooler output * rs1 (its RMS norm absorbs rs1 exactly, eps rescaled)

ROOTS = {"logits*rs1": logits_un, "pool*rs1": pool_un, "rs1^2": rs1sq, "gate": g}


# ---------------------------------------------------------------- cost model
@dataclasses.dataclass(frozen=True)
class Cost:
    kind: str                 # "mat", "cv", "s", "v", "str", "num"
    when: int = 0             # vec / scalar: max input class (0 pre, 1 early, 2 late)
    late: int = 0             # bytes streamed on the late (post-attention) path
    early: int = 0            # bytes streamed input-only (overlappable with attention)
    red: int = 0              # full-vector reductions of late vectors (grid-wide sums)
    rdepth: int = 0           # dependent levels of those reductions (each level is a pass / hand-off)
    ops: int = 0              # node count, tie-break
    rows: int = 0
    cols: int = 0
    bpe100: int = 0
    lit: object = None

    def key(self):
        if self.kind == "mat":
            return (self.rows * self.cols * self.bpe100, self.ops)
        return (self.late + EARLY_W * self.early, self.rdepth, self.red, self.ops)

    def __lt__(self, o):
        return self.key() < o.key()

    def __le__(self, o):
        return self.key() <= o.key()

    def __eq__(self, o):
        return isinstance(o, Cost) and self.key() == o.key()

    def __hash__(self):
        return hash(self.key())


PRODUCT_BPE100 = 150   # stored precision of a prepare-time product (BF16, 12-bit packed)
# Early (input-only) bytes overlap the latency-bound attention, but HPR measured that their HBM traffic stretches
# attention one for one (notes/dead/hpr_hoisted_pooler_router.md): EARLY_W = 1 charges them in full (the measured
# model); EARLY_W = 0 is the original lexicographic model that extracted HPR.
EARLY_W = 1.0


def _name(expr) -> str:
    return type(expr).__name__ + "." + str(expr).split("(")[0].split(".")[-1]


def cost_model(egraph, expr, kids):
    head = str(expr).split("(")[0]
    if isinstance(expr, (String,)):
        return Cost("str")
    if isinstance(expr, (i64, f64)):
        return Cost("num", lit=int(str(expr)[4:-1]) if isinstance(expr, i64) else 0)
    if isinstance(expr, CV):
        return Cost("cv", ops=1 + sum(k.ops for k in kids if k.kind == "cv"))
    if isinstance(expr, Mat):
        if head.endswith("diag"):
            return Cost("mat", rows=1, cols=1, bpe100=0, ops=1)
        if head.endswith("tr"):
            k = kids[0]
            return dataclasses.replace(k, rows=k.cols, cols=k.rows, ops=k.ops + 1)
        if head.endswith("mm"):
            l, r = kids
            if l.bpe100 == 0:   # diag on the left: same shape as the right
                return dataclasses.replace(r, ops=l.ops + r.ops + 1, bpe100=max(r.bpe100, PRODUCT_BPE100))
            if r.bpe100 == 0:
                return dataclasses.replace(l, ops=l.ops + r.ops + 1, bpe100=max(l.bpe100, PRODUCT_BPE100))
            return Cost("mat", rows=l.rows, cols=r.cols, bpe100=PRODUCT_BPE100, ops=l.ops + r.ops + 1)
        nums = [k.lit for k in kids if k.kind == "num"]
        return Cost("mat", rows=nums[0], cols=nums[1], bpe100=nums[2], ops=1)
    vk = [k for k in kids if k.kind in ("v", "s")]
    when = max([k.when for k in vk], default=0)
    late = sum(k.late for k in vk)
    early = sum(k.early for k in vk)
    red = sum(k.red for k in vk)
    rdepth = max([k.rdepth for k in vk], default=0)
    ops = 1 + sum(k.ops for k in kids)
    if isinstance(expr, V) and head == "V":
        return Cost("v", when=[k.lit for k in kids if k.kind == "num"][1], ops=1)
    if isinstance(expr, V) and head.endswith("mv"):
        m = kids[0]
        nbytes = m.rows * m.cols * m.bpe100 // 100
        if when == 2:
            late += nbytes
        else:
            early += nbytes
        return Cost("v", when=when, late=late, early=early, red=red, rdepth=rdepth, ops=ops)
    if head == "dot":
        return Cost("s", when=when, late=late, early=early, red=red + (when == 2), rdepth=rdepth + (when == 2), ops=ops)
    return Cost("v" if isinstance(expr, V) else "s", when=when, late=late, early=early, red=red, rdepth=rdepth, ops=ops)


# ---------------------------------------------------------------- the layer, back half (per token)
# Class 1 = exists before the MoE ends (s1, the routing weights), class 2 = the MoE's outputs. Three routed experts.
U = Mat("U", H, DE, 150)
alpha2, beta2, w3 = CV("alpha2", H), CV("beta2", H), CV("1+w_postffn", H)
s1_ = V("s1", H, 1)
vs_ = [V(f"v{e}", DE, 2) for e in range(3)]          # (1 + w_pe) y_e: the experts' rows before their RMS norm
cs_ = [S(f"c{e}") for e in range(3)]                 # wt_e / rms(v_e): needs each v_e's own reduction
mix = V.add(V.sc(cs_[0], vs_[0]), V.add(V.sc(cs_[1], vs_[1]), V.sc(cs_[2], vs_[2])))
fpost = V.had(w3, V.mv(U, mix))
rf = S.sqrt(S.add(S.mul(inv_n, dot(fpost, fpost)), eps_a))
s2 = V.add(V.had(alpha2, s1_), V.sc(S.inv(rf), V.had(beta2, fpost)))
rs2sq = S.add(S.mul(inv_n, dot(s2, s2)), eps_s)
BACK = {"f'": fpost, "rs2^2": rs2sq}


@function
def bundle(a: V, b: V, c: S, d: V) -> S: ...


# the layer's outputs together, so a subterm the outputs share (a = O z, the unpooled f) is charged once: the front
# half must also materialize s1 (the back half's input), the back half s2 (the layer's stream output)
LAYER_FRONT = {"front": bundle(logits_un, pool_un, rs1sq, s1)}
LAYER_BACK = {"back": bundle(fpost, s2, rs2sq, fpost)}


def extract_all(roots, rules, iters, label):
    eg = EGraph()
    for r in roots.values():
        eg.register(r)
    eg.run(rules * iters)
    print(f"#### {label} (EARLY_W = {EARLY_W})")
    for name, r in roots.items():
        term, cost = eg.extract(r, include_cost=True, cost_model=cost_model)
        print(f"== {name}: late {cost.late / 1e6:.1f} MB, early {cost.early / 1e6:.1f} MB, late reductions {cost.red}"
              f" in {cost.rdepth} dependent levels")
        print("  ", term)


def main():
    global EARLY_W
    for w in (0.0, 1.0):
        EARLY_W = w
        extract_all(ROOTS, linear, 7, "front half")
    EARLY_W = 1.0
    extract_all(BACK, linear, 6, "back half")


if __name__ == "__main__":
    main()
