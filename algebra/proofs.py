"""Proofs of the rewrites used in layer_egraph.py and of the restructured front half.

1. SymPy: every vector rewrite as a symbolic identity on explicit small matrices (all entries free symbols).
2. Z3: the scalar identities (the RMS-norm eps rescaling and the rs1^2 expansion) over the reals.
3. Float64 end to end: the restructured front half against reference.py's formulas at full size.

Run: .venv-alg/bin/python algebra/proofs.py
"""

import numpy as np
import sympy as sp
import z3


def sympy_rewrites():
    n, m, k = 3, 2, 4
    A = sp.Matrix(m, n, lambda i, j: sp.Symbol(f"A{i}{j}"))
    Bm = sp.Matrix(n, k, lambda i, j: sp.Symbol(f"B{i}{j}"))
    u = sp.Matrix(n, 1, lambda i, j: sp.Symbol(f"u{i}"))
    v = sp.Matrix(n, 1, lambda i, j: sp.Symbol(f"v{i}"))
    w = sp.Matrix(m, 1, lambda i, j: sp.Symbol(f"w{i}"))
    vk = sp.Matrix(k, 1, lambda i, j: sp.Symbol(f"q{i}"))
    c = sp.Matrix(n, 1, lambda i, j: sp.Symbol(f"c{i}"))
    d = sp.Matrix(n, 1, lambda i, j: sp.Symbol(f"d{i}"))
    x = sp.Symbol("x")
    had = lambda c_, v_: sp.Matrix([c_[i] * v_[i] for i in range(c_.rows)])  # noqa: E731
    D = sp.diag(*c)
    checks = {
        "mv linear (add)": A * (u + v) - (A * u + A * v),
        "mv linear (scale)": A * (x * u) - x * (A * u),
        "diag fold": A * had(c, u) - (A * D) * u,
        "diag fold (left)": had(w, A * u) - (sp.diag(*w) * A) * u,
        "product fold": A * (Bm * vk) - (A * Bm) * vk,
        "diag diag": sp.diag(*c) * sp.diag(*d) - sp.diag(*had(c, d)),
        "had distributes": had(c, u + v) - (had(c, u) + had(c, v)),
        "dot add": sp.Matrix([((u + v).T * c)[0] - ((u.T * c)[0] + (v.T * c)[0])]),
        "dot transpose": sp.Matrix([((A * u).T * w)[0] - (u.T * (A.T * w))[0]]),
        "dot had": sp.Matrix([(had(c, u).T * v)[0] - (u.T * had(c, v))[0]]),
        "transpose of product": (A * Bm).T - Bm.T * A.T,
    }
    for name, e in checks.items():
        assert sp.simplify(e) == sp.zeros(*e.shape), name
        print(f"  sympy  {name}: identity")


def z3_scalars():
    s = z3.Solver()
    # RMS norm absorbs a positive per-token scale exactly when its eps is rescaled:
    #   p / sqrt(mean(p^2) + eps) == u / sqrt(mean(u^2) + eps * r^2),  p = u / r,  r > 0.
    # Squared and cross-multiplied form (both sides share the sign of u); msq = mean(u^2) >= 0.
    uu, r, msq, eps = z3.Reals("u r msq eps")
    lhs_sq = (uu / r) ** 2 / (msq / r ** 2 + eps)
    rhs_sq = uu ** 2 / (msq + eps * r ** 2)
    s.add(r > 0, msq >= 0, eps > 0, lhs_sq != rhs_sq)
    print("  z3     rms eps rescale:", "proved" if s.check() == z3.unsat else s.model())
    # rs1^2 expansion: sum (a1 s + b1 a / ra)^2 = sum a1^2 s^2 + (2 / ra) sum a1 b1 s a + (1 / ra^2) sum b1^2 a^2
    # (per element; summation is linear, so the element identity suffices).
    a1, b1, ss, aa, ra = z3.Reals("a1 b1 s a ra")
    s2 = z3.Solver()
    s2.add(ra > 0, (a1 * ss + b1 * aa / ra) ** 2 != a1 ** 2 * ss ** 2 + 2 / ra * a1 * b1 * ss * aa
           + b1 ** 2 * aa ** 2 / ra ** 2)
    print("  z3     rs1^2 expansion:", "proved" if s2.check() == z3.unsat else s2.model())


def rms(v, eps):
    return np.sqrt((v * v).mean(-1, keepdims=True) + eps)


def end_to_end(T=4, seed=0):
    """reference.py's ref_front (attention output given) vs the hoisted form, float64, full size."""
    rng = np.random.default_rng(seed)
    H, DQ, DE, NE = 8192, 1024, 4096, 256
    O = rng.standard_normal((H, DQ)) / np.sqrt(DQ)
    P = rng.standard_normal((DE, H)) / np.sqrt(H)
    R = 0.045 * rng.standard_normal((NE, H))
    gate1, wpre = 0.3 * rng.standard_normal(H), 0.1 * rng.standard_normal(H)
    w_ = gate1 / 0.3
    beta1 = 1 / (1 + np.exp(-w_))
    alpha1 = np.sqrt(np.maximum((1 / (1 + np.exp(w_))) * (1 + beta1), 1e-3))
    s = rng.standard_normal((T, H))
    z = rng.standard_normal((T, DQ))
    # reference
    a = z @ O.T
    s1 = alpha1 * s + beta1 * a / rms(a, 1e-8)
    h = s1 / rms(s1, 1e-5) * (1 + wpre)
    logits_ref, pool = h @ R.T, h @ P.T
    p_ref = pool / rms(pool, 1e-5)
    # hoisted: early terms from s alone, late terms from z through prepare-time products
    c_s, c_a = alpha1 * (1 + wpre), beta1 * (1 + wpre)
    Q, M = P @ (c_a[:, None] * O), R @ (c_a[:, None] * O)       # prepare time
    u_s, l_s = s @ (P * c_s).T, s @ (R * c_s).T                 # early (the kernel scales s, not P)
    a_l = z @ O.T                                               # late, with its epilogue sums
    ra = rms(a_l, 1e-8)
    sum_ss = ((alpha1 * s) ** 2).sum(-1, keepdims=True)
    sum_sa = (alpha1 * beta1 * s * a_l).sum(-1, keepdims=True)
    sum_aa = ((beta1 * a_l) ** 2).sum(-1, keepdims=True)
    rs1 = np.sqrt((sum_ss + 2 * sum_sa / ra + sum_aa / ra ** 2) / H + 1e-5)
    u = u_s + (z @ Q.T) / ra
    logits = (l_s + (z @ M.T) / ra) / rs1
    p = u / np.sqrt((u * u).mean(-1, keepdims=True) + 1e-5 * rs1 ** 2)
    for name, got, ref in (("logits", logits, logits_ref), ("pooler rms", p, p_ref)):
        err = np.abs(got - ref).max() / np.abs(ref).max()
        print(f"  f64    {name}: max rel err {err:.2e}")
        assert err < 1e-12


if __name__ == "__main__":
    sympy_rewrites()
    z3_scalars()
    end_to_end()
