"""Sharing-aware (DAG) extraction from the serialized e-graph of algebra/layer_egraph.py, over the whole layer.

egglog-python's cost models see a node's children only as opaque e-classes, so a shared subterm (a = O z feeding both
s1 and ra; U feeding f) is charged once per use. Here the e-graph is serialized to JSON and extracted in Python:
class properties (matrix shapes, when a vector's inputs exist) by fixpoint, then greedy-DAG extraction whose cost of a
choice is the sum of marginal costs over the *set* of nodes it uses (each node once), the extraction-gym heuristic.

Cost of a node: a matrix-vector product streams its matrix's bytes, charged to the class of its vector (EARLY_W for
inputs available before the critical segment, as measured: early traffic stretches the latency-bound attention one for
one); a dot product of late vectors is one grid-wide reduction. Key: (bytes, reductions, nodes).

    .venv-alg/bin/python algebra/dag_extract.py
"""
from __future__ import annotations

import json
import sys

sys.argv = sys.argv[:1]
_src = open(__file__.replace("dag_extract.py", "layer_egraph.py")).read().replace(
    'if __name__ == "__main__":\n    main()', "")
exec(_src)   # the layer's terms, rules and leaves

EARLY_W = 1.0


def load(eg):
    j = json.loads(eg._serialize().to_json())
    nodes = j["nodes"]
    cls = {}
    for nid, n in nodes.items():
        cls.setdefault(n["eclass"], []).append(nid)
    return nodes, cls


def lit(nodes, nid):
    op = nodes[nid]["op"]
    try:
        return int(op)
    except ValueError:
        return op.strip('"')


def class_props(nodes, cls):
    """Mat classes -> (rows, cols, bytes per element x 100); V / S classes -> when (0 pre, 1 early, 2 late)."""
    shape, when = {}, {}
    changed = True
    while changed:
        changed = False
        for c, ids in cls.items():
            for nid in ids:
                n = nodes[nid]
                op, ch = n["op"], n["children"]
                cc = [nodes[x]["eclass"] for x in ch]
                if op == "Mat" and c not in shape:
                    shape[c] = (lit(nodes, ch[1]), lit(nodes, ch[2]), lit(nodes, ch[3])); changed = True
                elif op == "Mat.diag" and c not in shape:
                    shape[c] = (1, 1, 0); changed = True
                elif op == "Mat.tr" and c not in shape and cc[0] in shape:
                    r, k, b = shape[cc[0]]; shape[c] = (k, r, b); changed = True
                elif op == "Mat.mm" and c not in shape and cc[0] in shape and cc[1] in shape:
                    l, r = shape[cc[0]], shape[cc[1]]
                    if l[2] == 0:
                        shape[c] = (r[0], r[1], max(r[2], PRODUCT_BPE100))
                    elif r[2] == 0:
                        shape[c] = (l[0], l[1], max(l[2], PRODUCT_BPE100))
                    else:
                        shape[c] = (l[0], r[1], PRODUCT_BPE100)
                    changed = True
                elif op == "V" and c not in when:
                    when[c] = lit(nodes, ch[2]); changed = True
                elif op == "S" and c not in when:
                    when[c] = 2 if str(lit(nodes, ch[0])).startswith("c") else 0; changed = True
                elif (op.startswith("V.") or op.startswith("S.") or op == "dot") and op not in ("V", "S"):
                    ws = [when[x] for x in cc if x in when]
                    vcs = [x for x in cc if x.startswith("V-") or x.startswith("S-")]
                    if len(ws) == len(vcs):
                        w = max(ws, default=0)
                        if when.get(c, -1) < w:
                            when[c] = w; changed = True
    return shape, when


def marginal(nodes, nid, shape, when):
    n = nodes[nid]
    op = n["op"]
    cc = [nodes[x]["eclass"] for x in n["children"]]
    if op == "V.mv":
        r, k, b = shape[cc[0]]
        nb = r * k * b // 100
        return (nb, 0, 0, 1) if when.get(cc[1], 0) == 2 else (0, nb, 0, 1)
    if op == "dot":
        return (0, 0, int(when.get(n["eclass"], 0) == 2), 1)
    return (0, 0, 0, 1)


def key(t):
    return (t[0] + EARLY_W * t[1], t[2], t[3])


def extract(nodes, cls, shape, when):
    """Greedy DAG: best[c] = node set; a node's candidate set is itself plus its children classes' best sets."""
    best = {}
    margin = {nid: marginal(nodes, nid, shape, when) for nid in nodes}

    def total(s):
        t = [0, 0, 0, 0]
        for x in s:
            m = margin[x]
            for i in range(4):
                t[i] += m[i]
        return tuple(t)

    changed = True
    while changed:
        changed = False
        for c, ids in cls.items():
            for nid in ids:
                cc = [nodes[x]["eclass"] for x in nodes[nid]["children"]]
                if any(x not in best for x in cc):
                    continue
                s = frozenset([nid]).union(*[best[x][0] for x in cc])
                if any(nodes[x]["eclass"] == c for x in s if x != nid):   # a cycle through this class
                    continue
                t = total(s)
                if c not in best or key(t) < key(best[c][1]):
                    best[c] = (s, t, nid); changed = True
    return best


def ilp_extract(nodes, cls, shape, when, root):
    """Exact extraction (MILP, HiGHS through scipy): one node per used class, each used node's child classes used,
    no cycles (class order variables), minimum total marginal cost over the used node set (each node once)."""
    import numpy as np
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import lil_matrix

    nids = list(nodes)
    cids = list(cls)
    ni = {n: i for i, n in enumerate(nids)}
    ci = {c: len(nids) + i for i, c in enumerate(cids)}   # class order variables after the node variables
    nv = len(nids) + len(cids)
    cost = np.zeros(nv)
    for n in nids:
        m = marginal(nodes, n, shape, when)
        cost[ni[n]] = (m[0] + EARLY_W * m[1]) / 1e5 + m[2] * 1.0 + 1e-3   # 100 KB = 1 unit; a reduction = 1; a node
    rows = []
    lo, hi = [], []
    a = lil_matrix((0, nv))
    entries = []

    def add(coefs, lb, ub):
        entries.append((coefs, lb, ub))

    add({ni[n]: 1 for n in cls[root]}, 1, 1)
    bigm = len(cids) + 1
    for n in nids:
        c = nodes[n]["eclass"]
        for ch in set(nodes[x]["eclass"] for x in nodes[n]["children"]):
            co = {ni[m]: 1 for m in cls[ch]}
            co[ni[n]] = co.get(ni[n], 0) - 1
            add(co, 0, np.inf)                                   # a used node's child class is used
            add({ci[c]: 1, ci[ch]: -1, ni[n]: -bigm}, 1 - bigm, np.inf)   # order(c) > order(child) when used
    a = lil_matrix((len(entries), nv))
    for r, (co, lb, ub) in enumerate(entries):
        for k, v in co.items():
            a[r, k] = v
        lo.append(lb)
        hi.append(ub)
    integ = np.r_[np.ones(len(nids)), np.zeros(len(cids))]
    bounds = Bounds(np.zeros(nv), np.r_[np.ones(len(nids)), np.full(len(cids), bigm)])
    res = milp(cost, constraints=LinearConstraint(a.tocsr(), lo, hi), integrality=integ, bounds=bounds,
               options={"time_limit": 600})
    chosen = {nids[i] for i in range(len(nids)) if res.x is not None and res.x[i] > 0.5}
    best = {}
    for n in chosen:
        best[nodes[n]["eclass"]] = (None, None, n)
    t = [0, 0, 0, 0]
    for n in chosen:
        m = marginal(nodes, n, shape, when)
        for i in range(4):
            t[i] += m[i]
    return best, tuple(t), res.status


def render(nodes, best, c, depth=0):
    nid = best[c][2]
    n = nodes[nid]
    if not n["children"]:
        return n["op"]
    args = ", ".join(render(nodes, best, nodes[x]["eclass"], depth + 1) for x in n["children"])
    return f"{n['op'].split('.')[-1]}({args})"


def run_lets(roots, iters, label):
    eg = EGraph()
    for name, r in roots.items():
        eg.let(f"root_{name.replace('^', 'p').replace('*', 'x').replace(chr(39), 'q')}", r)
    eg.run(linear * iters)
    j = json.loads(eg._serialize().to_json())
    nodes = j["nodes"]
    cls = {}
    for nid, n in nodes.items():
        cls.setdefault(n["eclass"], []).append(nid)
    lets = {d.get("let"): c for c, d in j["class_data"].items() if d.get("let")}
    shape, when = class_props(nodes, cls)
    best = extract(nodes, cls, shape, when)
    print(f"#### {label}: {len(nodes)} nodes, {len(cls)} classes, EARLY_W = {EARLY_W}")
    for name in roots:
        ln = f"root_{name.replace('^', 'p').replace('*', 'x').replace(chr(39), 'q')}"
        c = next((cl for k, cl in lets.items() if ln in k.replace("$", "").split(", ")), None)
        if c is None or c not in best:
            print(f"== {name}: (no class found: lets {list(lets)[:6]})")
            continue
        s, t, nid = best[c]
        print(f"== {name} (greedy DAG): late {t[0] / 1e6:.1f} MB, early {t[1] / 1e6:.1f} MB, late reductions {t[2]},"
              f" {t[3]} nodes")
        ib, it, status = ilp_extract(nodes, cls, shape, when, c)
        print(f"== {name} (ILP, status {status}): late {it[0] / 1e6:.1f} MB, early {it[1] / 1e6:.1f} MB,"
              f" late reductions {it[2]}, {it[3]} nodes")
        print("   ", render(nodes, ib, c)[:3000])


if __name__ == "__main__":
    for w in (0.0, 1.0):
        EARLY_W = w
        run_lets(LAYER_FRONT, 7, "front half, outputs together")
    EARLY_W = 1.0
    run_lets(LAYER_BACK, 6, "back half, outputs together")
