"""Final source for a built candidate: python3 harness/freeze.py <candidate.py> <out.py>
The code after the base section patches source strings at import (X = X.replace(...), helpers applied at module
level, loops over the alternative sources). Here every such string gets its final value where the base defines it
(its one literal, edited in place), the small-MoE sources and source-table entries become readable final literals,
strings defined only by the patch code get one literal each, and the patching
statements and asserts go. Function and class definitions and non-string assignments stay. The module's final string
state is checked against the original's.
    python3 harness/freeze.py <candidate.py> <out.py> [--trace <builds.jsonl>]
--trace (the candidate's builds as cpu_prebuild.py's OKC_TRACE_BUILDS records them): the dead-code pass too, each CUDA
source's preprocessor conditionals fixed across all its builds resolved (harness/unifdef.py), each build proven to
preprocess the same."""
import ast
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
STUB = (ROOT / "harness" / "import_check.py").read_text().split("bad = 0")[0]
MARK = 'base_prepare, base_kernel_fn = globals().get("prepare"), kernel_fn\n'
TEMPS = {"_o", "_k", "_v", "_n", "_c", "_s", "_a", "_b", "_DR_OLD", "_DR_NEW", "_OKC_ORIG_0"}
TEMPS.update({"_moe_static_source", "_moe_static_it", "_moe_static_re"})
NEAR = {"moe_helper0", "moe_helper0df"}
BUILD_ONLY = {"PREBUILD_JOBS", "SHAPE_SRC_HOOKS", "MOE_LATE_PLAN", "MOE_BUILD_FUNCTIONS"}   # build metadata (cpu_prebuild.py's extra jobs; the shape build's source
# hooks, already applied to the class strings), never part of a submission
GONE = {"_okc_unpatch"}   # helpers the frozen source no longer needs (the small-MoE sources are written out in full)


def run(text, name):
    g = {"__name__": name, "__file__": name}
    exec(compile(STUB, "stub", "exec"), g)
    exec(compile(text, name, "exec"), g)
    return g


def strings(g):
    out = {}
    for k, v in g.items():
        if k.startswith("__"):
            continue
        if isinstance(v, str):
            out[k] = v
        elif isinstance(v, dict) and v and all(isinstance(x, str) for x in v) and all(isinstance(x, str) for x in v.values()):
            out[k] = dict(v)
    return out


def lit(s):
    if '"""' not in s and not s.endswith("\\") and "\r" not in s:
        try:
            if ast.literal_eval('r"""' + s + '"""') == s:
                return 'r"""' + s + '"""'
        except SyntaxError:
            pass
    return repr(s)


def multiline(s):
    """a long source as readable source text: a raw triple-quoted string, else one quoted line per source line"""
    r = lit(s)
    return r if r.startswith('r"""') else "(\n" + "".join(f"    {l!r}\n" for l in s.splitlines(keepends=True)) + ")"


def drop_unused_build_functions(text, names):
    """Remove plan steps only when no remaining code refers to the function."""
    pending, removed = set(names), set()
    while pending:
        tree = ast.parse(text)
        lines = text.splitlines(keepends=True)
        deletions, newly_removed = [], set()
        for name in pending:
            definitions = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                           and node.name == name]
            if not definitions:
                continue
            outside = [node for node in tree.body if node not in definitions]
            referenced = any((isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id == name)
                             or (isinstance(node, ast.Attribute) and node.attr == name)
                             or (isinstance(node, ast.Constant) and isinstance(node.value, str) and name in node.value)
                             for statement in outside for node in ast.walk(statement))
            if not referenced:
                newly_removed.add(name)
                for node in definitions:
                    first = min([node.lineno] + [decorator.lineno for decorator in node.decorator_list])
                    deletions.append((first - 1, node.end_lineno))
        if not newly_removed:
            break
        for first, end in sorted(deletions, reverse=True):
            lines[first:end] = []
        text = "".join(lines)
        pending -= newly_removed
        removed |= newly_removed
    return text, removed


def resolve_conditionals(text, trace):
    """each traced CUDA source written once, verbatim, in text: its preprocessor conditionals fixed across all its builds
    resolved (harness/unifdef.py; every build's original and resolved text preprocess the same), then its C++ branches
    fixed by macros constant across those builds (harness/cxxconst.py; proven by SASS, prove_sass.py). Returns
    (text, {old: new})"""
    import json
    sys.path.insert(0, str(ROOT / "harness"))
    import cxxconst, unifdef
    builds = [(b["cuda_sources"] if isinstance(b["cuda_sources"], str) else "\n".join(b["cuda_sources"]), list(b.get("extra_cuda_cflags", ())))
              for b in map(json.loads, open(trace)) if b.get("cuda_sources")]
    done = {}
    for v in strings(run(text, "okc_freeze_traced")).values():
        for s in (v.values() if isinstance(v, dict) else [v]):
            bs = [(t, f) for t, f in builds if s in t]
            if s in done or not bs or text.count(s) != 1:
                continue
            r = unifdef.resolve(s, bs)
            if r != s:
                assert all(unifdef.same(t, t.replace(s, r, 1), f) for t, f in bs), "a build preprocesses differently"
            r = cxxconst.resolve(r, cxxconst.constants([(t.replace(s, r, 1), f) for t, f in bs]))
            if r != s:
                done[s] = r
    for s, r in done.items():
        text = text.replace(s, r, 1)
    return text, done


def main():
    src, out = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
    text = src.read_text()
    if MARK not in text:   # a clean layer (clean/assemble.py): no patching section, its source is already final
        out.write_text(text)
        print(f"{src.name}: no patching section (a clean layer): copied unchanged")
        return
    i = text.index(MARK) + len(MARK)
    base, rest = text[:i], text[i:]
    g0 = strings(run(base, "okc_freeze_base"))
    final_namespace = run(text, "okc_freeze_full")
    TEMPS.update(k for k in final_namespace if k.startswith("_FRESH_"))
    g1 = strings(final_namespace)
    changed = [k for k, v in g1.items() if not isinstance(v, dict) and g0.get(k) != v and k not in TEMPS]
    # 1. base literals edited in place (top-level `NAME = <str constant>`, the only assignment in the base)
    btree = ast.parse(base)
    blines = base.splitlines(keepends=True)
    offs = [0]
    for l in blines:
        offs.append(offs[-1] + len(l))
    edits, region_only = [], []
    for k in changed:
        assigns = [n for n in btree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == k for t in n.targets)]
        if not assigns:
            region_only.append(k)
            continue
        assert len(assigns) == 1 and isinstance(assigns[0].value, ast.Constant), k
        v = assigns[0].value
        a = offs[v.lineno - 1] + v.col_offset
        b = offs[v.end_lineno - 1] + v.end_col_offset
        edits.append((a, b, lit(g1[k])))
    # 2. the small-MoE sources: their full final source, readable (not line edits of the MoE source); the line-edit
    # helper they used goes with them
    for node in btree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Subscript):
            t = node.targets[0]
            if isinstance(t.value, ast.Name) and t.value.id == "_MA_ALT_SRCS" and isinstance(t.slice, ast.Constant) and t.slice.value in NEAR:
                key = t.slice.value
                a = offs[node.lineno - 1] + node.col_offset
                b = offs[node.end_lineno - 1] + node.end_col_offset
                edits.append((a, b, f'_MA_ALT_SRCS[{key!r}] = {multiline(g1["_MA_ALT_SRCS"][key])}'))
        if isinstance(node, ast.FunctionDef) and node.name == "_okc_unpatch":
            edits.append((offs[node.lineno - 1], offs[node.end_lineno] if node.end_lineno < len(offs) else len(base), ""))
    for a, b, new in sorted(edits, reverse=True):
        base = base[:a] + new + base[b:]
    # 3. the code after the base: patching statements dropped, new strings defined once
    tree = ast.parse(rest)
    lines = rest.splitlines(keepends=True)
    keep, dropped, entries = [], [], set()
    for node in tree.body:
        first = min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])])
        seg = "".join(lines[first - 1:node.end_lineno])
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Import, ast.ImportFrom)):
            if (getattr(node, "name", None) in TEMPS or
                    isinstance(node, (ast.Import, ast.ImportFrom)) and all(alias.asname in TEMPS for alias in node.names)):
                dropped.append(seg)
                continue
            keep.append(seg)
            continue
        t = node.targets[0] if isinstance(node, ast.Assign) and len(node.targets) == 1 else None
        if (isinstance(t, ast.Subscript) and isinstance(t.value, ast.Name) and isinstance(t.slice, ast.Constant)
                and isinstance(g1.get(t.value.id), dict) and t.slice.value in g1[t.value.id]):
            # a string table's entry (a variant's source): its final value where it is first set, later sets dropped
            if (t.value.id, t.slice.value) not in entries:
                entries.add((t.value.id, t.slice.value))
                keep.append(f"{t.value.id}[{t.slice.value!r}] = {multiline(g1[t.value.id][t.slice.value])}\n")
            else:
                dropped.append(seg)
            continue
        if isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
            ts = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(t, ast.Name) and t.id in BUILD_ONLY for t in ts):
                dropped.append(seg)
                continue
            stringy = True
            for t in ts:
                if isinstance(t, ast.Name):
                    stringy &= isinstance(g1.get(t.id), str) or t.id in TEMPS
                elif isinstance(t, ast.Subscript) and isinstance(t.value, ast.Name):
                    stringy &= isinstance(g1.get(t.value.id), dict)
                elif isinstance(t, ast.Tuple):
                    stringy &= all(isinstance(e, ast.Name) and (isinstance(g1.get(e.id), str) or e.id in TEMPS) for e in t.elts)
                else:
                    stringy = False
            (dropped if stringy else keep).append(seg)
            continue
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
            continue
        if (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Attribute)
                and isinstance(node.value.func.value, ast.Name)
                and isinstance(final_namespace.get(node.value.func.value.id), list)):
            keep.append(seg)
            continue
        dropped.append(seg)
    block = "".join(f"{k} = {lit(g1[k])}\n" for k in region_only)
    new = base + "\n" + block + "\n" + "\n\n".join(s.rstrip("\n") + "\n" for s in keep)
    plan = final_namespace.get("MOE_LATE_PLAN", {})
    build_functions = [name for name, _ in plan.get("steps", ())] + list(final_namespace.get("MOE_BUILD_FUNCTIONS", ()))
    new, removed_steps = drop_unused_build_functions(new, build_functions)
    gone = GONE | removed_steps
    out.write_text(new)
    g2 = strings(run(new, "okc_freeze_new"))
    bad = sorted(k for k in g1 if k not in TEMPS | BUILD_ONLY | gone and g1[k] != g2.get(k))
    full, frz = run(text, "okc_freeze_full2"), run(new, "okc_freeze_new2")
    bad += sorted(k for k in full if not k.startswith("__") and k not in TEMPS and k in frz and type(full[k]).__name__ != type(frz[k]).__name__)
    bad += sorted(k for k in full if not k.startswith("__") and k not in TEMPS | BUILD_ONLY | gone and k not in frz and not isinstance(full[k], str))
    print(f"base literals edited: {[k for k in changed if k not in region_only]}; new literals: {region_only}")
    print(f"dropped {len(dropped)} statements, kept {len(keep)}")
    for d in dropped:
        print("  DROP:", d.strip().splitlines()[0][:110])
    print("final string state differs:", bad or "none")
    if "--trace" in sys.argv:
        new, done = resolve_conditionals(new, sys.argv[sys.argv.index("--trace") + 1])
        out.write_text(new)
        g3 = strings(run(new, "okc_freeze_resolved"))
        moved = sorted(k for k in g2 if g2[k] != g3.get(k) and not (isinstance(g2[k], str) and done.get(g2[k]) == g3.get(k))
                       and not (isinstance(g2[k], dict) and all(done.get(x, x) == g3[k].get(y) for y, x in g2[k].items())))
        print(f"conditionals resolved in {len(done)} sources ({sum(len(a) - len(b) for a, b in done.items())} bytes); "
              f"strings changed otherwise: {moved or 'none'}")
    import json
    print(f"wrote {out} ({len(new.encode())} bytes, {len(json.dumps(new))} as a JSON string)")


if __name__ == "__main__":
    main()
