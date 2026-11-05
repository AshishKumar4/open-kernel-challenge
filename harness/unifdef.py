"""Conditional resolution for a frozen submission's CUDA sources (freeze.py's dead-code pass): in each source, every
#if / #ifdef / #ifndef group that takes the same branch in every build of that source that reaches it (the builds traced
by cpu_prebuild.py with OKC_TRACE_BUILDS: each build's text and -D flags) becomes that branch's lines; every other group
stays as written. A macro is the source's own when some build defines it (#define or -D) or it is OKC_*: undefined, it
is 0, as in C; any other name (__CUDA_ARCH__, CUDART_VERSION, ...) is unknown, and a group that reads one stays.
Proof (same()): each build's original and resolved text preprocess to the same tokens (gcc -E, includes blanked, with
and without the device pass's macros).
    resolve(source, builds) -> resolved source        builds: [(full cuda text, [flags])]"""
import re
import subprocess
import tempfile

DIRECTIVE = re.compile(r"^\s*#\s*(if|ifdef|ifndef|elif|else|endif|define|undef)\b(.*)$", re.S)
IDENT = re.compile(r"[A-Za-z_]\w*")
UNKNOWN = "unknown"


def logical(text):
    """the text's logical lines (backslash continuations joined): (first physical line, last physical line, text)"""
    out, phys, i = [], text.split("\n"), 0
    while i < len(phys):
        j, cur = i, phys[i]
        while cur.endswith("\\") and j + 1 < len(phys):
            j += 1
            cur = cur[:-1] + " " + phys[j]
        out.append((i, j, cur))
        i = j + 1
    return out


def flag_defines(flags):
    d = {}
    for f in flags:
        m = re.match(r"-D([A-Za-z_]\w*)(?:=(.*))?$", f)
        if m:
            d[m.group(1)] = m.group(2) if m.group(2) is not None else "1"
    return d


def own_names(builds):
    names = set()
    for text, flags in builds:
        names |= set(flag_defines(flags))
        names |= {m.group(1) for m in re.finditer(r"^[ \t]*#[ \t]*define[ \t]+([A-Za-z_]\w*)", text, re.M)}
    return names


def evaluate(expr, macros, own):
    """a #if expression's value (an int) or UNKNOWN; macros: name -> replacement text, or UNKNOWN (maybe defined)"""
    expr = re.sub(r"/\*.*?\*/|//.*$", " ", expr, flags=re.S)

    def defined(m):
        n = m.group(1) or m.group(2)
        if n in macros:
            return "1" if macros[n] != UNKNOWN else "@"
        return "0" if n in own or n.startswith("OKC_") else "@"
    expr = re.sub(r"\bdefined\s*(?:\(\s*([A-Za-z_]\w*)\s*\)|([A-Za-z_]\w*))", defined, expr)
    for _ in range(32):   # object-like macros expanded until none is left
        changed = False

        def sub(m):
            nonlocal changed
            n = m.group(0)
            if n in macros:
                if macros[n] == UNKNOWN or re.match(r"\s*\(", expr[m.end():]):
                    return "@"
                changed = True
                return f"({macros[n].strip() or '1'})"
            if n in own or n.startswith("OKC_"):
                changed = True
                return "0"
            return "@"
        expr = IDENT.sub(sub, expr)
        if not changed:
            break
    if "@" in expr or IDENT.search(expr) or "?" in expr:
        return UNKNOWN
    py = re.sub(r"\b(0[xX][0-9a-fA-F]+|\d+)[uUlL]+\b", r"\1", expr)
    py = py.replace("&&", " and ").replace("||", " or ")
    py = re.sub(r"!(?!=)", " not ", py).replace("/", "//")
    try:
        return int(eval(py, {"__builtins__": {}}, {}))
    except Exception:  # noqa: BLE001
        return UNKNOWN


def scan(text, flags, own, first, last):
    """run one build's conditionals; for each group whose #if starts on a physical line in [first, last], the branch it
    takes (0, 1, ...; -1 none), UNKNOWN, or None (not reached: inside a branch not taken)"""
    macros, stack, taken, active = dict(flag_defines(flags)), [], {}, True
    for p0, _, line in logical(text):
        m = DIRECTIVE.match(line)
        if not m:
            continue
        kind, rest = m.group(1), m.group(2).strip()
        if kind in ("if", "ifdef", "ifndef"):
            g = {"parent": active, "key": p0, "idx": 0}
            if active is False:
                g["state"], active = "dead", False
            elif active == UNKNOWN:
                g["state"], active = UNKNOWN, UNKNOWN
            else:
                name = (rest.split() or ["?"])[0]
                v = evaluate(rest if kind == "if" else f"defined({name})" if kind == "ifdef" else f"!defined({name})", macros, own)
                if v == UNKNOWN:
                    g["state"], active = UNKNOWN, UNKNOWN
                elif v:
                    g["state"], g["taken"], active = "taken", 0, True
                else:
                    g["state"], active = "searching", False
            stack.append(g)
        elif kind in ("elif", "else"):
            g = stack[-1]
            g["idx"] += 1
            if g["state"] in ("dead", "taken"):
                active = False
            elif g["state"] == UNKNOWN:
                active = UNKNOWN
            else:
                v = 1 if kind == "else" else evaluate(rest, macros, own)
                if v == UNKNOWN:
                    g["state"], active = UNKNOWN, UNKNOWN
                elif v:
                    g["state"], g["taken"], active = "taken", g["idx"], True
                else:
                    active = False
        elif kind == "endif":
            g = stack.pop()
            if first <= g["key"] <= last:
                taken[g["key"] - first] = {"dead": None, UNKNOWN: UNKNOWN, "taken": g.get("taken"), "searching": -1}[g["state"]]
            active = g["parent"]
        elif active is not False:   # #define / #undef where the region runs (or may run)
            nm = IDENT.match(rest)
            if not nm:
                continue
            n = nm.group(0)
            if active == UNKNOWN:
                macros[n] = UNKNOWN
            elif kind == "undef":
                macros.pop(n, None)
            else:
                macros[n] = UNKNOWN if rest[len(n):len(n) + 1] == "(" else rest[len(n):]
    return taken


def groups(src):
    """the source's conditional groups: {first physical line of its #if: [(directive's first, last physical line), ...
    one per #if / #elif / #else, then the #endif's]}"""
    out, stack = {}, []
    for p0, p1, line in logical(src):
        m = DIRECTIVE.match(line)
        if not m:
            continue
        kind = m.group(1)
        if kind in ("if", "ifdef", "ifndef"):
            stack.append(p0)
            out[p0] = [(p0, p1)]
        elif kind in ("elif", "else"):
            out[stack[-1]].append((p0, p1))
        elif kind == "endif":
            out[stack.pop()].append((p0, p1))
    return out


def resolve(src, builds):
    """src with every group resolved that takes one branch in every build that reaches it; builds: [(text, flags)]
    whose text contains src (at a line start)"""
    own = own_names(builds)
    n = src.count("\n")
    seen = []
    for text, flags in builds:
        i = text.index(src)
        first = text[:i].count("\n")
        seen.append(scan(text, flags, own, first, first + n))
    G = groups(src)
    choice = {}
    for key in G:
        vals = {s.get(key) for s in seen} - {None}
        if len(vals) == 1 and UNKNOWN not in vals:
            choice[key] = vals.pop()
    phys = src.split("\n")
    keep = [True] * len(phys)

    def drop(a, b):
        for k in range(a, b + 1):
            keep[k] = False
    for key, parts in G.items():
        if key not in choice:
            continue
        v = choice[key]
        drop(*parts[0])   # the #if and every #elif / #else / #endif line go; so does every branch but the taken one
        for bi, (a, b) in enumerate(parts[1:]):
            drop(a, b)
        for bi in range(len(parts) - 1):
            if bi != v:
                body = (parts[bi][1] + 1, parts[bi + 1][0] - 1)
                if body[0] <= body[1]:
                    drop(*body)
    return "\n".join(l for l, k in zip(phys, keep) if k)


def preprocess(text, flags, device):
    blanked = re.sub(r"^[ \t]*#[ \t]*include\b.*$", "", text, flags=re.M)
    extra = ["-D__CUDA_ARCH__=900", "-D__CUDACC__"] if device else []
    with tempfile.NamedTemporaryFile("w", suffix=".cu") as f:
        f.write(blanked)
        f.flush()
        r = subprocess.run(["gcc", "-E", "-P", "-x", "c++", "-w", *[x for x in flags if x.startswith(("-D", "-U"))], *extra, f.name],
                           capture_output=True, text=True)
    return r.returncode, " ".join(r.stdout.split())


def same(text, new_text, flags):
    """the two texts preprocess to the same tokens for this build (host and device pass)"""
    return all(preprocess(text, flags, d) == preprocess(new_text, flags, d) for d in (False, True))
