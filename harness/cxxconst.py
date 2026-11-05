"""Constant C++ branches in a frozen submission's CUDA sources (freeze.py's dead-code pass, after harness/unifdef.py): an
`if (...)` / `if constexpr (...)` whose condition is fixed by macros that have one value in every build of the source
keeps only the branch it takes; `if (TRUE && x)` becomes `if (x)`, `if (FALSE || x)` too. A condition is read three-valued
over && / || / ! and parentheses: any other operand (a variable, a call, a template argument) is opaque, so only what the
macros alone decide is resolved. Proof: every kernel's SASS is identical before and after (sass_identity.py), every
build compiled both ways.
    resolve(source, macros) -> resolved source      macros: {name: value} constant across the source's builds
    constants(builds) -> those macros (gcc -E -dM of each build, includes blanked: every macro's final value)"""
import re
import subprocess
import tempfile

UNKNOWN = None


def skip_ws(s, i):
    """past whitespace and comments"""
    while i < len(s):
        if s[i].isspace():
            i += 1
        elif s.startswith("//", i):
            i = s.find("\n", i)
            i = len(s) if i < 0 else i
        elif s.startswith("/*", i):
            i = s.find("*/", i) + 2
        else:
            break
    return i


def match(s, i, open_, close):
    """index just past the bracket closing s[i] (== open_), skipping comments, strings and character literals"""
    depth = 0
    while i < len(s):
        c = s[i]
        if s.startswith("//", i):
            i = s.find("\n", i)
            continue
        if s.startswith("/*", i):
            i = s.find("*/", i) + 2
            continue
        if c in "\"'":
            j = i + 1
            while s[j] != c:
                j += 2 if s[j] == "\\" else 1
            i = j + 1
            continue
        if c == open_:
            depth += 1
        elif c == close:
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    raise ValueError("unbalanced")


def statement_end(s, i):
    """index just past the statement starting at i: a {block}, an if/else chain, or a simple statement ending in ;"""
    i = skip_ws(s, i)
    if s[i] == "{":
        return match(s, i, "{", "}")
    m = re.match(r"if\s*(?:constexpr\s*)?\(", s[i:])
    if m:
        j = match(s, i + m.end() - 1, "(", ")")
        j = statement_end(s, j)
        k = skip_ws(s, j)
        if re.match(r"else\b", s[k:]):
            return statement_end(s, k + 4)
        return j
    depth = 0
    while i < len(s):
        c = s[i]
        if c in "([{":
            i = match(s, i, c, {"(": ")", "[": "]", "{": "}"}[c])
            continue
        if c in "\"'":
            j = i + 1
            while s[j] != c:
                j += 2 if s[j] == "\\" else 1
            i = j + 1
            continue
        if c == ";":
            return i + 1
        i += 1
    raise ValueError("no statement end")


def split_top(expr, op):
    """expr's top-level operands around op (&& or ||), outside parentheses"""
    parts, depth, last, i = [], 0, 0, 0
    while i < len(expr):
        c = expr[i]
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
        elif depth == 0 and expr.startswith(op, i):
            parts.append(expr[last:i])
            last = i + len(op)
            i += len(op)
            continue
        i += 1
    parts.append(expr[last:])
    return [p.strip() for p in parts]


def value(expr, macros):
    """the condition's truth (True / False) or UNKNOWN, three-valued over && || ! ( )"""
    e = expr.strip()
    while e.startswith("(") and match(e, 0, "(", ")") == len(e):
        e = e[1:-1].strip()
    ors = split_top(e, "||")
    if len(ors) > 1:
        vs = [value(p, macros) for p in ors]
        return True if True in vs else False if all(v is False for v in vs) else UNKNOWN
    ands = split_top(e, "&&")
    if len(ands) > 1:
        vs = [value(p, macros) for p in ands]
        return False if False in vs else True if all(v is True for v in vs) else UNKNOWN
    if e.startswith("!") and not e.startswith("!="):
        v = value(e[1:], macros)
        return UNKNOWN if v is UNKNOWN else not v
    names = re.findall(r"[A-Za-z_]\w*", e)
    if not names or any(n not in macros for n in names):
        return UNKNOWN
    py = re.sub(r"[A-Za-z_]\w*", lambda m: f"({macros[m.group(0)]})", e)
    py = re.sub(r"\b(0[xX][0-9a-fA-F]+|\d+)[uUlL]+\b", r"\1", py).replace("&&", " and ").replace("||", " or ")
    py = re.sub(r"!(?!=)", " not ", py)
    if re.search(r"[A-Za-z_]", re.sub(r"\b(and|or|not)\b", "", py)):
        return UNKNOWN
    try:
        return bool(eval(py, {"__builtins__": {}}, {}))
    except Exception:  # noqa: BLE001
        return UNKNOWN


def simplify(expr, macros):
    """the condition with its decided && / || operands dropped (only operands the macros fix), or None if unchanged"""
    e = expr.strip()
    ands = split_top(e, "&&")
    if len(ands) > 1:
        keep = [p for p in ands if value(p, macros) is not True]
        if len(keep) < len(ands) and keep:
            return " && ".join(keep)
    ors = split_top(e, "||")
    if len(ors) > 1:
        keep = [p for p in ors if value(p, macros) is not False]
        if len(keep) < len(ors) and keep:
            return " || ".join(keep)
    return None


def constants(builds):
    """{name: value} of the object-like macros with one value in every build ((text, flags) pairs)"""
    seen = None
    for text, flags in builds:
        blanked = re.sub(r"^[ \t]*#[ \t]*include\b.*$", "", text, flags=re.M)
        with tempfile.NamedTemporaryFile("w", suffix=".cu") as f:
            f.write(blanked)
            f.flush()
            out = subprocess.run(["gcc", "-E", "-dM", "-x", "c++", "-w", *[x for x in flags if x.startswith(("-D", "-U"))], f.name],
                                 capture_output=True, text=True).stdout
        d = {}
        for line in out.splitlines():
            m = re.match(r"#define ([A-Za-z_]\w*) (.*)$", line)
            if m and not m.group(1).startswith("__"):
                d[m.group(1)] = m.group(2).strip()
        seen = d if seen is None else {k: v for k, v in seen.items() if d.get(k) == v}
    for _ in range(16):   # values written in terms of other macros, expanded
        seen = {k: re.sub(r"[A-Za-z_]\w*", lambda m: f"({seen[m.group(0)]})" if m.group(0) in seen else m.group(0), v)
                for k, v in seen.items()}
    return seen


IF = re.compile(r"(?<![\w#])if\s*(constexpr\s*)?\(")


def resolve(src, macros):
    out, i, n = [], 0, 0
    while True:
        m = IF.search(src, i)
        if not m:
            out.append(src[i:])
            break
        line_start = src.rfind("\n", 0, m.start()) + 1
        prefix = src[line_start:m.start()]
        logical = line_start   # (the start of its logical line: a #define's body continues over backslashes)
        while logical > 1 and src[logical - 2] == "\\":
            logical = src.rfind("\n", 0, logical - 1) + 1
        if ("//" in prefix or prefix.strip().startswith(("*", "/*", "#")) or src.count('"', line_start, m.start()) % 2
                or src[logical:].lstrip().startswith("#")):
            out.append(src[i:m.end()])   # (in a comment, a directive or a string)
            i = m.end()
            continue
        cl = match(src, m.end() - 1, "(", ")")
        cond = src[m.end():cl - 1]
        v = value(cond, macros)
        if v is UNKNOWN:
            s = simplify(cond, macros)
            out.append(src[i:m.end()] + (s if s is not None else cond) + ")")
            n += s is not None
            i = cl
            continue
        then_end = statement_end(src, cl)
        k = skip_ws(src, then_end)
        has_else = re.match(r"else\b", src[k:]) is not None
        else_start = skip_ws(src, k + 4) if has_else else None
        else_end = statement_end(src, else_start) if has_else else then_end
        out.append(src[i:m.start()])
        if v:
            out.append(src[skip_ws(src, cl):then_end])
        elif has_else:
            out.append(src[else_start:else_end])
        else:   # (nothing left: the statement's own line goes if it is now blank)
            if not src[line_start:m.start()].strip() and (else_end >= len(src) or src[else_end] == "\n"):
                out[-1] = out[-1][:len(out[-1]) - (m.start() - line_start)]
                else_end += 1
        i = else_end
        n += 1
    res = "".join(out)
    return (resolve(res, macros) if n and res != src else res)
