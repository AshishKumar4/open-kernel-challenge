"""Every assert statement of a Python file replaced by `pass` (same indentation): python3 harness/noassert.py <in> <out>"""
import ast
import sys

text = open(sys.argv[1]).read()
lines = text.splitlines(keepends=True)
spans = sorted(((n.lineno, n.end_lineno, n.col_offset) for n in ast.walk(ast.parse(text)) if isinstance(n, ast.Assert)), reverse=True)
for l0, l1, col in spans:
    first = lines[l0 - 1]
    assert first[:col].strip() == "" and lines[l1 - 1].rstrip("\n").endswith(lines[l1 - 1].rstrip("\n")), (l0, first)
    lines[l0 - 1:l1] = [" " * col + "pass\n"]
out = "".join(lines)
ast.parse(out)
open(sys.argv[2], "w").write(out)
print(f"{len(spans)} asserts removed -> {sys.argv[2]} ({len(out.encode())} bytes)")
