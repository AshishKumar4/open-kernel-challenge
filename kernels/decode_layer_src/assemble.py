"""Builds the submission file from its sources: python3 clean/assemble.py [out.py]  (default clean/muse_decode_layer.py)

clean/src/layer.py is the Python module (host code, weight layouts, dispatch); each @@<family>.cu@@ in it is replaced
by that CUDA source from clean/src, as a raw string. The CUDA sources may not contain a triple double quote."""
import ast
import json
import pathlib
import re
import sys

SRC = pathlib.Path(__file__).resolve().parent / "src"
LIMIT = int(0.95 * 2**20)   # the submission API's request limit, as a JSON string


def apply_case_switches(text, per_case, dims, apply, withdrawn):
    """Specialize an already globally configured module using the study's registry, without importing its harness."""
    tree = ast.parse(text)
    assignments = {target.id: node.value for node in tree.body if isinstance(node, ast.Assign)
                   for target in node.targets if isinstance(target, ast.Name)}
    moe_variant = ast.literal_eval(assignments["MOE_VARIANT"])
    moe_default = ast.literal_eval(assignments["MOE_DEFAULT"])
    moe_switches = {ast.literal_eval(key): {node.value for node in ast.walk(value)
                                          if isinstance(node, ast.Constant) and isinstance(node.value, str)}
                    for key, value in zip(assignments["MOE_SWITCHES"].keys, assignments["MOE_SWITCHES"].values)}
    scoped = {"stages": {"d16f", "d32f"}, "vunroll": {"v16f", "v32f"},
              "nsplit16": {"v16s"}, "v32s": {"v32s"}, "sdef": {"d1s", "d1f"}, "mid16": {"d16s"}}
    settings = {}
    for case, values in per_case.items():
        match = re.fullmatch(r"([dv])(1|16|32)([sf])", case)
        if match is None:
            raise ValueError(f"unknown scored case: {case}")
        mode, batch, layer = match.groups()
        length = 1 if mode == "d" else 8
        shape = (int(batch) * length, length, layer == "s")
        tokens = shape[0]
        cls = "t1" if tokens == 1 else "t8" if tokens <= 8 else "t16" if tokens <= 16 else "t32" if tokens <= 32 else "big"
        moe = moe_variant.get(shape, moe_default.get(cls, "big"))
        consumers = {"attention", "moe_" + moe, *moe_switches[moe]}
        if tokens <= 32:
            consumers.add("sk2")
            if tokens <= 8:
                consumers.add("sk2_gle")
                if tokens > 1:
                    consumers.add("sk2_narrow")
        else:
            consumers.add(f"rq{tokens}")
        flags, plan = [], {}
        for switch, value in sorted(values.items()):
            if switch in withdrawn:
                raise ValueError(f"withdrawn switch is ineligible on every case: {switch}")
            if switch not in dims or not any(type(value) is type(choice) and value == choice for choice in dims[switch]):
                raise ValueError(f"unsupported study value: {switch}={value!r}")
            if switch in scoped and case not in scoped[switch]:
                raise ValueError(f"{switch} does not execute on {case}")
            if switch in ("nsplit16", "v32s"):
                plan["n_splits"] = value
                continue
            if switch == "sdef":
                targets, templates = ("moe_d1s", "moe_d1f"), ("-DMOE_SDEF={}",)
            elif switch == "sc6s":
                targets, templates = ("MOE_TINY", "MOE_TINY2", "MOE_MID"), ("-DMOE_SC6S={}",)
            elif switch in apply:
                targets, templates = apply[switch]
            else:
                raise ValueError(f"study switch has no per-case application: {switch}")
            if not consumers.intersection(targets):
                raise ValueError(f"{switch} has no consuming module on {case}")
            overrides = []
            for template in templates:
                if "{}" in template:
                    flag = template.format(value)
                elif value == dims[switch][0] and template.startswith("-D"):
                    flag = template.partition("=")[0] + "=0"
                else:
                    flag = template
                overrides.append(flag)
            flags.append((tuple(targets), tuple(overrides)))
        settings[shape] = {"flags": tuple(flags), "attention_plan": plan, "switches": dict(values)}
    lines = text.splitlines(keepends=True)
    value = assignments["CASE_OVERRIDES"]
    start = sum(map(len, lines[:value.lineno - 1])) + value.col_offset
    end = sum(map(len, lines[:value.end_lineno - 1])) + value.end_col_offset
    return text[:start] + repr(settings) + text[end:]


def sk2_cases(layer):
    table = next(ast.literal_eval(node.value) for node in ast.parse(layer).body
                 if isinstance(node, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == "SK2_GEOMETRY" for t in node.targets))
    cases = {}
    for _, _, nt, nwr, wsk, upw, launches in table.values():
        for tokr, (cfg, mt, stages, p05, nacc, cl) in launches.items():
            depth = str(stages) if p05 == stages else f"SK2_P05 ? {p05} : {stages}"
            args = f"{mt}, {tokr}, {nt}, {nwr}, {wsk}, {upw}, {depth}, true, false, {nacc}, {cl}, __nv_bfloat16"
            cases[cfg] = (f"    case {cfg}: launch_sk2<{args}>"
                          "(Ws, x, out, ptab, pmax, base4, M, N, K, P); break;")
    return "\n".join(cases[cfg] for cfg in sorted(cases))


def assemble():
    text = (SRC / "layer.py").read_text()

    def source(m):
        cu = (SRC / m.group(1)).read_text()
        if m.group(1) == "sk2.cu":
            cu = cu.replace("@@SK2_CASES@@", sk2_cases(text))
        if '"""' in cu:
            raise SystemExit(f"{m.group(1)}: triple quote in a CUDA source")
        return "\n" + cu.strip("\n") + "\n"
    text = re.sub(r"@@(\w+\.cu)@@", source, text)
    unresolved = re.search(r"@@[^@\n]+@@", text)
    if unresolved:
        raise ValueError(f"unexpanded source placeholder: {unresolved[0]}")
    return text


def main():
    out = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else SRC.parent / "muse_decode_layer.py")
    text = assemble()
    size = len(json.dumps(text))
    if size > LIMIT:
        raise SystemExit(f"{size} bytes as JSON: over the {LIMIT} limit")
    out.write_text(text)
    print(f"{out}: {size} bytes as JSON, {text.count(chr(10))} lines")


if __name__ == "__main__":
    main()
