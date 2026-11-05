# okc: problem=muse_decode_layer
# Apache License 2.0 (see the LICENSE and NOTICE of the clean sources). Modified work: a rewrite by AshishKumar4 of
# Copyright 2026 team keell0renz (Bohdan Agarkov), https://github.com/keell0renz/muse-open-kernel-challenge. Their NOTICE:
#   muse-open-kernel-challenge
#   Copyright 2026 team keell0renz (Bohdan Agarkov)
#
#   This product includes code from the following projects. Each kernel file keeps the original attribution lines.
#
#   Apache License 2.0:
#   - vjeux/okc-2026 (https://github.com/vjeux/okc-2026, commit f35a41e): the starting point of every board.
#     "Portions Copyright vjeux, from https://github.com/vjeux/okc-2026 (commit f35a41e), licensed under the Apache License 2.0."
#   - alygg77/okc-muse-spark-kernels (commit 6cae49a): decode layer, MoE, prefill bases; pinned cuBLASLt plans and ctypes
#     wrapper in the bf16 projections. "Submitted by alygg77 under the Apache License 2.0 with this notice retained."
#   - chenwainuo/okc (muse_bf16_proj/submission.py @ 3d418e1): rs128 GEMVs (bf16 projections, decode layer).
#   - ZhimaoL/sglang (okc/x-vfgpdl, okc/cand12pre, okc/x-n72-naw okc_patch kernels): verify-attention kernels (wst, vfg) and the
#     prefill FMHA (fmha7, N72).
#   - Madhavan113/ParadigmKernelChallenge (c0f3460, dab77b3, 012d92b): TMA bulk-reduction merge in attention decode.
#   - Matsvei (kernarc user 'matvey', agents.kernarc.com; OKC submissions are licensed under the Apache License 2.0): the
#     per-thread CUDA-context bind helper and cuBLASLt plan pins.
#
#   MIT License:
#   - NathenFernandes/muse-bf16-proj (submission.py @ 7385e5d), Copyright (c) 2026 Nathen Fernandes: routertail_small and
#     ol_projection kernels in kernels/muse_bf16_proj.py, source unchanged.
#
#     Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
#     documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
#     rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
#     persons to whom the Software is furnished to do so, subject to the following conditions:
#
#     The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
#     Software.
#
#     THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
#     WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
#     COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
#     OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
# + alygg77/okc-muse-spark-kernels kernels/muse_decode_layer.py (Apache-2.0, commit 6cae49a); header below is theirs. Patched by team keell0renz: trig, rq, atrig0, rtrig8, pf4.
# + chenwainuo/okc muse_bf16_proj rs128 GEMVs (Apache-2.0) at T = 128 / 256 for qkv and T = 128 for the unpooler (team keell0renz).
# Portions (the per-thread CUDA-context bind helper; cuBLASLt plan pins) follow code posted by Matsvei (kernarc user
# 'matvey') on agents.kernarc.com; OKC submissions are licensed under the Apache License 2.0 (llms.txt), and that
# attribution is retained here.
# Portions Copyright vjeux, from https://github.com/vjeux/okc-2026 (commit f35a41e), licensed under the Apache License 2.0.
# Submitted by alygg77 under the Apache License 2.0 with this notice retained.
"""Muse Spark decode layer: one CUDA-graph-capturable kernel_fn, chosen by the step's token count T.

    class   T            qkv            o + boundary 1          pooler + router      route     experts        unpooler + boundary 2
    t1      1            sk2            sk2 + boundary (GLE)     sk2 + FP32 router    + carry   tiny build     sk2 + Triton finish
    t8      2..8         sk2 (narrow)   sk2 + cluster b1x        sk2 + FP32 router    Triton    tiny2 build    sk2 + Triton finish
    t16     9..16        sk2            sk2 + cluster b1         sk2 (fused)          Triton    mid build      sk2 + cluster b2c
    t32     17..32       sk2            sk2 + row (Gluon)        sk2 (fused)          Triton    b4 build       sk2 + row (Gluon)
    t128    128          rq             cuBLASLt + row (Gluon)   cuBLASLt (FP32 out)  Triton    WGS build      rq + row (Gluon)
    t256    256          rq             cuBLASLt + row (Gluon)   cuBLASLt (FP32 out)  Triton    WGS build      cuBLASLt + row (Gluon)
    wide    otherwise    cuBLAS         cuBLAS + row (Gluon)     cuBLAS (FP32 out)    Triton    WGS build      cuBLAS + row (Gluon)

Attention is one split-KV FP8 kernel family for every class (QK-norm, RoPE, cache write, merge, rmsnorm * sigmoid
gate); four scored shapes run their own builds of it (MODE bits 16..128). sk2: the skinny tensor-core GEMV over the
weights' lossless 12-bit encoding (BF16 exponents as a 3-bit offset from a per-matrix base, escapes corrected exactly);
rq: the 128 / 256-token GEMM over the same encoding; experts: one persistent NVFP4 MoE kernel, built per class.

Precision: BF16 weights stay BF16 (or their lossless 12-bit encoding), NVFP4 experts stay NVFP4, the FP32 router runs as
FP32 at T <= 8 and as BF16 hi + lo with FP32 accumulation above; FP8 queries and cache as SGLang; FP32 accumulation;
norms form (1 + w) in FP32. Every choice is fixed by the shape class: prepare() only repacks weights and compiles.
"""

import ctypes
import functools
import hashlib
import os
import threading
from concurrent.futures import ThreadPoolExecutor

import torch
import triton
import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.language.extra.cuda import gdc_launch_dependents, gdc_wait

HIDDEN = 8192
D_EXPERT = 4096
LOCAL_EXPERTS = 32
ROPE_THETA = 500000.0
LOG2E = 1.4426950408889634
PDL_MAX_T = 128   # programmatic dependent launches up to here (a launch-latency win at decode sizes, a loss at 256)
MOE_MMAX = 384    # tokens per expert launch (the scratch is sized for it)
NSM = 132


# ==================================================================================================================
# CUDA sources: one per kernel family, each compiled per shape class with its compile-time switches (BUILDS below)
# ==================================================================================================================

ATTENTION_CU = r"""@@attention.cu@@"""

MOE_CU = r"""@@moe.cu@@"""

SK2_CU = r"""@@sk2.cu@@"""

RQ_CU = r"""@@rq.cu@@"""

GLUE_CU = r"""@@glue.cu@@"""

# the experts builds' switches (the rest are 0): MOE_WGS the warpgroup-MMA phases (M > 16); MOE_B4FIN the column-parallel
# final pass (16 < M <= 32); MOE_TINY / TINY2 / MID the skinny builds for M <= 4 / 8 / 16; MOE_DF phase B reading x
# behind per-value data flags; MOE_DQM the DQM code layout; MOE_TUNED_* the tuned scored shapes (see the MoE source)
MOE_SC6S = 0   # 1: the skinny unit copies' scales as three six-bit planes (and -DMOE_SC6S=1 in every skinny build)
MOE_SDEF = 0   # 1: the t1 builds' deferred block scales (-DMOE_SDEF=1, their own natural-k-order unit copies)
MOE_SWITCHES = {
    'd1s': ('MOE_TINY', 'MOE_DQM',) + (('MOE_SDEF',) if MOE_SDEF else ()),
    'd1f': ('MOE_TINY',) + (('MOE_SDEF',) if MOE_SDEF else ()),
    'v1s': ('MOE_TINY2', 'MOE_DF', 'MOE_TUNED_V1',),
    'v1f': ('MOE_TINY2', 'MOE_DF',),
    'd16s': ('MOE_MID', 'MOE_DF', 'MOE_TUNED_D16',),
    'd16f': ('MOE_MID', 'MOE_DF',),
    'd32s': ('MOE_WGS', 'MOE_B4FIN', 'MOE_TUNED_D32',),
    'd32f': ('MOE_WGS', 'MOE_B4FIN',),
    'big': ('MOE_WGS',),
}
if MOE_SC6S:   # default skinny builds match the point's unit format; case views may bind another format
    MOE_SWITCHES = {c: sw + (() if 'MOE_WGS' in sw else ('MOE_SC6S',)) for c, sw in MOE_SWITCHES.items()}
# (source, exported functions, nvcc flags) per build
BUILDS = {
    "attention": (ATTENTION_CU, ["run"], ["-O3", "--expt-relaxed-constexpr"]),
    "sk2": (SK2_CU, ["sk2", "sk2f"], ["-O3", "-DSK2_NARROW=0", "-DSK2_GLE=0"]),
    "sk2_narrow": (SK2_CU, ["sk2"], ["-O3", "-DSK2_NARROW=1", "-DSK2_GLE=0"]),
    "sk2_gle": (SK2_CU, ["gle_o", "gle_pr"], ["-O3", "-DSK2_NARROW=0", "-DSK2_GLE=1"]),
    "rq128": (RQ_CU, ["run"], ["-O3", "-DNDEBUG", "--expt-relaxed-constexpr", "-DRQ_256=0"]),
    "rq256": (RQ_CU, ["run"], ["-O3", "-DNDEBUG", "--expt-relaxed-constexpr", "-DRQ_256=1"]),
    "glue": (GLUE_CU, ["b1", "b2c"], ["-O3", "-std=c++20", "--expt-relaxed-constexpr", "--fmad=false", "-DGLUE_ROUTER=0"]),
    "glue_router": (GLUE_CU, ["b1"], ["-O3", "-std=c++17", "--use_fast_math", "-DGLUE_ROUTER=1"]),
    **{"moe_" + c: (MOE_CU, ["run"], ["-O3"] + [f"-D{k}=1" for k in sw]) for c, sw in MOE_SWITCHES.items()},
}
_MODULES = {}
_BUILD_ALIASES = {}
CASE_OVERRIDES = {}   # assembly-time per-case flags and attention-plan overrides

# Per-shape overrides inherit the trial's other build flags.
BUILD_VARIANTS = {
    "sk2_d1": ("sk2", ("-DSK2_C7=1", "-DSK2_P05=1")),
    "sk2_decode": ("sk2", ("-DSK2_C7=1", "-DSK2_P05=0")),
    "sk2_gle_c7": ("sk2_gle", ("-DSK2_C7=1",)),
    "sk2_v1": ("sk2", ("-DSK2_C7=0",)),
    "sk2_narrow_v1": ("sk2_narrow", ("-DSK2_C7=0",)),
    "sk2_gle_v1": ("sk2_gle", ("-DSK2_C7=0",)),
    "sk2_gle_v1g": ("sk2_gle", ("-DSK2_C7=0", "-DSK2_GLE16=1")),
    "rq128_plain": ("rq128", ("-DRQ_C7=0",)),
    "rq256_plain": ("rq256", ("-DRQ_C7=0",)),
    "rq256_v32s": ("rq256", ("-DRQ_C7=1",)),
    "sk2_d16_32": ("sk2_decode", ("-DSK2_ELECT=1",)),   # (elected producer refills: +0.3..0.8% at d16s / d32 only)
}
BUILD_DEFINE_DEFAULTS = {"-DSK2_C7": "0", "-DSK2_P05": "0", "-DRQ_C7": "0",
                         "-DATT_FLAT_DECODE_STAGES": "3", "-DATT_FLAT_VERIFY_UNROLL": "2",
                         "-DATT_V1_POLL_NS": "64", "-DATT_V1_COPIES": "1",
                         "--register-usage-level": "5", "--def-load-cache": "none",
                         "-DGLUE_B1_ROWS": "16", "-DGLUE_B2_ROWS": "16"}
# (GLUE_B1_ROWS / GLUE_B2_ROWS: no code in glue.cu; the 9..32-token chain runs boundary 1 / 2 as the glue build's
# 8-CTA cluster kernel up to that many rows, the per-row Gluon boundary above)
SHAPE_MODULES = {
    (1, 1, True): {"sk2": "sk2_d1", "sk2_gle": "sk2_gle_c7"},
    (1, 1, False): {"sk2": "sk2_d1", "sk2_gle": "sk2_gle_c7"},
    (8, 8, True): {"sk2": "sk2_v1", "sk2_narrow": "sk2_narrow_v1", "sk2_gle": "sk2_gle_v1"},
    (8, 8, False): {"sk2": "sk2_v1", "sk2_narrow": "sk2_narrow_v1", "sk2_gle": "sk2_gle_v1"},
    (16, 1, True): {"sk2": "sk2_d16_32"},
    (32, 1, True): {"sk2": "sk2_d16_32"}, (32, 1, False): {"sk2": "sk2_d16_32"},
    (256, 8, True): {"rq256": "rq256_v32s"},
}
UN_STAGES = {(8, 8, True): 4}   # per shape: the T <= 8 unpooler's weight stages issued before its PDL wait (2 elsewhere; its whole K is 4)
GLE_T8 = 0   # 1: the verify-1 shapes run the T = 1 chain (boundary 1 in o's epilogue, the GLE pooler on 2T rows)
GLE_T8_SHAPES = {(8, 8, True), (8, 8, False)} if GLE_T8 else set()
for _shape in GLE_T8_SHAPES:
    SHAPE_MODULES[_shape] = {**SHAPE_MODULES[_shape], "sk2_gle": "sk2_gle_v1g"}


def _flag_groups(flags):
    it = iter(flags)
    for flag in it:
        yield (flag, next(it)) if flag == "-Xptxas" else (flag,)


def _flag_key(group):
    return (*group[:-1], group[-1].partition("=")[0])


def _flag_value(group):
    _, equals, value = group[-1].partition("=")
    return value if equals else "1"


def _flag_options(flags):
    return {_flag_key(group): _flag_value(group) for group in _flag_groups(flags)}


def _build_spec(name):
    if name not in BUILD_VARIANTS:
        return name, BUILDS[name]
    parent, overrides = BUILD_VARIANTS[name]
    resolved, (src, functions, flags) = _build_spec(parent)
    options = _flag_options(flags)
    changes = []
    for group in _flag_groups(overrides):
        key, wanted = _flag_key(group), _flag_value(group)
        if options.get(key, BUILD_DEFINE_DEFAULTS.get(key[-1], "0")) != wanted:
            changes.append(group)
    if not changes:
        return resolved, (src, functions, flags)
    replaced = {_flag_key(group) for group in changes}
    groups = [group for group in _flag_groups(flags) if _flag_key(group) not in replaced]
    groups += [group for group in changes if group[-1] != "--def-load-cache=none"]
    return name, (src, functions, [flag for group in groups for flag in group])


def _build_family(name):
    while name in BUILD_VARIANTS:
        name = BUILD_VARIANTS[name][0]
    return name


def _build_define(name, key):
    return int(_flag_options(_build_spec(name)[1][2]).get((key,), BUILD_DEFINE_DEFAULTS.get(key, "0")))


def _shape_families(shape):
    tokens, length, sliding = shape
    cls = _class(tokens)
    moe = MOE_VARIANT.get(shape, MOE_DEFAULT.get(cls, "big"))
    families = {"attention": "attention", "moe_" + moe: "moe_" + moe}
    if tokens <= 32:
        families["sk2"] = "sk2"
        if tokens <= 8:
            families["sk2_gle"] = "sk2_gle"
            if tokens > 1:
                families["sk2_narrow"] = "sk2_narrow"
    elif tokens in (128, 256):
        families[f"rq{tokens}"] = f"rq{tokens}_plain"
    families.update(SHAPE_MODULES.get(shape, {}))
    return families


def _case_builds():
    plans = {shape: _shape_families(shape) for shape in SHAPE_MODULES.keys() | CASE_OVERRIDES.keys()}
    for shape, setting in CASE_OVERRIDES.items():
        tokens, length, sliding = shape
        label = f"{'d' if length == 1 else 'v'}{tokens // length}{'s' if sliding else 'f'}"
        for family, parent in list(plans[shape].items()):
            switches = MOE_SWITCHES.get(family.removeprefix("moe_"), ()) if family.startswith("moe_") else ()
            flags = tuple(flag for targets, overrides in setting.get("flags", ())
                          if family in targets or any(s in targets for s in switches) for flag in overrides)
            if flags:
                name = f"{family}__{label}"
                BUILD_VARIANTS[name] = (parent, flags)
                plans[shape][family] = name
    return plans


def _build(name):
    from tvm_ffi.cpp import load_inline
    resolved, (src, functions, flags) = _build_spec(name)
    family = _build_family(resolved)
    tag = hashlib.md5((src + repr(flags)).encode()).hexdigest()[:12]
    module = load_inline(name=f"muse_{family}_{tag}", cuda_sources=src, functions=functions,
                        extra_cuda_cflags=flags)
    _MODULES[resolved] = _MODULES[name] = module
    return _MODULES[name]


def _start_builds():
    """every module, concurrently (one nvcc process each: the families are independent), for sm_90a"""
    os.environ["TVM_FFI_CUDA_ARCH_LIST"] = "9.0a"
    names, seen = [], {}
    _BUILD_ALIASES.clear()
    for name in (*BUILDS, *BUILD_VARIANTS):
        resolved, (src, functions, flags) = _build_spec(name)
        options = tuple(sorted((key, value) for key, value in _flag_options(flags).items()
                               if value != BUILD_DEFINE_DEFAULTS.get(key[-1], "0")))
        signature = (_build_family(name), src, tuple(functions), options)
        if signature not in seen:
            seen[signature] = resolved
            names.append(resolved)
        _BUILD_ALIASES[name] = seen[signature]
    pool = ThreadPoolExecutor(max_workers=max(1, min(len(names), os.cpu_count() or 1)))
    return pool, [pool.submit(_build, name) for name in names]


# ==================================================================================================================
# Triton / Gluon kernels: routing and the per-row residual boundaries
# ==================================================================================================================

@triton.jit
def _top8(sc, sel, ids_ptr, wts_ptr, lids_ptr, ls, t):
    """top-8 of sel (ties to the lower expert id), weights sc renormalized, local ids (-1: another GPU's expert)"""
    e = tl.arange(0, 256)
    k8 = tl.arange(0, 8)
    bits = sel.to(tl.uint32, bitcast=True)
    ordered = tl.where((bits & 0x80000000) != 0, ~bits, bits ^ 0x80000000).to(tl.uint32)
    packed = (ordered.to(tl.uint64) << 32) | (255 - e).to(tl.uint64)
    picked = tl.topk(packed, 8)
    ids = (255 - (picked & 255)).to(tl.int32)
    wv = tl.gather(sc, ids, 0)
    wv = wv / (tl.sum(wv, 0) + 1e-15)
    tl.store(ids_ptr + t * 8 + k8, ids)
    lid = (ids - ls + 256) % 256
    tl.store(lids_ptr + t * 8 + k8, tl.where(lid < 32, lid, -1))
    tl.store(wts_ptr + t * 8 + k8, wv)


@triton.jit
def _logit(LOGITS, t, e, RSUM: tl.constexpr):
    """router row t; RSUM: LOGITS is the pooler's part[32 tiles][4 quarters][32 rows][8], the quarters added in the
    order the pooler's last CTA adds them"""
    if RSUM == 2:   # LOGITS [T, 512]: the hi | lo router GEMM's FP32 rows
        return tl.load(LOGITS + t * 512 + e) + tl.load(LOGITS + t * 512 + 256 + e)
    if RSUM:
        r = LOGITS + (e >> 3) * 1024 + t * 8 + (e & 7)
        return tl.load(r) + tl.load(r + 256) + tl.load(r + 512) + tl.load(r + 768)
    return tl.load(LOGITS + t * 256 + e)


@triton.jit
def _route_split(POOLED, LOGITS, BIAS, GPRE, IDS, WTS, XOUT, LS, LIDS, PARTS: tl.constexpr, RSUM: tl.constexpr):
    """programs (t, p < PARTS): the pre-expert norm of pooled row t, column part p; (t, PARTS): its top-8"""
    t = tl.program_id(0)
    kind = tl.program_id(1)
    if kind < PARTS:
        c = tl.arange(0, 4096)
        oc = kind * (4096 // PARTS) + tl.arange(0, 4096 // PARTS)
        gpre = 1.0 + tl.load(GPRE + oc).to(tl.float32)
        gdc_launch_dependents()
        gdc_wait()
        pooled = tl.load(POOLED + t * 4096 + c).to(tl.float32)
        inv = tl.rsqrt(tl.sum(pooled * pooled, 0) / 4096.0 + 1e-5)
        if PARTS == 1:
            vals = pooled
        else:
            vals = tl.load(POOLED + t * 4096 + oc).to(tl.float32)
        tl.store(XOUT + t * 4096 + oc, (vals * inv * gpre).to(tl.bfloat16))
    else:
        e = tl.arange(0, 256)
        bias = tl.load(BIAS + e)
        ls = tl.load(LS)
        gdc_launch_dependents()
        gdc_wait()
        sc = tl.sigmoid(_logit(LOGITS, t, e, RSUM) * 0.1767766952966369)
        _top8(sc, sc + bias, IDS, WTS, LIDS, ls, t)


@triton.jit
def _route_packed(ALL, BIAS, GPRE, IDS, WTS, XOUT, LS, LIDS):
    """ALL [T, 4608] FP32: pooled [4096] | router hi [256] | router lo [256]; program (t, 0) norms, (t, 1) routes"""
    t = tl.program_id(0)
    kind = tl.program_id(1)
    c = tl.arange(0, 4096)
    e = tl.arange(0, 256)
    gpre = 1.0 + tl.load(GPRE + c, mask=(c < 4096) & (kind == 0), other=0.0).to(tl.float32)
    bias = tl.load(BIAS + e, mask=(e < 256) & (kind == 1), other=0.0)
    gdc_wait()
    gdc_launch_dependents()
    if kind == 0:
        pooled = tl.load(ALL + t * 4608 + c)
        x = pooled * tl.rsqrt(tl.sum(pooled * pooled, 0) / 4096.0 + 1e-5) * gpre
        tl.store(XOUT + t * 4096 + c, x.to(tl.bfloat16))
    else:
        logit = tl.load(ALL + t * 4608 + 4096 + e) + tl.load(ALL + t * 4608 + 4352 + e)
        sc = tl.sigmoid(logit * 0.1767766952966369)
        _top8(sc, sc + bias, IDS, WTS, LIDS, tl.load(LS), t)


@triton.jit
def _carry(PO, S, ALPHA, BETA, ALPHA2, t, cols, r1, ROWS1: tl.constexpr):
    # boundary 1's residual: ns = alpha * s + beta * bf16(r1 * P) (the cluster boundary's FMA order), carry = alpha2 * ns
    p = tl.load(PO + t * 8192 + cols, cache_modifier=".cg").to(tl.float32)
    s = tl.load(S + t * 8192 + cols)
    a = tl.load(ALPHA + cols)
    b = tl.load(BETA + cols)
    bn = (p * r1).to(tl.bfloat16).to(tl.float32)
    if ROWS1:
        ns = tl.fma(b, bn, a * s)
    else:
        ns = tl.where(cols < 4096, tl.fma(a, s, b * bn), tl.fma(b, bn, a * s))
    return tl.load(ALPHA2 + cols) * ns


@triton.jit
def _route_gle(POOLED, LOGITS, PART, T, BIAS, GPRE, IDS, WTS, XOUT, LS, LIDS, RT, PO, S, ALPHA, BETA, ALPHA2, S1,
               NCH: tl.constexpr, TT: tl.constexpr, RSUM: tl.constexpr):
    """t1: the routing on the GLE o epilogue's two-row (u, v) pooler / router outputs, whose norms r1 / r2 come from the
    o epilogue's partial sums; NCH * T more programs (no wait) form boundary 1's residual carry for the final boundary"""
    t = tl.program_id(0)
    kind = tl.program_id(1)
    k8 = tl.arange(0, 8)
    cta = tl.arange(0, 256)
    q = tl.sum(tl.load(PART + (cta[:, None] * T + t) * 8 + k8[None, :], cache_modifier=".cg"), 0)
    q0 = tl.sum(tl.where(k8 == 0, q, 0.0), 0)
    q1 = tl.sum(tl.where(k8 == 1, q, 0.0), 0)
    q2 = tl.sum(tl.where(k8 == 2, q, 0.0), 0)
    q3 = tl.sum(tl.where(k8 == 3, q, 0.0), 0)
    r1 = tl.rsqrt(q0 / 8192.0 + 1e-8)
    r2 = tl.rsqrt((q1 + 2.0 * r1 * q2 + r1 * r1 * q3) / 8192.0 + 1e-5)
    if kind > 1:
        # carry: column chunk (kind - 2) * T + t of 8192 // (NCH * T) columns, for every token
        cc = ((kind - 2) * TT + t) * (8192 // (NCH * TT)) + tl.arange(0, 8192 // (NCH * TT))
        for tok in tl.static_range(TT):
            r1t = tl.rsqrt(tl.sum(tl.load(PART + (cta * TT + tok) * 8, cache_modifier=".cg"), 0) / 8192.0 + 1e-8)
            tl.store(S1 + tok * 8192 + cc, _carry(PO, S, ALPHA, BETA, ALPHA2, tok, cc, r1t, TT == 1))
    elif kind == 0:
        c = tl.arange(0, 4096)
        gpre = 1.0 + tl.load(GPRE + c).to(tl.float32)
        gdc_wait()
        gdc_launch_dependents()
        pooled = (tl.load(POOLED + t * 4096 + c).to(tl.float32) + tl.load(POOLED + (T + t) * 4096 + c).to(tl.float32) * r1) * r2
        inv = tl.rsqrt(tl.sum(pooled * pooled, 0) / 4096.0 + 1e-5)
        tl.store(XOUT + t * 4096 + c, (pooled * inv * gpre).to(tl.bfloat16))
    else:
        e = tl.arange(0, 256)
        bias = tl.load(BIAS + e)
        ls = tl.load(LS)
        q4 = tl.sum(tl.where(k8 == 4, q, 0.0), 0)
        q5 = tl.sum(tl.where(k8 == 5, q, 0.0), 0)
        q6 = tl.sum(tl.where(k8 == 6, q, 0.0), 0)
        tail = q4 + 2.0 * r1 * q5 + r1 * r1 * q6
        gdc_wait()
        gdc_launch_dependents()
        tl.store(RT + t, r1)
        tl.store(RT + T + t, tail)
        logit = (_logit(LOGITS, t, e, RSUM) + _logit(LOGITS, T + t, e, RSUM) * r1) * r2
        sc = tl.sigmoid(logit * 0.1767766952966369)
        _top8(sc, sc + bias, IDS, WTS, LIDS, ls, t)


@triton.jit
def _finish(P, T, CARRY, TAIL, BETA, POST, NEXT, SOUT, HOUT, BC: tl.constexpr):
    """boundary 2 at T <= 8 (the unpooler's 1024 columns): s = carry + beta * rmsnorm(f * (1 + w)), h = next norm"""
    t = tl.program_id(0)
    tile = tl.program_id(1)
    c = tl.arange(0, 1024)
    beta = tl.load(BETA + c)
    gp = tl.load(POST + c).to(tl.float32)
    # Earlier-grid outputs are read before this kernel's wait; keep those reads L2-coherent.
    carry = tl.load(CARRY + t * 8192 + c, cache_modifier=".cg")
    tail = tl.load(TAIL + t, cache_modifier=".cg")
    o = tile * BC + tl.arange(0, BC)
    outcarry = tl.load(CARRY + t * 8192 + o, cache_modifier=".cg")
    gn = tl.load(NEXT + o).to(tl.float32)
    gdc_wait()
    gdc_launch_dependents()
    f = (tl.load(P + t * 1024 + c).to(tl.float32) * gp).to(tl.bfloat16).to(tl.float32)
    inv = tl.rsqrt(tl.sum(f * f, 0) / 8192.0 + 1e-8)
    n = (f * inv).to(tl.bfloat16).to(tl.float32)
    head = carry + beta * n
    invnext = tl.rsqrt((tail + tl.sum(head * head, 0)) / 8192.0 + 1e-5)
    result = tl.where(o < 1024, tl.gather(head, o % 1024, 0), outcarry)
    tl.store(SOUT + t * 8192 + o, result)
    tl.store(HOUT + t * 8192 + o, (result * invnext * gn).to(tl.bfloat16))


@gluon.jit
def _row_boundary(Branch, Stream, PostW, PreW, Alpha, Beta, Out, NewStream,
                  H: gl.constexpr, F: gl.constexpr, POST: gl.constexpr, V: gl.constexpr, NW: gl.constexpr):
    """one row per program: s' = alpha * s + beta * rmsnorm(b [* (1 + w_post)]), h = rmsnorm(s') * (1 + w_pre)"""
    gdc_wait()
    gdc_launch_dependents()
    row = gl.program_id(0)
    c = gl.arange(0, H, layout=gl.BlockedLayout([V], [32], [NW], [0]))
    b = gl.load(Branch + row * F + c, c < F, 0).to(gl.float32)
    if POST:
        w = gl.load(PostW + c).to(gl.float32)
        b = (b * w).to(gl.bfloat16).to(gl.float32)
    r = gl.rsqrt(gl.sum(b * b, 0) / H + 1.e-8)
    b = (b * r).to(gl.bfloat16).to(gl.float32)
    a = gl.load(Alpha + c)
    beta = gl.load(Beta + c)
    s = gl.load(Stream + row * H + c)
    ns = a * s + beta * b
    gl.store(NewStream + row * H + c, ns)
    r = gl.rsqrt(gl.sum(ns * ns, 0) / H + 1.e-5)
    w = gl.load(PreW + c).to(gl.float32)
    gl.store(Out + row * H + c, ns * r * w)


@gluon.jit
def _row_boundary_early(Branch, Stream, PostW, PreW, Alpha, Beta, Out, NewStream,
                        H: gl.constexpr, F: gl.constexpr, POST: gl.constexpr, V: gl.constexpr, NW: gl.constexpr):
    """_row_boundary with the weights and the stream row read before the wait (16 < T <= 192)"""
    row = gl.program_id(0)
    c = gl.arange(0, H, layout=gl.BlockedLayout([V], [32], [NW], [0]))
    pw = gl.load(PreW + c).to(gl.float32)
    a = gl.load(Alpha + c)
    beta = gl.load(Beta + c)
    if POST:
        post_w = gl.load(PostW + c).to(gl.float32)
    s = gl.load(Stream + row * H + c, cache_modifier=".cg")
    gdc_wait()
    gdc_launch_dependents()
    b = gl.load(Branch + row * F + c, c < F, 0).to(gl.float32)
    if POST:
        b = (b * post_w).to(gl.bfloat16).to(gl.float32)
    r = gl.rsqrt(gl.sum(b * b, 0) / H + 1.e-8)
    b = (b * r).to(gl.bfloat16).to(gl.float32)
    ns = a * s + beta * b
    gl.store(NewStream + row * H + c, ns)
    r = gl.rsqrt(gl.sum(ns * ns, 0) / H + 1.e-5)
    gl.store(Out + row * H + c, ns * r * pw)


def _boundary(branch, stream, g, post, s_out, h_out):
    """the per-row residual boundary above 16 tokens: post = False after attention, True after the experts"""
    T = branch.shape[0]
    i = "2" if post else "1"
    args = (branch, stream, g["post2"], g["pre2" if post else "pre1"], g["alpha" + i], g["beta" + i], h_out, s_out)
    if T <= 192:
        _row_boundary_early[(T,)](*args, HIDDEN, HIDDEN, post, 4, 32, num_warps=32, launch_pdl=True)
    else:
        _row_boundary[(T,)](*args, HIDDEN, HIDDEN, post, 4, 16, num_warps=16, launch_pdl=True)


# ==================================================================================================================
# cuBLASLt (ctypes, the image's libcublasLt): the 128 / 256-token o, pooler + router and unpooler GEMMs, pinned algos
# ==================================================================================================================

class _LtAlgo(ctypes.Structure):
    _fields_ = [("data", ctypes.c_uint64 * 8)]


class _Lt:
    """one handle and matmul descriptor (FP32 compute and scale), W transposed or not; a 64 MB workspace"""

    def __init__(self, transpose_w):
        P, I = ctypes.c_void_p, ctypes.c_int
        self.lib = ctypes.CDLL("libcublasLt.so.13")
        for name, args in (("cublasLtCreate", [ctypes.POINTER(P)]), ("cublasLtMatmulDescCreate", [ctypes.POINTER(P), I, I]),
                           ("cublasLtMatmulDescSetAttribute", [P, I, P, ctypes.c_size_t]),
                           ("cublasLtMatrixLayoutCreate", [ctypes.POINTER(P), I, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_int64]),
                           ("cublasLtMatrixLayoutSetAttribute", [P, I, P, ctypes.c_size_t]),
                           ("cublasLtMatmul", [P] * 12 + [ctypes.POINTER(_LtAlgo), P, ctypes.c_size_t, P])):
            fn = getattr(self.lib, name)
            fn.argtypes, fn.restype = args, I
        self.handle, self.desc = P(), P()
        self.lib.cublasLtCreate(ctypes.byref(self.handle))
        self.lib.cublasLtMatmulDescCreate(ctypes.byref(self.desc), 68, 0)   # CUBLAS_COMPUTE_32F, CUDA_R_32F
        op = I(1 if transpose_w else 0)
        self.lib.cublasLtMatmulDescSetAttribute(self.desc, 3, ctypes.byref(op), ctypes.sizeof(op))   # TRANSA
        self.workspace = torch.empty(64 << 20, device="cuda", dtype=torch.uint8)
        self.alpha, self.beta = ctypes.c_float(1), ctypes.c_float(0)

    def layouts(self, m, n, k, transpose_w, batch, fp32_out):
        """column-major W, x and out; the output features split into `batch` strided batches"""
        w = (k, n // batch, k, k * n // batch) if transpose_w else (n // batch, k, n, n // batch)
        out = []
        for i, (rows, cols, ld, stride) in enumerate((w, (k, m, k, 0), (n // batch, m, n, n // batch))):
            d = ctypes.c_void_p()
            self.lib.cublasLtMatrixLayoutCreate(ctypes.byref(d), 0 if fp32_out and i == 2 else 14, rows, cols, ld)   # R_32F / R_16BF
            count, offset = ctypes.c_int(batch), ctypes.c_int64(stride)
            self.lib.cublasLtMatrixLayoutSetAttribute(d, 5, ctypes.byref(count), ctypes.sizeof(count))     # BATCH_COUNT
            self.lib.cublasLtMatrixLayoutSetAttribute(d, 6, ctypes.byref(offset), ctypes.sizeof(offset))   # STRIDED_BATCH_OFFSET
            out.append(d)
        return out


# (matrix, T) -> (W transposed, output batches, FP32 output, algo words): the measured picks of our earlier entries
LT_PLANS = {
    ("o", 128): (False, 8, False, [77309411394, 4294967331, 0, 1407374883553283, 558779540176897, 60129542158, 292058693646, 0]),
    ("pr", 128): (True, 1, True, [575525617730, 4294967331, 0, 1970324836974598, 558779540176897, 60129542158, 292057776128, 0]),
    ("o", 256): (False, 2, False, [85899345986, 4294967331, 0, 1407374883553286, 558779540176897, 60129542158, 292058693646, 0]),
    ("pr", 256): (True, 2, True, [785979015234, 4294967331, 0, 562949953421312, 558779540176897, 60129542158, 292057776128, 0]),
    ("un", 256): (False, 2, False, [85899345986, 4294967331, 0, 1407374883553287, 558779540176897, 60129542158, 292058693646, 0]),
}


def _lt_run(x, plan):
    lt, (a, b, c), algo, weight, n, fp32 = plan
    out = torch.empty((x.shape[0], n), device=x.device, dtype=torch.float32 if fp32 else torch.bfloat16)
    P = ctypes.c_void_p
    lt.lib.cublasLtMatmul(lt.handle, lt.desc, ctypes.byref(lt.alpha), P(weight.data_ptr()), a, P(x.data_ptr()), b,
                          ctypes.byref(lt.beta), P(out.data_ptr()), c, P(out.data_ptr()), c, ctypes.byref(algo),
                          P(lt.workspace.data_ptr()), lt.workspace.numel(), P(torch.cuda.current_stream().cuda_stream))
    return out


_CTX = threading.local()


def _bind_context(device):
    """cuBLASLt through ctypes needs the device's primary context current on the calling thread"""
    index = device.index or 0
    if getattr(_CTX, "device", None) == index:
        return
    api = ctypes.CDLL("libcuda.so.1")
    ctx, cur = ctypes.c_void_p(), ctypes.c_int(-1)
    api.cuCtxGetCurrent(ctypes.byref(ctx))
    if ctx.value:
        api.cuCtxGetDevice(ctypes.byref(cur))
    if not ctx.value or cur.value != index:
        dev, primary = ctypes.c_int(), ctypes.c_void_p()
        api.cuDeviceGet(ctypes.byref(dev), index)
        api.cuDevicePrimaryCtxRetain(ctypes.byref(primary), dev)
        api.cuCtxSetCurrent(primary)
    _CTX.device = index


# ==================================================================================================================
# Weight layouts (prepare)
# ==================================================================================================================

# ---- sk2: the 12-bit lossless BF16 stream ----
# Each layout: matrix, streamed rows, NT n8 tiles/warp, NWR warp rows, WSK K-warps, UPW 64-k units/warp,
# then TOKR -> (cfg, MT, base stages, P05 stages, NACC, CL). assemble.py emits the standard CUDA launch switch
# from these same rows. The narrow, GLE and fused-pooler entry points keep their fixed launches.
SK2_GEOMETRY = {
    "qkv": ("qkv_gate", 2304, 3, 1, 4, 2,
            {8: (0, 1, 6, 5, 1, 1), 16: (1, 1, 5, 5, 1, 1), 32: (2, 2, 4, 4, 1, 1)}),
    "o8": ("o", 8192, 4, 1, 4, 1, {8: (3, 1, 4, 4, 2, 1), 16: (4, 1, 4, 3, 2, 1)}),
    "o32": ("o", 8192, 2, 2, 2, 2, {32: (5, 2, 3, 3, 1, 1)}),
    "pooler": ("pooler", 4096, 4, 1, 8, 1, {}),
    "router": ("router", 256, 1, 1, 8, 1, {}),
    "un8": ("unpooler", 1024, 1, 1, 8, 2, {8: (6, 1, 4, 4, 2, 1)}),
    "un16": ("unpooler", 8192, 4, 1, 4, 1, {16: (7, 1, 5, 3, 2, 1)}),
    "un32": ("unpooler", 8192, 4, 2, 2, 2, {32: (8, 2, 3, 3, 1, 1)}),
}
SK2_LAYOUTS = {name: (matrix, rows, nt * nwr * 8, nt, nwr, wsk, upw)
               for name, (matrix, rows, nt, nwr, wsk, upw, _) in SK2_GEOMETRY.items()}
# per class and matrix: (stream layout, configuration id of the sk2 build's launch table)
SK2_CONFIG = {({"qkv_gate": "qkv", "unpooler": "un"}.get(matrix, matrix), tokr): (name, launch[0])
              for name, (matrix, _, _, _, _, _, launches) in SK2_GEOMETRY.items()
              for tokr, launch in launches.items()}


def _sk2_tileunits(w):
    """W [N, K] -> [N/8, K/64, 32 lanes, 16] in mma.m16n8k16 B-fragment order: lane (g, t) holds row g and, for each
    k-step j of the 64-k unit, k = 16j + 2t, 2t+1, 2t+8, 2t+9"""
    n, k = w.shape
    t = w.view(n // 8, 8, k // 64, 4, 2, 4, 2)   # [tile, g, unit, j, h, t, p]: k = 64u + 16j + 8h + 2t + p
    return t.permute(0, 2, 1, 5, 3, 4, 6).reshape(n // 8, k // 64, 32, 16)


SK2_C7 = 0   # 1: the 7-op codec layout; goes with -DSK2_C7=1 in the sk2 / sk2_narrow / sk2_gle builds


def _sk2_compress(w, c7=None):
    """BF16 W [N, K] as tile-units [N/8, K/64, 768]: 512 B of low bytes (lane-major), then 256 B of 4-bit codes
    (sign, exponent - base in 0..7), plus the escapes (row, k, delta) the kernel corrects exactly"""
    c7 = SK2_C7 if c7 is None else c7
    if c7:
        return _sk2_compress_c7(w)
    n, k = w.shape
    tu = _sk2_tileunits(w)
    bits = tu.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
    hi, low = bits >> 8, bits & 0xFF
    e71, sign = hi & 0x7F, hi >> 7
    hist = torch.bincount(e71.flatten(), minlength=128)
    base = int(torch.stack([hist[b:b + 8].sum() for b in range(121)]).argmax())
    o = (e71 - base).clamp(0, 7)
    code = (sign << 3) | o
    esc = (e71 < base) | (e71 > base + 7)
    T, U = n // 8, k // 64
    c8 = code.to(torch.uint8)
    codes = torch.cat([c8[..., 0:4] | (c8[..., 4:8] << 4), c8[..., 8:12] | (c8[..., 12:16] << 4)], dim=-1)
    packed = torch.cat([low.to(torch.uint8).reshape(T, U, 512), codes.reshape(T, U, 256)], dim=-1).contiguous()
    idx = esc.nonzero()   # [E, 4]: tile, unit, lane, v
    dec = ((((sign << 7) | (base + o)) << 8) | low)[esc].to(torch.int16).view(torch.bfloat16).float()
    delta = tu[esc].float() - dec
    lane, v = idx[:, 2], idx[:, 3]
    rows = idx[:, 0] * 8 + (lane >> 2)
    cols = idx[:, 1] * 64 + 16 * (v >> 2) + 8 * ((v >> 1) & 1) + 2 * (lane & 3) + (v & 1)
    return {"packed": packed, "base4": base * 0x01010101, "rows": rows, "cols": cols, "delta": delta.contiguous()}


def _sk2_compress_c7(w):
    """SK2_C7: W times 2^sv (exact: its best 8-octave window moved onto the aligned [56, 64)), each code byte
    o_b | o_{4+b} << 4 | s_b << 7 | s_{4+((b-1)&3)} << 3, sv in base4's byte sign bits; escapes in the packed scale"""
    n, k = w.shape
    e = (w.contiguous().view(torch.int16).to(torch.int32) >> 8) & 0x7F
    hist = torch.bincount(e.flatten(), minlength=128)
    best = int(torch.stack([hist[b:b + 8].sum() for b in range(121)]).argmax())
    sv = 2 * (56 - best)
    tu = _sk2_tileunits((w.float() * 2.0 ** sv).to(torch.bfloat16))
    bits = tu.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
    hi, low = bits >> 8, bits & 0xFF
    e71, sign = hi & 0x7F, hi >> 7
    o = (e71 - 56).clamp(0, 7)
    esc = (e71 < 56) | (e71 > 63)
    T, U = n // 8, k // 64

    def word(c0):   # 8 codes (index c0 .. c0 + 7 of the lane's 16) -> 4 bytes
        oo, ss = o[..., c0:c0 + 8], sign[..., c0:c0 + 8]
        return [(oo[..., b] | (oo[..., 4 + b] << 4) | (ss[..., 4 + (b - 1) % 4] << 3) | (ss[..., b] << 7)) for b in range(4)]
    codes = torch.stack(word(0) + word(8), dim=-1).to(torch.uint8)
    packed = torch.cat([low.to(torch.uint8).reshape(T, U, 512), codes.reshape(T, U, 256)], dim=-1).contiguous()
    idx = esc.nonzero()
    dec = ((((sign << 7) | (56 + o)) << 8) | low)[esc].to(torch.int16).view(torch.bfloat16).float()
    delta = tu[esc].float() - dec
    lane, v = idx[:, 2], idx[:, 3]
    rows = idx[:, 0] * 8 + (lane >> 2)
    cols = idx[:, 1] * 64 + 16 * (v >> 2) + 8 * ((v >> 1) & 1) + 2 * (lane & 3) + (v & 1)
    s4 = sv & 15
    base4 = 0x38383838 | ((s4 & 1) << 7) | ((s4 >> 1 & 1) << 15) | ((s4 >> 2 & 1) << 23) | ((s4 >> 3 & 1) << 31)
    return {"packed": packed, "base4": base4, "rows": rows, "cols": cols, "delta": delta.contiguous()}


def _sk2_stream(tub, n, R, NT, NWR, WSK, UPW):
    """stage-major stream [stage][cta][warp][unit][tile][bytes] of the first n rows"""
    dev = tub.device
    U = tub.shape[1]
    ups = WSK * UPW
    ar = lambda v: torch.arange(v, device=dev)  # noqa: E731
    s = ar(U // ups).view(-1, 1, 1, 1, 1)
    c = ar(n // R).view(1, -1, 1, 1, 1)
    w = ar(NWR * WSK).view(1, 1, -1, 1, 1)
    uu = ar(UPW).view(1, 1, 1, -1, 1)
    nt = ar(NT).view(1, 1, 1, 1, -1)
    return tub[c * (R // 8) + (w // WSK) * NT + nt, s * ups + (w % WSK) * UPW + uu].contiguous()


def _sk2_table(comp, n, R):
    """per-CTA escape table int2 [n / R][pmax]: entry 0 = (count, 0); then ((row_local << 16) | k, delta bits) in k order"""
    dev = comp["rows"].device
    keep = comp["rows"] < n
    rows, cols, delta = comp["rows"][keep], comp["cols"][keep], comp["delta"][keep]
    ncta = n // R
    cta = rows // R
    counts = torch.bincount(cta, minlength=ncta)
    pmax = int(counts.max().item()) + 1 if rows.numel() else 1
    pmax += pmax % 2
    order = torch.argsort(cta.long() * 65536 + cols.long())
    cta, rows, cols, delta = cta[order], rows[order], cols[order], delta[order]
    start = torch.zeros(ncta + 1, dtype=torch.long, device=dev)
    start[1:] = torch.cumsum(counts, 0)
    slot = torch.arange(cta.numel(), device=dev) - start[cta] + 1
    tab = torch.zeros(ncta, pmax, 2, dtype=torch.int32, device=dev)
    tab[:, 0, 0] = counts.to(torch.int32)
    tab[cta, slot, 0] = (((rows % R) << 16) | cols).to(torch.int32)
    tab[cta, slot, 1] = delta.view(torch.int32)
    return tab.contiguous(), pmax


def _sk2_prepare(weights, layouts=None, c7=None):
    """name -> (stream, escape table, pmax, base4); the router: BF16 hi + lo tile-units, no escapes"""
    out, comp = {}, {}
    layouts = SK2_LAYOUTS if layouts is None else layouts
    for name in layouts:
        layer, n, R, NT, NWR, WSK, UPW = SK2_LAYOUTS[name]
        w = weights[layer]
        if layer == "router":
            hi = w.to(torch.bfloat16)
            lo = (w - hi.float()).to(torch.bfloat16)
            T, U = w.shape[0] // 8, w.shape[1] // 64
            tub = torch.cat([_sk2_tileunits(hi).contiguous().view(torch.uint8).reshape(T, U, 1024),
                             _sk2_tileunits(lo).contiguous().view(torch.uint8).reshape(T, U, 1024)], dim=-1)
            out[name] = (_sk2_stream(tub, n, R, NT, NWR, WSK, UPW), None, 0, 0)
            continue
        if layer not in comp:
            comp[layer] = _sk2_compress(w.contiguous(), c7)
        c = comp[layer]
        out[name] = (_sk2_stream(c["packed"], n, R, NT, NWR, WSK, UPW),) + _sk2_table(c, n, R) + (c["base4"],)
    return out


# ---- rq: the 12-bit lossless BF16 layout of the 128 / 256-token GEMM ----
_RQ_OFF = (0, 8, 16, 24, 4, 12, 20, 28)    # bit offset of each weight's 3-bit exponent offset in the code word
_RQ_SIGN = (7, 15, 23, 31, 11, 19, 27, 3)  # bit of its sign
RQ_KS = {"qkv_gate": 7, "unpooler": 2, "pooler": 4, "o": 2}     # the K split: one wave of N/128 x KS CTAs
RQ_POOL = 0   # 1: at T = 128 the pooler streams 12-bit through rq128 and the router (BF16 hi | lo rows, FP32 out) runs
              # as its own GEMM before it, instead of the one BF16 pooler + router GEMM (75.5 MB -> 50.3 + 8.4 MB)
RQ_O = 0   # 1: at T = 128 the o projection streams 12-bit through rq128 (KS 2) instead of the BF16 Lt GEMM


RQ_C7 = 0   # 1: the aligned exponent window (W x 2^sv); goes with -DRQ_C7=1 in the rq128 / rq256 builds


def _rq_pack(w, c7=None):
    """W [N, K] bf16 -> (layout [N/128][K/64][8][1536] uint8, exponent base word, escapes [E, 6])"""
    N, K = w.shape
    dev = w.device
    sv = 0
    c7 = RQ_C7 if c7 is None else c7
    if c7:   # align the exponent window by an exact 2^sv
        h = (w.contiguous().view(torch.int16).int() >> 8) & 127
        B0 = int(torch.bincount(h.flatten(), minlength=128).unfold(0, 8, 1).sum(1).argmax())
        d = (8 - B0 % 8) % 8
        d = d - 8 if d > 3 else d
        sv = 2 * d
        w = (w.float() * 2.0 ** sv).to(torch.bfloat16)
    fr = w.view(N // 16, 2, 8, K // 64, 4, 2, 4, 2).permute(0, 3, 4, 2, 6, 5, 1, 7).reshape(N // 16, K // 64, 4, 32, 8)
    T, KC = fr.shape[:2]
    b = fr.reshape(T, KC, 4, 32, 8).view(torch.int16).int() & 0xffff
    h7, sg = (b >> 8) & 127, b >> 15
    B = int(torch.bincount(h7.flatten(), minlength=128).unfold(0, 8, 1).sum(1).argmax())
    d = h7 - B
    inw = (d >= 0) & (d < 8)
    d = torch.where(inw, d, 0).long()
    C = ((d << torch.tensor(_RQ_OFF, device=dev)) | (sg.long() << torch.tensor(_RQ_SIGN, device=dev))).sum(-1)
    C = torch.where(C >= 2 ** 31, C - 2 ** 32, C).int()
    lo = (b & 255).to(torch.uint8)
    p0 = lo[:, :, 0:2].permute(0, 1, 3, 2, 4).reshape(T, KC, 32, 16)
    p1 = lo[:, :, 2:4].permute(0, 1, 3, 2, 4).reshape(T, KC, 32, 16)
    p2 = C.permute(0, 1, 3, 2).contiguous().view(torch.uint8).view(T, KC, 32, 16)
    data = torch.stack((p0, p1, p2), 2).contiguous()
    ei = (~inw).nonzero()
    esc = torch.cat((ei, (h7[tuple(ei.t())] ^ B)[:, None].long()), 1)
    lay = data.reshape(N // 128, 8, KC, 1536).permute(0, 2, 1, 3).contiguous()
    s4 = sv & 15
    sbits = ((s4 & 1) << 7) | ((s4 >> 1 & 1) << 15) | ((s4 >> 2 & 1) << 23) | ((s4 >> 3 & 1) << 31)
    return lay, B * 0x01010101 | sbits, esc


def _rq_escapes(esc6, NB, KC, KS, dev):
    """(per-(ks, nb, warp) sentinel-terminated escape lists, per-(warp, lane) sub-lists of at most 4)"""
    tile, ch, q, ln, v, xr = [t.long() for t in esc6.t()]
    nb, wt = tile // 8, tile % 8
    s0 = torch.arange(KS + 1, device=dev, dtype=torch.long) * KC // KS
    ks = torch.searchsorted(s0, ch.contiguous(), right=True) - 1
    u = ch - s0[ks]
    ent = (u << 17) | (ln << 12) | ((q * 4 + v // 2) << 8) | ((v % 2) << 7) | xr
    G = KS * NB * 8
    gw = (ks * NB + nb) * 8 + wt
    o = torch.argsort(gw * 65536 + u)
    gw1, ent1 = gw[o], ent[o]
    rank = torch.arange(len(gw1), device=dev) - torch.searchsorted(gw1, gw1)
    cap = (int(torch.bincount(gw1, minlength=G).max()) + 1 + 31) // 32 * 32
    lists = torch.full((G * cap + 32,), 0xffffffff, dtype=torch.long, device=dev)
    lists[gw1 * cap + rank] = ent1
    slot = gw * 32 + ln
    o = torch.argsort(slot * 65536 + u)
    slot, ent2 = slot[o], ent[o]
    rank = torch.arange(len(slot), device=dev) - torch.searchsorted(slot, slot)
    sub = torch.full((G * 32 * 4,), 0xffffffff, dtype=torch.long, device=dev)
    sub[slot * 4 + rank] = ent2
    as32 = lambda a: torch.where(a >= 2 ** 31, a - 2 ** 32, a).int().contiguous()  # noqa: E731
    return as32(lists), as32(sub)


def _rq_fits(esc6, NB, KC, KS):
    """rq128 corrects at most 4 escapes per (k slice, row block, warp, lane)"""
    tile, ch, ln = esc6[:, 0].long(), esc6[:, 1].long(), esc6[:, 3].long()
    s0 = torch.arange(KS + 1, device=esc6.device, dtype=torch.long) * KC // KS
    ks = torch.searchsorted(s0, ch.contiguous(), right=True) - 1
    slot = ((ks * NB + tile // 8) * 8 + tile % 8) * 32 + ln
    return len(slot) == 0 or int(torch.bincount(slot).max()) <= 4


def _rq_prepare(weights, layers=("qkv_gate", "unpooler"), c7=None):
    out = {}
    for layer in layers:
        lay, tab, esc6 = _rq_pack(weights[layer], c7)
        if layer in ("pooler", "o") and not _rq_fits(esc6, lay.shape[0], lay.shape[1], RQ_KS[layer]):
            continue   # (the chain keeps its BF16 GEMM)
        esc, esc2 = _rq_escapes(esc6, lay.shape[0], lay.shape[1], RQ_KS[layer], lay.device)
        out[layer] = (lay, esc, esc2, torch.zeros(8 * lay.shape[0], dtype=torch.int32, device=lay.device), RQ_KS[layer],
                      tab - 2 ** 64 if tab >= 2 ** 63 else tab)
    return out


# ---- the experts: NVFP4 codes in the kernel's decode layouts, scales in its fragment order ----
_REG_HJ = [(0, 0), (1, 0), (0, 2), (1, 2)]
# the stored word's image of each code bit (register, weight of the pair, bit): the kernel's 9-op decode of 8 codes
_DQX = [0x440, 0x200a80, 0x5115, 0x200a8000, 0x4400000, 0xa800020, 0x51150000, 0x8000200a, 0x400, 0x200a00, 0x5015,
        0x200a0000, 0x4000000, 0xa000020, 0x50150000, 0x200a, 0x4014, 0x2002, 0x10, 0x200200, 0x40140000, 0x20020000,
        0x100000, 0x2000020, 0x1, 0x2, 0x4010, 0x200, 0x10000, 0x20000, 0x40100000, 0x2000000]
# the t1 sliding build's bit positions (MOE_DQM decode)
_DQM = {(0, 0, 0): 6, (0, 0, 1): 7, (0, 0, 2): 8, (0, 0, 3): 15, (0, 1, 0): 22, (0, 1, 1): 23, (0, 1, 2): 24, (0, 1, 3): 31,
        (1, 0, 0): 9, (1, 0, 1): 10, (1, 0, 2): 11, (1, 0, 3): 18, (1, 1, 0): 25, (1, 1, 1): 26, (1, 1, 2): 27, (1, 1, 3): 2,
        (2, 0, 0): 12, (2, 0, 1): 13, (2, 0, 2): 14, (2, 0, 3): 21, (2, 1, 0): 28, (2, 1, 1): 29, (2, 1, 2): 30, (2, 1, 3): 5,
        (3, 0, 0): 4, (3, 0, 1): 0, (3, 0, 2): 1, (3, 0, 3): 3, (3, 1, 0): 20, (3, 1, 1): 16, (3, 1, 2): 17, (3, 1, 3): 19}


def _int32(word):
    return torch.where(word >= 2**31, word - 2**32, word).to(torch.int32).contiguous().view(-1)


def _dqx_word(regs):
    """XOR of each set code bit's stored-word image; regs[r] = the code tensors of register r's two weights"""
    word = torch.zeros(regs[0][0].shape, dtype=torch.int64, device=regs[0][0].device)
    for r in range(4):
        for up in range(2):
            for b in range(4):
                word ^= ((regs[r][up] >> b) & 1) * _DQX[8 * r + 4 * up + b]
    return _int32(word)


def _nibbles(codes, rows):
    c = codes.index_select(0, rows)
    return torch.stack((c & 0xF, c >> 4), -1).reshape(c.shape[0], -1)


def _pack_codes(codes, rows):
    """codes uint8 [R0, K/2] (low nibble first) -> int32 words [rb][w][c][g][t][s] (a uint4 per lane per chunk)"""
    nib = _nibbles(codes, rows)
    n = nib.view(nib.shape[0] // 16, 2, 8, 8, 8, 4, 4, 4).permute(0, 3, 4, 2, 5, 6, 1, 7).to(torch.int64)
    return _dqx_word([(n[..., h, j0], n[..., h, j0 + 1]) for h, j0 in _REG_HJ])


def _pack_codes_dqm(codes, rows):
    nib = _nibbles(codes, rows)
    n = nib.view(nib.shape[0] // 16, 2, 8, 8, 8, 4, 4, 4).permute(0, 3, 4, 2, 5, 6, 1, 7).to(torch.int64)
    word = torch.zeros(n.shape[:-2], dtype=torch.int64, device=codes.device)
    for r, (h, j0) in enumerate(_REG_HJ):
        for up in range(2):
            for b in range(4):
                word |= ((n[..., h, j0 + up] >> b) & 1) << _DQM[(r, up, b)]
    return _int32(word)


def _pack_codes_nat(codes, rows):
    """the WGS copy's natural mma k order"""
    nib = _nibbles(codes, rows)
    n = nib.view(nib.shape[0] // 16, 2, 8, 8, 8, 4, 2, 4, 2).permute(0, 3, 4, 2, 7, 5, 1, 6, 8).to(torch.int64)
    return _dqx_word([(n[..., h, q, 0], n[..., h, q, 1]) for h, q in ((0, 0), (1, 0), (0, 1), (1, 1))])


def _pack_codes_dqm_nat(codes, rows):
    """MOE_SDEF: the DQM bit layout in natural mma k order (each word one block's k16)"""
    nib = _nibbles(codes, rows)
    n = nib.view(nib.shape[0] // 16, 2, 8, 8, 8, 4, 2, 4, 2).permute(0, 3, 4, 2, 7, 5, 1, 6, 8).to(torch.int64)
    word = torch.zeros(n.shape[:-3], dtype=torch.int64, device=codes.device)
    for r, (h, q) in enumerate(((0, 0), (1, 0), (0, 1), (1, 1))):
        for up in range(2):
            for b in range(4):
                word |= ((n[..., h, q, up] >> b) & 1) << _DQM[(r, up, b)]
    return _int32(word)


def _pack_scales_sdef(sf, rows):
    """MOE_SDEF: sf fp8 [R0, K/16] -> bytes [rb][w][g][t][q][h][p]: block 32 w + 8 q + 2 t + p, the scale of lane
    (g, t)'s C column 2t + p of chunk pair q for row g (h 0) / g + 8 (h 1)"""
    s = sf.view(torch.uint8).index_select(0, rows)
    return s.view(s.shape[0] // 16, 2, 8, 8, 4, 4, 2).permute(0, 3, 2, 5, 4, 1, 6).contiguous().view(-1)


def _pack_scales(sf, rows):
    """sf fp8 [R0, K/16] -> bytes [rb][w][g][t][c][h]"""
    s = sf.view(torch.uint8).index_select(0, rows)
    return s.view(s.shape[0] // 16, 2, 8, 8, 8, 4).permute(0, 3, 2, 5, 4, 1).contiguous().view(-1)


def _unit_order(codes, scales, sc6s=None):
    """per 16-row block: its 32 KB of codes, then its 4 KB of scales (one TMA copy per skinny unit); MOE_SC6S: the
    scales as three 1024-byte planes [w][warp][lane][4], byte n = v[4w + n] | (bits 2w+1:2w of v[12 + n]) << 6,
    v = scale byte - 0x40 (every scale byte is 0x40 + a six-bit value)"""
    nblk = codes.numel() // (8 * 8 * 32 * 4)
    sc = scales.contiguous().view(nblk, 4096)
    sc6s = MOE_SC6S if sc6s is None else sc6s
    if sc6s:
        v = (sc.view(nblk, 8, 32, 16).to(torch.int32) - 0x40)
        planes = [v[..., 4 * w:4 * w + 4] | (((v[..., 12:16] >> (2 * w)) & 3) << 6) for w in range(3)]
        sc = torch.stack(planes, 1).to(torch.uint8).contiguous().view(nblk, 3072)
    return torch.cat([codes.contiguous().view(torch.uint8).view(nblk, 32768), sc], dim=1).contiguous().view(-1)


def _kperm():
    """stage position p -> stage column k: lane t's columns of k16 step j in all four load groups are scale block 4t + j"""
    p = torch.arange(256)
    s, c = p >> 4, p & 15
    lg, j, t, i = s >> 2, s & 3, (c >> 1) & 3, (c & 1) | ((c >> 3) << 1)
    return 16 * (4 * t + j) + 4 * lg + i


def _kperm_codes(codes):
    R, K = codes.shape[0], codes.shape[1] * 2
    nib = torch.stack((codes & 0xF, codes >> 4), -1).reshape(R, K // 256, 256)
    nib = nib.index_select(2, _kperm().to(codes.device)).reshape(R, K // 2, 2)
    return (nib[..., 0] | (nib[..., 1] << 4)).contiguous()


def _kperm_rows(rows_a):
    """gate / up row order whose 16-row block r16, lane g holds the inter column phase B reads at K position 8 r16 + g"""
    r = rows_a.view(-1, 2, 8)
    gate, up = r[:, 0, :].reshape(-1), r[:, 1, :].reshape(-1)
    P = torch.arange(gate.numel(), device=rows_a.device)
    cm = (P >> 8) * 256 + _kperm().to(rows_a.device)[P & 255]
    return torch.stack((gate.index_select(0, cm).view(-1, 8), up.index_select(0, cm).view(-1, 8)), 1).reshape(-1)


def _sc6(s):
    """WGS piece scales [..., 1024] -> [..., 768], lossless: every scale byte is 0x40 + a 6-bit value"""
    lead = s.shape[:-1]
    v = s.view(*lead, 128, 8).to(torch.int32) - 0x40
    v6, v7 = v[..., 6], v[..., 7]
    piece = [(v6 >> 0) & 3, (v6 >> 2) & 3, (v6 >> 4) & 3, v7 & 3, (v7 >> 2) & 3, (v7 >> 4) & 3]
    wb = torch.stack([(v[..., n] << 2) | piece[n] for n in range(4)], -1).to(torch.uint8)
    hb = torch.stack([(v[..., 4 + n] << 2) | piece[4 + n] for n in range(2)], -1).to(torch.uint8)
    return torch.cat([wb.reshape(*lead, 512), hb.reshape(*lead, 256)], -1).contiguous()


def _wgs_copy(codes, sf, rows):
    """[256-row unit][stage (16)][warpgroup (4)][8192 B codes + 768 B six-bit scales]"""
    wn = _pack_codes_nat(_kperm_codes(codes), rows)
    nb, nub = rows.numel() // 16, rows.numel() // 256
    w = wn.view(nb, 8, 8, 8, 4, 4).permute(0, 1, 2, 5, 3, 4).contiguous().view(nb, 256, 32)
    w = w.view(nub, 4, 4, 16, 4, 4, 32).permute(0, 3, 1, 4, 2, 6, 5).contiguous()
    cb = w.view(torch.uint8).view(nub, 16, 4, 8192)
    s = sf.view(torch.uint8).index_select(0, rows)
    s = s.view(nub, 4, 4, 2, 8, 16, 4, 4).permute(0, 5, 1, 2, 4, 6, 7, 3).contiguous().view(nub, 16, 4, 1024)
    return torch.cat([cb, _sc6(s)], dim=3).contiguous().view(-1)


def _moe_layout(name):
    """The selected skinny decoder determines its code order and scale-plane format; WGS uses separate copies."""
    if _build_define(name, "-DMOE_WGS"):
        return None
    sdef = _build_define(name, "-DMOE_SDEF")
    if _build_define(name, "-DMOE_DQM"):
        kind = "dqm_sdef" if sdef else "dqm"
    else:
        kind = "sdef" if sdef else "plain"
    return kind, _build_define(name, "-DMOE_SC6S")


def _moe_prepare(weights, layouts):
    """the experts' 14 run arguments, and the t1 sliding build's copy (DQM codes); the kernels read the global scales,
    the norm weight, the unit-order copies (skinny builds) and the WGS copies (wgmma builds): the other argument slots
    (codes, scales, LUT and fat-order copies, which the kernel never reads) get a placeholder"""
    dev = weights["w13"].device
    rb, g = torch.arange(D_EXPERT // 8, device=dev), torch.arange(8, device=dev)
    rows_a = torch.stack((8 * rb[:, None] + g[None, :], D_EXPERT + 8 * rb[:, None] + g[None, :]), 1).reshape(-1)
    rows_b = torch.arange(D_EXPERT, device=dev)
    w13 = [weights["w13"][e].view(torch.uint8) for e in range(LOCAL_EXPERTS)]
    w2 = [weights["w2"][e].view(torch.uint8) for e in range(LOCAL_EXPERTS)]
    sa = [_pack_scales(weights["w13_sf"][e], rows_a) for e in range(LOCAL_EXPERTS)]
    sb = [_pack_scales(weights["w2_sf"][e], rows_b) for e in range(LOCAL_EXPERTS)]

    def units(pack, scale_a, scale_b, formats):
        result = {fmt: [None, None] for fmt in formats}
        for side, codes, scales, rows in ((0, w13, scale_a, rows_a), (1, w2, scale_b, rows_b)):
            pieces = {fmt: [] for fmt in formats}
            for code, scale in zip(codes, scales):
                packed = pack(code, rows)
                for fmt in formats:
                    pieces[fmt].append(_unit_order(packed, scale, sc6s=fmt))
                del packed
            for fmt in formats:
                result[fmt][side] = torch.cat(pieces[fmt])
            del pieces
        return result
    unused = torch.zeros(16, dtype=torch.uint8, device=dev)
    gs_a, gs_b = weights["w13_gs"].float().contiguous(), weights["w2_gs"].float().contiguous()
    wgs = (torch.cat([_wgs_copy(w13[e], weights["w13_sf"][e], _kperm_rows(rows_a)) for e in range(LOCAL_EXPERTS)]),
           torch.cat([_wgs_copy(w2[e], weights["w2_sf"][e], rows_b) for e in range(LOCAL_EXPERTS)]))

    def args(unit):
        return [unused, unused, gs_a, unused, unused, gs_b, unused, weights["post_expert_norm"].contiguous(),
                unused, unused, *unit, *wgs]
    packed, natural_scales = {}, None
    for kind, pack in (("plain", _pack_codes), ("dqm", _pack_codes_dqm),
                       ("dqm_sdef", _pack_codes_dqm_nat), ("sdef", _pack_codes_nat)):
        formats = sorted(fmt for name, fmt in layouts if name == kind)
        if not formats:
            continue
        if kind in ("dqm_sdef", "sdef"):
            if natural_scales is None:
                natural_scales = ([_pack_scales_sdef(weights["w13_sf"][e], rows_a) for e in range(LOCAL_EXPERTS)],
                                  [_pack_scales_sdef(weights["w2_sf"][e], rows_b) for e in range(LOCAL_EXPERTS)])
            scale_a, scale_b = natural_scales
        else:
            scale_a, scale_b = sa, sb
        for fmt, unit in units(pack, scale_a, scale_b, formats).items():
            packed[(kind, fmt)] = args(unit)
    return packed


def _moe_scratch(dev):
    """counters + flags (int32), sums + y rows + stream-K fixups (fp32), inter + the WGS x copies (bf16)"""
    emax = MOE_MMAX * 8
    fixf = 2 * 160 * 4 * 4096 + 2 * 160 * 4 + 136 * 16
    xrows = emax + 8 * (32 + emax // 32 + 8)
    inter = torch.zeros((emax + 2 * xrows) * 4096, dtype=torch.bfloat16, device=dev)
    inter[128 * 4096:256 * 4096].view(torch.int16).fill_(-1)   # DF: rows [128, 256) start as "not yet written"
    return (torch.zeros(8 + 1024 + emax + 4096, dtype=torch.int32, device=dev),
            torch.zeros(emax + emax * 4096 + fixf, dtype=torch.float32, device=dev), inter)


# ==================================================================================================================
# prepare
# ==================================================================================================================

def _bind_shape_states(st, plain_sk2, coded_sk2, rq_weights, rq_selected, case_modules=None,
                       rq_coded=None, moe_variants=None):
    """Keep each selected codec paired with its packed weights; unrelated state is shared."""
    for name in BUILD_VARIANTS:
        resolved, _ = _build_spec(name)
        _MODULES[name] = _MODULES[_BUILD_ALIASES.get(resolved, resolved)]
    st["modules"] = dict(_MODULES)
    st["attention_plan"] = {}
    st["moe_sdef"] = MOE_SDEF
    st["sk2_modules"] = {name: _MODULES[name] for name in ("sk2", "sk2_narrow", "sk2_gle")}
    st["glue_rows"] = (_build_define("glue", "-DGLUE_B1_ROWS"), _build_define("glue", "-DGLUE_B2_ROWS"))
    st["sk2_rsum"] = {name: _build_define(name, "-DSK2_RSUM") for name in ("sk2", "sk2_gle")}
    st["rq"] = {
        (layer, tokens): (_MODULES[f"rq{tokens}_plain"], packed)
        for layer, packed in rq_weights.items() for tokens in (128, 256)
    }
    case_modules = SHAPE_MODULES if case_modules is None else case_modules
    wst_shape = (128, 8, False)
    wst_module = case_modules.get(wst_shape, {}).get("attention", "attention")
    wst_active = bool(_build_define(wst_module, "-DATT_VERIFY_WST"))
    if wst_active:
        case_modules = dict(case_modules)
        case_modules.setdefault(wst_shape, {})
    rq_coded = {"qkv_gate": rq_selected} if rq_coded is None else rq_coded
    states = {}
    for shape, names in case_modules.items():
        view = dict(st, modules=dict(st["modules"]))
        view["modules"].update({family: _MODULES[name] for family, name in names.items()})
        if "sk2" in names:
            codec = _build_define(names["sk2"], "-DSK2_C7")
            view["sk2"] = coded_sk2 if codec else plain_sk2
            view["sk2_modules"] = {family: view["modules"][family] for family in st["sk2_modules"]}
        for tokens in (128, 256):
            family = f"rq{tokens}"
            if family in names:
                codec = _build_define(names[family], "-DRQ_C7")
                packed = rq_coded if codec else rq_weights
                view["rq"] = dict(st["rq"])
                for matrix in ("qkv_gate", "unpooler", "pooler", "o") if tokens == 128 else ("qkv_gate",):
                    if matrix in packed:
                        view["rq"][(matrix, tokens)] = (view["modules"][family], packed[matrix])
        if moe_variants is not None:
            # Each shape launches one MoE build; rebind only the argument tuple that build consumes.
            for family, name in names.items():
                if family.startswith("moe_") and (layout := _moe_layout(name)) is not None:
                    view["moe_sdef"] = _build_define(name, "-DMOE_SDEF")
                    field = "moe_w_dqm" if family == "moe_d1s" else \
                        "moe_w_sdef" if family == "moe_d1f" and view["moe_sdef"] else "moe_w"
                    view[field] = moe_variants[layout]
        view["sk2_rsum"] = {f: _build_define(names.get(f, f), "-DSK2_RSUM") for f in ("sk2", "sk2_gle")}
        view["attention_plan"] = CASE_OVERRIDES.get(shape, {}).get("attention_plan", {})
        if wst_active and shape == wst_shape:
            # WST overwrites its partials and keeps monotonic tickets; base attention requires zero-at-rest sums.
            view["att_wst_ws"] = tuple(torch.zeros_like(buffer) for buffer in st["att_ws"])
        states[shape] = view
    return states


def prepare(weights):
    case_modules = _case_builds()
    pool, builds = _start_builds()
    dev = weights["qkv_gate"].device
    st = {}
    for k in ("qkv_gate", "o", "unpooler"):
        st[k] = weights[k].contiguous()
    router = weights["router"].float()
    hi = router.to(torch.bfloat16)
    st["router_fp32"] = router.contiguous()
    st["pr"] = torch.cat([weights["pooler"], hi, (router - hi.float()).to(torch.bfloat16)], 0).contiguous()
    st["g_preffn"] = weights["pre_ffn_norm"].contiguous()
    st["g_preexp"] = weights["pre_expert_norm"].contiguous()
    st["bias"] = weights["router_bias"].float().contiguous()
    g = {}
    for i, name in ((1, "attn_residual_gate"), (2, "ffn_residual_gate")):
        gw = weights[name].float() / 0.3
        beta = torch.sigmoid(gw)
        g["alpha%d" % i] = torch.sqrt(torch.clamp(torch.sigmoid(-gw) * (1. + beta), min=1.e-3)).contiguous()
        g["beta%d" % i] = beta.contiguous()
    g["pre1"] = (weights["pre_ffn_norm"].float() + 1).to(torch.bfloat16).contiguous()
    g["post2"] = (weights["post_ffn_norm"].float() + 1).to(torch.bfloat16).contiguous()
    g["pre2"] = (weights["next_pre_attn_norm"].float() + 1).to(torch.bfloat16).contiguous()
    st["g"] = g
    st["inv_freq"] = (1.0 / (ROPE_THETA ** (torch.arange(0, 64, 2, device=dev).float() / 64))).contiguous()
    st["inv_freq0"] = torch.zeros(32, device=dev)
    sk2_weights = {k: weights[k] for k in ("qkv_gate", "o", "pooler", "unpooler", "router")}
    st["sk2"] = _sk2_prepare(sk2_weights)
    # Keep the point's complete layouts for nearby shapes; additional codecs pack only the layouts their cases use.
    required = set()
    for shape, names in case_modules.items():
        if "sk2" in names and _build_define(names["sk2"], "-DSK2_C7") != SK2_C7:
            tokens = shape[0]
            required.update(("qkv", "o8" if tokens <= 16 else "o32", "pooler",
                             "un8" if tokens <= 8 else "un16" if tokens <= 16 else "un32"))
    other_sk2 = dict(st["sk2"])
    if required:
        other_sk2.update(_sk2_prepare(sk2_weights, tuple(k for k in SK2_LAYOUTS if k in required), c7=not SK2_C7))
    plain_sk2, coded_sk2 = (other_sk2, st["sk2"]) if SK2_C7 else (st["sk2"], other_sk2)
    st["sk2_part"] = torch.empty(32768, dtype=torch.float32, device=dev)
    st["sk2_cnt"] = torch.zeros(32, dtype=torch.int32, device=dev)
    # Each codec owns its counter with its packed matrix; plain remains the generic fallback.
    rq_weights = _rq_prepare(weights, ("qkv_gate", "unpooler") + (("pooler",) if RQ_POOL else ()) + (("o",) if RQ_O else ()),
                             c7=False)
    coded_matrices = set()
    for names in case_modules.values():
        for tokens in (128, 256):
            family = f"rq{tokens}"
            if family in names and _build_define(names[family], "-DRQ_C7"):
                coded_matrices.update(("qkv_gate", "unpooler") if tokens == 128 else ("qkv_gate",))
    rq_coded = _rq_prepare(weights, tuple(k for k in ("qkv_gate", "unpooler") if k in coded_matrices), c7=True)
    st["rq_sink"] = torch.zeros(4, dtype=torch.int32, device=dev)
    dqm_kind = "dqm_sdef" if MOE_SDEF else "dqm"
    moe_layouts = {("plain", MOE_SC6S), (dqm_kind, MOE_SC6S)}
    if MOE_SDEF:
        moe_layouts.add(("sdef", MOE_SC6S))
    moe_layouts.update(layout for names in case_modules.values() for family, name in names.items()
                       if family.startswith("moe_") and (layout := _moe_layout(name)) is not None)
    moe_variants = _moe_prepare(weights, moe_layouts)
    st["moe_w"] = moe_variants[("plain", MOE_SC6S)]
    st["moe_w_dqm"] = moe_variants[(dqm_kind, MOE_SC6S)]
    st["moe_w_sdef"] = moe_variants.get(("sdef", MOE_SC6S)) if MOE_SDEF else None
    st["moe_scratch"] = _moe_scratch(dev)
    st["att_ws"] = (torch.zeros((1 << 15) * 72, dtype=torch.float32, device=dev),
                    torch.zeros(1 << 15, dtype=torch.float32, device=dev), torch.zeros(1 << 15, dtype=torch.int32, device=dev))
    _bind_context(dev)
    lt = {True: _Lt(True), False: _Lt(False)}
    st["o_t"], st["unpooler_t"] = st["o"].t().contiguous(), st["unpooler"].t().contiguous()
    src = {"o": st["o_t"], "pr": st["pr"], "un": st["unpooler_t"]}
    st["lt"] = {}
    for (name, m), (tw, batch, fp32, words) in LT_PLANS.items():
        w = src[name]
        n, k = w.shape if tw else (w.shape[1], w.shape[0])
        st["lt"][(name, m)] = (lt[tw], lt[tw].layouts(m, n, k, tw, batch, fp32), _LtAlgo((ctypes.c_uint64 * 8)(*words)), w, n, fp32)
    for b in builds:
        b.result()
    pool.shutdown()
    st["shape_states"] = _bind_shape_states(st, plain_sk2, coded_sk2, rq_weights, rq_coded.get("qkv_gate"),
                                           case_modules, rq_coded, moe_variants)
    torch.cuda.synchronize(dev)
    return st


# ==================================================================================================================
# kernel_fn
# ==================================================================================================================

# exact scored shapes (T, q_len, sliding) with their own attention build (MODE bits) and MoE build
ATTENTION_VARIANT = {(1, 1, True): 16, (32, 1, True): 32, (256, 8, False): 64, (8, 8, True): 128, (256, 8, True): 256}
MOE_VARIANT = {(1, 1, True): "d1s", (8, 8, True): "v1s", (16, 1, True): "d16s", (32, 1, True): "d32s"}
MOE_DEFAULT = {"t1": "d1f", "t8": "v1f", "t16": "d16f", "t32": "d32f"}
# the attention launches' plan overrides (B, q_len, sliding, KV pairs)
ATTENTION_TUNED = {
    (32, 8, True, 1): dict(n_splits=4, nc=1, coop=1),
    (16, 8, True, 1): dict(n_splits=11, nc=1, coop=1),
    (1, 1, False, 1): dict(nc=8, coop=1, n_splits=252),
}


def _class(T):
    return "t1" if T == 1 else "t8" if T <= 8 else "t16" if T <= 16 else "t32" if T <= 32 else \
        "t128" if T == 128 else "t256" if T == 256 else "wide"


def _attention_plan(B, L, window, max_pages, P):
    """n_splits CTAs per (request, KV pair) or, flat, per KV pair over all requests; nc workspace copies; coop: every
    split finalizes a slice of the rows (all resident)"""
    target = 2 * NSM
    if not window and 1 < B <= 512:
        # full layer, several requests: equal shares of all requests' keys, one CTA pass per SM pair
        return dict(n_splits=target, flat=1, nc=1)
    keys_max = min(window + L - 1, max_pages * P) if window else max_pages * P
    tiles = max(1, (keys_max + 63) // 64)
    reqs = B
    if not window:   # one long request: plain split-KV over 2 CTAs per SM
        if L == 8:
            n = max(1, min(tiles, ((target * 21) // 22) // reqs))
            return dict(n_splits=n, flat=0, nc=max(1, min(n, 8)))
        n = max(1, min(tiles, target // reqs))
        return dict(n_splits=n, flat=0, nc=max(1, min(n, 4)), coop=0)
    if L == 1 and reqs < 24:   # plain split-KV with one tile per CTA, one workspace copy
        return dict(n_splits=max(1, min(tiles, target // reqs)), flat=0, nc=1)
    if L == 8 and reqs < 16:   # few requests: one tile per CTA, the cooperative finalize
        return dict(n_splits=max(1, min(tiles, target // reqs)), flat=0, nc=1, coop=1)
    if L == 8:   # many requests: about one CTA per SM (two from 24 requests)
        return dict(n_splits=max(1, min(tiles, (target if reqs >= 24 else NSM) // reqs)), flat=0, nc=1, coop=1)
    return dict(n_splits=max(1, min(tiles, target // reqs)), flat=0, nc=1)


def _attention(data, st, qkv, T):
    cfg = data["config"]
    B, L, window, P = cfg["batch"], cfg["q_len"], cfg["window"], cfg["page_size"]
    page_table = data["page_table"]
    plan = _attention_plan(B, L, window, page_table.shape[1], P)
    plan.update(ATTENTION_TUNED.get((B, L, bool(window), 1), {}))
    plan.update(st["attention_plan"])
    n_splits, flat, nc = plan["n_splits"], plan["flat"], plan["nc"]
    if flat:
        coop = plan.get("coop", int(L == 8 and n_splits <= 2 * NSM))
    else:
        coop = plan.get("coop", 0 if n_splits < 64 else int(n_splits * B <= 2 * NSM))
    # the first CTA's key share on each SM (/1024) of the 2-CTA-per-SM grids
    w0 = (1024 if L == 1 else 1135) if flat and (coop or L == 1) else 1024
    variant = ATTENTION_VARIANT.get((T, L, bool(window)), 0)
    pf = (0, 0, 0, 0)
    if T in (1, 8):   # the pooler's weight stream into L2 during attention (decode-sized steps)
        stream, tab = st["sk2"]["pooler"][:2]
        nbytes = stream.numel()
        if variant == 128:   # 16 B-aligned prefixes of each CTA segment of the stream, 1 / 16 of each
            seg16 = nbytes // tab.shape[0] >> 4
            pf = (stream.data_ptr(), (seg16 // 16) << 4, seg16 << 4, tab.shape[0])
        else:
            ctas = n_splits * (1 if flat else B)
            pf = (stream.data_ptr(), (min(4 << 20, nbytes) // ctas) & ~15, 0, 0)
    out = torch.empty((B, L, 1024), dtype=torch.bfloat16, device=qkv.device)
    ws_acc, ws_l, ws_cnt = st.get("att_wst_ws", st["att_ws"]) if P == 1 else st["att_ws"]
    st["modules"]["attention"].run(qkv, data["seq_lens"], data["cache_loc"], data["k_cache"], data["v_cache"], page_table,
                              data["k_scale"], data["v_scale"], st["inv_freq"] if cfg["rope_theta"] else st["inv_freq0"],
                              out, ws_acc, ws_l, ws_cnt, B, L, 16, 2, P, window, 1 if cfg["rope_theta"] else 0, n_splits,
                              flat, nc, cfg["sm_scale"] * cfg["qk_scale_factor"] * LOG2E, coop, w0, int(T <= PDL_MAX_T),
                              variant, *pf)
    return out.view(T, 1024)


def _sk2_pdl(T, stages=2):
    """the sk2 launches' PDL word: programmatic launch with `stages` weight stages issued before the wait"""
    return (1 << 30) | stages if T <= PDL_MAX_T else 0


def _sk2(st, matrix, x, tokr, n, pdl, narrow=False):
    layout, cfg = SK2_CONFIG[(matrix, tokr)]
    stream, tab, pmax, b4 = st["sk2"][layout]
    out = torch.empty((x.shape[0], n), dtype=torch.bfloat16, device=x.device)
    if narrow:
        st["sk2_modules"]["sk2_narrow"].sk2(stream, x, out, tab, pmax, b4, pdl)
    else:
        st["sk2_modules"]["sk2"].sk2(stream, x, out, tab, pmax, b4, cfg, pdl)
    return out


def _rq(st, layer, x, n):
    module, packed = st["rq"][(layer, x.shape[0])]
    lay, esc, esc2, cnt, ks, tab = packed
    out = torch.empty((x.shape[0], n), dtype=torch.bfloat16, device=x.device)
    module.run(lay, x, out, esc, esc2, cnt, st["rq_sink"], ks, tab, 1)
    return out


def _experts(st, build, xm, lids, wts, T):
    """the routed experts of this GPU (local ids 0..31, -1 elsewhere): MOE_MMAX tokens per launch"""
    out = torch.empty((T, D_EXPERT), device=xm.device, dtype=torch.bfloat16)
    w = st["moe_w_dqm"] if build == "d1s" else st["moe_w_sdef"] if st["moe_sdef"] and build == "d1f" else st["moe_w"]
    upf = (0, 0)
    if T <= 16:   # the small builds prefetch the unpooler's stream into L2
        stream = st["sk2"][SK2_CONFIG[("un", 8 if T <= 8 else 16)][0]][0]
        upf = (stream.data_ptr(), stream.numel())
    mod = st["modules"]["moe_" + build]
    for m0 in range(0, T, MOE_MMAX):
        m1 = min(T, m0 + MOE_MMAX)
        mod.run(*w, *st["moe_scratch"], xm[m0:m1], lids[m0:m1], wts[m0:m1], out[m0:m1], 0, LOCAL_EXPERTS, NSM,
                int(T <= PDL_MAX_T), *upf)
    return out


def _routing_outputs(T, dev):
    return (torch.empty((T, 8), device=dev, dtype=torch.int32), torch.empty((T, 8), device=dev, dtype=torch.float32),
            torch.empty((T, 8), device=dev, dtype=torch.int32), torch.empty((T, D_EXPERT), device=dev, dtype=torch.bfloat16))


def _chain_t1(data, st, o_in, T, moe):
    """T = 1: the o projection carries the first boundary in its epilogue (GLE): rows u = r1 * P and v = the stream
    part feed the pooler and the FP32 router; the route kernel folds the norms back in and forms the residual carry"""
    dev, g = o_in.device, st["g"]
    f32 = dict(device=dev, dtype=torch.float32)
    stream, tab, pmax, b4 = st["sk2"]["o8"]
    P = torch.empty((T, HIDDEN), device=dev, dtype=torch.bfloat16)
    uv = torch.empty((2 * T, HIDDEN), device=dev, dtype=torch.bfloat16)
    uvf = torch.empty((2 * T, HIDDEN), **f32)
    part = torch.empty((256 * T * 8,), **f32)
    rt = torch.empty((2, T), **f32)
    s1 = torch.empty((T, HIDDEN), **f32)
    gle = st["sk2_modules"]["sk2_gle"]
    gle.gle_o(stream, o_in, P, tab, pmax, b4, data["stream"], g["alpha1"], g["beta1"], g["alpha2"], st["g_preffn"], uv,
              part, uvf, _sk2_pdl(T))
    pooled = torch.empty((2 * T, D_EXPERT), device=dev, dtype=torch.bfloat16)
    logits = torch.empty((2 * T, 256), **f32)
    pstream, ptab, ppmax, pb4 = st["sk2"]["pooler"]
    gle.gle_pr(pstream, st["sk2"]["router"][0], uv, pooled, logits, ptab, ppmax, pb4, st["sk2_part"], st["sk2_cnt"], uvf,
               st["router_fp32"], 1, _sk2_pdl(T))
    ids, wts, lids, xm = _routing_outputs(T, dev)
    nch = 8
    rsum = st["sk2_rsum"]["sk2_gle"]
    _route_gle[(T, 2 + nch)](pooled, st["sk2_part"] if rsum else logits, part, T, st["bias"], st["g_preexp"], ids, wts,
                             xm, data["local_start"], lids, rt, P, data["stream"], g["alpha1"], g["beta1"], g["alpha2"],
                             s1, NCH=nch, TT=T, RSUM=rsum, num_warps=4, launch_pdl=True)
    m = _experts(st, moe, xm, lids, wts, T)
    U = _sk2(st, "un", m, 8, 1024, _sk2_pdl(T))
    s2 = torch.empty((T, HIDDEN), **f32)
    out = torch.empty((T, HIDDEN), device=dev, dtype=torch.bfloat16)
    _finish[(T, HIDDEN // 256)](U, T, s1, rt[1], g["beta2"], g["post2"], g["pre2"], s2, out, BC=256, num_warps=8,
                                launch_pdl=True)
    return out, s2, ids


def _chain_t8(data, st, o_in, T, moe, o_stages, un_stages):
    """2 <= T <= 8: the cluster boundary also writes the router's exact FP32 input; pooler + FP32 router in one sk2"""
    dev, g = o_in.device, st["g"]
    f32 = dict(device=dev, dtype=torch.float32)
    P = _sk2(st, "o", o_in, 8, HIDDEN, _sk2_pdl(T, o_stages))
    tail = torch.empty((T,), **f32)
    s1 = torch.empty((T, HIDDEN), **f32)
    h = torch.empty((T, HIDDEN), device=dev, dtype=torch.bfloat16)
    hr = torch.empty((T, HIDDEN), **f32)
    _MODULES["glue_router"].b1(P, data["stream"], g["pre1"], g["alpha1"], g["beta1"], h, s1, g["alpha2"], tail, hr, True,
                               st["g_preffn"])
    pooled = torch.empty((T, D_EXPERT), device=dev, dtype=torch.bfloat16)
    logits = torch.empty((T, 256), **f32)
    pstream, ptab, ppmax, pb4 = st["sk2"]["pooler"]
    st["sk2_modules"]["sk2_gle"].gle_pr(
        pstream, st["sk2"]["router"][0], h, pooled, logits, ptab, ppmax, pb4, st["sk2_part"],
        st["sk2_cnt"], hr, st["router_fp32"], 1, _sk2_pdl(T))
    ids, wts, lids, xm = _routing_outputs(T, dev)
    rsum = st["sk2_rsum"]["sk2_gle"]
    _route_split[(T, 5)](pooled, st["sk2_part"] if rsum else logits, st["bias"], st["g_preexp"], ids, wts, xm,
                         data["local_start"], lids, PARTS=4, RSUM=rsum, num_warps=4, launch_pdl=True)
    m = _experts(st, moe, xm, lids, wts, T)
    U = _sk2(st, "un", m, 8, 1024, _sk2_pdl(T, un_stages))
    s2 = torch.empty((T, HIDDEN), **f32)
    out = torch.empty((T, HIDDEN), device=dev, dtype=torch.bfloat16)
    _finish[(T, HIDDEN // 1024)](U, T, s1, tail, g["beta2"], g["post2"], g["pre2"], s2, out, BC=1024, num_warps=16,
                                 launch_pdl=True)
    return out, s2, ids


def _chain_t32(data, st, o_in, T, moe):
    """9 <= T <= 32: sk2 projections, the fused pooler + router; boundaries as 8-CTA clusters (T <= 16) or per row"""
    dev, g = o_in.device, st["g"]
    f32 = dict(device=dev, dtype=torch.float32)
    tokr = 16 if T <= 16 else 32
    P = _sk2(st, "o", o_in, tokr, HIDDEN, _sk2_pdl(T))
    s1 = torch.empty((T, HIDDEN), **f32)
    h = torch.empty((T, HIDDEN), device=dev, dtype=torch.bfloat16)
    if T <= st["glue_rows"][0]:
        _MODULES["glue"].b1(P, data["stream"], g["pre1"], g["alpha1"], g["beta1"], h, s1, s1, False)
    else:
        _boundary(P, data["stream"], g, False, s1, h)
    pooled = torch.empty((T, D_EXPERT), device=dev, dtype=torch.bfloat16)
    logits = torch.empty((T, 256), **f32)
    pstream, ptab, ppmax, pb4 = st["sk2"]["pooler"]
    st["sk2_modules"]["sk2"].sk2f(
        pstream, st["sk2"]["router"][0], h, pooled, logits, ptab, ppmax, pb4, st["sk2_part"],
        st["sk2_cnt"], _sk2_pdl(T))
    ids, wts, lids, xm = _routing_outputs(T, dev)
    rsum = st["sk2_rsum"]["sk2"]
    _route_split[(T, 2)](pooled, st["sk2_part"] if rsum else logits, st["bias"], st["g_preexp"], ids, wts, xm,
                         data["local_start"], lids, PARTS=1, RSUM=rsum, num_warps=8, launch_pdl=True)
    m = _experts(st, moe, xm, lids, wts, T)
    U = _sk2(st, "un", m, tokr, HIDDEN, _sk2_pdl(T))
    s2 = torch.empty((T, HIDDEN), **f32)
    out = torch.empty((T, HIDDEN), device=dev, dtype=torch.bfloat16)
    if T <= st["glue_rows"][1]:
        _MODULES["glue"].b2c(U, s1, g["post2"], g["pre2"], g["alpha2"], g["beta2"], out, s2)
    else:
        _boundary(U, s1, g, True, s2, out)
    return out, s2, ids


def _chain_wide(data, st, o_in, T, cls):
    """T > 32: GEMMs (cuBLASLt pinned plans at 128 / 256, rq for the 128-token unpooler, cuBLAS otherwise), per-row
    boundaries, the packed FP32 pooler + router output routed in one kernel, the WGS experts"""
    dev, g = o_in.device, st["g"]
    f32 = dict(device=dev, dtype=torch.float32)
    plan = st["lt"].get(("o", T))
    if RQ_O and cls == "t128" and ("o", T) in st["rq"]:
        P = _rq(st, "o", o_in, HIDDEN)
    else:
        P = _lt_run(o_in, plan) if plan else torch.mm(o_in, st["o"].t())
    s1 = torch.empty((T, HIDDEN), **f32)
    h = torch.empty((T, HIDDEN), device=dev, dtype=torch.bfloat16)
    _boundary(P, data["stream"], g, False, s1, h)
    ids, wts, lids, xm = _routing_outputs(T, dev)
    if RQ_POOL and cls == "t128" and ("pooler", T) in st["rq"]:
        logits = torch.mm(h, st["pr"][D_EXPERT:].t(), out_dtype=torch.float32)
        pooled = _rq(st, "pooler", h, D_EXPERT)
        _route_split[(T, 2)](pooled, logits, st["bias"], st["g_preexp"], ids, wts, xm, data["local_start"], lids,
                             PARTS=1, RSUM=2, num_warps=8, launch_pdl=True)
    else:
        plan = st["lt"].get(("pr", T))
        allp = _lt_run(h, plan) if plan else torch.mm(h, st["pr"].t(), out_dtype=torch.float32)
        _route_packed[(T, 2)](allp, st["bias"], st["g_preexp"], ids, wts, xm, data["local_start"], lids, num_warps=4,
                              launch_pdl=True)
    m = _experts(st, "big", xm, lids, wts, T)
    if cls == "t128":
        U = _rq(st, "unpooler", m, HIDDEN)
    else:
        plan = st["lt"].get(("un", T))
        U = _lt_run(m, plan) if plan else torch.mm(m, st["unpooler"].t())
    s2 = torch.empty((T, HIDDEN), **f32)
    out = torch.empty((T, HIDDEN), device=dev, dtype=torch.bfloat16)
    _boundary(U, s1, g, True, s2, out)
    return out, s2, ids


def kernel_fn(data, state):
    cfg = data["config"]
    T, L = cfg["tokens"], cfg["q_len"]
    shape = (T, L, bool(cfg["window"]))
    state = state["shape_states"].get(shape, state)
    cls = _class(T)
    x = data["attn_input"]
    _bind_context(x.device)
    if cls in ("t128", "t256"):
        qkv = _rq(state, "qkv_gate", x, 2304)
    elif cls == "wide":
        qkv = torch.mm(x, state["qkv_gate"].t())
    else:
        tokr = 8 if T <= 8 else 16 if T <= 16 else 32
        qkv = _sk2(state, "qkv", x, tokr, 2304, _sk2_pdl(T), narrow=cls == "t8")
    o_in = _attention(data, state, qkv, T)
    moe = MOE_VARIANT.get(shape, MOE_DEFAULT.get(cls, "big"))
    if cls == "t1" or shape in GLE_T8_SHAPES:
        return _chain_t1(data, state, o_in, T, moe)
    if cls == "t8":
        return _chain_t8(data, state, o_in, T, moe, 3 if shape == (8, 8, True) else 2, UN_STAGES.get(shape, 2))
    if cls in ("t16", "t32"):
        return _chain_t32(data, state, o_in, T, moe)
    return _chain_wide(data, state, o_in, T, cls)
