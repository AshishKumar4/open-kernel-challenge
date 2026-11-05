# Kernels

One file per board, each a single Python module with its CUDA sources inline, as the challenge's API takes it. Each
file exposes `prepare(weights)` (untimed, up to 300 s) and `kernel_fn(...)`, and needs the challenge's image (PyTorch,
the CUDA 13.0 toolkit, tvm-ffi).

| File | Board | Judge result | Job | md5 as submitted |
|---|---|---|---|---|
| `decode_layer_best_run.py` | decode layer | **2.342x**, 9,659.4 µs (95%: 2.307 to 2.377x), our board row | `5fcec3e5` | `49f7340b2ac83b7f3552c9fff84ebdc3` |
| `decode_layer_final.py` | decode layer | 2.323x, 9,736.8 µs | `99f60821` | `87126498f9e45932762389d1d491acff` |
| `attention_decode.py` | attention decode | 2.6525x | `41ea463c` | `2a9eb343306c5dfdeff2d15e66121fd4` |
| `moe_w4a16.py` | MoE W4A16 | 2.3878x | `8fae6971` | `a5dc9fd7f202b8a8de6d7fcb983ee2b6` |
| `bf16_proj.py` | bf16 projections | 1.5780x | `461710a3` | `02d402a0fcdec69a8872ef1a3d2332b6` |
| `prefill_layer.py` | prefill layer | 1.9096x | `c623ac38` | `d54ae58a20f8342c40df5a0e439b6fe9` |

The two decode-layer files differ from the submitted bytes only in six comments, where I removed the names of the
agents that made each change. `bf16_proj.py` differs only in eight comments, where I trimmed notes on where a
measurement was taken. The other three files are byte-identical to what we submitted.

**Which decode-layer file to read.** `decode_layer_final.py` is the last incumbent: every change adopted on the last
night, stacked and confirmed, and fully validated (cold `prepare` 67 s, 13/13 task tests, 52/52 nearby shapes, 12 x 200
judge-style replays with no failures). `decode_layer_best_run.py` is an earlier bundle from about 04:00 UTC on
November 5 that drew the best score in the final burst. All 11 files in that burst scored within the judge's noise of
each other (2.306 to 2.342x).

## The decode layer as sources

[`decode_layer_src/`](decode_layer_src/) is the final decode layer split by kernel family. `assemble.py` inlines the
CUDA sources into `layer.py` and writes one submission file:

```
python3 kernels/decode_layer_src/assemble.py /tmp/muse_decode_layer.py
```

| Source | What it holds |
|---|---|
| `src/attention.cu` | split-KV FP8 attention with QK-norm, RoPE and the cache write; `MODE` bits select per-shape builds |
| `src/moe.cu` | the persistent NVFP4 experts kernel; `MOE_*` switches select per-size-class builds |
| `src/sk2.cu` | the skinny GEMV over the lossless 12-bit BF16 weight stream (up to 32 tokens) |
| `src/rq.cu` | the 128- and 256-token GEMM over the same stream |
| `src/glue.cu` | cluster boundaries: residual, RMSNorm, router glue |
| `src/layer.py` | host code: weight layouts, `prepare`, dispatch per token count, the per-case build table |

`decode_layer_final.py` is this assembled output plus a block at the end that appends the TPE study's chosen switches
to the build flags (the point is in [`../data/final_incumbent_point.json`](../data/final_incumbent_point.json)).

## Attribution

These files are modified versions of team keell0renz's public kernels, which build on vjeux, alygg77, chenwainuo,
ZhimaoL, Madhavan113, Matsvei and NathenFernandes. Each file's header keeps every original copyright and notice line.
See [`../NOTICE`](../NOTICE).
