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

ATTENTION_CU = r"""
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cstring>
#include <type_traits>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda.h>
#include <stdint.h>

#ifndef ATT_FLAT_DECODE_STAGES
#define ATT_FLAT_DECODE_STAGES 3
#endif
#ifndef ATT_FLAT_VERIFY_UNROLL
#define ATT_FLAT_VERIFY_UNROLL 2
#endif
#ifndef ATT_EPOCH_ASYNC
#define ATT_EPOCH_ASYNC 0
#endif
#ifndef ATT_V8_QDIR
#define ATT_V8_QDIR 0   // q_len 8 per-request kernels: queries straight to the e4m3 A tile in the prologue
#endif
#ifndef ATT_V8C_BF16RED
#define ATT_V8C_BF16RED 0   // batched sliding verify: split partials reduced as BF16 (half the L2 atomics)
#endif
#ifndef ATT_V8_SCHEXP
#define ATT_V8_SCHEXP 0   // sliding verify: P's e4m3 codes by linear-mantissa exp2 (no MUFU, no F2FP)
#endif
#ifndef ATT_V8F_SCHEXP
#define ATT_V8F_SCHEXP 0   // full-context verify: the same linear-mantissa P codes
#endif
#ifndef ATT_V8C_CLUSTER
#define ATT_V8C_CLUSTER 0   // v32s (variant 256): the request's splits merge in one cluster through DSMEM
#endif
#ifndef ATT_V8_OVL
#define ATT_V8_OVL 0   // v32s (variant 256): QK^T of tile i+1 in flight during tile i's softmax, PV across the loop edge
#endif
#ifndef ATT_TRIM_FINAL_WARPS
#define ATT_TRIM_FINAL_WARPS 0
#endif
#ifndef ATT_V1_GATE_EARLY
#define ATT_V1_GATE_EARLY 0
#endif
#ifndef ATT_V1_POLL_NS
#define ATT_V1_POLL_NS 64
#endif
#ifndef ATT_V1_RED_ARRIVE
#define ATT_V1_RED_ARRIVE 0
#endif
#ifndef ATT_V1_BF16RED
#define ATT_V1_BF16RED 0
#endif
#ifndef ATT_V1_COPIES
#define ATT_V1_COPIES 1
#endif
#ifndef ATT_V1_WARP_ROW
#define ATT_V1_WARP_ROW 0
#endif
#ifndef ATT_V1_ELECT_COUNTER
#define ATT_V1_ELECT_COUNTER 0
#endif
#ifndef ATT_V1_RELAXED_POLL
#define ATT_V1_RELAXED_POLL 0
#endif
#ifndef ATT_V1_FINAL_INT
#define ATT_V1_FINAL_INT 0
#endif
#ifndef ATT_V1_CTA_BARRIER
#define ATT_V1_CTA_BARRIER 0
#endif
#ifndef ATT_VERIFY_CLUSTER
#define ATT_VERIFY_CLUSTER 0
#endif
#ifndef ATT_VERIFY_WST
#define ATT_VERIFY_WST 0
#endif
#if ATT_VERIFY_CLUSTER == 2
#include <cooperative_groups.h>
#endif

using TV = tvm::ffi::TensorView;

namespace {

// the model's attention geometry: 16 query heads, 2 KV heads, cache rows of 2 heads x 64 FP8 bytes
constexpr int HQ = 16, HKV = 2, ROW_BYTES = 128, LOG_ROW_BYTES = 7;

template <int L, bool F>
struct Cfg {
  static constexpr int NT = L == 1 ? 128 : 256;   // consumer threads per CTA
  // flat kernel on sm_90: one extra producer warp issues every K/V copy (warp specialization), so the
  // consumers never stall in cp.async issue; the stages are handed over with mbarriers
  static constexpr bool WS = F;
  static constexpr int NP = WS ? 128 : 0;   // four passes of 16 rows, eight threads per row
  static constexpr int NTX = NT + NP;   // threads per CTA
  static constexpr int MINB = 2;   // CTAs per SM the registers must allow
  // setmaxnreg split of the CTA's registers (launch: 65536 / (MINB * NTX), rounded down to 8)
  static constexpr int BASEREGS = (65536 / (MINB * NTX)) / 8 * 8 > 255 ? 248 : (65536 / (MINB * NTX)) / 8 * 8;
  static constexpr int PREGS = L == 1 ? 56 : 32;
  static constexpr int CREGS = ((BASEREGS * NTX - PREGS * NP) / NT) / 8 * 8 > 248
                                   ? 248
                                   : ((BASEREGS * NTX - PREGS * NP) / NT) / 8 * 8;
  // cp.async pipeline depth (64-key tiles): deep for the long flat runs, shallow (smaller shared memory,
  // more CTAs per SM, all clusters resident) for the per-request grids of sliding windows / one request
  static constexpr int STAGES = F ? ((L == 1) ? ATT_FLAT_DECODE_STAGES : (4)) : (3);
};
constexpr int TILE = 64;
constexpr int ROWB = 128;   // bytes per tile row: 2 KV heads x 64
constexpr int STAGE_BYTES = TILE * ROWB;   // 8 KB per stage for K (same for V)
constexpr int MAXB = 512;   // flat partition: requests per call
constexpr int MAXSEG = 64;   // flat partition: segments per CTA

// Members marked (layout) are unused: they keep the parameter layout the kernels were tuned with (removing them
// moves every later member's offset, and the compiler then schedules the kernels differently)
struct Params {
  const __nv_bfloat16* qkv;
  const int* seq_lens;
  const int64_t* cache_loc;
  uint8_t* k_cache;
  uint8_t* v_cache;
  const int* page_table;
  const float* k_scale;
  const float* v_scale;
  const float* inv_freq;
  __nv_bfloat16* out;
  float* ws_acc;
  float* ws_l;
  int* ws_cnt;
  int B, Hq, Hkv, P, max_pages, window, rope, n_splits, flat, nc;
  void* layout0;   // (layout)
  int cs;   // cluster mode: cs CTAs per cluster (DSMEM reduction)
  int layout1;   // (layout)
  void* layout2;   // (layout)
  int* bar;   // sm_80 emulation of the cluster barrier
  int ld_qkv, row_bytes;
  float scale_log2;   // sm_scale * qk_scale_factor * log2(e)
  int layout3;   // (layout)
  int coop;   // per-request grids that are all resident: every CTA finalizes a slice of the rows
  int lp;   // log2(page size)
  int lrb;   // log2(row bytes of a cache row: Hkv * 64)
  int off32;   // both caches below 4 GB (32-bit cache offsets); else the 64-bit path
  int w0;   // flat grids, 2 CTAs per SM: key share of the first CTA on each SM (1024 = even)
  int resident;   // the host checked: the grid is exactly 2 CTAs per SM and all of them are resident
  unsigned ns_magic;   // ceil(2^32 / n_splits) (0: n_splits == 1)
  int layout4;   // (layout)
  int pdlf;   // launched with programmatic dependent launch (pdlkv prologue order)
  CUtensorMap layout5[3];   // (layout)
  const uint8_t* pf;   // L2 prefetch source (the pooler's weight stream), null = off
  unsigned pf_chunk;   // bytes per CTA (MODE 128: per segment)
  unsigned pf_seg, pf_nseg;   // (MODE 128) segment stride, segment count
};

__device__ __forceinline__ uint32_t smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void cp_async16(uint32_t dst, const void* src, int src_bytes) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(src), "r"(src_bytes));
}
__device__ __forceinline__ void cp_async16_pred(uint32_t dst, const void* src, int src_bytes, bool pred) {
  // predicated copy (no branch): pred = 0 issues nothing, src_bytes = 0 zero-fills
  asm volatile(
      "{\n .reg .pred q;\n setp.ne.b32 q, %3, 0;\n @q cp.async.cg.shared.global [%0], [%1], 16, %2;\n}\n" ::"r"(dst),
      "l"(src), "r"(src_bytes), "r"((int)pred));
}
__device__ __forceinline__ void cp_async4(uint32_t dst, const void* src) {
  // Immutable input metadata only; mutable epochs and predecessor outputs use explicit L2-coherent reads.
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" ::"r"(dst), "l"(src));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void cp_wait() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

// ---- sm_90 TMA row loads tracked by one mbarrier per pipeline stage ----
__device__ __forceinline__ void mbar_fence_init() {
  asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
}
__device__ __forceinline__ bool elect_one() {
  uint32_t elected;
  asm volatile("{\n.reg .pred p;\nelect.sync _|p, 0xffffffff;\nselp.u32 %0, 1, 0, p;\n}" : "=r"(elected));
  return elected != 0;
}
__device__ __forceinline__ void mbar_expect_tx(uint32_t bar, int bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(bar), "r"(bytes) : "memory");
}
__device__ __forceinline__ void mbar_init_n(uint64_t* bar, int count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(smem_u32(bar)), "r"(count) : "memory");
}
__device__ __forceinline__ void mbar_arrive(uint64_t* bar) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" ::"r"(smem_u32(bar)) : "memory");
}
// unbounded wait for a phase (producer / consumer hand-off)
__device__ __forceinline__ void mbar_wait_full(uint64_t* bar, int parity) {
  asm volatile(
      "{\n.reg .pred q;\nWAIT_%=:\nmbarrier.try_wait.parity.shared::cta.b64 q, [%0], %1;\n@!q bra WAIT_%=;\n}\n" ::"r"(
          smem_u32(bar)),
      "r"(parity)
      : "memory");
}
// the mbarrier receives one arrival when all of this thread's earlier cp.async copies have landed
__device__ __forceinline__ void cp_async_mbar_arrive(uint64_t* bar) {
  asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];\n" ::"r"(smem_u32(bar)) : "memory");
}
// plain bulk copy global -> shared (16-byte aligned, multiple of 16 bytes), completion on an mbarrier
__device__ __forceinline__ void bulk_load(uint32_t dst, const void* src, int bytes, uint32_t bar) {
  asm volatile(
      "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n" ::"r"(dst),
      "l"(src), "r"(bytes), "r"(bar)
      : "memory");
}
// 16 zero bytes into shared memory through the same cp.async path (no global read)
__device__ __forceinline__ void cp_async16_zero(uint32_t dst, const void* any_src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, 0;\n" ::"r"(dst), "l"(any_src));
}
// barrier over the NT consumer threads only (named barrier 1); __syncthreads when there is no producer
// barrier over one warpgroup (named barriers 3 and 4)
__device__ __forceinline__ void bar_sync_n(int id, int n) {
  asm volatile("bar.sync %0, %1;\n" ::"r"(id), "r"(n) : "memory");
}
__device__ __forceinline__ void bar_arrive_n(int id, int n) {
  asm volatile("bar.arrive %0, %1;\n" ::"r"(id), "r"(n) : "memory");
}
__device__ __forceinline__ void wgsync(int wg) { asm volatile("bar.sync %0, 128;\n" ::"r"(3 + wg) : "memory"); }
template <bool WS, int NT>
__device__ __forceinline__ void csync() {
  if constexpr (WS)
    asm volatile("bar.sync 1, %0;\n" ::"n"(NT) : "memory");
  else
    __syncthreads();
}

__device__ __forceinline__ float ex2(float x) {
  float y;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}
// 2^x on the FMA / ALU pipes (round-to-integer by the 1.5 * 2^23 shifter, degree-3 minimax of 2^f on [-1/2, 1/2],
// relative error ~1e-4: far below the e4m3 step of P), for a share of the q_len 8 softmax's exponentials
__device__ __forceinline__ float ex2p(float x) {
  x = fmaxf(x, -60.f);
  const float t = x + 12582912.f;
  const float f = x - (t - 12582912.f);
  float q = fmaf(f, 0.0555041086648216f, 0.2402264923172690f);
  q = fmaf(q, f, 0.6931471805599453f);
  q = fmaf(q, f, 1.0f);
  return __int_as_float(__float_as_int(q) + (__float_as_int(t) << 23));
}
__device__ __forceinline__ uint32_t prmt(uint32_t a, uint32_t b, uint32_t sel) {
  uint32_t d;
  asm("prmt.b32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(sel));
  return d;
}
// ATT_V8_SCHEXP: four e4m3 codes of 2^x, x = s * ssc - d <= 8, without MUFU or F2FP: the code is 8 * (x + 7) rounded
// (exponent bias 7, three mantissa bits), i.e. 2^x with a linearly interpolated mantissa (Schraudolph; within +6.1%,
// a per-element weight error the row sums share). a = 8 ssc, c = 1.5 * 2^23 + 56 - 8 d: one FMA lands the code in the
// low mantissa bits; x < -7 clamps to code 0 (true value below 2^-7 of the row offset).
// HI (kernels that check P after packing instead of before): codes clamp to 127, so x > 8 can't wrap the byte and the
// check (codes >= 0x79) still sees it.
template <bool HI = false>
__device__ __forceinline__ uint32_t sch_e4m3x4(float s0, float s1, float s2, float s3, float a, float c) {
  auto code = [&](float s) {
    const float v = fmaxf(fmaf(s, a, c), 12582912.f);
    return __float_as_uint(HI ? fminf(v, 12583039.f) : v);
  };
  const uint32_t t0 = code(s0), t1 = code(s1), t2 = code(s2), t3 = code(s3);
  return prmt(prmt(t0, t1, 0x0040u), prmt(t2, t3, 0x0040u), 0x5410u);
}
// two e4m3 (HIGH 16 bits of x) -> f16x2: ptxas folds the half select into F2FP's operand (.H1), no SHF
__device__ __forceinline__ uint32_t cvt2h(uint32_t x) {
  uint32_t d;
  asm("{\n\t.reg .b16 t;\n\tmov.b32 {_, t}, %1;\n\tcvt.rn.f16x2.e4m3x2 %0, t;\n\t}" : "=r"(d) : "r"(x));
  return d;
}
// two e4m3 (low 16 bits of x) -> f16x2 (low byte -> low half)
__device__ __forceinline__ uint32_t cvt2(uint32_t x) {
  uint32_t d;
  asm("{\n\t.reg .b16 t;\n\tmov.b32 {t, _}, %1;\n\tcvt.rn.f16x2.e4m3x2 %0, t;\n\t}" : "=r"(d) : "r"(x));
  return d;
}
__device__ __forceinline__ uint32_t pack_h2(float lo, float hi) {
  __half2 h = __floats2half2_rn(lo, hi);
  return *reinterpret_cast<uint32_t*>(&h);
}
__device__ __forceinline__ uint8_t to_e4m3(float x) {
  return (uint8_t)__nv_cvt_float_to_fp8(x, __NV_SATFINITE, __NV_E4M3);
}

__device__ __forceinline__ void mma_f16(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
                                        uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
      "{%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint4 lds128(const uint8_t* p) { return *reinterpret_cast<const uint4*>(p); }
__device__ __forceinline__ uint2 lds64(const uint8_t* p) { return *reinterpret_cast<const uint2*>(p); }

// V B-fragments for 16 keys starting at kb (rows kb+2tig, +1, +8, +9), head hk, dims 8g..8g+7.
__device__ __forceinline__ void load_v_frags(const uint8_t* V, int kb, int hk, int g, int tig, uint32_t* b0,
                                             uint32_t* b1) {
  const int ch = hk * 4 + (g >> 1), off = (g & 1) * 8;
  const int k0 = kb + 2 * tig;
  const uint2 v0 = lds64(V + k0 * ROWB + ((ch ^ (k0 & 7)) << 4) + off);
  const uint2 v1 = lds64(V + (k0 + 1) * ROWB + ((ch ^ ((k0 + 1) & 7)) << 4) + off);
  const uint2 v2 = lds64(V + (k0 + 8) * ROWB + ((ch ^ ((k0 + 8) & 7)) << 4) + off);
  const uint2 v3 = lds64(V + (k0 + 9) * ROWB + ((ch ^ ((k0 + 9) & 7)) << 4) + off);
  uint32_t x;
  x = prmt(v0.x, v1.x, 0x5140);
  b0[0] = cvt2(x);
  b0[1] = cvt2h(x);
  x = prmt(v0.x, v1.x, 0x7362);
  b0[2] = cvt2(x);
  b0[3] = cvt2h(x);
  x = prmt(v0.y, v1.y, 0x5140);
  b0[4] = cvt2(x);
  b0[5] = cvt2h(x);
  x = prmt(v0.y, v1.y, 0x7362);
  b0[6] = cvt2(x);
  b0[7] = cvt2h(x);
  x = prmt(v2.x, v3.x, 0x5140);
  b1[0] = cvt2(x);
  b1[1] = cvt2h(x);
  x = prmt(v2.x, v3.x, 0x7362);
  b1[2] = cvt2(x);
  b1[3] = cvt2h(x);
  x = prmt(v2.y, v3.y, 0x5140);
  b1[4] = cvt2(x);
  b1[5] = cvt2h(x);
  x = prmt(v2.y, v3.y, 0x7362);
  b1[6] = cvt2(x);
  b1[7] = cvt2h(x);
}

__device__ __forceinline__ void gpu_fence() { __threadfence(); }
__device__ __forceinline__ void red_f32(float* a, float x) {
  asm volatile("red.global.add.f32 [%0], %1;" ::"l"(a), "f"(x) : "memory");
}
__device__ __forceinline__ void red_v4(float* a, float x, float y, float z, float w) {
  asm volatile("red.global.add.v4.f32 [%0], {%1, %2, %3, %4};" ::"l"(a), "f"(x), "f"(y), "f"(z), "f"(w) : "memory");
}


__device__ __forceinline__ void ldsm_x4_t(uint32_t addr, uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
               : "r"(addr));
}
// four 8x8 b16 matrices from registers to shared memory (each lane gives one row address: lanes 8m..8m+7 the
// rows of matrix m; thread t holds row t/4, bytes 4(t%4)..4(t%4)+3 of each matrix)
__device__ __forceinline__ void stsm_x4(uint32_t addr, uint32_t r0, uint32_t r1, uint32_t r2, uint32_t r3) {
  asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1,%2,%3,%4};\n" ::"r"(addr), "r"(r0), "r"(r1), "r"(r2),
               "r"(r3)
               : "memory");
}


// ---- sm_90 warpgroup MMA helpers ----
__device__ __forceinline__ void fence_proxy_async() { asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory"); }
__device__ __forceinline__ void wgmma_fence() { asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void wgmma_commit() { asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory"); }
template <int N>
__device__ __forceinline__ void wgmma_wait() {
  asm volatile("wgmma.wait_group.sync.aligned %0;\n" ::"n"(N) : "memory");
}
__device__ __forceinline__ void reg_fence(float& r) { asm volatile("" : "+f"(r)::"memory"); }
// D[64x64] (+)= A[64x32] (e4m3, shared memory descriptor) * B[64x32]^T (e4m3, shared memory descriptor)
__device__ __forceinline__ void wgmma_m64n64k32_e4m3_ss(float* d, uint64_t da, uint64_t db, int scale_d) {
  asm volatile(
      "{\n.reg .pred p;\nsetp.ne.b32 p, %34, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3 "
      "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
      "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, "
      "%32, %33, p, 1, 1;\n}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]),
        "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]),
        "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]),
        "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])
      : "l"(da), "l"(db), "r"(scale_d));
}
// K-major operand with 64-byte rows, 64B swizzle (8-row atoms of 512 bytes)
__device__ __forceinline__ uint64_t desc_sw64(uint32_t addr) {
  return (uint64_t)((addr & 0x3FFFF) >> 4) | ((uint64_t)1 << 16) | ((uint64_t)(512 >> 4) << 32) | ((uint64_t)2 << 62);
}
// D[64x72] += A[64x32] (e4m3, registers) * B[72x32]^T (e4m3, K-major shared memory): d = 32 + 4 accumulators
__device__ __forceinline__ void wgmma_m64n72k32_e4m3(float* d, float* e, const uint32_t* a, uint64_t desc) {
  asm volatile(
      "{\n"
      "wgmma.mma_async.sync.aligned.m64n72k32.f32.e4m3.e4m3 "
      "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
      "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35}, "
      "{%36, %37, %38, %39}, %40, 1, 1, 1;\n}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]),
        "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]),
        "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]),
        "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(e[0]),
        "+f"(e[1]), "+f"(e[2]), "+f"(e[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc));
}
// floor(x / d) for 0 <= x < 2^24 and 1 <= d < 2^24 through the float reciprocal (one correction step):
// ~8 instructions instead of the ~20 of an integer division, which matters in once-per-launch (cold) code
__device__ __forceinline__ int fdiv_i(int x, int d) {
  // approximate reciprocal (MUFU.RCP, no IEEE slow-path call: the correction below absorbs its error)
  const float rd = __fdividef(1.f, __int2float_rn(d));
  int q = __float2int_rz(__int2float_rn(x) * rd);
  const int r = x - q * d;
  q += (r >= d) - (r < 0);
  return q;
}
// sigmoid for the output gate: 0.5 + 0.5 tanh(x / 2) with the hardware tanh (3 instructions; abs error ~2^-12,
// below the BF16 output's rounding) or exp + fast reciprocal
__device__ __forceinline__ float sigm(float x) { return __fdividef(1.f, 1.f + __expf(-x)); }
// Programmatic dependent launch (sm_90): wait for the preceding grid (the one writing qkv) before the first qkv read;
// let the next grid start its own prologue once the main loop is done. Both are no-ops for a plain launch.
__device__ __forceinline__ void pdl_wait() { asm volatile("griddepcontrol.wait;" ::: "memory"); }
__device__ __forceinline__ void pdl_launch_dependents() {
  // (no early trigger at the end of the tile loop -- the dependent kernel launches when this grid
  // exits; for decode_layer's chain A/B, no effect on standalone launches)
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
}
// 32-bit offsets for qkv, the workspace and the caches (the host checks the sizes): no 64-bit multiplies
// in the once-per-launch address math
typedef int ix_t;
__device__ __forceinline__ const uint8_t* cache_row(const uint8_t* base, const Params& p, int64_t row, int col_bytes) {
  return base + (((uint32_t)row << LOG_ROW_BYTES) + (uint32_t)col_bytes);
  return base + row * ROW_BYTES + col_bytes;
}
// `for (e = tid; e < N; e += NTH)` with a compile-time trip count: fully unrolled, the tail predicated (the
// once-executed prologue loops otherwise keep 3-4 branches each; a taken branch costs 60-100 cycles cold)
template <int N, int NTH, class F>
__device__ __forceinline__ void tfor(int tid, F&& f) {
#pragma unroll
  for (int e0 = 0; e0 < N; e0 += NTH) {
    const int e = e0 + tid;
    if ((N % NTH) == 0 || e < N) f(e);
  }
}
// four floats -> four e4m3 bytes (x0 in the low byte), saturating
__device__ __forceinline__ uint32_t e4m3x4f(float x0, float x1, float x2, float x3) {
  uint16_t lo, hi;
  asm("cvt.rn.satfinite.e4m3x2.f32 %0, %2, %1;" : "=h"(lo) : "f"(x0), "f"(x1));
  asm("cvt.rn.satfinite.e4m3x2.f32 %0, %2, %1;" : "=h"(hi) : "f"(x2), "f"(x3));
  return (uint32_t)lo | ((uint32_t)hi << 16);
}
// four f16 (two f16x2 words, low first) -> four e4m3 bytes (low byte first)
__device__ __forceinline__ uint32_t pack_e4m3x4(uint32_t h01, uint32_t h23) {
  __half2_raw a, b;
  a.x = (unsigned short)(h01 & 0xffffu);
  a.y = (unsigned short)(h01 >> 16);
  b.x = (unsigned short)(h23 & 0xffffu);
  b.y = (unsigned short)(h23 >> 16);
  const uint32_t lo = __nv_cvt_halfraw2_to_fp8x2(a, __NV_SATFINITE, __NV_E4M3);
  const uint32_t hi = __nv_cvt_halfraw2_to_fp8x2(b, __NV_SATFINITE, __NV_E4M3);
  return (lo & 0xffffu) | (hi << 16);
}


__device__ __forceinline__ float quad_max(float v) {
  v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 1));
  v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 2));
  return v;
}
__device__ __forceinline__ float quad_sum(float v) {
  v += __shfl_xor_sync(0xffffffffu, v, 1);
  v += __shfl_xor_sync(0xffffffffu, v, 2);
  return v;
}
__device__ __forceinline__ float oct_sum(float v) {
  v += __shfl_xor_sync(0xffffffffu, v, 1);
  v += __shfl_xor_sync(0xffffffffu, v, 2);
  v += __shfl_xor_sync(0xffffffffu, v, 4);
  return v;
}

template <int L, bool F>
struct Smem {
  uint8_t k[Cfg<L, F>::STAGES * STAGE_BYTES];
  uint8_t v[Cfg<L, F>::STAGES * STAGE_BYTES];
  uint64_t mbar[8];   // TMA mode: one per stage (64 bytes keeps the members after it 16-byte aligned)
  uint64_t pbar[4];   // producer mode: page-index slots landed (TMA)
  int pgc;   // page-index fills issued in this CTA's earlier segments
  int dbgn;   // debug builds: progress-mark count
  int vidx;   // flat weighted partition: this CTA's slot (2 * SM + arrival rank), -1 = uniform
  alignas(128) int pidx[4][80];   // producer mode: page indices of upcoming tiles (16-byte aligned window)
  uint64_t full[8];   // producer mode: stage landed (32 producer lanes arrive as their copies complete)
  uint64_t empty[8];   // producer mode: stage consumed (one arrival per consumer warp)
  uint64_t vtf[2];   // V^T slot written by the producers (one arrival per producer thread)
  uint64_t vte[2];   // V^T slot read by both heads' PV wgmmas (one arrival per consumer warp)
  uint64_t qdone;   // consumers have read the q staging (S.kvq) and staged S.kn / S.vn
  int claim[2];   // 0: next unclaimed tile per KV head
  int nxt[4][2];   // 0: each warpgroup's next claimed tile (double buffered)
  int idx[2 * Cfg<L, F>::STAGES * TILE];
  // raw q rows of the block (bf16, prologue only) | L==8: the current tile's K and V in f16 (2 x 2 x 64 x 64)
  // L == 8: q staging (producer mode) before the loop, then the per-tile f16 V copy -- or, FP8 PV, per tile
  // (double buffer) and KV head, V^T as 72 rows (64 dims + 8 rows of ones for the row sums) x 64 keys (permuted to
  // the P register order), K-major with 64B swizzle: 4 x 4608 bytes
  alignas(1024) uint8_t kvq[L == 1 ? L * 16 * 64 * 2 : 18432];
  // L == 8 with FP8 PV and qss: the block's e4m3 queries per KV head (64 rows x 64 dims, K-major, 64B swizzle)
  alignas(1024) uint8_t q8s[L == 8 ? 8192 : 16];
  __device__ __forceinline__ __nv_bfloat16* qst() { return reinterpret_cast<__nv_bfloat16*>(kvq); }
  __nv_bfloat16 kvst[L * 4 * 64];   // raw k (2 heads) then v (2 heads) rows per token
  __nv_bfloat16 gst[L * 4 * 64];   // cluster mode: gate rows of this CTA's output slice (<= L*16/4 rows)
  uint8_t kn[L * ROWB];   // new keys of the block, [j][2 heads x 64] e4m3
  uint8_t vn[L * ROWB];
  float cs[L * 32];
  float sn[L * 32];
  float ifreq[32];   // RoPE inverse frequencies (staged with cp.async)
  float scl[4];   // k_scale, v_scale
  long long cloc[L];   // cache rows of the block's tokens
  int flag;
  int sense;   // synchronous epoch staging; retained in both builds to preserve shared layout
  int cum[F ? MAXB + 1 : 1];
  int seg[F ? MAXSEG : 1][3];
  int nseg;
  int fcs, fce;   // this CTA's key range (warp 0 -> everyone)
  // flat dynamic chunks : the producer's segment list of its current chunk, and a queue that hands the
  // segments (request, first key, end key, copy; request -1 = done) to the consumers
  int pseg[F ? MAXSEG : 1][3];
  int pnseg, pchunk;
  int dq[F ? 16 : 1][4];
  uint64_t dqf[F ? 16 : 1];
  uint64_t dqe[F ? 16 : 1];
};

// Epochs occupy isolated 16-byte slots in EVERY build: per-case variants share one counter workspace.
// The other three words stay untouched, so a coherent vector copy never reads another request's live counter.
constexpr int REQUEST_EPOCH_BASE = 24576, FLAT_EPOCH = 30024;
template <int L, bool F>
__device__ __forceinline__ int* epoch_smem(Smem<L, F>& S) {
#if ATT_EPOCH_ASYNC
  // gst is unused; its base is 16-byte aligned. Leave its first 256 bytes available to legacy probes.
  return reinterpret_cast<int*>(S.gst + 128);
#else
  return &S.sense;
#endif
}
template <int L, bool F>
__device__ __forceinline__ void stage_epoch(Smem<L, F>& S, const int* src) {
#if ATT_EPOCH_ASYNC
  cp_async16(smem_u32(epoch_smem(S)), src, 16);
#else
  *epoch_smem(S) = __ldcg(src);
#endif
}

// One (request b, KV pair) segment [start, end) of the request's key positions.
// Sum the nc workspace copies of one (row, 8-dim part) of request-pair bp, token j, head hh, and zero them. Copies in
// groups of 4 (loads of a group in flight together): one compact loop body, since this runs once per launch from a
// cold instruction cache (an 8-way unrolled version cost ~2x the instructions).
template <int L, bool W72 = false>
__device__ __forceinline__ void sum_copies(const Params& p, const int bp, const int j, const int hh, const int part,
                                           const bool act, float (&y)[8], float& lsum) {
  if (p.nc == 1) {   // one copy (the common case): no copy loop, no per-copy branches (scout's census)
    if (act) {
      float* acc = p.ws_acc + ((ix_t)(bp * L + j) * 16 + hh) * (W72 ? 72 : 64) + part * 8;
      const float4 xa = __ldcg(reinterpret_cast<const float4*>(acc));
      const float4 xb = __ldcg(reinterpret_cast<const float4*>(acc) + 1);
      const float lc =
          __ldcg(W72 ? p.ws_acc + ((ix_t)(bp * L + j) * 16 + hh) * 72 + 64 : p.ws_l + (bp * L + j) * 16 + hh);
      y[0] += xa.x;
      y[1] += xa.y;
      y[2] += xa.z;
      y[3] += xa.w;
      y[4] += xb.x;
      y[5] += xb.y;
      y[6] += xb.z;
      y[7] += xb.w;
      lsum += lc;
      *reinterpret_cast<float4*>(acc) = make_float4(0.f, 0.f, 0.f, 0.f);
      *(reinterpret_cast<float4*>(acc) + 1) = make_float4(0.f, 0.f, 0.f, 0.f);
      if (part == 0)
        *(W72 ? p.ws_acc + ((ix_t)(bp * L + j) * 16 + hh) * 72 + 64 : p.ws_l + (bp * L + j) * 16 + hh) = 0.f;
    }
    return;
  }
  if (p.nc == 8) {   // eight copies (the one-request q_len 8 plan): every copy's loads in one round trip
    if (act) {   // (the rolled 4-copy loop below took two dependent L2 round trips)
      float4 xa[8], xb[8];
      float lc[8];
#pragma unroll
      for (int c = 0; c < 8; ++c) {
        const float* acc = p.ws_acc + ((ix_t)((bp * 8 + c) * L + j) * 16 + hh) * (W72 ? 72 : 64) + part * 8;
        xa[c] = __ldcg(reinterpret_cast<const float4*>(acc));
        xb[c] = __ldcg(reinterpret_cast<const float4*>(acc) + 1);
        lc[c] = __ldcg(W72 ? p.ws_acc + ((ix_t)((bp * 8 + c) * L + j) * 16 + hh) * 72 + 64
                           : p.ws_l + ((bp * 8 + c) * L + j) * 16 + hh);
      }
#pragma unroll
      for (int c = 0; c < 8; ++c) {
        y[0] += xa[c].x;
        y[1] += xa[c].y;
        y[2] += xa[c].z;
        y[3] += xa[c].w;
        y[4] += xb[c].x;
        y[5] += xb[c].y;
        y[6] += xb[c].z;
        y[7] += xb[c].w;
        lsum += lc[c];
        float* acc = p.ws_acc + ((ix_t)((bp * 8 + c) * L + j) * 16 + hh) * (W72 ? 72 : 64) + part * 8;
        *reinterpret_cast<float4*>(acc) = make_float4(0.f, 0.f, 0.f, 0.f);
        *(reinterpret_cast<float4*>(acc) + 1) = make_float4(0.f, 0.f, 0.f, 0.f);
        if (part == 0)
          *(W72 ? p.ws_acc + ((ix_t)((bp * 8 + c) * L + j) * 16 + hh) * 72 + 64
                : p.ws_l + ((bp * 8 + c) * L + j) * 16 + hh) = 0.f;
      }
    }
    return;
  }
#pragma unroll 1
  for (int c0 = 0; c0 < p.nc; c0 += 4) {
    float4 xa[4], xb[4];
    float lc[4];
#pragma unroll
    for (int u = 0; u < 4; ++u) {
      const int c = c0 + u;
      xa[u] = make_float4(0.f, 0.f, 0.f, 0.f);
      xb[u] = make_float4(0.f, 0.f, 0.f, 0.f);
      lc[u] = 0.f;
      if (act && c < p.nc) {
        const float* acc = p.ws_acc + ((ix_t)((bp * p.nc + c) * L + j) * 16 + hh) * (W72 ? 72 : 64) + part * 8;
        xa[u] = __ldcg(reinterpret_cast<const float4*>(acc));
        xb[u] = __ldcg(reinterpret_cast<const float4*>(acc) + 1);
        lc[u] = __ldcg(W72 ? p.ws_acc + ((ix_t)((bp * p.nc + c) * L + j) * 16 + hh) * 72 + 64
                           : p.ws_l + ((bp * p.nc + c) * L + j) * 16 + hh);
      }
    }
#pragma unroll
    for (int u = 0; u < 4; ++u) {
      const int c = c0 + u;
      if (act && c < p.nc) {
        y[0] += xa[u].x;
        y[1] += xa[u].y;
        y[2] += xa[u].z;
        y[3] += xa[u].w;
        y[4] += xb[u].x;
        y[5] += xb[u].y;
        y[6] += xb[u].z;
        y[7] += xb[u].w;
        lsum += lc[u];
        float* acc = p.ws_acc + ((ix_t)((bp * p.nc + c) * L + j) * 16 + hh) * (W72 ? 72 : 64) + part * 8;
        *reinterpret_cast<float4*>(acc) = make_float4(0.f, 0.f, 0.f, 0.f);
        *(reinterpret_cast<float4*>(acc) + 1) = make_float4(0.f, 0.f, 0.f, 0.f);
        if (part == 0)
          *(W72 ? p.ws_acc + ((ix_t)((bp * p.nc + c) * L + j) * 16 + hh) * 72 + 64
                : p.ws_l + ((bp * p.nc + c) * L + j) * 16 + hh) = 0.f;
      }
    }
  }
}
// The same for BF16 partials (ATT_V8C_BF16RED): 64-dim rows of 128 bytes, row sums FP32 in ws_l; zeroes them.
template <int L>
__device__ __forceinline__ void sum_copies_bf16(const Params& p, const int bp, const int j, const int hh,
                                                const int part, const bool act, float (&y)[8], float& lsum) {
  if (!act) return;
#pragma unroll 1
  for (int c = 0; c < p.nc; ++c) {
    const ix_t row = (ix_t)((bp * p.nc + c) * L + j) * 16 + hh;
    uint4* acc = reinterpret_cast<uint4*>(p.ws_acc) + row * 8 + part;
    const uint4 x = __ldcg(acc);
    const __nv_bfloat162* x2 = reinterpret_cast<const __nv_bfloat162*>(&x);
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      const float2 v = __bfloat1622float2(x2[e]);
      y[2 * e] += v.x;
      y[2 * e + 1] += v.y;
    }
    lsum += __ldcg(p.ws_l + row);
    *acc = make_uint4(0, 0, 0, 0);
    if (part == 0) p.ws_l[row] = 0.f;
  }
}

__device__ __forceinline__ float rms_softmax_mean(float (&y)[8], float lsum, float vscale) {
  // Fixed-reference sums can be finite while their squares overflow. Form the attention mean first.
  // Inactive rows have zero numerator and denominator; selecting 1 avoids 0/0 without a divergent divide path.
  const float inv = __fdividef(vscale, lsum > 0.f ? lsum : 1.f);
  float ss = 0.f;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    y[i] *= inv;
    ss += y[i] * y[i];
  }
  ss = oct_sum(ss);   // every lane participates, including inactive rows with zero numerator and denominator
  return rsqrtf(ss * (1.f / 64.f) + 1e-5f);
}

// Last-CTA finalize of a request with one workspace copy: sum, normalize, rmsnorm * sigmoid(gate), store,
// reset the workspace.
template <int L, int NT, bool W72 = false>
__device__ __forceinline__ void fin_nc1(const Params& p, const int b, const int pair, const int bp, const float vscale) {
  const int tid = threadIdx.x;
  // loads of a group of passes first (one L2 round trip per group), then math and stores; groups of 2 passes in a
  // rolled loop (this runs once per launch from a cold instruction cache: q_len 8 has 4 passes)
  constexpr int PASSES = L * 16 / (NT / 8);
  constexpr int GP = PASSES >= 2 ? 2 : 1;
  const int part = tid & 7;
#pragma unroll 1
  for (int p0 = 0; p0 < PASSES; p0 += GP) {
    float4 xa[GP], xb[GP];
    float lc[GP];
    uint4 graw[GP];
#pragma unroll
    for (int u = 0; u < GP; ++u) {
      const int rr = (p0 + u) * (NT / 8) + (tid >> 3);
      const int j = rr >> 4, hh = rr & 15;
      const float* acc = p.ws_acc + ((ix_t)(bp * L + j) * 16 + hh) * (W72 ? 72 : 64) + part * 8;
      xa[u] = __ldcg(reinterpret_cast<const float4*>(acc));
      xb[u] = __ldcg(reinterpret_cast<const float4*>(acc) + 1);
      lc[u] = __ldcg(W72 ? p.ws_acc + ((ix_t)(bp * L + j) * 16 + hh) * 72 + 64 : p.ws_l + (bp * L + j) * 16 + hh);
      graw[u] = *reinterpret_cast<const uint4*>(p.qkv + (ix_t)(b * L + j) * p.ld_qkv + (HQ + 2 * HKV) * 64 +
                                                (16 * pair + hh) * 64 + part * 8);
    }
#pragma unroll
    for (int u = 0; u < GP; ++u) {
      const int rr = (p0 + u) * (NT / 8) + (tid >> 3);
      const int j = rr >> 4, hh = rr & 15;
      float* acc = p.ws_acc + ((ix_t)(bp * L + j) * 16 + hh) * (W72 ? 72 : 64) + part * 8;
      *reinterpret_cast<float4*>(acc) = make_float4(0.f, 0.f, 0.f, 0.f);
      *(reinterpret_cast<float4*>(acc) + 1) = make_float4(0.f, 0.f, 0.f, 0.f);
      if (part == 0)
        *(W72 ? p.ws_acc + ((ix_t)(bp * L + j) * 16 + hh) * 72 + 64 : p.ws_l + (bp * L + j) * 16 + hh) = 0.f;
      float y[8] = {xa[u].x, xa[u].y, xa[u].z, xa[u].w, xb[u].x, xb[u].y, xb[u].z, xb[u].w};
      const float rn = rms_softmax_mean(y, lc[u], vscale);
      const __nv_bfloat162* g2 = reinterpret_cast<const __nv_bfloat162*>(&graw[u]);
      uint4 ov;
      uint32_t* ow = reinterpret_cast<uint32_t*>(&ov);
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const float2 gt = __bfloat1622float2(g2[e]);
        const float s0 = sigm(gt.x), s1 = sigm(gt.y);
        __nv_bfloat162 r2 = __floats2bfloat162_rn(y[2 * e] * rn * s0, y[2 * e + 1] * rn * s1);
        ow[e] = *reinterpret_cast<uint32_t*>(&r2);
      }
      *reinterpret_cast<uint4*>(p.out + (ix_t)(b * L + j) * (HQ * 64) + (16 * pair + hh) * 64 + part * 8) = ov;
    }
  }
}

// the q_len 8 flat loop's rare new-row patch as a call, so ptxas cannot hoist its bounds math into the
// common path of every tile
template <int L, int NT, bool PTR, class SM>
__device__ __noinline__ bool ws_patch_ni(SM* Sp, int i, int st, int start, int seq, int end, int tid) {
  SM& S = *Sp;
  const int t0 = start + i * TILE;
  const int lo = max(t0, seq);
  const int hi = min(t0 + TILE, end);
  if (lo < hi) {
    for (int e = tid; e < (hi - lo) * 8; e += NT) {
      const int pos = lo + (e >> 3), c = e & 7, r = pos - t0;
      const int sw = ((c ^ (r & 7)) << 4);
      *reinterpret_cast<uint4*>(S.k + st * STAGE_BYTES + r * ROWB + sw) =
          *reinterpret_cast<const uint4*>(S.kn + (pos - seq) * ROWB + c * 16);
      if constexpr (!PTR)
        *reinterpret_cast<uint4*>(S.v + st * STAGE_BYTES + r * ROWB + sw) =
            *reinterpret_cast<const uint4*>(S.vn + (pos - seq) * ROWB + c * 16);
    }
    if (L == 8) fence_proxy_async();
  }
  return lo < hi;
}

// ATT_V8C_CLUSTER: the request's splits are one cluster (rank = split). The splits with keys (ranks 0..nact-1) own
// R = ceil(L*16 / nact) consecutive output rows each. After the cluster barrier that follows every split's tile loop
// (the K/V stages are then free everywhere), every active split pushes its rescaled partial rows to their owners'
// K/V stage memory with st.async, which completes bytes on the owner's barrier (S.mbar[0], unused in this build):
// [CL_ROWS][64] rows (source-major slices) then [CL_ROWS] row sums. No counter, no global workspace, no GPU fence.
constexpr int CL_ROWS = 136;   // nact * R <= L*16 + nact - 1 rows for nact <= 8
template <int L, bool F>
__device__ __forceinline__ float* cl_base(Smem<L, F>& S) {
  return reinterpret_cast<float*>(S.k);   // S.k then S.v: 48 KB >= CL_ROWS * 65 floats
}
// Owner barriers expect nact slices of their rows (64 floats + the sum each); armed at kernel start, ordered before
// every remote complete_tx by the init fence and the post-loop cluster barrier
template <int L, bool F>
__device__ __forceinline__ void cl_arm(Smem<L, F>& S, const int rank, const int n_keys, const int chunk) {
  if (threadIdx.x == 0) {
    const int nact = fdiv_i(n_keys + chunk - 1, chunk);
    const int R = fdiv_i(L * 16 + nact - 1, nact);
    if (rank < nact) {
      const int rows = min(L * 16, (rank + 1) * R) - rank * R;
      const uint32_t bar = smem_u32(&S.mbar[0]);
      asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(bar) : "memory");
      asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
      asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(bar), "r"(nact * rows * 65 * 4)
                   : "memory");
    }
  }
}
template <int L, int NT, bool F>
__device__ __forceinline__ void cl_merge(const Params& p, Smem<L, F>& S, const int b, const int pair, const int n_keys,
                                         const int chunk, const float vscale, const float (&o)[1][8][4],
                                         const float (&m_r)[2], const float (&l_r)[2], const int wsub, const int hk,
                                         const int g, const int tig, const int tid) {
  const int rank = (int)blockIdx.x;
  const int nact = fdiv_i(n_keys + chunk - 1, chunk);
  const int R = fdiv_i(L * 16 + nact - 1, nact);
  float* rb = cl_base(S);
  float* rl = rb + CL_ROWS * 64;
  const uint32_t bar = smem_u32(&S.mbar[0]);
  // every split is past its tile loop (its stage memory is free: the loop waited for its last copies and wgmmas) and
  // every owner's barrier is armed (init fence). Relaxed arrive: the release form drains global memory (MEMBAR.GPU);
  // the pushed data is published by complete_tx and acquired by the owner's try_wait
  asm volatile("barrier.cluster.arrive.relaxed.aligned;\nbarrier.cluster.wait.aligned;" ::: "memory");
  const bool odd = tig & 1;
#pragma unroll
  for (int r = 0; r < 2; ++r) {
    const int rr = (2 * wsub + r) * 16 + hk * 8 + g;
    const int ow = fdiv_i(rr, R);
    const int slot = rank * R + rr - ow * R;   // this source's slice in the owner's buffer
    const float lsum = quad_sum(l_r[r]);
    const float f = (m_r[r] == -INFINITY) ? 0.f : ex2(fminf(m_r[r], 126.f));
    uint32_t dst, rbar;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(dst) : "r"(smem_u32(rb + slot * 64)), "r"(ow));
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(rbar) : "r"(bar), "r"(ow));
#pragma unroll
    for (int t = 0; t < 8; t += 2) {   // lane pairs swap halves: each lane sends 4 consecutive dims (as the red path)
      const float a0 = o[0][t][2 * r] * f, a1 = o[0][t][2 * r + 1] * f;
      const float b0 = o[0][t + 1][2 * r] * f, b1 = o[0][t + 1][2 * r + 1] * f;
      const float s0 = odd ? a0 : b0, s1 = odd ? a1 : b1;
      const float r0 = __shfl_xor_sync(0xffffffffu, s0, 1), r1 = __shfl_xor_sync(0xffffffffu, s1, 1);
      asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.f32 [%0], {%1, %2, %3, %4}, [%5];" ::"r"(
                       dst + 4 * (8 * (t + (odd ? 1 : 0)) + 4 * (tig >> 1))),
                   "f"(odd ? r0 : a0), "f"(odd ? r1 : a1), "f"(odd ? b0 : r0), "f"(odd ? b1 : r1), "r"(rbar)
                   : "memory");
    }
    if (tig == 0) {
      uint32_t dl;
      asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(dl) : "r"(smem_u32(rl + slot)), "r"(ow));
      asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.f32 [%0], %1, [%2];" ::"r"(dl),
                   "f"(lsum * f), "r"(rbar)
                   : "memory");
    }
  }
  if (tid == 0) {   // every active split's slice of this CTA's rows has landed (acquire pairs with complete_tx)
    asm volatile("{\n .reg .pred p;\n CLW_%=:\n mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [%0], 0;\n"
                 " @!p bra CLW_%=;\n}\n" ::"r"(bar)
                 : "memory");
  }
  __syncthreads();
  const int r_lo = rank * R, r_hi = min(L * 16, r_lo + R);
  const int part = tid & 7;
  for (int rbase = r_lo; rbase < r_hi; rbase += NT / 8) {   // warp-uniform trip count (shuffles below)
    const int rr = rbase + (tid >> 3);
    const bool act = rr < r_hi;
    const int j = act ? rr >> 4 : 0, hh = act ? rr & 15 : 0;
    const uint4 graw = act ? *reinterpret_cast<const uint4*>(p.qkv + (ix_t)(b * L + j) * p.ld_qkv + (HQ + 2 * HKV) * 64 +
                                                             (16 * pair + hh) * 64 + part * 8)
                           : make_uint4(0, 0, 0, 0);
    float y[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    float lsum = 0.f;
    if (act) {
#pragma unroll 1
      for (int sr = 0; sr < nact; ++sr) {
        const float* row = rb + (sr * R + rr - r_lo) * 64 + part * 8;
        const float4 xa = *reinterpret_cast<const float4*>(row), xb = *reinterpret_cast<const float4*>(row + 4);
        y[0] += xa.x;
        y[1] += xa.y;
        y[2] += xa.z;
        y[3] += xa.w;
        y[4] += xb.x;
        y[5] += xb.y;
        y[6] += xb.z;
        y[7] += xb.w;
        lsum += rl[sr * R + rr - r_lo];
      }
    }
    const float rn = rms_softmax_mean(y, lsum, vscale);
    const __nv_bfloat162* g2 = reinterpret_cast<const __nv_bfloat162*>(&graw);
    uint4 ov;
    uint32_t* ow = reinterpret_cast<uint32_t*>(&ov);
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      const float2 gt = __bfloat1622float2(g2[e]);
      const float s0 = sigm(gt.x), s1 = sigm(gt.y);
      __nv_bfloat162 r2 = __floats2bfloat162_rn(y[2 * e] * rn * s0, y[2 * e + 1] * rn * s1);
      ow[e] = *reinterpret_cast<uint32_t*>(&r2);
    }
    if (act) *reinterpret_cast<uint4*>(p.out + (ix_t)(b * L + j) * (HQ * 64) + (16 * pair + hh) * 64 + part * 8) = ov;
  }
}

template <int L, bool FLAT, int ROLE, int MODE = 0>
__device__ __forceinline__ bool segment(const Params& p, Smem<L, FLAT>& S, const int b_, const int pair_,
                                        const bool first, const int start_, const int end_, const int seq_,
                                        const int n_keys_, const int copy_, const int chunk_, const int tb_ = 0,
                                        const int seq_real_ = 0, const bool skip_init_ = false) {
  // (returns false only in the spec builds, when the speculated length missed: nothing global was written yet and
  // the caller re-runs the segment with the real length)
  // per-segment values as lane-0 broadcasts : provably warp-uniform, so ptxas keeps the address math
  // derived from them in uniform registers (they come from shared memory or per-CTA math, uniform but unproven)
#define UNI(x) __shfl_sync(0xffffffffu, (x), 0)
  const int b = UNI(b_);
  const int pair = UNI(pair_);
  const int start = UNI(start_);
  const int end = UNI(end_);
  const int seq = UNI(seq_);
  const int n_keys = UNI(n_keys_);
  const int copy = UNI(copy_);
  const int chunk = UNI(chunk_);
  const int tb = UNI(tb_);
#undef UNI
  // tb: tiles this CTA's pipeline has already run (flat producer mode: the stage ring and its mbarrier phases
  // continue across segments, so the producer runs ahead into the next segment instead of draining)
  // MODE: the per-request (non-flat) kernel is built once per launch mode, so each build carries only its own
  // epilogue (once-per-launch code runs from a cold instruction cache): 1 plain split-KV, 2 clusters,
  // 3 cooperative; 0 = decided at run time (flat kernels)
  // MODE | 4 = a build for grids whose every split holds at most one tile (the host checks): the tile loop,
  // the in-loop copies and the first-tile tests fold at compile time (scout's once-path census, s1 / vs1)
  constexpr bool ONE = !FLAT && (MODE & 4) != 0;
  // q_len 1 per-request merges: 72-float workspace rows (the sum in column 64), one bulk reduce-add per CTA; or
  // (MODE 16) 64-float rows and the sums apart, two bulk reduce-adds; or (MODE 32) red.v4 atomics
  constexpr bool W72 = L == 1 && !FLAT && !ONE && (MODE & 48) == 0;
  constexpr bool BULK64 = L == 1 && !FLAT && (MODE & 16) != 0;
  // ATT_V8C_BF16RED: the batched sliding verify build (MODE 3 exactly: B > 1, cooperative finalize) adds BF16 partials
  constexpr bool V8B = L == 8 && !FLAT &&
      ((ATT_V8C_BF16RED && MODE == 3) || (ATT_V1_BF16RED && (MODE & 128) != 0));
  // ATT_V8_SCHEXP: the sliding verify builds (MODE 3: B > 1; | 128: v1s) form P's e4m3 codes on the FMA/ALU pipes
  constexpr bool SCH = ATT_V8_SCHEXP && L == 8 && !FLAT && ((MODE & ~256) == 3 || (MODE & 128) != 0);
  // ATT_V8C_CLUSTER (MODE | 256): the request's splits are one cluster; partials go to their rows' owners by DSMEM
  constexpr bool CL = ATT_V8C_CLUSTER && L == 8 && !FLAT && (MODE & 256) != 0;
  // ATT_V8_OVL (MODE | 256): the tile loop overlaps QK^T(i+1) with softmax(i) (two score sets, one CTA per SM)
  constexpr bool OVL = ATT_V8_OVL && L == 8 && !FLAT && (MODE & 256) != 0;
  // ATT_V8F_SCHEXP: the same for the full-context verify builds (flat; MODE 11: one request)
  constexpr bool SCHF = ATT_V8F_SCHEXP && L == 8 && (FLAT || MODE == 11);
  const bool M_COOP = (MODE & 3) == 0 ? p.coop != 0 : (MODE & 3) == 3;
  constexpr int NT = Cfg<L, FLAT>::NT;
  constexpr int STAGES = Cfg<L, FLAT>::STAGES;
  // flat q_len 8 with FP8 PV: the producer warps (idle most of the time) build each tile's V^T for the PV wgmma
  constexpr bool PTR = false;
  constexpr int IDX_SLOTS = 2 * STAGES;
  constexpr bool WS = Cfg<L, FLAT>::WS;
  // (the warp index as a lane-0 broadcast, so ptxas can prove it warp-uniform and keep the addresses
  // derived from it (KV head, descriptors, buffers) in uniform registers; CUTLASS's canonical_warp_idx_sync idiom)
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = __shfl_sync(0xffffffffu, tid >> 5, 0);
  const int g = lane >> 2, tig = lane & 3;
  const int split = first ? 0 : 1;
  const int pairs = HKV >> 1;
  const int bp = b * pairs + pair;
  const int ntiles = ONE ? 1 : (end - start + TILE - 1) / TILE;   // (ONE: a CTA without keys returned earlier)

  if constexpr (WS) {
    constexpr int NP = Cfg<L, FLAT>::NP;
    constexpr int PR = NP / 8;   // rows per producer pass
    constexpr int PK = TILE / PR;   // passes per tile
    if (!skip_init_) {   // (the first segment's barriers were set up at kernel start)
      if (ROLE == 2 && tid == 0) {   // fresh phases for this segment (the previous one ended with every stage consumed)
        for (int st = 0; st < STAGES; ++st) {
          mbar_init_n(&S.full[st], NP);
          mbar_init_n(&S.empty[st], NT);
        }
        if constexpr (PTR) {
          for (int k = 0; k < 2; ++k) {
            mbar_init_n(&S.vtf[k], NP);
            mbar_init_n(&S.vte[k], NT / 32);
          }
          mbar_init_n(&S.qdone, NT / 32);
        }
        mbar_fence_init();
      }
      __syncthreads();
    }
    if constexpr (ROLE == 1) {
      // ---- producer threads: 16-byte chunk (q & 7) of rows (q >> 3) + PR*k; the cache rows of every tile,
      // zeros past the range; the block's own new keys are patched in by the consumers
      const int q = tid - NT;
      const int ch = q & 7;
      const int r0 = q >> 3;
      const int* ptr = p.page_table + (ix_t)b * p.max_pages;
      const int cache_end = min(end, seq);
      int pg[PK];

      const bool pfast = p.lp == 0;
      const uint32_t pcol = (uint32_t)(pair * 128 + ch * 16);
      const uint32_t pso = r0 * ROWB + ((ch ^ (r0 & 7)) << 4);
      // page indices by TMA (q_len 1 flat only: f16 -2.4%, f32 -1.7%; the q_len 8 flat kernel lost ~0.7%)
      constexpr bool TMAPG_ON = L == 1;
      int pgb = 0;   // fills issued by earlier segments (slot / phase base)
      if constexpr (TMAPG_ON) {
        {
          if (q == 0) {
            for (int st = 0; st < 4; ++st) mbar_init_n(&S.pbar[st], 1);
            mbar_fence_init();
          }
          asm volatile("bar.sync 2, %0;\n" ::"n"(NP) : "memory");
        }
      }
      // the 16-byte aligned window holding the tile's page indices; reads stay inside the table's allocation
      // (CUDA allocations are 256-byte aligned and sized in larger multiples than 16 bytes)
      const long long pt_total = (long long)p.B * p.max_pages;
      const long long pt_lim = (pt_total + 3) & ~3LL;
      auto fetch_pages = [&](int x) {
        if (warp == NT / 32 && x < ntiles) {
          const int slot = (pgb + x) % 4;
          const long long e0 = (long long)b * p.max_pages + ((start + x * TILE) >> p.lp);
          const long long w0 = e0 & ~3LL;
          const long long w1 = min(pt_lim, (e0 + 64 + 3) & ~3LL);
          const int bytes = (int)(w1 - w0) * 4;
          const uint32_t dst = smem_u32(&S.pidx[slot][0]), bar = smem_u32(&S.pbar[slot]);
          const int* src = p.page_table + w0;
          if (elect_one()) {
            mbar_expect_tx(bar, bytes);
            bulk_load(dst, src, bytes, bar);
          }
          __syncwarp();
        }
      };
      if constexpr (TMAPG_ON) {
        for (int x = 0; x < 3; ++x) fetch_pages(x);
      }
      auto load_pages_t = [&](int x, int* dst) {
        const int slot = (pgb + x) % 4;
        mbar_wait_full(&S.pbar[slot], ((pgb + x) / 4) & 1);
        const int fp = (start + x * TILE) >> p.lp;
        const int dl = (int)(((long long)b * p.max_pages + fp) & 3);   // window offset of the first page
#pragma unroll
        for (int k = 0; k < PK; ++k) {
          const int pos = start + x * TILE + r0 + PR * k;
          dst[k] = (pos < cache_end) ? S.pidx[slot][(pos >> p.lp) - fp + dl] : 0;
        }
      };
      auto load_pages = [&](int x, int* dst) {
#pragma unroll
        for (int k = 0; k < PK; ++k) {
          const int pos = start + x * TILE + r0 + PR * k;
          dst[k] = (x < ntiles && pos < cache_end) ? __ldg(ptr + (pos >> p.lp)) : 0;
        }
      };
      if constexpr (!TMAPG_ON) load_pages(0, pg);

      for (int x = 0; x < ntiles; ++x) {
        const int st = (tb + x) % STAGES;
        int pgn[PK];
        if constexpr (TMAPG_ON) {
          fetch_pages(x + 3);
          load_pages_t(x, pg);
        } else {
          load_pages(x + 1, pgn);   // in flight while this tile's copies issue
        }
        if (tb + x >= STAGES) mbar_wait_full(&S.empty[st], (((tb + x) / STAGES) - 1) & 1);
        const uint32_t kb = smem_u32(S.k + st * STAGE_BYTES);
        const uint32_t vb = smem_u32(S.v + st * STAGE_BYTES);
        // common tiles (every row cached, page size 1, caches below 4 GB): one shift-add per row, no per-row
        // branches (the producer's instructions take issue slots from the consumers on the same SMSPs)
        if (pfast && start + x * TILE + TILE <= cache_end) {
#pragma unroll
          for (int k = 0; k < PK; ++k) {
            const uint32_t off = ((uint32_t)pg[k] << LOG_ROW_BYTES) + pcol;
            cp_async16(kb + pso + k * (PR * ROWB), p.k_cache + off, 16);
            cp_async16(vb + pso + k * (PR * ROWB), p.v_cache + off, 16);
          }
        } else
#pragma unroll
          for (int k = 0; k < PK; ++k) {
            const int r = r0 + PR * k;
            const int pos = start + x * TILE + r;
            const uint32_t so = r * ROWB + ((ch ^ (r & 7)) << 4);
            if (pos < cache_end) {
              {   // caches below 4 GB: 32-bit offsets (uniform branch)
                const uint32_t off = ((((uint32_t)pg[k] << p.lp) + (uint32_t)(pos & (p.P - 1))) << LOG_ROW_BYTES) +
                                     (uint32_t)(pair * 128 + ch * 16);
                cp_async16(kb + so, p.k_cache + off, 16);
                cp_async16(vb + so, p.v_cache + off, 16);
              }
            } else if (pos >= end) {
              cp_async16_zero(kb + so, p.k_cache);
              cp_async16_zero(vb + so, p.v_cache);
            }
          }
        cp_async_mbar_arrive(&S.full[st]);
        if constexpr (TMAPG_ON) {
          asm volatile("bar.sync 2, %0;\n" ::"n"(NP) : "memory");   // slot x % PS is free for tile x + PS
        } else {
#pragma unroll
          for (int k = 0; k < PK; ++k) pg[k] = pgn[k];
        }
      }

      {
        cp_wait<0>();
        __syncthreads();   // matches the consumers' end-of-segment barrier
      }
      return true;
    }
  }
  if constexpr (ROLE != 1) {
    const int prow = tid >> 1, phalf = tid & 1;
    const int* pt_row = p.page_table + (ix_t)b * p.max_pages;
    const int part = tid & 7;

    // ---- producer helpers ----
    // K/V: thread -> 16-byte chunk (tid & 7) of rows (tid >> 3) + 16*k: a warp copies 4 whole 128-byte rows.
    // New-token rows come from S.kn / S.vn; rows past the range are zeroed.
    const int cchunk = tid & 7;
    const int crow0 = tid >> 3;
    constexpr int RPP = NT / 8;   // rows per pass
    auto issue_kv = [&](int x, bool with_new) {
      const int st = x % STAGES;
      const int* idxs = &S.idx[(x % IDX_SLOTS) * TILE];
      bool generic = false;   // st.shared into a K stage: the wgmma (async proxy) needs a proxy fence
      // every row of the tile cached (pos < min(end, seq)), page size 1, caches below 4 GB: one shift-add
      // per row, no per-row branches (the per-request issue runs from a cold instruction cache, mostly once)
      if (with_new && p.lp == 0 && ((start + (x * TILE)) + TILE) <= min(end, seq)) {   // (in-loop issues only: the
        // prologue's cold copy of the tile-0 issue stays the plain one, s1 measured +1.5% with it)
        const uint32_t kst = smem_u32(S.k + st * STAGE_BYTES);
        const uint32_t vst = smem_u32(S.v + st * STAGE_BYTES);
        const uint32_t col = (uint32_t)(pair * 128 + cchunk * 16);
#pragma unroll
        for (int k = 0; k < TILE / RPP; ++k) {
          const int r = crow0 + RPP * k;
          const uint32_t so = r * ROWB + ((cchunk ^ (r & 7)) << 4);
          const uint32_t off = ((uint32_t)idxs[r] << LOG_ROW_BYTES) + col;
          cp_async16(kst + so, p.k_cache + off, 16);
          cp_async16(vst + so, p.v_cache + off, 16);
        }
        return;
      }
#pragma unroll
      for (int k = 0; k < TILE / RPP; ++k) {
        const int r = crow0 + RPP * k;
        const int pos = start + x * TILE + r;
        const int sw = ((cchunk ^ (r & 7)) << 4);
        uint8_t* dk = S.k + st * STAGE_BYTES + r * ROWB + sw;
        uint8_t* dv = S.v + st * STAGE_BYTES + r * ROWB + sw;
        if (pos < end && pos < seq) {
          {   // caches below 4 GB: 32-bit offsets (uniform branch)
            const uint32_t off = ((((uint32_t)idxs[r] << p.lp) + (uint32_t)(pos & (p.P - 1))) << LOG_ROW_BYTES) +
                                 (uint32_t)(pair * 128 + cchunk * 16);
            cp_async16(smem_u32(dk), p.k_cache + off, 16);
            cp_async16(smem_u32(dv), p.v_cache + off, 16);
          }
        } else if (pos < end) {
          if (with_new) {
            *reinterpret_cast<uint4*>(dk) = *reinterpret_cast<const uint4*>(S.kn + (pos - seq) * ROWB + cchunk * 16);
            *reinterpret_cast<uint4*>(dv) = *reinterpret_cast<const uint4*>(S.vn + (pos - seq) * ROWB + cchunk * 16);
            generic = true;
          }
        } else {   // past the range: masked out of S; V must be finite (P = 0 times V)
          *reinterpret_cast<uint4*>(dk) = make_uint4(0, 0, 0, 0);
          *reinterpret_cast<uint4*>(dv) = make_uint4(0, 0, 0, 0);
        }
      }
      if (L == 8 && generic) fence_proxy_async();
    };
    auto issue_idx = [&](int y) {
      if (phalf == 0 && prow < TILE && y < ntiles) {
        const int pos = start + y * TILE + prow;
        if (pos < end && pos < seq) cp_async4(smem_u32(&S.idx[(y % IDX_SLOTS) * TILE + prow]), pt_row + (pos >> p.lp));
      }
    };
    // producer mode: wait for stage `st` (tile i) and write the block's own new keys / values into it
    auto ws_patch = [&](int i, int st) {   // the block's own new rows (positions >= seq) into tile i's stage
      const int t0 = start + i * TILE;
      const int lo = max(t0, seq);
      const int hi = min(t0 + TILE, end);
      if (lo < hi) {   // same for every consumer thread
        for (int e = tid; e < (hi - lo) * 8; e += NT) {
          const int pos = lo + (e >> 3), c = e & 7, r = pos - t0;
          const int sw = ((c ^ (r & 7)) << 4);
          *reinterpret_cast<uint4*>(S.k + st * STAGE_BYTES + r * ROWB + sw) =
              *reinterpret_cast<const uint4*>(S.kn + (pos - seq) * ROWB + c * 16);
          if constexpr (!PTR)
            *reinterpret_cast<uint4*>(S.v + st * STAGE_BYTES + r * ROWB + sw) =
                *reinterpret_cast<const uint4*>(S.vn + (pos - seq) * ROWB + c * 16);
        }
        if (L == 8) fence_proxy_async();
      }
      return lo < hi;
    };
    auto ws_acquire = [&](int i, int st) {
      mbar_wait_full(&S.full[st], ((tb + i) / STAGES) & 1);
      const int t0 = start + i * TILE;
      const int lo = max(t0, seq);
      const int hi = min(t0 + TILE, end);
      if (lo < hi) {   // same for every consumer thread
        for (int e = tid; e < (hi - lo) * 8; e += NT) {
          const int pos = lo + (e >> 3), c = e & 7, r = pos - t0;
          const int sw = ((c ^ (r & 7)) << 4);
          *reinterpret_cast<uint4*>(S.k + st * STAGE_BYTES + r * ROWB + sw) =
              *reinterpret_cast<const uint4*>(S.kn + (pos - seq) * ROWB + c * 16);
          if constexpr (!PTR)   // (the producers patch the values before their transpose)
            *reinterpret_cast<uint4*>(S.v + st * STAGE_BYTES + r * ROWB + sw) =
                *reinterpret_cast<const uint4*>(S.vn + (pos - seq) * ROWB + c * 16);
        }
        if (L == 8) fence_proxy_async();
      }
      return lo < hi;
    };

    // ---- 1. page indices of the first STAGES-1 tiles; raw q / k / v rows into registers ----
    int tidL = tid, partL = part;   // (re-read after the page-index copies are issued)
    constexpr int QPASS = L * 16 / RPP;   // L*16 rows, RPP rows per pass
    const bool has_new = end > seq;
    constexpr int KPASS = (L * 4 + RPP - 1) / RPP;   // L*4 rows (k, v of 2 heads), RPP rows per pass
    constexpr bool IDXF = !WS;
    constexpr bool QDIR = ATT_V8_QDIR && L == 8 && !WS;
    {
      if constexpr (!WS) {
#pragma unroll
        for (int y = 0; y < STAGES - 1; ++y) issue_idx(y);
      }
      cp_commit();
      if (!WS &&
          (MODE &
           8)) {   // (speculative builds only: Oct 26 01:33, vs1 -3%, vf1 -1.6%, but s32/vs32 +1.5%) a late copy of tid: the q / new-row / table math below cannot be hoisted above the page-index
        // copies (or out of the speculative retry loop to the kernel top: scout's v200 census, segment head 21 -> 62/97)
        asm volatile("mov.u32 %0, %%tid.x;" : "=r"(tidL));
        partL = tidL & 7;
      }

      pdl_wait();   // qkv (q, new k / v, gate) is the preceding kernel's output; nothing above reads or writes it
      pdl_launch_dependents();
      // raw q rows: L*16 rows x 128 B = 8 chunks of 16 B per row
      tfor<L * 16 * 8, NT>(tidL, [&](int e) {   // (straight-line: no loop branches on the once-path)
        const int rr = e >> 3, ch = e & 7;
        const int j = rr >> 4, hh = rr & 15;
        cp_async16(smem_u32(&S.qst()[rr * 64 + ch * 8]),
                   p.qkv + (ix_t)(b * L + j) * p.ld_qkv + (16 * pair + hh) * 64 + ch * 8, 16);
      });
      if (has_new) {
        tfor<L * 4 * 8, NT>(tidL, [&](int e) {   // (straight-line: no loop branches on the once-path)
          const int rr = e >> 3, ch = e & 7;
          const int kind = rr / (2 * L);
          const int jj = (rr % (2 * L)) >> 1, hh = rr & 1;
          const int col = (kind == 0 ? HQ : HQ + HKV) * 64 + (2 * pair + hh) * 64 + ch * 8;
          cp_async16(smem_u32(&S.kvst[rr * 64 + ch * 8]), p.qkv + (ix_t)(b * L + jj) * p.ld_qkv + col, 16);
        });
      }

      // small per-call constants: staged with the same group instead of cold loads in the middle of the prologue
      if (p.rope && tidL < 8)
        cp_async16(smem_u32(&S.ifreq[tidL * 4]), p.inv_freq + tidL * 4, 16);
      else if (tidL == 8)
        cp_async4(smem_u32(&S.scl[0]), p.k_scale);
      else if (tidL == 9)
        cp_async4(smem_u32(&S.scl[1]), p.v_scale);
      else if (!FLAT && M_COOP && tidL == 10
#if ATT_V1_CTA_BARRIER
               && !(L == 8 && (MODE & 128) != 0)
#endif
      )   // mutable epoch: bypass L1 even under --def-load-cache=ca
        stage_epoch(S, p.ws_cnt + REQUEST_EPOCH_BASE + 4 * bp);
      else if (tidL >= 16 && tidL < 16 + L) {
        const int jj = tidL - 16;
        asm volatile("cp.async.ca.shared.global [%0], [%1], 8;\n" ::"r"(smem_u32(&S.cloc[jj])),
                     "l"(p.cache_loc + b * L + jj));
      }
      cp_commit();
      if (split == 0) {   // warm L2 with the gate rows for whichever CTA finishes last
        tfor<L * 16, NT>(tidL, [&](int e) {   // (straight-line: no loop branches on the once-path)
          const __nv_bfloat16* gp =
              p.qkv + (ix_t)(b * L + (e >> 4)) * p.ld_qkv + (HQ + 2 * HKV) * 64 + (16 * pair + (e & 15)) * 64;
          asm volatile("prefetch.global.L2 [%0];" ::"l"(gp));
        });
      }

      // ---- 2. K/V of the first STAGES-1 tiles in flight as soon as their page indices land ----
      // wait for the page-index group only (committed first, 256 B); the q / new
      // k,v / constants group (up to 16 KB) may still be in flight while the K/V copies go out
      cp_wait<IDXF ? 1 : 0>();
      csync<WS, NT>();
      if constexpr (!FLAT && (MODE & 8) != 0) {
        // this CTA's ranges and page indices came from the speculated length (see the kernel); the real
        // cache_seqlens value has landed by now (it was loaded alongside the indices). On a miss, nothing global has been
        // written: drain the copies and let the caller redo the segment with the real length (today's path)
        if (seq_real_ != seq) {
          cp_wait<0>();
          __syncthreads();
          return false;
        }
      }
      if constexpr (!WS) {
        constexpr int PRE = STAGES - 1;
        bool one = false;
        // a CTA with a single fully cached tile (page size 1, caches < 4 GB): its copies straight-line (one
        // shift-add per row), none of the loop / per-row tests / issue_idx past the last tile, then the commits the
        // pipeline's group count expects (scout's census: ~168 instr, 11 taken branches here on s1, cold)
        if (ONE && p.lp == 0) {
          one = true;
          const uint32_t kst = smem_u32(S.k);
          const uint32_t vst = smem_u32(S.v);
          const uint32_t col = (uint32_t)(pair * 128 + cchunk * 16);
          const int cend = min(end, seq);
#pragma unroll
          for (int k = 0; k < TILE / RPP; ++k) {
            const int r = crow0 + RPP * k;
            const uint32_t so = r * ROWB + ((cchunk ^ (r & 7)) << 4);
            {
              // branch-free (scout's v200 census: the per-row branches cost 16 instr + 4 taken branches per warp, cold):
              // cached rows copy, rows past the range zero-fill (masked out of S; V must be finite), the block's new
              // rows are skipped (written by the patch pass after the new-k/v pass)
              const int pr = start + r;
              const bool cached = pr < cend;
              const bool doit = cached || pr >= end;
              const uint32_t off = ((uint32_t)S.idx[r] << LOG_ROW_BYTES) + col;   // (zero-filled index past the cached rows)
              cp_async16_pred(kst + so, p.k_cache + off, cached ? 16 : 0, doit);
              cp_async16_pred(vst + so, p.v_cache + off, cached ? 16 : 0, doit);
            }   // (the block's new rows: written by the patch pass after the new-k/v pass, as with the general issue)
          }
#pragma unroll
          for (int k = 0; k < STAGES - 1; ++k) cp_commit();
        }
        if (!one) {
#pragma unroll
          for (int k = 0; k < STAGES - 1; ++k) {   // (unrolled: no loop branches; rolled: one copy of the issue code)
            if (k < PRE && k < ntiles) issue_kv(k, false);
            issue_idx(k + STAGES - 1);
            if (k < (PRE - 1) || k == (STAGES - 2)) cp_commit();
          }
        }
      }
    }

    if constexpr (IDXF) {   // the q group (all but the STAGES-1 K/V groups just committed), visible to every thread
      cp_wait<STAGES - 1>();
      csync<WS, NT>();
    }
    // ---- 3. while they load: RoPE table, queries, the block's new keys / values ----
    if (p.rope) {
      tfor<L * 32, NT>(tidL, [&](int e) {   // (straight-line: no loop branches on the once-path)
        const int j = e >> 5, i = e & 31;
        const float ang = (float)(seq + j) * S.ifreq[i];   // fp32 angle, as the reference builds it
        float sv, cv;
        // Cody-Waite reduction mod 2 pi with a 3-part constant (6.25 has 5 significant bits: exact products for
        // |k| < 2^19, angles up to ~3e6), then the hardware sin / cos on [-pi, pi]: ~5e-7 absolute, against
        // ~30 cold instructions (and a Payne-Hanek call above |x| ~ 1e5) for the accurate sincosf
        const float kq = rintf(ang * 0.15915493667125702f);
        float rr = fmaf(-kq, 6.25f, ang);
        rr = fmaf(-kq, 0.03318530693650246f, rr);
        rr = fmaf(-kq, 2.430837753308879e-10f, rr);
        __sincosf(rr, &sv, &cv);
        S.cs[e] = cv;
        S.sn[e] = sv;
      });
    }
    csync<WS, NT>();

    const float kscale = S.scl[0], vscale = S.scl[1];
    const float sl2 = p.scale_log2 * kscale;

    if constexpr (QDIR) {
      // Mechanism from keell0renz/muse-open-kernel-challenge @ 890809cee421500b00126d084e81b338ef9181ab
      // (kernels/muse_attention_decode.py, QDIR; Apache-2.0): four threads per query row, 16 dims each, one
      // fp32 -> e4m3 rounding straight into the SS QK^T A tile (64-byte rows, 16-byte chunks swizzled by
      // (row >> 1) & 3); no f16 staging and no re-read before the loop
      constexpr int RP4 = NT / 4;   // rows per pass
#pragma unroll
      for (int ps = 0; ps < L * 16 / RP4; ++ps) {
        const int q4 = tidL & 3;
        const int rr = ps * RP4 + (tidL >> 2);
        const int j = rr >> 4, hh = rr & 15;
        const uint4 qa = *reinterpret_cast<const uint4*>(&S.qst()[rr * 64 + q4 * 16]);
        const uint4 qb = *reinterpret_cast<const uint4*>(&S.qst()[rr * 64 + q4 * 16 + 8]);
        const __nv_bfloat162* ha = reinterpret_cast<const __nv_bfloat162*>(&qa);
        const __nv_bfloat162* hb = reinterpret_cast<const __nv_bfloat162*>(&qb);
        float x[16];
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const float2 fa = __bfloat1622float2(ha[e]);
          const float2 fb = __bfloat1622float2(hb[e]);
          x[2 * e] = fa.x;
          x[2 * e + 1] = fa.y;
          x[8 + 2 * e] = fb.x;
          x[8 + 2 * e + 1] = fb.y;
        }
        float ss = 0.f;
#pragma unroll
        for (int e = 0; e < 16; ++e) ss += x[e] * x[e];
        ss += __shfl_xor_sync(0xffffffffu, ss, 1);
        ss += __shfl_xor_sync(0xffffffffu, ss, 2);
        const float r = rsqrtf(ss * (1.f / 64.f) + 1e-5f);
#pragma unroll
        for (int e = 0; e < 16; ++e) x[e] *= r;
        if (p.rope) {
#pragma unroll
          for (int e = 0; e < 8; ++e) {
            const int i = q4 * 8 + e;
            const float c = S.cs[j * 32 + i], sn = S.sn[j * 32 + i];
            const float a0 = x[2 * e], a1 = x[2 * e + 1];
            x[2 * e] = a0 * c - a1 * sn;
            x[2 * e + 1] = a0 * sn + a1 * c;
          }
        }
        const int r4 = j * 8 + (hh & 7), kb = q4 * 16;
        *reinterpret_cast<uint4*>(S.q8s + (hh >> 3) * 4096 + r4 * 64 + (((kb >> 4) ^ ((r4 >> 1) & 3)) << 4)) =
            make_uint4(e4m3x4f(x[0], x[1], x[2], x[3]), e4m3x4f(x[4], x[5], x[6], x[7]),
                       e4m3x4f(x[8], x[9], x[10], x[11]), e4m3x4f(x[12], x[13], x[14], x[15]));
      }
      fence_proxy_async();   // the e4m3 q tile (st.shared) is read by the wgmma after the loop's first barrier
    } else
#pragma unroll
    for (int ps = 0; ps < QPASS; ++ps) {
      const int rr = ps * RPP + (tidL >> 3);
      const int j = rr >> 4, hh = rr & 15;
      const uint4 qv = *reinterpret_cast<const uint4*>(&S.qst()[rr * 64 + partL * 8]);
      const __nv_bfloat162* h2 = reinterpret_cast<const __nv_bfloat162*>(&qv);
      float x[8];
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const float2 f = __bfloat1622float2(h2[e]);
        x[2 * e] = f.x;
        x[2 * e + 1] = f.y;
      }
      float ss = 0.f;
#pragma unroll
      for (int e = 0; e < 8; ++e) ss += x[e] * x[e];
      ss = oct_sum(ss);
      const float r = rsqrtf(ss * (1.f / 64.f) + 1e-5f);
#pragma unroll
      for (int e = 0; e < 8; ++e) x[e] *= r;
      if (p.rope) {
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const int i = partL * 4 + e;
          const float c = S.cs[j * 32 + i], sn = S.sn[j * 32 + i];
          const float a0 = x[2 * e], a1 = x[2 * e + 1];
          x[2 * e] = a0 * c - a1 * sn;
          x[2 * e + 1] = a0 * sn + a1 * c;
        }
      }
      // f16 query pre-scaled to log2 units, staged in the (still free) last pipeline stage
      uint4 qw;
      uint32_t* qwp = reinterpret_cast<uint32_t*>(&qw);
      // (q_len 8 on sm_90 rounds the query to e4m3: keep it unit-RMS like SGLang does -- pre-scaling it by the
      // kv scale would push it into e4m3's subnormals -- and scale the scores instead)
      const float qsc = (L == 8) ? 1.f : sl2;
      const int hk = hh >> 3, gg = hh & 7;

#pragma unroll
      for (int e = 0; e < 4; ++e) qwp[e] = pack_h2(x[2 * e] * qsc, x[2 * e + 1] * qsc);
      if constexpr (L == 1) {
        const uint32_t t1 = qwp[1];
        qwp[1] = qwp[2];
        qwp[2] = t1;
      }   // (scout: quad order 0,2,1,3)
      if constexpr (WS) {   // in place over the raw row (same thread, same 16 bytes)
        *reinterpret_cast<uint4*>(S.kvq + rr * 128 + partL * 16) = qw;
      } else {
        uint8_t* qs = (hk ? S.v : S.k) + (STAGES - 1) * STAGE_BYTES;
        *reinterpret_cast<uint4*>(qs + (j * 8 + gg) * 128 + partL * 16) = qw;
      }
    }
    // the e4m3 q tile (st.shared) is read by the wgmma after the loop's first barrier

    if (has_new) {
#pragma unroll
      for (int ps = 0; ps < KPASS; ++ps) {
        const int rr = ps * RPP + (tidL >> 3);
        const bool live = rr < L * 4;
        const int kind = rr / (2 * L);
        const int jj = (rr % (2 * L)) >> 1, hh = rr & 1;
        float x[8];
        const uint4 kvv = live ? *reinterpret_cast<const uint4*>(&S.kvst[rr * 64 + partL * 8]) : make_uint4(0, 0, 0, 0);
        const __nv_bfloat162* h2 = reinterpret_cast<const __nv_bfloat162*>(&kvv);
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const float2 f = __bfloat1622float2(h2[e]);
          x[2 * e] = f.x;
          x[2 * e + 1] = f.y;
        }
        float ss = 0.f;
#pragma unroll
        for (int e = 0; e < 8; ++e) ss += x[e] * x[e];
        ss = oct_sum(ss);   // every lane of the warp (rows of both kinds share warps)
        float sc;
        if (kind == 0) {
          const float r = rsqrtf(ss * (1.f / 64.f) + 1e-5f);
#pragma unroll
          for (int e = 0; e < 8; ++e) x[e] *= r;
          if (p.rope && live) {
#pragma unroll
            for (int e = 0; e < 4; ++e) {
              const int i = partL * 4 + e;
              const float c = S.cs[jj * 32 + i], sn = S.sn[jj * 32 + i];
              const float a0 = x[2 * e], a1 = x[2 * e + 1];
              x[2 * e] = a0 * c - a1 * sn;
              x[2 * e + 1] = a0 * sn + a1 * c;
            }
          }
          sc = __fdividef(1.f, kscale);   // (MUFU.RCP: no IEEE slow-path call and its branches on the once-path;
        } else {   //  exact for the power-of-two scales)
          sc = __fdividef(1.f, vscale);
        }
        if (live) {
          uint32_t w0 = 0, w1 = 0;
#pragma unroll
          for (int e = 0; e < 4; ++e) {
            w0 |= (uint32_t)to_e4m3(x[e] * sc) << (8 * e);
            w1 |= (uint32_t)to_e4m3(x[4 + e] * sc) << (8 * e);
          }
          uint8_t* dst = (kind == 0 ? S.kn : S.vn) + jj * ROWB + hh * 64 + partL * 8;
          *reinterpret_cast<uint2*>(dst) = make_uint2(w0, w1);
          const int pos = seq + jj;
          if (pos >= start && pos < end) {
            const int64_t row = S.cloc[jj];
            uint8_t* g8 = const_cast<uint8_t*>(
                cache_row(kind == 0 ? p.k_cache : p.v_cache, p, row, (2 * pair + hh) * 64 + partL * 8));
            *reinterpret_cast<uint2*>(g8) = make_uint2(w0, w1);
          }
        }
      }
    }
    csync<WS, NT>();
    if (!WS && has_new) {
      // new-token rows of the tiles already issued (their cached rows are in flight with cp.async)
      for (int x = 0; x < (STAGES - 1) && x < ntiles; ++x) {
        const int st = x % STAGES;
#pragma unroll
        for (int k = 0; k < TILE / RPP; ++k) {
          const int r = crow0 + RPP * k;
          const int pos = start + x * TILE + r;
          if (pos >= seq && pos < end) {
            const int sw = ((cchunk ^ (r & 7)) << 4);
            *reinterpret_cast<uint4*>(S.k + st * STAGE_BYTES + r * ROWB + sw) =
                *reinterpret_cast<const uint4*>(S.kn + (pos - seq) * ROWB + cchunk * 16);
            *reinterpret_cast<uint4*>(S.v + st * STAGE_BYTES + r * ROWB + sw) =
                *reinterpret_cast<const uint4*>(S.vn + (pos - seq) * ROWB + cchunk * 16);
          }
        }
      }
      if (L == 8) fence_proxy_async();   // new keys written with st.shared, read by the QK wgmma
    }

    constexpr int NROW = (L == 1) ? 1 : 2;   // rows per thread
    // (lane-0 broadcasts, so ptxas keeps them in uniform registers instead of re-deriving them every tile)
    const int hk = __shfl_sync(0xffffffffu, L == 1 ? (warp & 1) : ((warp & 7) >> 2), 0);   // L==8: warpgroup = KV head
    const int wsub = __shfl_sync(0xffffffffu, L == 1 ? (warp >> 1) : (warp & 3), 0);   // L==1: key half; L==8: m-tile
    const int wgi = hk;   // named barrier of this warpgroup
    float o[1][8][4];
#pragma unroll
    for (int mt = 0; mt < 1; ++mt)
#pragma unroll
      for (int t = 0; t < 8; ++t)
#pragma unroll
        for (int e = 0; e < 4; ++e) o[mt][t][e] = 0.f;
    float m_r[NROW], l_r[NROW];
#pragma unroll
    for (int r = 0; r < NROW; ++r) {
      m_r[r] = -INFINITY;
      l_r[r] = 0.f;
    }
    // tiles with no window / causal / range edge skip the per-element mask
    const int lo_all = (L == 1 || !p.window) ? start : max(start, seq + (L - 1) - p.window + 1);
    const int hi_all = __shfl_sync(0xffffffffu, min(end - 1, seq), 0);

    if constexpr (L == 1) {
      // Q A-fragments (f16, pre-scaled to log2 units) for 4 k-steps of 16 dims: thread holds dims
      // 16*tig + 4*ks + {0,1} (a0) and + {2,3} (a2); the K B-fragments use the same dims.
      uint32_t qa[4][2];
      {
        const uint8_t* qs = WS ? S.kvq + hk * 8 * 128 : (hk ? S.v : S.k) + (STAGES - 1) * STAGE_BYTES;
        const uint4 l0 = *reinterpret_cast<const uint4*>(qs + g * 128 + 32 * tig);
        const uint4 l1 = *reinterpret_cast<const uint4*>(qs + g * 128 + 32 * tig + 16);
        // (scout: chunk words stored (0, 2, 1, 3): l0 = {ks0 a0, ks1 a0, ks0 a2, ks1 a2}, l1 the same for ks 2, 3)
        qa[0][0] = l0.x;
        qa[0][1] = l0.y;
        qa[1][0] = l0.z;
        qa[1][1] = l0.w;
        qa[2][0] = l1.x;
        qa[2][1] = l1.y;
        qa[3][0] = l1.z;
        qa[3][1] = l1.w;
      }

      for (int i = 0; i < ntiles; ++i) {
        if constexpr (WS) {
          if (ws_acquire(i, (tb + i) % STAGES)) csync<WS, NT>();
        } else {
          cp_wait<STAGES - 2>();
          __syncthreads();
          if (((i + STAGES) - 1) < ntiles) issue_kv(i + STAGES - 1, true);
          issue_idx(i + 2 * STAGES - 2);
          cp_commit();
        }

        const uint8_t* K = S.k + ((tb + i) % STAGES) * STAGE_BYTES;
        const uint8_t* V = S.v + ((tb + i) % STAGES) * STAGE_BYTES;
        const int t0 = start + i * TILE;
        const bool interior = (t0 >= lo_all) && (t0 + TILE - 1 <= hi_all);
        // warp: kv head hk, keys [32*wsub, 32*wsub+32) of the tile
        float s[4][4];
        uint32_t kw[4][4];
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
          const int key = 32 * wsub + 8 * nt + g;
          const uint4 kb = lds128(K + key * ROWB + (((hk * 4 + tig) ^ (key & 7)) << 4));
          kw[nt][0] = kb.x;
          kw[nt][1] = kb.y;
          kw[nt][2] = kb.z;
          kw[nt][3] = kb.w;
        }
#pragma unroll
        for (int nt = 0; nt < 4;
             ++nt) {   // (scout: unpadded quads; even k-steps -> rows g of se, odd -> rows g+8 of so)
          float se[4] = {0.f, 0.f, 0.f, 0.f}, so[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
          for (int h = 0; h < 2; ++h) {
            mma_f16(se, qa[2 * h][0], qa[2 * h][1], qa[2 * h + 1][0], qa[2 * h + 1][1], cvt2(kw[nt][2 * h]),
                    cvt2h(kw[nt][2 * h]));
            mma_f16(so, qa[2 * h][0], qa[2 * h][1], qa[2 * h + 1][0], qa[2 * h + 1][1], cvt2(kw[nt][2 * h + 1]),
                    cvt2h(kw[nt][2 * h + 1]));
          }
          s[nt][0] = se[0] + so[2];
          s[nt][1] = se[1] + so[3];
          s[nt][2] = 0.f;
          s[nt][3] = 0.f;
        }
        if (!interior) {
#pragma unroll
          for (int nt = 0; nt < 4; ++nt)
#pragma unroll
            for (int e = 0; e < 2; ++e) {
              const int pos = t0 + 32 * wsub + 8 * nt + 2 * tig + e;
              if (pos >= end) s[nt][e] = -INFINITY;
            }
        }
        float mx = -INFINITY;
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) mx = fmaxf(mx, fmaxf(s[nt][0], s[nt][1]));
        mx = quad_max(mx);
        const float m_new = fmaxf(m_r[0], mx);
        if (m_new != -INFINITY) {
          const float alpha = ex2(m_r[0] - m_new);
          m_r[0] = m_new;
          float ls = 0.f;
#pragma unroll
          for (int nt = 0; nt < 4; ++nt)
#pragma unroll
            for (int e = 0; e < 2; ++e) {
              const float pv = ex2(s[nt][e] - m_new);
              s[nt][e] = pv;
              ls += pv;
            }
          if constexpr (ONE) {   // (scout: the only tile: o = 0 and l = 0, alpha would only multiply zeros)
            l_r[0] = ls;
          } else {
            l_r[0] = l_r[0] * alpha + ls;
#pragma unroll
            for (int t = 0; t < 8; ++t) {
              o[0][t][0] *= alpha;
              o[0][t][1] *= alpha;
            }
          }
#pragma unroll
          for (int ks = 0; ks < (2); ++ks) {
            uint32_t b0[8], b1[8];
            load_v_frags(V, 32 * wsub + 16 * ks, hk, g, tig, b0, b1);
            const uint32_t a0 = pack_h2(s[2 * ks][0], s[2 * ks][1]);
            const uint32_t a2 = pack_h2(s[2 * ks + 1][0], s[2 * ks + 1][1]);
#pragma unroll
            for (int t = 0; t < 8; ++t) mma_f16(o[0][t], a0, 0u, a2, 0u, b0[t], b1[t]);
          }
        }
        if constexpr (WS) {   // this warp is done with the stage
          __syncwarp();
          mbar_arrive(&S.empty[(tb + i) % STAGES]);
        }
      }
    } else {
      // ---- L == 8 on sm_90: QK^T with FP8 wgmma straight from the FP8 K tile in shared memory (warpgroup =
      // KV head, 64 rows = 8 tokens x 8 heads, K-major 128B-swizzled B operand), PV with f16 mma.sync from a
      // per-tile f16 copy of V. Softmax with a lazily updated per-row offset; row sums from a ones column.
      uint8_t* V16 = S.kvq;
      const uint32_t v16_base = smem_u32(V16) + hk * 8192;
      if constexpr (!QDIR) {   // (QDIR: the prologue wrote the e4m3 A tile)
        uint32_t qa8[2][4];   // e4m3 A fragments, k-steps of 32 dims
        const uint8_t* qs = (hk ? S.v : S.k) + (STAGES - 1) * STAGE_BYTES;   // f16 staging, row = j*8 + g
        const int r0 = 16 * wsub + g;
        const int r1 = r0 + 8;
        // producer mode: in place in S.kvq, row j*16 + hk*8 + g
        const uint8_t* q0 = WS ? S.kvq + ((r0 >> 3) * 16 + hk * 8 + (r0 & 7)) * 128 : qs + r0 * 128;
        const uint8_t* q1 = WS ? S.kvq + ((r1 >> 3) * 16 + hk * 8 + (r1 & 7)) * 128 : qs + r1 * 128;
#pragma unroll
        for (int ks = 0; ks < 2; ++ks) {
#pragma unroll
          for (int hf = 0; hf < 2; ++hf) {   // dims 32*ks + 16*hf + 4*tig .. +3
            const int dofs = (32 * ks + 16 * hf + 4 * tig) * 2;
            const uint2 lo = *reinterpret_cast<const uint2*>(q0 + dofs);
            const uint2 hi = *reinterpret_cast<const uint2*>(q1 + dofs);
            qa8[ks][2 * hf] = pack_e4m3x4(lo.x, lo.y);
            qa8[ks][2 * hf + 1] = pack_e4m3x4(hi.x, hi.y);
          }
        }
        {   // the A tile for the SS QK^T wgmma
#pragma unroll
          for (int ks = 0; ks < 2; ++ks)
#pragma unroll
            for (int hf = 0; hf < 2; ++hf) {
              const int kb = 32 * ks + 16 * hf + 4 * tig;
              *reinterpret_cast<uint32_t*>(S.q8s + hk * 4096 + r0 * 64 + (((kb >> 4) ^ ((r0 >> 1) & 3)) << 4) +
                                           (kb & 15)) = qa8[ks][2 * hf];
              *reinterpret_cast<uint32_t*>(S.q8s + hk * 4096 + r1 * 64 + (((kb >> 4) ^ ((r1 >> 1) & 3)) << 4) +
                                           (kb & 15)) = qa8[ks][2 * hf + 1];
            }
          fence_proxy_async();   // read by the wgmma after the loop's first barrier
        }
        if constexpr (PTR) {   // q staging read, S.kn / S.vn staged: the producers may write V^T into S.kvq
          __syncwarp();
          if (lane == 0) mbar_arrive(&S.qdone);
        }
      }
      float cr[2] = {0.f, 0.f};   // per-row offsets (log2 units), rows g and g+8
      const float ssc = sl2;   // raw score -> log2 units
      float lacc[4] = {0.f, 0.f, 0.f, 0.f};
      bool first = true;
      {   // the zero init stays before the loop: ptxas had sunk these moves into the first QK^T
        float* oz = &o[0][0][0];   // wgmma's pipeline stage (warning C7515: every wgmma of the kernel serialized)
#pragma unroll
        for (int e = 0; e < 32; ++e) reg_fence(oz[e]);
#pragma unroll
        for (int e = 0; e < 4; ++e) reg_fence(lacc[e]);
      }
      // transpose roles (warpgroup = head): warp wq takes keys 16wq..16wq+15, lane = (q, w) of the ldmatrix fragments
      const int vwq = warp & 3;
      const int vq = lane & 3;
      const int vw = lane >> 2;   // (warp: uniform, uwarp)
      const uint32_t vsel0 = (vw >= 4) ? 0x7531u : 0x6420u, vsel1 = (vw >= 4) ? 0x6420u : 0x7531u;
      const int vr0 = (vw >= 4) ? 1 : 0;   // rows d + vr0 then d + 1 - vr0 (bank spread)
      // hoisted addressing (registers are tight in the flat kernel): V^T row n = 16*dgrp + 2*vw + r has swizzle chunk
      // vwq ^ (vw & 3) for every dgrp, so the 8 stores per tile are one per-thread base plus immediates; the two
      // ldmatrix rows are key = 16*vwq + 8*(mat & 1) + (lane & 7) with chunk (4*hk + 2u + (mat >> 1)) ^ (lane & 7)
      const uint32_t vt_off0 = 128 * vw + ((vwq ^ (vw & 3)) << 4) + 4 * vq + 64 * vr0;
      const uint32_t vt_off1 = 128 * vw + ((vwq ^ (vw & 3)) << 4) + 4 * vq + 64 * (1 - vr0);
      const int vkey_t = 16 * vwq + 8 * ((lane >> 3) & 1) + (lane & 7);
      const uint32_t vld_off0 = vkey_t * ROWB + (((4 * hk + 0 + ((lane >> 3) >> 1)) ^ (lane & 7)) << 4);
      // the V^T stores as two stmatrix.x4 per thread (instead of eight 4-byte stores). For ldmatrix group u,
      // matrix m = 2*g2 + ab holds, in row j, the 16 key bytes of V^T row 16(2u+g2) + 2j + (ab ^ (j >= 4)) (the same
      // rows and the same half swap for bank spread as the per-thread stores: thread (q, w) supplies row j = w)
      const int stj = lane & 7;
      const int stm = lane >> 3;
      const uint32_t vst_off = 1024 * (stm >> 1) + 128 * stj + 64 * ((stm & 1) ^ (stj >> 2)) + ((vwq ^ (stj & 3)) << 4);
      constexpr bool STSM_ON = !FLAT;
      // P fragments of the in-flight PV wgmma: kept live (fenced) until the next wgmma wait, so the compiler cannot
      // hand their registers to anything the next tile writes before that
      uint32_t pa[2][4] = {{0u, 0u, 0u, 0u}, {0u, 0u, 0u, 0u}};
      constexpr bool PVSPLIT = !PTR;
      // per-request (non-producer) q8 kernels with the split PV and the consumers' own V^T
      constexpr bool LSYNCN = !WS && !FLAT && !PTR && PVSPLIT;
      int st_c = tb % STAGES;
      uint32_t ph_c = (uint32_t)(tb / STAGES) & 1u;
      // descriptor low words broadcast from lane 0: a shuffle result is uniform and is not rematerialized, so the
      // bases stay in uniform registers (the high words are constants); +st*512 / +2 never carry out of the low word
      const uint32_t qlo0 = __shfl_sync(0xffffffffu, (uint32_t)((smem_u32(S.q8s) + hk * 4096) >> 4) | 0x10000u, 0);
      const uint32_t klo0 = __shfl_sync(0xffffffffu, (uint32_t)((smem_u32(S.k) + hk * 64) >> 4) | 0x10000u, 0);
      const uint64_t qdesc0 = ((uint64_t)((512 >> 4) | (2u << 30)) << 32) | qlo0;
      const uint32_t vsrc0 = smem_u32(S.v);
      uint8_t* vtb_c = S.kvq + hk * 4608;
      const uintptr_t vtb_x = (uintptr_t)(S.kvq + hk * 4608) ^ (uintptr_t)(S.kvq + (2 + hk) * 4608);
      uint64_t vdesc_c = desc_sw64(smem_u32(S.kvq + hk * 4608));
      const uint64_t vdesc_x = vdesc_c ^ desc_sw64(smem_u32(S.kvq + (2 + hk) * 4608));
      const int i_new = seq > start ? (seq - start) / TILE : 0;   // first tile that can hold the block's new rows
      // the common tiles -- no mask, no new rows (every key before the block's first new
      // position seq) -- are i in [r_lo, r_lo + r_span], found once; the per-tile test is then one compare, and t0, the
      // exact mask test and the new-row patch run only on the rare tiles
      uint32_t r_lo = 0x7fffffffu, r_span = 0u;
      if constexpr (WS) {
        const int hi2 = min(end - 1, seq - 1);
        const int a = (lo_all - start + TILE - 1) / TILE;
        const int z = hi2 - (TILE - 1) - start;
        const int bz = z >= 0 ? z / TILE : -1;
        r_lo = __shfl_sync(0xffffffffu, bz >= a ? (uint32_t)a : 0x7fffffffu, 0);
        r_span = __shfl_sync(0xffffffffu, bz >= a ? (uint32_t)(bz - a) : 0u, 0);
      }
      // the two KV heads' warpgroups take turns in the exponential phase (named barriers
      // 5 / 6), so the 4 consumer warps of an SMSP do not all queue on the MUFU unit at once (scout: the exps segment
      // takes ~650-820 cycles for ~110 instructions; both warpgroups start each tile together on the same stage)
      constexpr bool TURN = (MODE & 64) != 0 && WS;
      if constexpr (TURN) {
        if (hk == 1 && ntiles > 0) bar_arrive_n(5, 256);   // head 0 goes first
      }
      constexpr int UNR8 = WS ? ATT_FLAT_VERIFY_UNROLL : 1;   // (pragma arguments are not macro-expanded;
      // plain split-KV with many tiles per CTA (b32 q8 sliding: 9) gains from the unroll, 1-tile coop grids lose)
      if constexpr (OVL) {
        // Per tile x (FlashAttention-3's intra-warpgroup order): QK^T(x) and PV(x-1) issued back to back, V^T(x) built
        // under them, wait<1> retires QK^T(x), softmax(x) runs while PV(x-1) is on the tensor cores, wait<0>, then a
        // pending offset move rescales o. Nothing is in flight across the loop edge. P alternates between two register
        // sets (unrolled by 2): PV(x-1) still reads one while softmax(x) writes the other.
        float s[32];
        uint32_t pA[2][4], pB[2][4];
        int stc = st_c;   // stage of tile x
        float alo = 1.f, ahi = 1.f;
        auto qk = [&]() {
          const uint32_t kl = klo0 + (uint32_t)(stc * (STAGE_BYTES >> 4));
          const uint64_t kd = ((uint64_t)((1024 >> 4) | (1u << 30)) << 32) | kl;
          const uint64_t kd2 = ((uint64_t)((1024 >> 4) | (1u << 30)) << 32) | (kl + 2u);
          const uint64_t qd2 = ((uint64_t)((512 >> 4) | (2u << 30)) << 32) | (qlo0 + 2u);
          wgmma_fence();
          wgmma_m64n64k32_e4m3_ss(s, qdesc0, kd, 0);
          wgmma_m64n64k32_e4m3_ss(s, qd2, kd2, 1);
          wgmma_commit();
        };
        auto transpose = [&]() {   // this head's V tile -> V^T (FP8, 72 x 64, keys in P-register order)
          const uint32_t vsrc = vsrc0 + stc * STAGE_BYTES;
#pragma unroll
          for (int u = 0; u < 2; ++u) {
            uint32_t r0, r1, r2, r3;
            ldsm_x4_t(vsrc + (u ? (vld_off0 ^ 32u) : vld_off0), r0, r1, r2, r3);
            stsm_x4(smem_u32(vtb_c) + vst_off + 2048 * u, prmt(r0, r1, vsel0), prmt(r0, r1, vsel1), prmt(r2, r3, vsel0),
                    prmt(r2, r3, vsel1));
          }
          fence_proxy_async();   // read by the PV wgmma after the next CTA barrier
        };
        auto pv = [&](uint32_t (&pp)[2][4], const uint64_t vd) {
          float* od = &o[0][0][0];
#pragma unroll
          for (int kk = 0; kk < 2; ++kk) {
#pragma unroll
            for (int e = 0; e < 4; ++e) asm volatile("" : "+r"(pp[kk][e]));
            wgmma_fence();
            wgmma_m64n72k32_e4m3(od, lacc, pp[kk], vd + 2 * kk);
            wgmma_commit();
          }
        };
        // softmax(x) of s into pc; an offset move only records its factors (o is applied after PV(x-1) retires)
        auto soft = [&](const int x, uint32_t (&pc)[2][4]) {
#pragma unroll
          for (int e = 0; e < 32; ++e) reg_fence(s[e]);
          const int t0 = start + x * TILE;
          if (!(t0 >= lo_all && t0 + TILE - 1 <= hi_all)) {
#pragma unroll
            for (int hf = 0; hf < 2; ++hf) {
              const int pj = seq + 2 * wsub + hf;
              const int vlo = p.window ? max(start, pj - p.window + 1) : start;
              const int vhi = min(end - 1, pj);
#pragma unroll
              for (int nt = 0; nt < 8; ++nt)
#pragma unroll
                for (int e = 0; e < 2; ++e) {
                  const int pos = t0 + 8 * nt + 2 * tig + e;
                  if (pos < vlo || pos > vhi) s[4 * nt + 2 * hf + e] = -INFINITY;
                }
            }
          }
          float mlo = -INFINITY, mhi = -INFINITY;
#pragma unroll
          for (int nt = 0; nt < 8; ++nt) {
            mlo = fmaxf(mlo, fmaxf(s[4 * nt], s[4 * nt + 1]));
            mhi = fmaxf(mhi, fmaxf(s[4 * nt + 2], s[4 * nt + 3]));
          }
          mlo *= ssc;
          mhi *= ssc;
          alo = 1.f;
          ahi = 1.f;
          if (first) {
            mlo = quad_max(mlo);
            mhi = quad_max(mhi);
            cr[0] = (mlo == -INFINITY) ? 0.f : mlo;
            cr[1] = (mhi == -INFINITY) ? 0.f : mhi;
            first = false;
          } else if (__any_sync(0xffffffffu, mlo > cr[0] + 8.f || mhi > cr[1] + 8.f)) {
            // rare: a score grew past 2^8 relative to the offset -> move the offset up
            mlo = quad_max(mlo);
            mhi = quad_max(mhi);
            const float nlo = fmaxf(mlo, cr[0]), nhi = fmaxf(mhi, cr[1]);
            alo = ex2(cr[0] - nlo);
            ahi = ex2(cr[1] - nhi);
            cr[0] = nlo;
            cr[1] = nhi;
          }
          const float dlo = cr[0], dhi = cr[1];
#pragma unroll
          for (int kk = 0; kk < 2; ++kk) {
            const float* sv = s + 16 * kk;
            if constexpr (SCH) {
              const float a8 = 8.f * ssc, clo = 12582968.f - 8.f * dlo, chi = 12582968.f - 8.f * dhi;
              pc[kk][0] = sch_e4m3x4(sv[0], sv[1], sv[4], sv[5], a8, clo);
              pc[kk][1] = sch_e4m3x4(sv[2], sv[3], sv[6], sv[7], a8, chi);
              pc[kk][2] = sch_e4m3x4(sv[8], sv[9], sv[12], sv[13], a8, clo);
              pc[kk][3] = sch_e4m3x4(sv[10], sv[11], sv[14], sv[15], a8, chi);
            } else {
              pc[kk][0] = e4m3x4f(ex2(fmaf(sv[0], ssc, -dlo)), ex2(fmaf(sv[1], ssc, -dlo)),
                                  ex2(fmaf(sv[4], ssc, -dlo)), ex2(fmaf(sv[5], ssc, -dlo)));
              pc[kk][1] = e4m3x4f(ex2(fmaf(sv[2], ssc, -dhi)), ex2(fmaf(sv[3], ssc, -dhi)),
                                  ex2(fmaf(sv[6], ssc, -dhi)), ex2(fmaf(sv[7], ssc, -dhi)));
              pc[kk][2] = e4m3x4f(ex2(fmaf(sv[8], ssc, -dlo)), ex2(fmaf(sv[9], ssc, -dlo)),
                                  ex2(fmaf(sv[12], ssc, -dlo)), ex2(fmaf(sv[13], ssc, -dlo)));
              pc[kk][3] = e4m3x4f(ex2(fmaf(sv[10], ssc, -dhi)), ex2(fmaf(sv[11], ssc, -dhi)),
                                  ex2(fmaf(sv[14], ssc, -dhi)), ex2(fmaf(sv[15], ssc, -dhi)));
            }
          }
        };
        auto settle = [&]() {   // PV(x-1) retired: o, lacc settled; apply a pending offset move
          float* od = &o[0][0][0];
#pragma unroll
          for (int e = 0; e < 32; ++e) reg_fence(od[e]);
#pragma unroll
          for (int e = 0; e < 4; ++e) reg_fence(lacc[e]);
          if (alo != 1.f || ahi != 1.f) {
#pragma unroll
            for (int t = 0; t < 8; ++t) {
              o[0][t][0] *= alo;
              o[0][t][1] *= alo;
              o[0][t][2] *= ahi;
              o[0][t][3] *= ahi;
            }
            lacc[0] *= alo;
            lacc[1] *= alo;
            lacc[2] *= ahi;
            lacc[3] *= ahi;
          }
        };
        auto advance = [&]() {
          stc = stc + 1 == STAGES ? 0 : stc + 1;
          vtb_c = reinterpret_cast<uint8_t*>(reinterpret_cast<uintptr_t>(vtb_c) ^ vtb_x);
          vdesc_c ^= vdesc_x;
        };
        auto copies = [&](const int x) {   // tile x landed and visible; tile x+2 into the stage of x-1
          cp_wait<1>();
          __syncthreads();
          if (x + STAGES - 1 < ntiles) issue_kv(x + STAGES - 1, true);
          issue_idx(x + 2 * STAGES - 2);
          cp_commit();
        };
        auto it = [&](const int x, uint32_t (&pp)[2][4], uint32_t (&pc)[2][4]) {
          copies(x);
          qk();
          pv(pp, vdesc_c ^ vdesc_x);   // P(x-1) x V^T(x-1)
          transpose();
          wgmma_wait<1>();   // QK^T(x) retired; PV(x-1) in flight under the softmax
          soft(x, pc);
          wgmma_wait<0>();
          settle();
          advance();
        };
        {   // tile 0
          copies(0);
          {   // rows 64..71 of this head's two V^T buffers: e4m3 ones (the PV wgmma's row sums)
            const int wt = tid & 127;
            *reinterpret_cast<uint2*>(S.kvq + ((wt >> 6) * 2 + hk) * 4608 + 4096 + (wt & 63) * 8) =
                make_uint2(0x38383838u, 0x38383838u);
          }
          qk();
          transpose();
          wgmma_wait<0>();
          soft(0, pA);
          advance();
        }
        int x = 1;
#pragma unroll 1
        for (; x + 1 < ntiles; x += 2) {
          it(x, pA, pB);
          it(x + 1, pB, pA);
        }
        if (x < ntiles) {
          it(x, pA, pB);
          __syncthreads();   // every warp's V^T of the last tile written
          pv(pB, vdesc_c ^ vdesc_x);
        } else {
          __syncthreads();
          pv(pA, vdesc_c ^ vdesc_x);
        }
        wgmma_wait<0>();
      } else
#pragma unroll(UNR8)
      for (int jj = 0; jj < ntiles; ++jj) {
        const int i = jj;

        if constexpr (WS) {
          // tile i landed (+ new rows); previous tile's PV of this warpgroup done (its V16 half is free). The two
          // warpgroups (one per KV head) only meet when both must: tile 0 (q staging) and tiles with new rows.
          bool patched;
          {
            mbar_wait_full(&S.full[st_c], ph_c);
            patched = ((uint32_t)i - r_lo > r_span) && (i >= i_new) &&
                      ws_patch_ni<L, NT, PTR>(&S, i, st_c, start, seq, end, tid);
          }
          if (patched || i == 0)
            csync<WS, NT>();
          else {
          }
          // (FP8 PV: V^T is double-buffered and each warp's wgmma wait already covers the PV that read this tile's
          // buffer, so the tile needs no warpgroup barrier before its transpose)
        } else {
          cp_wait<STAGES - 2>();
          __syncthreads();   // tile i landed; previous tile's PV done (V16 free)
        }

        const int st = st_c;
        const uint32_t kaddr = smem_u32(S.k + st * STAGE_BYTES) + hk * 64;

        float s[32];
        wgmma_fence();
        {
          const uint32_t kl = klo0 + (uint32_t)(st * (STAGE_BYTES >> 4));
          const uint64_t kd = ((uint64_t)((1024 >> 4) | (1u << 30)) << 32) | kl;
          const uint64_t kd2 = ((uint64_t)((1024 >> 4) | (1u << 30)) << 32) | (kl + 2u);
          const uint64_t qd2 = ((uint64_t)((512 >> 4) | (2u << 30)) << 32) | (qlo0 + 2u);
          wgmma_m64n64k32_e4m3_ss(s, qdesc0, kd, 0);
          wgmma_m64n64k32_e4m3_ss(s, qd2, kd2, 1);
        }
        wgmma_commit();
        if constexpr (!WS) {   // the copies for tile i + STAGES - 1
          if (((i + STAGES) - 1) < ntiles) issue_kv(i + STAGES - 1, true);
          issue_idx(i + 2 * STAGES - 2);
          cp_commit();
        }
        if constexpr (FLAT) {
          // the previous tile's PV (committed before this QK^T) is done: its P registers are free for the transpose
          // below (the flat kernel's 104-register consumers spilled there) and o / lacc are settled
          wgmma_wait<1>();
#pragma unroll
          for (int e = 0; e < 8; ++e) asm volatile("" : "+r"(pa[e >> 2][e & 3]));
          float* od0 = &o[0][0][0];
#pragma unroll
          for (int e = 0; e < 32; ++e) reg_fence(od0[e]);
#pragma unroll
          for (int e = 0; e < 4; ++e) reg_fence(lacc[e]);
        }
        uint8_t* V16t = V16 + (0);   // PV reads it asynchronously: double buffer
        const uint32_t v16b = smem_u32(V16t) + hk * 8192;
        // this head's V^T for this tile (PV reads it asynchronously: double buffer per stream)
        uint8_t* vtb = vtb_c;
        if constexpr (!PTR) {
          if (jj == 0) {   // rows 64..71 of this head's two V^T buffers: e4m3 ones (0x38), so the PV
            // wgmma also produces the row sums (after the first barrier: the q staging shares this space)
            const int wt = tid & 127;
            *reinterpret_cast<uint2*>(S.kvq + ((wt >> 6) * 2 + hk) * 4608 + 4096 +
                                      (wt & 63) * 8) = make_uint2(0x38383838u, 0x38383838u);
          }
        }
        if constexpr (PTR) {
          // the producers build V^T
        } else {   // meanwhile: this head's V tile -> V^T (FP8, 72 x 64, keys in P-register order: k = 4q + 2h + b for
          // key = 8h + 2q + b within each 16-key block), two ldmatrix.trans (b16 pairs) + byte permutes per thread
          const uint32_t vsrc = vsrc0 + st * STAGE_BYTES;
#pragma unroll
          for (int u = 0; u < (2); ++u) {
            uint32_t r0, r1, r2, r3;
            ldsm_x4_t(vsrc + (u ? ((vld_off0 ^ 32u)) : vld_off0), r0, r1, r2, r3);   // (scout: off1 == off0 ^ 32)
            // r0/r1: dims group 2u (h = 0 / 1), r2/r3: dims group 2u+1
            if constexpr (STSM_ON) {
              stsm_x4(smem_u32(vtb) + vst_off + 2048 * u, prmt(r0, r1, vsel0), prmt(r0, r1, vsel1), prmt(r2, r3, vsel0),
                      prmt(r2, r3, vsel1));
            } else {
#pragma unroll
              for (int g2 = 0; g2 < 2; ++g2) {
                const uint32_t R0 = g2 ? r2 : r0;
                const uint32_t R1 = g2 ? r3 : r1;
                const uint32_t w0 = prmt(R0, R1, vsel0);
                const uint32_t w1 = prmt(R0, R1, vsel1);
                *reinterpret_cast<uint32_t*>(vtb + vt_off0 + 1024 * (2 * u + g2)) = w0;   // row 16(2u+g2) + 2vw + vr0
                *reinterpret_cast<uint32_t*>(vtb + vt_off1 + 1024 * (2 * u + g2)) =
                    w1;   // row 16(2u+g2) + 2vw + 1 - vr0
              }
            }
          }
          if (!LSYNCN) fence_proxy_async();   // V^T written with st.shared, read by the PV wgmma
        }
        if constexpr (PTR) {
        } else if (LSYNCN) {   // (fence + barrier just before the PV)
        } else {
          wgsync(wgi);
        }   // V16 ready
        wgmma_wait<0>();
#pragma unroll
        for (int e = 0; e < 32; ++e) reg_fence(s[e]);
        {
#pragma unroll
          for (int e = 0; e < 8; ++e) asm volatile("" : "+r"(pa[e >> 2][e & 3]));
        }
        if constexpr (WS) {   // K (wgmma) and V (conversion) of the stage are consumed
          mbar_arrive(&S.empty[st]);
        }
        if constexpr (PTR) {   // the previous tile's PV is done with its V^T slot
          if (i >= 1 && lane == 0) mbar_arrive(&S.vte[(i - 1) & 1]);
        }

        const bool common_t = WS && ((uint32_t)i - r_lo <= r_span);
        const int t0 = start + i * TILE;
        const bool interior = common_t || (t0 >= lo_all && t0 + TILE - 1 <= hi_all);
        if (!interior) {
#pragma unroll
          for (int hf = 0; hf < 2; ++hf) {
            const int pj = seq + 2 * wsub + hf;
            const int vlo = p.window ? max(start, pj - p.window + 1) : start;
            const int vhi = min(end - 1, pj);
#pragma unroll
            for (int nt = 0; nt < 8; ++nt)
#pragma unroll
              for (int e = 0; e < 2; ++e) {
                const int pos = t0 + 8 * nt + 2 * tig + e;
                if (pos < vlo || pos > vhi) s[4 * nt + 2 * hf + e] = -INFINITY;
              }
          }
        }
        // after a segment's first tile the offset test runs on the packed P instead of on 32 row-max steps: an
        // e4m3 byte above 0x78 (P > 256, i.e. a score past the offset + 8) sets bit 7 when 0x07 is added to every byte
        // (P >= 0: no carry between bytes); the (rare) hit moves the offset and repacks that half (see the PV loop)
        constexpr bool PCHK = (MODE & 64) == 0 && !PTR && WS;
        const bool chk = PCHK && !first;
        float mlo = -INFINITY, mhi = -INFINITY;
        if (!chk) {
#pragma unroll
          for (int nt = 0; nt < 8; ++nt) {
            mlo = fmaxf(mlo, fmaxf(s[4 * nt], s[4 * nt + 1]));
            mhi = fmaxf(mhi, fmaxf(s[4 * nt + 2], s[4 * nt + 3]));
          }
        }
        // the threshold check needs no row reduction: any element past the offset + 8 shows in its own thread's
        // partial max; only the (rare) offset move and the first tile reduce over the row's quad
        if (!chk) {
          mlo *= ssc;   // scores -> log2 units (ssc > 0; ssc = 1 when the query was pre-scaled)
          mhi *= ssc;
        }
        float dlo, dhi;
        if (first) {
          {
            mlo = quad_max(mlo);
            mhi = quad_max(mhi);
          }

          dlo = (mlo == -INFINITY) ? 0.f : mlo;
          dhi = (mhi == -INFINITY) ? 0.f : mhi;
          cr[0] = dlo;
          cr[1] = dhi;

          first = false;
        } else {
          dlo = cr[0];
          dhi = cr[1];
          if (!PCHK && __any_sync(0xffffffffu, mlo > cr[0] + 8.f || mhi > cr[1] + 8.f)) {
            // rare: a score grew past 2^8 relative to the offset -> move the offset up, rescale the state
            {
              mlo = quad_max(mlo);
              mhi = quad_max(mhi);
            }   // (warp-uniform branch: full-mask shuffles)

            const float nlo = fmaxf(mlo, cr[0]), nhi = fmaxf(mhi, cr[1]);
            const float alo = ex2(cr[0] - nlo), ahi = ex2(cr[1] - nhi);
            // (the previous tile's PV wgmma completed at this tile's wgmma wait)
#pragma unroll
            for (int t = 0; t < 8; ++t) {
              o[0][t][0] *= alo;
              o[0][t][1] *= alo;
              o[0][t][2] *= ahi;
              o[0][t][3] *= ahi;
            }
            lacc[0] *= alo;
            lacc[1] *= alo;
            lacc[2] *= ahi;
            lacc[3] *= ahi;   // (FP8 PV: the row sums live here too)
            cr[0] = nlo;
            cr[1] = nhi;
            dlo = nlo;
            dhi = nhi;
          }
        }
        if constexpr (TURN) bar_sync_n(5 + hk, 256);   // my turn for the exponentials

        {   // P (e4m3, registers, k-steps of 32 permuted keys) x V^T (e4m3, K-major, 72 columns: 64 dims + 8 copies of

#pragma unroll
          for (int kk = 0; kk < (2); ++kk) {
            const float* sv = s + 16 * kk;
#define EXQ(x, q) ((MODE & 128) && (q) == 3 ? ex2p(x) : ex2(x))   // one of four on the FMA pipe
            bool rchk = chk;   // one retry at most: after the offset move every finite P is <= 1 (NaN / garbage scores
            // would otherwise retrigger the check forever: the SASS oracle's value-perturbing probe hung)
#pragma unroll 1
            for (;;) {   // (repeated once after a rare offset move; otherwise a single pass)
              if constexpr (SCH || SCHF) {
                const float a8 = 8.f * ssc, clo = 12582968.f - 8.f * dlo, chi = 12582968.f - 8.f * dhi;
                pa[kk][0] = sch_e4m3x4<PCHK>(sv[0], sv[1], sv[4], sv[5], a8, clo);
                pa[kk][1] = sch_e4m3x4<PCHK>(sv[2], sv[3], sv[6], sv[7], a8, chi);
                pa[kk][2] = sch_e4m3x4<PCHK>(sv[8], sv[9], sv[12], sv[13], a8, clo);
                pa[kk][3] = sch_e4m3x4<PCHK>(sv[10], sv[11], sv[14], sv[15], a8, chi);
                if constexpr (!PCHK) break;
              } else {
                pa[kk][0] = e4m3x4f(EXQ(fmaf(sv[0], ssc, -dlo), 0), EXQ(fmaf(sv[1], ssc, -dlo), 1),
                                    EXQ(fmaf(sv[4], ssc, -dlo), 2), EXQ(fmaf(sv[5], ssc, -dlo), 3));
                pa[kk][1] = e4m3x4f(EXQ(fmaf(sv[2], ssc, -dhi), 0), EXQ(fmaf(sv[3], ssc, -dhi), 1),
                                    EXQ(fmaf(sv[6], ssc, -dhi), 2), EXQ(fmaf(sv[7], ssc, -dhi), 3));
                pa[kk][2] = e4m3x4f(EXQ(fmaf(sv[8], ssc, -dlo), 0), EXQ(fmaf(sv[9], ssc, -dlo), 1),
                                    EXQ(fmaf(sv[12], ssc, -dlo), 2), EXQ(fmaf(sv[13], ssc, -dlo), 3));
                pa[kk][3] = e4m3x4f(EXQ(fmaf(sv[10], ssc, -dhi), 0), EXQ(fmaf(sv[11], ssc, -dhi), 1),
                                    EXQ(fmaf(sv[14], ssc, -dhi), 2), EXQ(fmaf(sv[15], ssc, -dhi), 3));
              }
              if constexpr (!PCHK)
                break;
              else {
                if (!rchk) break;
                {
                  constexpr uint32_t PB = 0x07070707u;
                  const uint32_t ov =
                      ((pa[kk][0] + PB) | (pa[kk][1] + PB) | (pa[kk][2] + PB) | (pa[kk][3] + PB)) & 0x80808080u;
                  if (!__any_sync(0xffffffffu, ov != 0u)) break;
                  {
                    // rare: a score of this half passed its row's offset + 8: offsets up to the half's row maxima,
                    // state rescaled (half 0's PV wgmma finished first), this half repacked
                    float hlo = fmaxf(fmaxf(fmaxf(sv[0], sv[1]), fmaxf(sv[4], sv[5])),
                                      fmaxf(fmaxf(sv[8], sv[9]), fmaxf(sv[12], sv[13])));
                    float hhi = fmaxf(fmaxf(fmaxf(sv[2], sv[3]), fmaxf(sv[6], sv[7])),
                                      fmaxf(fmaxf(sv[10], sv[11]), fmaxf(sv[14], sv[15])));
                    hlo = quad_max(hlo) * ssc;
                    hhi = quad_max(hhi) * ssc;
                    const float nlo = fmaxf(hlo, cr[0]), nhi = fmaxf(hhi, cr[1]);
                    const float alo = ex2(cr[0] - nlo), ahi = ex2(cr[1] - nhi);
                    if (PVSPLIT && kk == 1) {   // (a PV wgmma may still accumulate into o)
                      wgmma_wait<0>();
                      float* odw = &o[0][0][0];
#pragma unroll
                      for (int e = 0; e < 32; ++e) reg_fence(odw[e]);
#pragma unroll
                      for (int e = 0; e < 4; ++e) reg_fence(lacc[e]);
                    }
#pragma unroll
                    for (int t = 0; t < 8; ++t) {
                      o[0][t][0] *= alo;
                      o[0][t][1] *= alo;
                      o[0][t][2] *= ahi;
                      o[0][t][3] *= ahi;
                    }
                    lacc[0] *= alo;
                    lacc[1] *= alo;
                    lacc[2] *= ahi;
                    lacc[3] *= ahi;
                    cr[0] = nlo;
                    cr[1] = nhi;
                    dlo = nlo;
                    dhi = nhi;
                    rchk = false;
                  }
                }
              }
            }   // (pack / check loop)
#undef EXQ
            if constexpr (PVSPLIT) {   // each half of P goes to the tensor cores as soon as it is packed
              float* od = &o[0][0][0];
              if constexpr (LSYNCN) {   // every warp's V^T for this tile written (and visible to the async proxy)
                if (kk == 0) {
                  fence_proxy_async();
                  wgsync(wgi);
                }
              }
              if (kk == 0) {
#pragma unroll
                for (int e = 0; e < 32; ++e) reg_fence(od[e]);
#pragma unroll
                for (int e = 0; e < 4; ++e) reg_fence(lacc[e]);
              }
#pragma unroll
              for (int e = 0; e < 4; ++e) asm volatile("" : "+r"(pa[kk][e]));
              wgmma_fence();
              wgmma_m64n72k32_e4m3(od, lacc, pa[kk], vdesc_c + 2 * kk);
              wgmma_commit();
            }
          }
          if constexpr (TURN) {   // the other head's turn (head 1 skips its last hand-back)
            if (!(hk == 1 && i == (ntiles - 1))) bar_arrive_n(6 - hk, 256);
          }

          float* od = &o[0][0][0];
          if constexpr (PVSPLIT) {
            {   // done before the next tile (no wgmma in flight across the loop edge)
              wgmma_wait<0>();
#pragma unroll
              for (int e = 0; e < 32; ++e) reg_fence(od[e]);
#pragma unroll
              for (int e = 0; e < 4; ++e) reg_fence(lacc[e]);
#pragma unroll
              for (int e = 0; e < 8; ++e) asm volatile("" : "+r"(pa[e >> 2][e & 3]));
            }
          } else {
#pragma unroll
            for (int e = 0; e < 32; ++e) reg_fence(od[e]);
#pragma unroll
            for (int e = 0; e < 4; ++e) reg_fence(lacc[e]);
#pragma unroll
            for (int e = 0; e < 8; ++e) asm volatile("" : "+r"(pa[e >> 2][e & 3]));
            if constexpr (PTR) mbar_wait_full(&S.vtf[i & 1], (i >> 1) & 1);   // this tile's V^T is written
            wgmma_fence();
            {
              wgmma_m64n72k32_e4m3(od, lacc, pa[0], vdesc_c);
              wgmma_m64n72k32_e4m3(od, lacc, pa[1], vdesc_c + 2);
            }
            wgmma_commit();

          }   // (!PVSPLIT)
        }

        {
          if (++st_c == STAGES) {
            st_c = 0;
            ph_c ^= 1u;
          }
          vtb_c = reinterpret_cast<uint8_t*>(reinterpret_cast<uintptr_t>(vtb_c) ^ vtb_x);
          vdesc_c ^= vdesc_x;
        }
      }
      {
        wgmma_wait<0>();
        {
#pragma unroll
          for (int e = 0; e < 8; ++e) asm volatile("" : "+r"(pa[e >> 2][e & 3]));
        }
        {
          float* od = &o[0][0][0];
#pragma unroll
          for (int e = 0; e < 32; ++e) reg_fence(od[e]);
#pragma unroll
          for (int e = 0; e < 4; ++e) reg_fence(lacc[e]);
        }
      }
      m_r[0] = first ? -INFINITY : cr[0];
      m_r[1] = first ? -INFINITY : cr[1];
      l_r[0] = 0.25f * lacc[0];   // every lane of the quad holds the full row sum; the epilogue quad-sums
      l_r[1] = 0.25f * lacc[2];
    }
    if constexpr (CL) {
      cl_merge<L, NT>(p, S, b, pair, n_keys, chunk, vscale, o, m_r, l_r, wsub, hk, g, tig, tid);
      return true;
    }

    // ---- merge: rescale to the fixed reference 2^0 and accumulate with FP32 atomics ----
    bool do_red = true;
    // stream 1's state was merged into stream 0
    if (L == 1) {
      // warps 2,3 (second key half) hand their state to warps 0,1 (same heads) through shared memory
      // (outside cluster mode both halves add their own rescaled state to the workspace instead)
      cp_wait<0>();
      csync<WS, NT>();
      float* xs = reinterpret_cast<float*>(S.k) + ((warp & 1) * 32 + lane) * (BULK64 ? 19 : 18);
      if (wsub == 1) {
        xs[0] = m_r[0];
        xs[1] = l_r[0];
#pragma unroll
        for (int t = 0; t < 8; ++t) {
          xs[2 + 2 * t] = o[0][t][0];
          xs[3 + 2 * t] = o[0][t][1];
        }
      }
      csync<WS, NT>();
      if (wsub == 0) {
        const float m1 = xs[0], l1 = xs[1];
        const float mm = fmaxf(m_r[0], m1);
        const float f0 = (m_r[0] == -INFINITY) ? 0.f : ex2(m_r[0] - mm);
        const float f1 = (m1 == -INFINITY) ? 0.f : ex2(m1 - mm);
        m_r[0] = mm;
        l_r[0] = l_r[0] * f0 + l1 * f1;
#pragma unroll
        for (int t = 0; t < 8; ++t) {
          o[0][t][0] = o[0][t][0] * f0 + xs[2 + 2 * t] * f1;
          o[0][t][1] = o[0][t][1] * f0 + xs[3 + 2 * t] * f1;
        }
      } else {
        do_red = false;
      }
    }

    if (do_red) {   // (one branch, not one per row: the post-loop merge runs straight-line from a cold cache)
#pragma unroll
      for (int r = 0; r < NROW; ++r) {
        const int mt = r >> 1, hf = r & 1;
        const int j = (L == 1) ? 0 : 2 * (wsub + mt) + hf;
        const int hh = hk * 8 + g;
        const float lsum = quad_sum(l_r[r]);
        const float f = (m_r[r] == -INFINITY) ? 0.f : ex2(fminf(m_r[r], 126.f));
        float* acc = p.ws_acc + ((ix_t)((bp * p.nc + copy) * L + j) * 16 + hh) * 64;
        if constexpr (
            W72) {   // staged rows [16][72] at S.k + 8 KB (the exchange used the first 4.6 KB): 72-float rows, no bank conflicts
          float* stg = reinterpret_cast<float*>(S.k + 8192) + hh * 72;
#pragma unroll
          for (int c = 0; c < 2; ++c)
#pragma unroll
            for (int t4 = 0; t4 < 8; t4 += 4)
              *reinterpret_cast<float4*>(stg + 16 * tig + 8 * c + t4) =
                  make_float4(o[mt][t4][2 * hf + c] * f, o[mt][t4 + 1][2 * hf + c] * f, o[mt][t4 + 2][2 * hf + c] * f,
                              o[mt][t4 + 3][2 * hf + c] * f);
        } else if constexpr (BULK64) {   // staged rows [16][64] at S.k + 8 KB
          float* stg = reinterpret_cast<float*>(S.k + 8192) + hh * 64;
#pragma unroll
          for (int c = 0; c < 2; ++c)
#pragma unroll
            for (int t4 = 0; t4 < 8; t4 += 4)
              *reinterpret_cast<float4*>(stg + 16 * tig + 8 * c + t4) =
                  make_float4(o[mt][t4][2 * hf + c] * f, o[mt][t4 + 1][2 * hf + c] * f, o[mt][t4 + 2][2 * hf + c] * f,
                              o[mt][t4 + 3][2 * hf + c] * f);
        } else if constexpr (L == 1) {
#pragma unroll
          for (int c = 0; c < 2; ++c)
#pragma unroll
            for (int t4 = 0; t4 < 8; t4 += 4)
              red_v4(acc + 16 * tig + 8 * c + t4, o[mt][t4][2 * hf + c] * f, o[mt][t4 + 1][2 * hf + c] * f,
                     o[mt][t4 + 2][2 * hf + c] * f, o[mt][t4 + 3][2 * hf + c] * f);
        } else if constexpr (V8B) {
          // BF16 partials: the quad transposes its bf16x2 words (two 4 x 4 transposes over t = 0..3, 4..7) so lane
          // tig holds dims 8t..8t+7 for t = tig and tig + 4: two 16-byte atomics per row, half the bytes and operations
          // of the FP32 form
          uint32_t* accb = reinterpret_cast<uint32_t*>(p.ws_acc) + ((ix_t)((bp * p.nc + copy) * L + j) * 16 + hh) * 32;
#pragma unroll
          for (int t0 = 0; t0 < 8; t0 += 4) {
            uint32_t a[4];
#pragma unroll
            for (int k = 0; k < 4; ++k) {
              __nv_bfloat162 w = __floats2bfloat162_rn(o[mt][t0 + k][2 * hf] * f, o[mt][t0 + k][2 * hf + 1] * f);
              a[k] = *reinterpret_cast<uint32_t*>(&w);
            }
            {   // pairs of lanes (tig, tig ^ 1): each keeps the words whose k & 1 is its own tig & 1
              const bool o1 = tig & 1;
              const uint32_t r0 = __shfl_xor_sync(0xffffffffu, o1 ? a[0] : a[1], 1);
              const uint32_t r1 = __shfl_xor_sync(0xffffffffu, o1 ? a[2] : a[3], 1);
              a[0] = o1 ? r0 : a[0];
              a[1] = o1 ? a[1] : r0;
              a[2] = o1 ? r1 : a[2];
              a[3] = o1 ? a[3] : r1;
            }
            {   // pairs (tig, tig ^ 2): each keeps the half whose k & 2 is its own tig & 2
              const bool o2 = tig & 2;
              const uint32_t r0 = __shfl_xor_sync(0xffffffffu, o2 ? a[0] : a[2], 2);
              const uint32_t r1 = __shfl_xor_sync(0xffffffffu, o2 ? a[1] : a[3], 2);
              a[0] = o2 ? r0 : a[0];
              a[1] = o2 ? r1 : a[1];
              a[2] = o2 ? a[2] : r0;
              a[3] = o2 ? a[3] : r1;
            }
            asm volatile("red.global.add.noftz.v4.bf16x2 [%0], {%1, %2, %3, %4};" ::"l"(accb + 4 * (t0 + tig)),
                         "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3])
                         : "memory");
          }
        } else {
          // 16-byte atomics (half the L2 atomic operations of 8-byte ones): lane pairs (tig, tig ^ 1) swap halves so
          // each lane holds 4 consecutive dims: even tig keeps t even, odd tig keeps t odd (dims 8t + 4(tig >> 1) + 0..3)
          const bool odd = tig & 1;
#pragma unroll
          for (int t = 0; t < 8; t += 2) {
            // mine: (t, 2tig..2tig+1) and (t+1, same); partner's: same t's at 2(tig^1)..
            const float a0 = o[mt][t][2 * hf] * f, a1 = o[mt][t][2 * hf + 1] * f;
            const float b0 = o[mt][t + 1][2 * hf] * f, b1 = o[mt][t + 1][2 * hf + 1] * f;
            // even lane sends its t+1 pair, odd lane sends its t pair
            const float s0 = odd ? a0 : b0, s1 = odd ? a1 : b1;
            const float r0 = __shfl_xor_sync(0xffffffffu, s0, 1), r1 = __shfl_xor_sync(0xffffffffu, s1, 1);
            // even lane: t (dims 2tig, 2tig+1, then the partner's); odd lane: t+1 (the partner's, then mine) -- selects,
            // one atomic, no branch
            red_v4(acc + 8 * (t + (odd ? 1 : 0)) + 4 * (tig >> 1), odd ? r0 : a0, odd ? r1 : a1, odd ? b0 : r0,
                   odd ? b1 : r1);
          }
        }
        if (tig == 0) {
          if constexpr (W72) {   // the sum in column 64, zero padding to 72
            float* stg = reinterpret_cast<float*>(S.k + 8192) + hh * 72 + 64;
            *reinterpret_cast<float4*>(stg) = make_float4(lsum * f, 0.f, 0.f, 0.f);
            *reinterpret_cast<float4*>(stg + 4) = make_float4(0.f, 0.f, 0.f, 0.f);
          } else if constexpr (BULK64) {
            reinterpret_cast<float*>(S.k + 8192)[1024 + hh] = lsum * f;
          } else {
            red_f32(p.ws_l + ((bp * p.nc + copy) * L + j) * 16 + hh, lsum * f);
          }
        }
      }
    }
    if constexpr (W72 || BULK64) asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    const bool mode1 = !((FLAT && L == 8) && M_COOP) && !(!FLAT && M_COOP);
    // the same for the cooperative grids (the barrier's arrive is an acq_rel atomic after a CTA barrier)
    if (!(mode1 || !mode1)) gpu_fence();
    csync<WS, NT>();
    if constexpr (W72) {
      if (tid ==
          0) {   // the CTA's 16 rows (sums included): one bulk reduce-add, complete before thread 0's counter atomic
        float* acc0 = p.ws_acc + (ix_t)(bp * p.nc + copy) * 16 * 72;
        asm volatile("cp.reduce.async.bulk.global.shared::cta.bulk_group.add.f32 [%0], [%1], 4608;" ::"l"(acc0),
                     "r"(smem_u32(S.k + 8192))
                     : "memory");
        asm volatile("cp.async.bulk.commit_group;" ::: "memory");
        asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
        asm volatile("fence.proxy.async.global;" ::: "memory");
      }
    }
    if constexpr (BULK64) {
      if (tid ==
          0) {   // the CTA's 16 rows and 16 sums: two bulk reduce-adds, complete before thread 0's counter atomic
        const uint32_t src = smem_u32(S.k + 8192);
        float* acc0 = p.ws_acc + (ix_t)(bp * p.nc + copy) * 16 * 64;
        float* l0 = p.ws_l + (bp * p.nc + copy) * 16;
        asm volatile("cp.reduce.async.bulk.global.shared::cta.bulk_group.add.f32 [%0], [%1], 4096;" ::"l"(acc0),
                     "r"(src)
                     : "memory");
        asm volatile("cp.reduce.async.bulk.global.shared::cta.bulk_group.add.f32 [%0], [%1], 64;" ::"l"(l0),
                     "r"(src + 4096)
                     : "memory");
        asm volatile("cp.async.bulk.commit_group;" ::: "memory");
        asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
        asm volatile("fence.proxy.async.global;" ::: "memory");
      }
    }
    if ((FLAT && L == 8) && M_COOP) {
      // flat cooperative mode: every request is finalized after all CTAs have merged all their segments
    } else if (!FLAT && M_COOP) {
      // ---- all splits of this (request, kv pair) are resident: wait for every split's merge, then each
      // CTA finalizes its own slice of the rows (instead of one CTA doing all of them)
      // (the slice arithmetic first: it does not depend on the merge, so its divisions leave the barrier's chain)
      const int split_idx = (int)blockIdx.x;
      int nact, R;
#if ATT_V1_FINAL_INT
      if constexpr (L == 8 && (MODE & 128) != 0 && ONE) {
        // ONE's launch condition gives one 64-key tile per active CTA. For 128 output rows,
        // ceil(128/nact) is four throughout nact=32..42; shorter contexts retain the exact division.
        nact = (n_keys + TILE - 1) >> 6;
        R = (unsigned)(nact - 32) <= 10 ? 4 : fdiv_i(L * 16 + nact - 1, nact);
      } else
#endif
      {
        nact = fdiv_i(n_keys + chunk - 1, chunk);   // splits with keys
        R = fdiv_i(L * 16 + nact - 1, nact);   // rows per CTA
      }
      // keep both divisions above the barrier wait (ptxas otherwise sinks them below the spin, where
      // they are first used: scout's v200 census found them after the poll); volatile asm keeps program order
      asm volatile("" : "+r"(nact), "+r"(R));
#if ATT_V1_GATE_EARLY
      constexpr bool EARLY_GATE_MODE = L == 8 && !FLAT && (MODE & 128) != 0;
      const bool early_gate = EARLY_GATE_MODE && R == 4;
      float gate_value[8] = {};
      const int counter_thread = early_gate ? NT - 32 : 0;
#else
      constexpr int counter_thread = 0;
#endif
      bool counter_owner = tid == counter_thread;
#if ATT_V1_ELECT_COUNTER
      if constexpr (L == 8 && (MODE & 128) != 0) {
        counter_owner = false;
        // Every lane of exactly one warp participates; the elected lane alone publishes this CTA's keys.
        if ((tid >> 5) == (counter_thread >> 5)) counter_owner = elect_one();
      }
#endif
      if (counter_owner) {
#if ATT_V1_CTA_BARRIER
        if constexpr (L == 8 && (MODE & 128) != 0) {
          // A distinct, zero-at-rest counter: after our own completed arrival, zero can mean only that
          // all nact CTAs have arrived. The last RMW wraps it; no waiter can miss a terminal reset.
          int* arrivals = p.ws_cnt + 16384 + bp;
          const unsigned last = (unsigned)nact - 1;
          unsigned prior;
          asm volatile("atom.acq_rel.gpu.global.inc.u32 %0, [%1], %2;\n"
                       : "=r"(prior) : "l"(arrivals), "r"(last) : "memory");
          if (prior != last) {
            for (;;) {
              unsigned value;
#if ATT_V1_RELAXED_POLL
              asm volatile("ld.relaxed.gpu.global.u32 %0, [%1];\n" : "=r"(value) : "l"(arrivals) : "memory");
              if (value == 0)
                asm volatile("ld.acquire.gpu.global.u32 %0, [%1];\n" : "=r"(value) : "l"(arrivals) : "memory");
#else
              asm volatile("ld.acquire.gpu.global.u32 %0, [%1];\n" : "=r"(value) : "l"(arrivals) : "memory");
#endif
              if (value == 0) break;
              if constexpr (ATT_V1_POLL_NS > 0) __nanosleep(ATT_V1_POLL_NS);
            }
          }
        } else
#endif
        {
        // two key counters alternating per launch (parity = this slot's epoch, read in the prologue): the last split
        // to arrive resets the OTHER counter (unused until the next launch) and bumps the epoch, so nobody needs a
        // departure counter (no atomic round trip at the very end of the kernel)
        const int ep = *epoch_smem(S);
        int* c = p.ws_cnt + ((ep & 1) ? 12288 : 8192) + bp;
        const int mine = end - start;
#if ATT_V1_RED_ARRIVE
        if constexpr (L == 8 && (MODE & 128) != 0) {
          asm volatile("red.release.gpu.global.add.s32 [%0], %1;\n" :: "l"(c), "r"(mine) : "memory");
          for (;;) {
            int value;
            asm volatile("ld.acquire.gpu.global.s32 %0, [%1];\n" : "=r"(value) : "l"(c) : "memory");
            if (value >= n_keys) break;
            if constexpr (ATT_V1_POLL_NS > 0) __nanosleep(ATT_V1_POLL_NS);
          }
          // No returned arrival rank: split 0 owns next-phase preparation after observing every arrival.
          // All other CTAs staged this epoch before publishing, and no next call can consume these slots yet.
          if (split_idx == 0) {
            p.ws_cnt[((ep & 1) ? 8192 : 12288) + bp] = 0;
            p.ws_cnt[REQUEST_EPOCH_BASE + 4 * bp] = ep + 1;
          }
        } else
#endif
        {
        int old;
        asm volatile("atom.acq_rel.gpu.global.add.s32 %0, [%1], %2;\n" : "=r"(old) : "l"(c), "r"(mine) : "memory");
        if (old + mine == n_keys) {
          p.ws_cnt[((ep & 1) ? 8192 : 12288) + bp] = 0;
          p.ws_cnt[REQUEST_EPOCH_BASE + 4 * bp] = ep + 1;
        } else {
          int v;
          for (;;) {
#if ATT_V1_RELAXED_POLL
            if constexpr (L == 8 && (MODE & 128) != 0) {
              asm volatile("ld.relaxed.gpu.global.s32 %0, [%1];\n" : "=r"(v) : "l"(c) : "memory");
              // The current counter cannot reset in this call. Only the terminal observation needs to
              // acquire the other CTAs' published partials; failed probes need no L1 invalidation.
              if (v >= n_keys)
                asm volatile("ld.acquire.gpu.global.s32 %0, [%1];\n" : "=r"(v) : "l"(c) : "memory");
            } else
#endif
            {
            asm volatile("ld.acquire.gpu.global.s32 %0, [%1];\n" : "=r"(v) : "l"(c) : "memory");
            }
            if (v >= n_keys) break;
            if constexpr (L == 8 && (MODE & 128) != 0) {
              if constexpr (ATT_V1_POLL_NS > 0) __nanosleep(ATT_V1_POLL_NS);
            } else {
              __nanosleep(64);
            }
          }
        }
        }
        }
      }
#if ATT_V1_GATE_EARLY
      if constexpr (EARLY_GATE_MODE) {
        const int row = split_idx * R + (tid >> 3);
        if (early_gate && tid < 32 && row < L * 16) {
          const uint4 packed = __ldcg(reinterpret_cast<const uint4*>(p.qkv +
              (ix_t)(b * L + (row >> 4)) * p.ld_qkv + (HQ + 2 * HKV + 16 * pair + (row & 15)) * 64 + (tid & 7) * 8));
          const auto* gate = reinterpret_cast<const __nv_bfloat162*>(&packed);
#pragma unroll
          for (int e = 0; e < 4; ++e) {
            const float2 value = __bfloat1622float2(gate[e]);
            gate_value[2 * e] = sigm(value.x);
            gate_value[2 * e + 1] = sigm(value.y);
          }
#if ATT_V1_GATE_EARLY == 2
          // The epoch occupies gst bytes 256..271. These four gate rows use bytes 512..1535 instead.
          const uint32_t dst = smem_u32(S.gst + 256) + tid * 32;
#pragma unroll
          for (int e = 0; e < 8; e += 4)
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(dst + e * 4),
                         "f"(gate_value[e]), "f"(gate_value[e + 1]), "f"(gate_value[e + 2]), "f"(gate_value[e + 3])
                         : "memory");
#endif
        }
#if ATT_V1_GATE_EARLY != 2
#pragma unroll
        for (int e = 0; e < 8; ++e) reg_fence(gate_value[e]);
#endif
      }
#endif
      csync<WS, NT>();
      const int r_lo = split_idx * R;
      const int r_hi = min(L * 16, r_lo + R);
#if ATT_V1_WARP_ROW
      if constexpr (L == 8 && (MODE & 128) != 0) {
        if (R == 4) {
          const int row = r_lo + (tid >> 5), dim = 2 * lane;
          if (tid < 128 && row < r_hi) {
            const int j = row >> 4, hh = row & 15;
            float x = 0.f, z = 0.f, denominator = 0.f;
#pragma unroll 1
            for (int copy = 0; copy < p.nc; ++copy) {
              const ix_t index = (ix_t)((bp * p.nc + copy) * L + j) * 16 + hh;
              if constexpr (V8B) {
                auto* address = reinterpret_cast<uint32_t*>(p.ws_acc) + index * 32 + lane;
                const uint32_t bits = __ldcg(address);
                const float2 value = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&bits));
                x += value.x;
                z += value.y;
                *address = 0;
              } else {
                auto* address = reinterpret_cast<float2*>(p.ws_acc + index * 64 + dim);
                const float2 value = __ldcg(address);
                x += value.x;
                z += value.y;
                *address = make_float2(0.f, 0.f);
              }
              const float sum = lane == 0 ? __ldcg(p.ws_l + index) : 0.f;
              denominator += __shfl_sync(0xffffffffu, sum, 0);
              if (lane == 0) p.ws_l[index] = 0.f;
            }
            // Keep the RMS-overflow fix: normalize the softmax numerator before forming any square.
            const float scale = __fdividef(vscale, denominator > 0.f ? denominator : 1.f);
            x *= scale;
            z *= scale;
            float square_sum = x * x + z * z;
#pragma unroll
            for (int shift = 16; shift; shift >>= 1)
              square_sum += __shfl_xor_sync(0xffffffffu, square_sum, shift);
            const float norm = rsqrtf(square_sum / 64.f + 1e-5f);
            float2 gate;
#if ATT_V1_GATE_EARLY == 2
            gate = *reinterpret_cast<const float2*>(reinterpret_cast<const float*>(S.gst + 256) +
                                                    (tid >> 5) * 64 + dim);
#else
            const uint32_t packed = __ldcg(reinterpret_cast<const uint32_t*>(p.qkv +
                (ix_t)(b * L + j) * p.ld_qkv + (HQ + 2 * HKV + 16 * pair + hh) * 64 + dim));
            gate = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&packed));
            gate.x = sigm(gate.x);
            gate.y = sigm(gate.y);
#endif
            *reinterpret_cast<__nv_bfloat162*>(p.out + (ix_t)(b * L + j) * HQ * 64 + (16 * pair + hh) * 64 + dim) =
                __floats2bfloat162_rn(x * norm * gate.x, z * norm * gate.y);
          }
          return true;   // non-flat cooperative path has no remaining CTA joins or workspace updates
        }
      }
#endif
      const int part = tid & 7;
      // Each warp owns four rows. Skip only wholly inactive warps; partial warps still run every shuffle.
      const int warp_row = ATT_TRIM_FINAL_WARPS ? (tid >> 5) * 4 : 0;
      for (int rbase = r_lo; rbase + warp_row < r_hi; rbase += NT / 8) {   // warp-uniform trip count
        const int rr = rbase + (tid >> 3);
        const bool act = rr < r_hi;
        const int j = act ? rr >> 4 : 0, hh = act ? rr & 15 : 0;
        const uint4 graw =
            (act
#if ATT_V1_GATE_EARLY
             && !early_gate
#endif
            ) ? *reinterpret_cast<const uint4*>(p.qkv + (ix_t)(b * L + j) * p.ld_qkv + (HQ + 2 * HKV) * 64 +
                                                  (16 * pair + hh) * 64 + part * 8)
                : make_uint4(0, 0, 0, 0);
        float y[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
        float lsum = 0.f;
        if constexpr (V8B)
          sum_copies_bf16<L>(p, bp, j, hh, part, act, y, lsum);
        else
          sum_copies<L, W72>(p, bp, j, hh, part, act, y, lsum);
        const float rn = rms_softmax_mean(y, lsum, vscale);
        const __nv_bfloat162* g2 = reinterpret_cast<const __nv_bfloat162*>(&graw);
        uint4 ov;
        uint32_t* ow = reinterpret_cast<uint32_t*>(&ov);
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          float s0, s1;
#if ATT_V1_GATE_EARLY
          if (early_gate) {
#if ATT_V1_GATE_EARLY == 2
            const float2 gate = act ? *reinterpret_cast<const float2*>(
                reinterpret_cast<const float*>(S.gst + 256) + tid * 8 + 2 * e) : make_float2(0.f, 0.f);
            s0 = gate.x;
            s1 = gate.y;
#else
            s0 = gate_value[2 * e];
            s1 = gate_value[2 * e + 1];
#endif
          } else
#endif
          {
            const float2 gt = __bfloat1622float2(g2[e]);
            s0 = sigm(gt.x);
            s1 = sigm(gt.y);
          }
          __nv_bfloat162 r2 = __floats2bfloat162_rn(y[2 * e] * rn * s0, y[2 * e + 1] * rn * s1);
          ow[e] = *reinterpret_cast<uint32_t*>(&r2);
        }
        if (act)
          *reinterpret_cast<uint4*>(p.out + (ix_t)(b * L + j) * (HQ * 64) + (16 * pair + hh) * 64 + part * 8) = ov;
      }

    } else {
      if (tid == 0) {
        const int mine = end - start;
        int old;
        asm volatile("atom.acq_rel.gpu.global.add.s32 %0, [%1], %2;\n"
                     : "=r"(old)
                     : "l"(p.ws_cnt + bp), "r"(mine)
                     : "memory");
        S.flag = (old + mine == n_keys);
      }
      csync<WS, NT>();
      if (S.flag) {
        // ---- last CTA: sum the copies, normalize, rmsnorm * sigmoid(gate), store; reset the workspace ----
        if (p.nc == 1) {
          fin_nc1<L, NT, W72>(p, b, pair, bp, vscale);
        } else {
          const int part = tid & 7;
          for (int rr = tid >> 3; rr < L * 16; rr += NT / 8) {
            const int j = rr >> 4, hh = rr & 15;
            float y[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
            float lsum = 0.f;
            // gate row in flight with the copies (one L2 round trip per row instead of two)
            const uint4 graw = *reinterpret_cast<const uint4*>(
                p.qkv + (ix_t)(b * L + j) * p.ld_qkv + (HQ + 2 * HKV) * 64 + (16 * pair + hh) * 64 + part * 8);
            sum_copies<L, W72>(p, bp, j, hh, part, true, y, lsum);
            const float rn = rms_softmax_mean(y, lsum, vscale);
            const __nv_bfloat162* g2 = reinterpret_cast<const __nv_bfloat162*>(&graw);
            uint4 ov;
            uint32_t* ow = reinterpret_cast<uint32_t*>(&ov);
#pragma unroll
            for (int e = 0; e < 4; ++e) {
              const float2 gt = __bfloat1622float2(g2[e]);
              const float s0 = sigm(gt.x), s1 = sigm(gt.y);
              __nv_bfloat162 r2 = __floats2bfloat162_rn(y[2 * e] * rn * s0, y[2 * e + 1] * rn * s1);
              ow[e] = *reinterpret_cast<uint32_t*>(&r2);
            }
            *reinterpret_cast<uint4*>(p.out + (ix_t)(b * L + j) * (HQ * 64) + (16 * pair + hh) * 64 + part * 8) = ov;
          }
        }
        if (tid == 0) p.ws_cnt[bp] = 0;
      }
    }
    if constexpr (FLAT) __syncthreads();   // flat kernels may reuse shared memory in the next segment
  }
  return true;
}

// Flat grids whose CTAs are all resident: once every CTA has merged all its segments (grid-wide count),
// each CTA finalizes its own slice of all requests' rows (sum the copies, normalize, rmsnorm * gate, store,
// zero the workspace). Runs on the NT consumer threads; the last CTA out resets the two counters.
template <int L, bool WS, int NT, bool NEED_WAIT = true, bool ZLAST = false>
__device__ __forceinline__ void flat_finalize(const Params& p, const int* sense_s) {
  int zlast = 0;   // (ZLAST) this CTA arrived last: warp 0 zeroes the per-SM rank counters below
  const int tid = threadIdx.x;
  if constexpr (NEED_WAIT) pdl_wait();   // a CTA without segments has not waited yet (the gate rows are qkv)
  const int G = gridDim.x * gridDim.y * gridDim.z;
  const int c = blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z);
  int* cnt = p.ws_cnt + 30000;
  // (thread 0's acq_rel arrive releases the CTA's reds)
  csync<WS, NT>();
  if (tid == 0) {
    // two grid counters alternating per launch (parity = the epoch thread 0 read at kernel start): the last CTA
    // to arrive resets the other counter and bumps the epoch; no departure counter at the end of the kernel
    cp_wait<0>();
    const int ep = *reinterpret_cast<const volatile int*>(sense_s);
    int* ac = p.ws_cnt + 30020 + (ep & 1);
    int old;
    asm volatile("atom.acq_rel.gpu.global.add.s32 %0, [%1], 1;\n" : "=r"(old) : "l"(ac) : "memory");
    if ((old + 1) == G) {
      p.ws_cnt[30020 + ((ep & 1) ^ 1)] = 0;
      p.ws_cnt[FLAT_EPOCH] = ep + 1;
      {   // the per-SM rank registration (read by every CTA at kernel start, before arriving here)
        cnt[2] = 0;
        cnt[3] = 0;
        if constexpr (ZLAST)
          zlast = 1;
        else
          for (int i = 0; i < G / 2 && i < 1700; ++i) p.ws_cnt[31000 + i] = 0;
      }
    } else {
      int v;
      for (;;) {
        asm volatile("ld.acquire.gpu.global.s32 %0, [%1];\n" : "=r"(v) : "l"(ac) : "memory");
        if (v >= G) break;
        __nanosleep(64);
      }
    }
  } else {
    if (tid == 0) {
      atomicAdd(cnt, 1);
      int v;
      for (;;) {
        asm volatile("ld.acquire.gpu.global.s32 %0, [%1];\n" : "=r"(v) : "l"(cnt) : "memory");
        if (v >= G) break;
        __nanosleep(64);
      }
    }
  }
  if (ZLAST && tid < 32 && __shfl_sync(0xffffffffu, zlast, 0))
    for (int i = tid; i < G / 2 && i < 1700; i += 32) p.ws_cnt[31000 + i] = 0;
  csync<WS, NT>();
  const int pairs = HKV >> 1;
  const int rows = p.B * pairs * L * 16;
  const int R = fdiv_i(rows + G - 1, G);
  const int r_lo = c * R;
  const int r_hi = min(rows, r_lo + R);
  const float vscale = __ldg(p.v_scale);
  const int part = tid & 7;
  const int warp_row = ATT_TRIM_FINAL_WARPS ? (tid >> 5) * 4 : 0;
  for (int rbase = r_lo; rbase + warp_row < r_hi; rbase += NT / 8) {   // warp-uniform trip count
    const int r = rbase + (tid >> 3);
    const bool act = r < r_hi;
    const int bp = act ? r / (L * 16) : 0;
    const int rr = act ? r % (L * 16) : 0;
    const int j = rr >> 4, hh = rr & 15;
    const int b = (pairs == 1) ? bp : fdiv_i(bp, pairs);
    const int pair = bp - b * pairs;
    const uint4 graw = act ? *reinterpret_cast<const uint4*>(p.qkv + (ix_t)(b * L + j) * p.ld_qkv +
                                                           (HQ + 2 * HKV) * 64 + (16 * pair + hh) * 64 + part * 8)
                           : make_uint4(0, 0, 0, 0);
    float y[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    float lsum = 0.f;
    sum_copies<L>(p, bp, j, hh, part, act, y, lsum);
    const float rn = rms_softmax_mean(y, lsum, vscale);
    const __nv_bfloat162* g2 = reinterpret_cast<const __nv_bfloat162*>(&graw);
    uint4 ov;
    uint32_t* ow = reinterpret_cast<uint32_t*>(&ov);
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      const float2 gt = __bfloat1622float2(g2[e]);
      const float s0 = sigm(gt.x), s1 = sigm(gt.y);
      __nv_bfloat162 r2 = __floats2bfloat162_rn(y[2 * e] * rn * s0, y[2 * e + 1] * rn * s1);
      ow[e] = *reinterpret_cast<uint32_t*>(&r2);
    }
    if (act) *reinterpret_cast<uint4*>(p.out + (ix_t)(b * L + j) * (HQ * 64) + (16 * pair + hh) * 64 + part * 8) = ov;
  }
}

template <int L, bool FLAT, int MODE = 0>
__global__ void __launch_bounds__(Cfg<L, FLAT>::NTX, (ATT_V8_OVL && L == 8 && !FLAT && (MODE & 256) != 0)
                                                         ? 1
                                                         : Cfg<L, FLAT>::MINB) attn_kernel(const __grid_constant__ Params p) {
  extern __shared__ __align__(1024) uint8_t smem_raw[];
  // 1024-byte aligned base: the 128B-swizzled tiles must start on swizzle-pattern boundaries (wgmma)
  Smem<L, FLAT>& S = *reinterpret_cast<Smem<L, FLAT>*>(smem_raw + ((1024u - (smem_u32(smem_raw) & 1023u)) & 1023u));
  const int pair = blockIdx.z;
  if constexpr ((MODE & 128) != 0) {   // segments of pf_seg bytes, a pf_chunk prefix of each, spread over the CTAs
    if (threadIdx.x == 0) {
      const unsigned nb = gridDim.x * gridDim.y * gridDim.z;
      for (unsigned sg = blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z); sg < p.pf_nseg; sg += nb) {
        const uint8_t* src = p.pf + (size_t)sg * p.pf_seg;
        for (unsigned o = 0; o < p.pf_chunk; o += 32768u) {
          const unsigned n = min(32768u, p.pf_chunk - o);
          asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(src + o), "r"(n) : "memory");
        }
      }
    }
  } else if (p.pf_chunk && threadIdx.x == 0) {   // one pf_chunk per CTA
    const uint8_t* src = p.pf + (size_t)(blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z)) * p.pf_chunk;
    for (unsigned o = 0; o < p.pf_chunk; o += 32768u) {
      const unsigned n = min(32768u, p.pf_chunk - o);
      asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(src + o), "r"(n) : "memory");
    }
  }
  if (!FLAT) {
    // grid (n_splits, B): request b, split s of its window / context
    const int split = blockIdx.x;
    const int b = blockIdx.y;
    // SGLang's FA3 backend builds
    // page_table = req_to_token[req_pool_indices, :max_seq_len_k] (sglang/srt/layers/attention/
    // flashattention_backend.py, init_forward_metadata: decode / target-verify paths), so with one request its row
    // is exactly seq_len + q_len entries wide. The page table's width is a shape: speculate seq = width - q_len and
    // issue the window's page indices in the same trip as cache_seqlens; the real length is checked (segment) before
    // the indices are used for any K/V address or any softmax work, and a miss re-runs everything with the real one
    const int seq_ld = __ldg(p.seq_lens + b);
    constexpr bool SPEC = (MODE & 8) != 0;
    int seq = SPEC ? p.max_pages - L : seq_ld;
    for (;;) {
      const int lo = p.window ? max(0, seq + 1 - p.window) : 0;
      const int n_keys = seq + L - lo;
      // plain split-KV: whole tiles per split (the per-CTA tile count is what costs); trailing splits may get no keys:
      // they skip the workspace merge (the merge counts keys, the cooperative finalize counts the splits with keys).
      const int ntl = ((n_keys + 63) >> 6) + p.n_splits - 1;
      // ceil(2^32 / n_splits) from the host (exact for ntl < 2^20, n_splits <= 1024): one IMAD.HI.
      const int chunk = (p.ns_magic ? (int)__umulhi((unsigned)ntl, p.ns_magic) : ntl) << 6;
      int start = lo + split * chunk;
      int end = min(seq + L, start + chunk);
      if constexpr (ATT_V8C_CLUSTER && L == 8 && (MODE & 256) != 0) {
        cl_arm<L, FLAT>(S, split, n_keys, chunk);
        if (start >= end) {   // an empty split owns no rows and sends nothing; it still meets the cluster barrier
          asm volatile("barrier.cluster.arrive.relaxed.aligned;\nbarrier.cluster.wait.aligned;" ::: "memory");
          return;
        }
      }
      if (start >= end) {
        if (SPEC && seq != seq_ld) {
          seq = seq_ld;
          continue;
        }   // (an empty split only on the real length)

        return;
      }
      if (segment<L, FLAT, 0, MODE>(p, S, b, pair, split == 0, start, end, seq, n_keys,
                                    (p.nc == 1 ? 0 : split - fdiv_i(split, p.nc) * p.nc), chunk, 0,
                                    SPEC ? seq_ld : seq))
        return;
      seq = seq_ld;   // (spec miss: today's path with the real length)
    }
  }
  if constexpr (FLAT && L == 8) {
    if (p.coop && threadIdx.x == 0) {   // the grid barrier's epoch, long before it is needed
      stage_epoch(S, p.ws_cnt + FLAT_EPOCH);
      cp_commit();
    }
  }
  // the per-SM registration's atomics go out from warp 1 while warp 0 builds the prefix sums
  const bool w0on = p.coop && p.resident && p.w0 != 1024 && gridDim.y * gridDim.z == 1 && (gridDim.x & 1) == 0;
  unsigned w0_smid = 0;
  int w0_rank = 0;
  if (w0on && threadIdx.x == 32) {
    asm volatile("mov.u32 %0, %%smid;" : "=r"(w0_smid));
    const int Gx = gridDim.x;
    w0_rank = ((int)w0_smid < Gx / 2) ? atomicAdd(p.ws_cnt + 31000 + w0_smid, 1) : 2;
    if (w0_rank >= 2) atomicExch(p.ws_cnt + 30003, 1);
    if (w0_rank < 2) p.ws_cnt[31200 + 2 * w0_smid + w0_rank] = (int)blockIdx.x + 1;
    gpu_fence();
    atomicAdd(p.ws_cnt + (30002), 1);   // (8 counters: no single hot address)
  }

  if constexpr (L == 1 && Cfg<L, FLAT>::WS) {
    if (threadIdx.x == 32) {   // the first segment's stage barriers (segment() skips its init for that segment)
      for (int st = 0; st < Cfg<L, FLAT>::STAGES; ++st) {
        mbar_init_n(&S.full[st], Cfg<L, FLAT>::NP);
        mbar_init_n(&S.empty[st], Cfg<L, FLAT>::NT);
      }
      mbar_fence_init();
    }
  }
  if (threadIdx.x < 32) {
    int run = 0;
    int incl_l = 0, n_l = 0, seq_l = 0;
    for (int b0 = 0; b0 < p.B; b0 += 32) {
      const int b = b0 + (int)threadIdx.x;
      int n = 0;
      if (b < p.B) {
        const int seq = __ldg(p.seq_lens + b);
        const int lo = p.window ? max(0, seq + 1 - p.window) : 0;
        n = seq + L - lo;
        seq_l = seq;
      }
      int incl = n;
#pragma unroll
      for (int o = 1; o < 32; o <<= 1) {
        const int t = __shfl_up_sync(0xffffffffu, incl, o);
        if ((int)threadIdx.x >= o) incl += t;
      }
      if (b < p.B) S.cum[b + 1] = run + incl;
      run += __shfl_sync(0xffffffffu, incl, 31);
      incl_l = incl;
      n_l = n;
    }
    if (threadIdx.x == 0) S.cum[0] = 0;
  }
  __syncthreads();
  const int64_t T0 = S.cum[p.B];
  // every request boundary costs a CTA that crosses it a whole extra segment (q staging, pipeline refill,
  // merge: ~7 us at q_len 8), so the shares are cut in a virtual key space with SEGK keys inserted before each
  // request after the first, then mapped back (a CTA crossing a boundary gets SEGK fewer real keys)
  constexpr int SEGK = (L == 8) ? 600 : 400;
  const int64_t T = T0 + (SEGK > 0 ? (int64_t)(p.B - 1) * SEGK : 0);
  const int c = blockIdx.x;
  const int G = gridDim.x;
  const int cpart = c;
  int cs, ce;
  {
    cs = (int)((int64_t)cpart * T / G);
    ce = (int)((int64_t)(cpart + 1) * T / G);
  }   // (a float-estimate + correction

  int vc = c;
  if (w0on) {
    // The two CTAs sharing an SM do not progress equally (the first one dispatched gets more issue slots), so
    // give the first one a larger share: slot 2 * SM + arrival rank. Every CTA waits until all have registered;
    // any SM with more than two (another kernel holding SMs) falls back to the even split.
    if (threadIdx.x == (32)) {
      unsigned smid = w0_smid;
      int* arrive = p.ws_cnt + 30002;
      int* bad = p.ws_cnt + 30003;
      int rank = w0_rank;

      int v;
      {
        for (;;) {
          asm volatile("ld.acquire.gpu.global.s32 %0, [%1];\n" : "=r"(v) : "l"(arrive) : "memory");
          if (v >= G) break;
        }
      }
      asm volatile("ld.acquire.gpu.global.s32 %0, [%1];\n" : "=r"(v) : "l"(bad) : "memory");
      if (!v) {   // rank by blockIdx order on the SM (the lower index gets the issue-slot priority)
        int other;
        asm volatile("ld.acquire.gpu.global.s32 %0, [%1];\n"
                     : "=r"(other)
                     : "l"(p.ws_cnt + 31200 + 2 * smid + (rank ^ 1))
                     : "memory");
        rank = (other - 1 < c) ? 1 : 0;
      }
      S.vidx = v ? -1 : (int)(smid * 2 + rank);
    }
    __syncthreads();
    if (S.vidx >= 0) {
      vc = S.vidx;
      const int64_t sp = vc >> 1, r = vc & 1;
      cs = (int)(T * (2048 * sp + r * p.w0) / (1024LL * G));
      ce = (int)(T * (2048 * sp + (r ? 2048 : p.w0)) / (1024LL * G));
    }
  }

  if (SEGK > 0) {   // virtual -> real key positions
    auto vmap = [&](int v) -> int {
      int lo_b = 0, hi_b = p.B - 1;   // the last request whose virtual start is <= v
      while (lo_b < hi_b) {
        const int mid = (lo_b + hi_b + 1) >> 1;
        if (S.cum[mid] + mid * SEGK <= v)
          lo_b = mid;
        else
          hi_b = mid - 1;
      }
      return min(v - lo_b * SEGK, S.cum[lo_b + 1]);
    };
    cs = vmap(cs);
    ce = vmap(ce);
  }
  if (cs >= ce) {
    if constexpr (L == 8) {
      if (p.coop) {   // still part of the grid-wide finalize
        if constexpr (Cfg<L, FLAT>::WS) {
          if (threadIdx.x < Cfg<L, FLAT>::NT)
            flat_finalize<L, true, Cfg<L, FLAT>::NT, true, (MODE & 64) != 0>(p, epoch_smem(S));
        } else {
          flat_finalize<L, false, Cfg<L, FLAT>::NTX, true, (MODE & 64) != 0>(p, epoch_smem(S));
        }
      }
    }
    return;
  }

  // segment list in shared memory (keeps the per-segment loop state out of registers)
  if (threadIdx.x == 0) {
    int b = 0, n = 0;
    while (S.cum[b + 1] <= cs) ++b;
    for (; b < p.B && S.cum[b] < ce && n < MAXSEG; ++b) {
      S.seg[n][0] = b;
      S.seg[n][1] = max(cs, S.cum[b]) - S.cum[b];
      S.seg[n][2] = min(ce, S.cum[b + 1]) - S.cum[b];
      ++n;
    }
    S.nseg = n;
  }
  __syncthreads();   // (warp 0 wrote the list before the barrier above)
  const int nseg = S.nseg;
  auto run_segments = [&](auto role) {
    constexpr int R = decltype(role)::value;
#pragma unroll 1
    for (int k = 0; k < nseg; ++k) {
      const int b = S.seg[k][0];
      const int seq = !p.window ? S.cum[b + 1] - S.cum[b] - L : __ldg(p.seq_lens + b);
      const int lo = p.window ? max(0, seq + 1 - p.window) : 0;
      const int n_keys = seq + L - lo;
      const int s0 = S.seg[k][1] + lo;
      const int s1 = S.seg[k][2] + lo;
      if (s0 < s1) {
        segment<L, FLAT, R, MODE>(p, S, b, pair, s0 == lo, s0, s1, seq, n_keys,
                                  (p.nc == 1 ? 0 : vc - fdiv_i(vc, p.nc) * p.nc), 0, 0, 0,
                                  L == 1 && Cfg<L, FLAT>::WS && k == 0);
      }
    }
    // the producer leaves with its copies landed
  };
  if constexpr (Cfg<L, FLAT>::WS) {
    // separate code for the two roles, so the producer warpgroup can hand its registers to the consumers
    if (threadIdx.x >= Cfg<L, FLAT>::NT) {
      asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" ::"n"(Cfg<L, FLAT>::PREGS));
      run_segments(std::integral_constant<int, 1>());
    } else {
      asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" ::"n"(Cfg<L, FLAT>::CREGS));
      run_segments(std::integral_constant<int, 2>());
      if constexpr (L == 8) {
        if (p.coop) flat_finalize<L, true, Cfg<L, FLAT>::NT, false, (MODE & 64) != 0>(p, epoch_smem(S));
      }
    }
  } else {
    run_segments(std::integral_constant<int, 0>());
    if constexpr (FLAT && L == 8) {
      if (p.coop) flat_finalize<L, false, Cfg<L, FLAT>::NTX, false, (MODE & 64) != 0>(p, epoch_smem(S));
    }
  }
}

#if ATT_VERIFY_CLUSTER == 1
// One cluster per KV head. Each of its 16 CTAs handles 128 past keys and owns four query rows.
// Warpgroup 0 computes cached-key partials; warpgroup 1 computes the owner's eight new-key contributions.
// Every cached partial is pushed into the owner's shared memory, so neither a global counter nor a tail CTA
// is needed. The two cluster joins protect destination lifetime and publish the completed partials.
struct VerifyClusterSmem {
  alignas(1024) uint8_t k[128 * 64];
  alignas(1024) uint8_t v[128 * 128];
  alignas(1024) uint8_t q[64 * 64];
  union alignas(1024) {
    __nv_bfloat16 raw_q[64 * 64];
    uint8_t vt[2 * 4608];
  } scratch;
  __nv_bfloat16 raw_kv[16 * 64];
  uint8_t new_kv[16 * 64];
  float cs[8 * 32], sn[8 * 32], freq[32], scale[4];
  int64_t loc[8];
  int page[128];
  alignas(16) float partial[16][4][68];   // source CTA, owner row, 64 numerator components and denominator
};

__device__ __forceinline__ void verify_cluster_join() {
  asm volatile("barrier.cluster.arrive.release.aligned;\nbarrier.cluster.wait.acquire.aligned;\n" ::: "memory");
}
__device__ __forceinline__ uint32_t verify_remote(const void* local, int rank) {
  uint32_t remote;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(remote) : "r"(smem_u32(local)), "r"(rank));
  return remote;
}
__device__ __forceinline__ void verify_store4(uint32_t dst, float x, float y, float z, float w) {
  asm volatile("st.shared::cluster.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(dst), "f"(x), "f"(y), "f"(z), "f"(w)
               : "memory");
}
__device__ __forceinline__ void verify_qk(uint32_t* d, uint64_t q, uint64_t k, int accumulate) {
  asm volatile(
      "{ .reg .pred p; setp.ne.b32 p, %34, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n128k32.f16.e4m3.e4m3 "
      "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
      "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, %32, %33, p, 1, 1; }"
      : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3]), "+r"(d[4]), "+r"(d[5]), "+r"(d[6]), "+r"(d[7]),
        "+r"(d[8]), "+r"(d[9]), "+r"(d[10]), "+r"(d[11]), "+r"(d[12]), "+r"(d[13]), "+r"(d[14]), "+r"(d[15]),
        "+r"(d[16]), "+r"(d[17]), "+r"(d[18]), "+r"(d[19]), "+r"(d[20]), "+r"(d[21]), "+r"(d[22]), "+r"(d[23]),
        "+r"(d[24]), "+r"(d[25]), "+r"(d[26]), "+r"(d[27]), "+r"(d[28]), "+r"(d[29]), "+r"(d[30]), "+r"(d[31])
      : "l"(q), "l"(k), "r"(accumulate));
}
// Piecewise-linear exp2 encoded directly as E4M3, with the row maximum represented by 256. Around 1024,
// half's unit step lets one HFMA2 form two integer E4M3 codes in the low bytes; no query precision is lost.
__device__ __forceinline__ uint32_t verify_prob2(uint32_t scores, float scale, float row_max) {
  __half2 value = __hfma2(*reinterpret_cast<__half2*>(&scores), __float2half2_rn(8.f * scale),
                        __float2half2_rn(1144.f - 8.f * row_max));
  value = __hmin2(__hmax2(value, __float2half2_rn(1024.f)), __float2half2_rn(1144.f));
  return *reinterpret_cast<uint32_t*>(&value);
}
__device__ __forceinline__ float verify_warp_sum(float value) {
#pragma unroll
  for (int shift = 16; shift; shift >>= 1) value += __shfl_xor_sync(0xffffffffu, value, shift);
  return value;
}

__global__ void __launch_bounds__(256, 1) verify_cluster_kernel(const __grid_constant__ Params p) {
  extern __shared__ __align__(1024) uint8_t raw[];
  auto& S = *reinterpret_cast<VerifyClusterSmem*>(raw);
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int rank = blockIdx.x, head = blockIdx.y, b = blockIdx.z;
  const int seq = __ldg(p.seq_lens + b), lo = max(0, seq - 2047), start = lo + rank * 128;
  if (tid == 0 && p.pf) {   // retain the layer's early pooler-weight prefetch; standalone passes a null source
    for (unsigned sg = rank + 16 * head; sg < p.pf_nseg; sg += 32) {
      const uint8_t* src = p.pf + (size_t)sg * p.pf_seg;
      for (unsigned off = 0; off < p.pf_chunk; off += 32768u) {
        const unsigned bytes = min(32768u, p.pf_chunk - off);
        asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(src + off), "r"(bytes) : "memory");
      }
    }
  }
  if (tid < 128) {
    if (start + tid < seq)
      cp_async4(smem_u32(S.page + tid), p.page_table + (ix_t)b * p.max_pages + start + tid);
    else
      S.page[tid] = 0;
  }
  if (tid < 8) cp_async16(smem_u32(S.freq + 4 * tid), p.inv_freq + 4 * tid, 16);
  if (tid >= 8 && tid < 12) cp_async16(smem_u32(S.loc + 2 * (tid - 8)), p.cache_loc + b * 8 + 2 * (tid - 8), 16);
  if (tid == 12) cp_async4(smem_u32(S.scale), p.k_scale);
  if (tid == 13) cp_async4(smem_u32(S.scale + 1), p.v_scale);
  cp_commit();
  pdl_wait();
  pdl_launch_dependents();
  tfor<512, 256>(tid, [&](int e) {
    const int row = e >> 3, chunk = e & 7;
    cp_async16(smem_u32(S.scratch.raw_q + row * 64 + chunk * 8),
               p.qkv + (ix_t)(b * 8 + (row >> 3)) * p.ld_qkv + (8 * head + (row & 7)) * 64 + chunk * 8, 16);
  });
  if (tid < 128) {
    const int row = tid >> 3, chunk = tid & 7;
    cp_async16(smem_u32(S.raw_kv + row * 64 + chunk * 8),
               p.qkv + (ix_t)(b * 8 + (row & 7)) * p.ld_qkv + (HQ + (row >> 3) * HKV + head) * 64 + chunk * 8, 16);
  }
  cp_commit();
  cp_wait<1>();
  __syncthreads();
  tfor<512, 256>(tid, [&](int e) {
    const int row = e >> 2, chunk = e & 3;
    const uint32_t offset = ((uint32_t)S.page[row] << 7) + head * 64 + chunk * 16;
    const int bytes = start + row < seq ? 16 : 0;
    cp_async16(smem_u32(S.k + row * 64 + ((chunk ^ ((row >> 1) & 3)) << 4)), p.k_cache + offset, bytes);
    cp_async16(smem_u32(S.v + row * 128 + ((chunk ^ (row & 7)) << 4)), p.v_cache + offset, bytes);
  });
  cp_commit();
  {
    const float angle = (seq + (tid >> 5)) * S.freq[tid & 31];
    const float turns = rintf(angle * 0.15915493667125702f);
    float reduced = fmaf(-turns, 6.25f, angle);
    reduced = fmaf(-turns, 0.03318530693650246f, reduced);
    reduced = fmaf(-turns, 2.430837753308879e-10f, reduced);
    float sn, cs;
    __sincosf(reduced, &sn, &cs);
    S.cs[tid] = p.rope ? cs : 1.f;
    S.sn[tid] = p.rope ? sn : 0.f;
  }
  cp_wait<1>();
  __syncthreads();
  const int part = tid & 7;
#pragma unroll
  for (int pass = 0; pass < 2; ++pass) {
    const int row = pass * 32 + (tid >> 3), j = row >> 3;
    const uint4 words = *reinterpret_cast<const uint4*>(S.scratch.raw_q + row * 64 + part * 8);
    const auto* packed = reinterpret_cast<const __nv_bfloat162*>(&words);
    float x[8], ss = 0.f;
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      const float2 pair = __bfloat1622float2(packed[e]);
      x[2 * e] = pair.x;
      x[2 * e + 1] = pair.y;
      ss += pair.x * pair.x + pair.y * pair.y;
    }
    const float norm = rsqrtf(oct_sum(ss) / 64.f + 1e-5f);
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      const int rotary = j * 32 + part * 4 + e;
      const float a = x[2 * e] * norm, z = x[2 * e + 1] * norm;
      x[2 * e] = a * S.cs[rotary] - z * S.sn[rotary];
      x[2 * e + 1] = a * S.sn[rotary] + z * S.cs[rotary];
    }
    *reinterpret_cast<uint2*>(S.q + row * 64 + (((part >> 1) ^ ((row >> 1) & 3)) << 4) + (part & 1) * 8) =
        make_uint2(e4m3x4f(x[0], x[1], x[2], x[3]), e4m3x4f(x[4], x[5], x[6], x[7]));
  }
  if (tid < 128) {
    const int row = tid >> 3, j = row & 7;
    const uint4 words = *reinterpret_cast<const uint4*>(S.raw_kv + row * 64 + part * 8);
    const auto* packed = reinterpret_cast<const __nv_bfloat162*>(&words);
    float x[8], ss = 0.f;
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      const float2 pair = __bfloat1622float2(packed[e]);
      x[2 * e] = pair.x;
      x[2 * e + 1] = pair.y;
      ss += pair.x * pair.x + pair.y * pair.y;
    }
    const float norm = row < 8 ? rsqrtf(oct_sum(ss) / 64.f + 1e-5f) : 1.f;
    const float inverse = __fdividef(norm, S.scale[row >> 3]);
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      const float a = x[2 * e] * inverse, z = x[2 * e + 1] * inverse;
      const int rotary = j * 32 + part * 4 + e;
      x[2 * e] = row < 8 ? a * S.cs[rotary] - z * S.sn[rotary] : a;
      x[2 * e + 1] = row < 8 ? a * S.sn[rotary] + z * S.cs[rotary] : z;
    }
    const uint2 converted = make_uint2(e4m3x4f(x[0], x[1], x[2], x[3]), e4m3x4f(x[4], x[5], x[6], x[7]));
    *reinterpret_cast<uint2*>(S.new_kv + row * 64 + part * 8) = converted;
    if (rank == 0) {
      uint8_t* cache = row < 8 ? p.k_cache : p.v_cache;
      *reinterpret_cast<uint2*>(cache + (uint32_t)S.loc[j] * 128 + head * 64 + part * 8) = converted;
    }
  }
  cp_wait<0>();
  fence_proxy_async();
  verify_cluster_join();
  const float score_scale = p.scale_log2 * S.scale[0];
  float own_x = 0.f, own_y = 0.f, own_l = 0.f, gate_x = 0.f, gate_y = 0.f;
  if (tid < 128) {
    uint32_t scores[32] = {};
#pragma unroll
    for (int e = 0; e < 32; ++e) asm volatile("" : "+r"(scores[e]));
    wgmma_fence();
    const uint64_t qdesc = desc_sw64(smem_u32(S.q)), kdesc = desc_sw64(smem_u32(S.k));
    verify_qk(scores, qdesc, kdesc, 0);
    verify_qk(scores, qdesc + 2, kdesc + 2, 1);
    wgmma_commit();
    // Reuse our existing warp transpose, one 64-key slab at a time. Its key permutation matches the
    // register operand of the FP8 PV instruction; the extra eight rows of ones produce the denominators.
    const int vw = lane >> 2, wq = warp, stj = lane & 7, stm = lane >> 3;
    const uint32_t sel0 = vw >= 4 ? 0x7531u : 0x6420u, sel1 = vw >= 4 ? 0x6420u : 0x7531u;
    const int key = 16 * wq + 8 * ((lane >> 3) & 1) + (lane & 7);
    const uint32_t load_off = key * 128 + (((lane >> 4) ^ (lane & 7)) << 4);
    const uint32_t store_off = 1024 * (stm >> 1) + 128 * stj + 64 * ((stm & 1) ^ (stj >> 2)) + ((wq ^ (stj & 3)) << 4);
#pragma unroll
    for (int slab = 0; slab < 2; ++slab) {
#pragma unroll
      for (int u = 0; u < 2; ++u) {
        uint32_t a, z, c, d;
        ldsm_x4_t(smem_u32(S.v + slab * 8192) + (load_off ^ (u * 32)), a, z, c, d);
        stsm_x4(smem_u32(S.scratch.vt + slab * 4608) + store_off + 2048 * u,
                 prmt(a, z, sel0), prmt(a, z, sel1), prmt(c, d, sel0), prmt(c, d, sel1));
      }
    }
    *reinterpret_cast<uint2*>(S.scratch.vt + (tid >> 6) * 4608 + 4096 + (tid & 63) * 8) =
        make_uint2(0x38383838u, 0x38383838u);
    fence_proxy_async();
    wgsync(0);
    wgmma_wait<0>();
#pragma unroll
    for (int e = 0; e < 32; ++e) asm volatile("" : "+r"(scores[e]));
    const int g = lane >> 2, tig = lane & 3;
    float maximum[2] = {-INFINITY, -INFINITY};
#pragma unroll
    for (int column = 0; column < 16; ++column) {
#pragma unroll
      for (int half = 0; half < 2; ++half) {
        const int row = warp * 16 + g + 8 * half, j = row >> 3;
        float2 value = __half22float2(*reinterpret_cast<__half2*>(scores + 2 * column + half));
        const int position = start + 8 * column + 2 * tig;
        const int lower = max(0, seq + j - 2047);
        if (position < lower || position >= seq) value.x = -65504.f;
        if (position + 1 < lower || position + 1 >= seq) value.y = -65504.f;
        scores[2 * column + half] = pack_h2(value.x, value.y);
        maximum[half] = fmaxf(maximum[half], fmaxf(value.x, value.y));
      }
    }
    maximum[0] = quad_max(maximum[0]) * score_scale;
    maximum[1] = quad_max(maximum[1]) * score_scale;
    float values[32] = {}, sums[4] = {};
#pragma unroll
    for (int e = 0; e < 32; ++e) reg_fence(values[e]);
#pragma unroll
    for (int e = 0; e < 4; ++e) reg_fence(sums[e]);
    uint32_t probability[4][4];
#pragma unroll
    for (int group = 0; group < 4; ++group) {
      uint32_t encoded[8];
#pragma unroll
      for (int e = 0; e < 8; ++e)
        encoded[e] = verify_prob2(scores[group * 8 + e], score_scale, maximum[e & 1]);
      probability[group][0] = prmt(encoded[0], encoded[2], 0x6420);
      probability[group][1] = prmt(encoded[1], encoded[3], 0x6420);
      probability[group][2] = prmt(encoded[4], encoded[6], 0x6420);
      probability[group][3] = prmt(encoded[5], encoded[7], 0x6420);
      wgmma_fence();
      wgmma_m64n72k32_e4m3(values, sums, probability[group],
                           desc_sw64(smem_u32(S.scratch.vt + (group >> 1) * 4608)) + 2 * (group & 1));
      wgmma_commit();
    }
    wgmma_wait<0>();
#pragma unroll
    for (int e = 0; e < 32; ++e) reg_fence(values[e]);
#pragma unroll
    for (int e = 0; e < 4; ++e) reg_fence(sums[e]);
#pragma unroll
    for (int e = 0; e < 16; ++e) asm volatile("" : "+r"(probability[e >> 2][e & 3]));
#pragma unroll
    for (int half = 0; half < 2; ++half) {
      const int row = warp * 16 + g + 8 * half, owner = row >> 2;
      const float factor = ex2(fminf(maximum[half] - 8.f, 118.f));
      const uint32_t remote = verify_remote(&S.partial[rank][row & 3][0], owner);
      const bool odd = tig & 1;
#pragma unroll
      for (int column = 0; column < 8; column += 2) {
        const float a = values[4 * column + 2 * half] * factor, z = values[4 * column + 2 * half + 1] * factor;
        const float c = values[4 * (column + 1) + 2 * half] * factor, d = values[4 * (column + 1) + 2 * half + 1] * factor;
        const float other0 = __shfl_xor_sync(0xffffffffu, odd ? a : c, 1);
        const float other1 = __shfl_xor_sync(0xffffffffu, odd ? z : d, 1);
        verify_store4(remote + 4 * (8 * (column + int(odd)) + 4 * (tig >> 1)),
                      odd ? other0 : a, odd ? other1 : z, odd ? c : other0, odd ? d : other1);
      }
      if (tig == 0) {
        const float denominator = sums[2 * half] * factor;
        asm volatile("st.shared::cluster.f32 [%0], %1;" :: "r"(remote + 256), "f"(denominator) : "memory");
      }
    }
  } else {
    const int row = 4 * rank + (warp - 4), j = row >> 3, dim = 2 * lane;
    const uint16_t qbytes = *reinterpret_cast<const uint16_t*>(S.q + row * 64 +
        (((dim >> 4) ^ ((row >> 1) & 3)) << 4) + (dim & 15));
    uint32_t qhalf = cvt2(qbytes);
    const float2 query = __half22float2(*reinterpret_cast<__half2*>(&qhalf));
    float logits[8], maximum = -INFINITY;
#pragma unroll
    for (int token = 0; token < 8; ++token) {
      uint32_t key_half = cvt2(*reinterpret_cast<const uint16_t*>(S.new_kv + token * 64 + dim));
      const float2 key = __half22float2(*reinterpret_cast<__half2*>(&key_half));
      logits[token] = verify_warp_sum(query.x * key.x + query.y * key.y) * score_scale;
      if (token <= j) maximum = fmaxf(maximum, logits[token]);
    }
    const float factor = ex2(fminf(maximum - 8.f, 118.f));
#pragma unroll
    for (int token = 0; token < 8; ++token) {
      const uint32_t code = token <= j ? (uint32_t)max(0, min(120, __float2int_rn(120.f + 8.f * (logits[token] - maximum)))) : 0;
      uint32_t weight_half = cvt2(code), value_half = cvt2(*reinterpret_cast<const uint16_t*>(S.new_kv + (8 + token) * 64 + dim));
      const float weight = __half2float(__low2half(*reinterpret_cast<__half2*>(&weight_half))) * factor;
      const float2 value = __half22float2(*reinterpret_cast<__half2*>(&value_half));
      own_x += weight * value.x;
      own_y += weight * value.y;
      own_l += weight;
    }
    const uint32_t raw_gate = __ldcg(reinterpret_cast<const uint32_t*>(p.qkv + (ix_t)(b * 8 + j) * p.ld_qkv +
                         (HQ + 2 * HKV + head * 8 + (row & 7)) * 64 + dim));
    const float2 gate = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&raw_gate));
    gate_x = sigm(gate.x);
    gate_y = sigm(gate.y);
  }
  verify_cluster_join();
  if (tid >= 128) {
    const int local_row = warp - 4, row = rank * 4 + local_row;
#pragma unroll
    for (int source = 0; source < 16; ++source) {
      const float2 values = *reinterpret_cast<const float2*>(&S.partial[source][local_row][2 * lane]);
      own_x += values.x;
      own_y += values.y;
      own_l += S.partial[source][local_row][64];
    }
    const float scale = __fdividef(S.scale[1], own_l);
    own_x *= scale;
    own_y *= scale;
    const float norm = rsqrtf(verify_warp_sum(own_x * own_x + own_y * own_y) / 64.f + 1e-5f);
    const __nv_bfloat162 result = __floats2bfloat162_rn(own_x * norm * gate_x, own_y * norm * gate_y);
    *reinterpret_cast<__nv_bfloat162*>(p.out + (ix_t)(b * 8 + (row >> 3)) * 1024 +
                                      (head * 8 + (row & 7)) * 64 + 2 * lane) = result;
  }
}

void launch_verify_cluster(const Params& p, cudaStream_t stream, bool pdl) {
  static bool initialized = false;
  if (!initialized) {
    cudaFuncSetAttribute(verify_cluster_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, sizeof(VerifyClusterSmem));
    cudaFuncSetAttribute(verify_cluster_kernel, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
    initialized = true;
  }
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(16, 2, p.B);
  cfg.blockDim = dim3(256, 1, 1);
  cfg.dynamicSmemBytes = sizeof(VerifyClusterSmem);
  cfg.stream = stream;
  cudaLaunchAttribute attrs[2] = {};
  attrs[0].id = cudaLaunchAttributeClusterDimension;
  attrs[0].val.clusterDim.x = 16;
  attrs[0].val.clusterDim.y = 1;
  attrs[0].val.clusterDim.z = 1;
  attrs[1].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[1].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = attrs;
  cfg.numAttrs = pdl ? 2 : 1;
  cudaLaunchKernelEx(&cfg, verify_cluster_kernel, p);
}
#endif

#if ATT_VERIFY_CLUSTER == 2
// Adapted from keell0renz/muse-open-kernel-challenge, 890809cee421500b00126d084e81b338ef9181ab
// (Apache-2.0), including ZhimaoL/sglang's verify-attention code. Changes: Params ABI, layer PDL/prefetch,
// and L2-coherent predecessor reads. Upstream's normalize-before-square RMS and ZH_SCH=1 are retained.
namespace keell_vp2 {
namespace cg = cooperative_groups;
#define V_L 8
#define V_HQ 16
#define V_HKV 2
#define V_D 64
#define V_QKVW ((2 * V_HQ + 2 * V_HKV) * V_D)
#define V_MINIT (-1.0e30f)
#define ZH_SCH 1
#define V_QK16_OUT "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}"
__device__ __forceinline__ uint32_t v_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void v_cpa16(void* dst, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(v_u32(dst)), "l"(src) : "memory");
}
__device__ __forceinline__ void v_sincos(float ang, float* sn, float* cs) {
  double const a = (double)ang;
  float const r = (float)(a - 6.283185307179586 * floor(a * 0.15915494309189535 + 0.5));
  __sincosf(r, sn, cs);
}
__device__ __forceinline__ uint32_t v_e4m3x2(float lo, float hi) {
  return (uint32_t)__nv_cvt_float2_to_fp8x2(make_float2(lo, hi), __NV_SATFINITE, __NV_E4M3);
}
__device__ __forceinline__ float2 v_e4m3x2_dec(uint32_t two) {
  __half2_raw const h = __nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)(two & 0xFFFFu), __NV_E4M3);
  return __half22float2(*reinterpret_cast<const __half2*>(&h));
}
__device__ __forceinline__ void v_fence_async() { asm volatile("fence.proxy.async.shared::cta;" ::: "memory"); }
__device__ __forceinline__ void wg_fence() { asm volatile("wgmma.fence.sync.aligned;" ::: "memory"); }
__device__ __forceinline__ void wg_qk16_0(uint32_t (&d)[32], const uint32_t (&a)[4], unsigned long long db) {
  asm volatile(
      "{\n.reg .pred p;\nsetp.ne.b32 p, 0, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n128k32.f16.e4m3.e4m3\n" V_QK16_OUT ", {%32,%33,%34,%35}, %36, p, 1, 1;\n}\n"
      : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3]), "=r"(d[4]), "=r"(d[5]), "=r"(d[6]), "=r"(d[7]),
        "=r"(d[8]), "=r"(d[9]), "=r"(d[10]), "=r"(d[11]), "=r"(d[12]), "=r"(d[13]), "=r"(d[14]), "=r"(d[15]),
        "=r"(d[16]), "=r"(d[17]), "=r"(d[18]), "=r"(d[19]), "=r"(d[20]), "=r"(d[21]), "=r"(d[22]), "=r"(d[23]),
        "=r"(d[24]), "=r"(d[25]), "=r"(d[26]), "=r"(d[27]), "=r"(d[28]), "=r"(d[29]), "=r"(d[30]), "=r"(d[31])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(db));
}
__device__ __forceinline__ unsigned long long v_desc(uint32_t saddr) {
  return (unsigned long long)((saddr >> 4) & 0x3FFF) | (1ull << 16) | ((unsigned long long)(1024 >> 4) << 32) | (1ull << 62);
}
__device__ __forceinline__ void wg_qk16_1(uint32_t (&d)[32], const uint32_t (&a)[4], unsigned long long db) {
  asm volatile(
      "{\n.reg .pred p;\nsetp.ne.b32 p, 1, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n128k32.f16.e4m3.e4m3\n" V_QK16_OUT ", {%32,%33,%34,%35}, %36, p, 1, 1;\n}\n"
      : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3]), "+r"(d[4]), "+r"(d[5]), "+r"(d[6]), "+r"(d[7]),
        "+r"(d[8]), "+r"(d[9]), "+r"(d[10]), "+r"(d[11]), "+r"(d[12]), "+r"(d[13]), "+r"(d[14]), "+r"(d[15]),
        "+r"(d[16]), "+r"(d[17]), "+r"(d[18]), "+r"(d[19]), "+r"(d[20]), "+r"(d[21]), "+r"(d[22]), "+r"(d[23]),
        "+r"(d[24]), "+r"(d[25]), "+r"(d[26]), "+r"(d[27]), "+r"(d[28]), "+r"(d[29]), "+r"(d[30]), "+r"(d[31])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(db));
}
__device__ __forceinline__ void wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory"); }
__device__ __forceinline__ void wg_wait0() { asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory"); }
__device__ __forceinline__ __half2 v_h2(uint32_t v) { return *reinterpret_cast<__half2*>(&v); }
__device__ __forceinline__ uint32_t v_u(__half2 v) { return *reinterpret_cast<uint32_t*>(&v); }
__device__ __forceinline__ uint32_t v_ex2h2(uint32_t x) {
  uint32_t r;
  asm("ex2.approx.f16x2 %0, %1;" : "=r"(r) : "r"(x));
  return r;
}
__device__ __forceinline__ uint32_t v_prmt(uint32_t a, uint32_t b, uint32_t sel) {
  uint32_t r;
  asm("prmt.b32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(sel));
  return r;
}
__device__ __forceinline__ uint32_t v_cvt_e4m3x2h(uint32_t h2) {
  unsigned short r;
  asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(r) : "r"(h2));
  return (uint32_t)r;
}
__device__ __forceinline__ float v_ex2(float x) {
  float y;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}
__device__ __forceinline__ void v_ldsm4t(uint32_t& a, uint32_t& b, uint32_t& c, uint32_t& d, uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(a), "=r"(b), "=r"(c), "=r"(d) : "r"(addr));
}
__device__ __forceinline__ void wg_pv(float (&d)[32], const uint32_t (&a)[4], unsigned long long db) {
  asm volatile(
      "{\n.reg .pred p;\nsetp.ne.b32 p, %37, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3\n"
      "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
      "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31},"
      "{%32,%33,%34,%35}, %36, p, 1, 1;\n}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]),
        "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]),
        "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]),
        "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(db), "n"(1));
}
#define VP2_PUSH 1
#define VP2_LATEQ 0
#define VP2_ST 0
#define VP2_QSM 1
#define VP2_LWG0 1
#define VP2_MLB 0
#define VP2_P64 0
#define VP2_NOTF 0
#define VP2_STRM 0
#define VP2_WG1A 1

#define Q_BLK 1120
#define Q_TX 1056
struct VP2Smem {
  uint8_t kt[128 * 128];
  uint8_t vt[128 * 128];
  uint8_t pt[64 * 128];

  uint8_t mrb[16 * Q_BLK];
  float2 tab[256];
  float kdec[8 * 65];
  float qdec[4 * 65];
  float vdec[8 * 64];
  float pp[4 * 8];
  float lrow[64];
  uint8_t qs[64 * 160];
  alignas(16) uint32_t nks[128 * 8];
  alignas(8) unsigned long long mbm;
  alignas(8) unsigned long long mbml;
  alignas(8) unsigned long long mbs[16];
};
#define Q_ST(i) do { } while (0)

__global__ void __launch_bounds__(256, 1)
vp2_kernel(const __grid_constant__ Params p) {
  const __nv_bfloat16* qkv = p.qkv;
  const int* page_table = p.page_table;
  const int* seq_lens = p.seq_lens;
  const long long* cache_loc = reinterpret_cast<const long long*>(p.cache_loc);
  uint8_t* k_cache = p.k_cache;
  uint8_t* v_cache = p.v_cache;
  const float* inv_freq = p.inv_freq;
  __nv_bfloat16* out = p.out;
  const int window = p.window, pt_stride = p.max_pages;
  const float k_scale = __ldcg(p.k_scale), v_scale = __ldcg(p.v_scale);
  const float c2 = p.scale_log2 * k_scale, k_rscale = __fdividef(1.f, k_scale);
  const float v_rscale = __fdividef(1.f, v_scale), eps = 1e-5f;
  if (threadIdx.x == 0 && p.pf) {
    const unsigned block = blockIdx.x + gridDim.x * blockIdx.y;
    for (unsigned sg = block; sg < p.pf_nseg; sg += gridDim.x * gridDim.y) {
      const uint8_t* src = p.pf + (size_t)sg * p.pf_seg;
      for (unsigned off = 0; off < p.pf_chunk; off += 32768u) {
        const unsigned bytes = min(32768u, p.pf_chunk - off);
        asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(src + off), "r"(bytes) : "memory");
      }
    }
  }
  pdl_wait();
  pdl_launch_dependents();

  {
    int const b = (int)blockIdx.y;
    qkv += (long long)b * 8 * V_QKVW;
    out += (long long)b * 8 * (V_HQ * V_D);
    page_table += (long long)b * pt_stride;
    seq_lens += b;
    cache_loc += 8 * b;
  }
  extern __shared__ uint8_t smem_raw[];
  VP2Smem& sm = *reinterpret_cast<VP2Smem*>((reinterpret_cast<uintptr_t>(smem_raw) + 1023) & ~uintptr_t(1023));
  int const tid = threadIdx.x, lane = tid & 31;
  int const wg = tid >> 7;
  int const w = (tid >> 5) & 3;
  int const g = lane >> 2, t = lane & 3;
  unsigned const r = cg::this_cluster().block_rank();
  int const hk = (int)blockIdx.x >> 4;
  int const otok = (int)r >> 1, oh0 = 8 * hk + 4 * ((int)r & 1);
  __nv_bfloat162 gpre = __float2bfloat162_rn(0.f);
  Q_ST(0 + 16 * wg);
  if (wg == 1) {

    uint32_t const d_ = v_u32(&sm.nks[(tid & 127) * 8]);
#pragma unroll
    for (int i = 0; i < 2; i++) {
      int const j = w + 4 * i;
      sm.nks[(tid & 127) * 8 + i] = __ldcg(reinterpret_cast<const uint32_t*>(
          qkv + (long long)j * V_QKVW + (V_HQ + hk) * V_D + 2 * lane));
      sm.nks[(tid & 127) * 8 + 2 + i] = __ldcg(reinterpret_cast<const uint32_t*>(
          qkv + (long long)j * V_QKVW + (V_HQ + V_HKV + hk) * V_D + 2 * lane));
    }
    sm.nks[(tid & 127) * 8 + 4] = __ldcg(reinterpret_cast<const uint32_t*>(
        qkv + (long long)otok * V_QKVW + (oh0 + w) * V_D + 2 * lane));
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8;" :: "r"(d_ + 24), "l"(cache_loc + otok) : "memory");
    asm volatile("cp.async.commit_group;" ::: "memory");
  }
  if (tid == 0) {
    uint32_t const mb = v_u32(&sm.mbm);
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(mb) : "memory");
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(mb), "r"(VP2_MLB ? 16 * 512 : 16 * Q_TX) : "memory");
  }
  asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");
  int const seq = seq_lens[0];
  int const lo = max(0, seq - window + 1);
  int const n = seq - lo;
  int const chunk = (((n + 15) / 16) + 127) / 128 * 128;
  int const p0 = min(seq, lo + (int)r * chunk), p1 = min(seq, p0 + chunk);
  int const ntiles = (p1 - p0 + 127) >> 7;
  float const nk_ifr = inv_freq[lane];

  __nv_bfloat162 nk_kraw[2], nk_vraw[2], nk_qraw = __float2bfloat162_rn(0.f);
  long long nk_crow = 0;
  auto small_loads = [&]() {
    if (wg == 1) {
    }
  };
  small_loads();

  int const gch = tid & 7, grs = tid >> 3;
  {
    int rws[4];
#pragma unroll
    for (int k = 0; k < 4; k++) {
      int const pos = p0 + grs + 32 * k;
      rws[k] = (ntiles > 0 && pos < p1) ? page_table[pos] : -1;
    }
    if (wg == 0) {
      int const qrow = tid >> 1, qh = tid & 1;
      const uint8_t* const qsrc = reinterpret_cast<const uint8_t*>(qkv)
          + 2 * ((long long)(qrow >> 3) * V_QKVW + (8 * hk + (qrow & 7)) * V_D) + 64 * qh;
#pragma unroll
      for (int c = 0; c < 4; c++) v_cpa16(sm.qs + qrow * 160 + 64 * qh + 16 * c, qsrc + 16 * c);
      asm volatile("cp.async.commit_group;" ::: "memory");
      const uint32_t raw_gate = __ldcg(reinterpret_cast<const uint32_t*>(qkv + (long long)otok * V_QKVW +
          (V_HQ + 2 * V_HKV) * V_D + (oh0 + w) * V_D + 2 * lane));
      gpre = *reinterpret_cast<const __nv_bfloat162*>(&raw_gate);
    }
#pragma unroll
    for (int k = 0; k < 4; k++) {
      int const gi = grs + 32 * k;
      int const off = gi * 128 + ((gch ^ (gi & 7)) << 4);
      if (rws[k] >= 0) {
        v_cpa16(sm.kt + off, k_cache + (long long)rws[k] * 128 + gch * 16);
        v_cpa16(sm.vt + off, v_cache + (long long)rws[k] * 128 + gch * 16);
      } else {
        *reinterpret_cast<uint4*>(sm.kt + off) = make_uint4(0u, 0u, 0u, 0u);
        *reinterpret_cast<uint4*>(sm.vt + off) = make_uint4(0u, 0u, 0u, 0u);
      }
    }
    asm volatile("cp.async.commit_group;" ::: "memory");
  }
  Q_ST(3 + 16 * wg);
  {
    float sn_, cs_;
    v_sincos((float)(seq + (tid >> 5)) * nk_ifr, &sn_, &cs_);
    sm.tab[tid] = make_float2(cs_, sn_);
  }
  asm volatile("cp.async.wait_group 1;" ::: "memory");
  __syncthreads();
  Q_ST(4 + 16 * wg);
  uint32_t qf[2][4];
  if (wg == 0) {
    float x[2][16];
#pragma unroll
    for (int e = 0; e < 2; e++) {
      int const tok = 2 * w + e;
      float ss = 0.f;
#pragma unroll
      for (int c4 = 0; c4 < 4; c4++) {
        uint2 const raw = *reinterpret_cast<const uint2*>(sm.qs + (8 * tok + g) * 160 + 32 * c4 + 8 * t);
        __nv_bfloat162 const u0 = *reinterpret_cast<const __nv_bfloat162*>(&raw.x);
        __nv_bfloat162 const u1 = *reinterpret_cast<const __nv_bfloat162*>(&raw.y);
        x[e][4 * c4 + 0] = __bfloat162float(u0.x); x[e][4 * c4 + 1] = __bfloat162float(u0.y);
        x[e][4 * c4 + 2] = __bfloat162float(u1.x); x[e][4 * c4 + 3] = __bfloat162float(u1.y);
#pragma unroll
        for (int i = 0; i < 4; i++) ss += x[e][4 * c4 + i] * x[e][4 * c4 + i];
      }
      ss += __shfl_xor_sync(0xffffffffu, ss, 1);
      ss += __shfl_xor_sync(0xffffffffu, ss, 2);
      float const rn = rsqrtf(ss * (1.0f / V_D) + eps);
#pragma unroll
      for (int c4 = 0; c4 < 4; c4++)
#pragma unroll
        for (int i2 = 0; i2 < 2; i2++) {
          float2 const t2_ = sm.tab[tok * 32 + 8 * c4 + 2 * t + i2];
          float const cs = t2_.x, sn = t2_.y;
          float const ev = x[e][4 * c4 + 2 * i2] * rn, od = x[e][4 * c4 + 2 * i2 + 1] * rn;
          x[e][4 * c4 + 2 * i2] = ev * cs - od * sn;
          x[e][4 * c4 + 2 * i2 + 1] = ev * sn + od * cs;
        }
    }
#pragma unroll
    for (int ks = 0; ks < 2; ks++)
#pragma unroll
      for (int e = 0; e < 2; e++)
#pragma unroll
        for (int hi = 0; hi < 2; hi++) {
          int const c4 = 2 * ks + hi;
          qf[ks][e + 2 * hi] = v_e4m3x2(x[e][4 * c4], x[e][4 * c4 + 1]) | (v_e4m3x2(x[e][4 * c4 + 2], x[e][4 * c4 + 3]) << 16);
        }
    Q_ST(5);
  } else {
    {
      uint32_t const* s_ = &sm.nks[(tid & 127) * 8];
      nk_kraw[0] = *reinterpret_cast<const __nv_bfloat162*>(s_); nk_kraw[1] = *reinterpret_cast<const __nv_bfloat162*>(s_ + 1);
      nk_vraw[0] = *reinterpret_cast<const __nv_bfloat162*>(s_ + 2); nk_vraw[1] = *reinterpret_cast<const __nv_bfloat162*>(s_ + 3);
      nk_qraw = *reinterpret_cast<const __nv_bfloat162*>(s_ + 4);
      nk_crow = *reinterpret_cast<const long long*>(s_ + 6);
    }
#pragma unroll
    for (int i = 0; i < 2; i++) {
      int const j = w + 4 * i;
      float2 kv = __bfloat1622float2(nk_kraw[i]);
      float2 const vv = __bfloat1622float2(nk_vraw[i]);
      float ks = kv.x * kv.x + kv.y * kv.y;
#pragma unroll
      for (int o2 = 16; o2 > 0; o2 >>= 1) ks += __shfl_xor_sync(0xffffffffu, ks, o2);
      float const krn = rsqrtf(ks * (1.0f / V_D) + eps);
      float2 const tk_ = sm.tab[j * 32 + lane];
      float const ck = tk_.x, sk = tk_.y;
      kv.x *= krn; kv.y *= krn;
      uint32_t const k8 = v_e4m3x2((kv.x * ck - kv.y * sk) * k_rscale, (kv.x * sk + kv.y * ck) * k_rscale);
      uint32_t const v8 = v_e4m3x2(vv.x * v_rscale, vv.y * v_rscale);
      if (j == otok && (r & 1) == 0) {
        *reinterpret_cast<uint16_t*>(k_cache + nk_crow * 128 + hk * 64 + 2 * lane) = (uint16_t)k8;
        *reinterpret_cast<uint16_t*>(v_cache + nk_crow * 128 + hk * 64 + 2 * lane) = (uint16_t)v8;
      }
      float2 const kd = v_e4m3x2_dec(k8), vd = v_e4m3x2_dec(v8);
      sm.kdec[j * 65 + 2 * lane] = kd.x; sm.kdec[j * 65 + 2 * lane + 1] = kd.y;
      *reinterpret_cast<float2*>(sm.vdec + j * 64 + 2 * lane) = vd;
    }
    {
      float2 qv = __bfloat1622float2(nk_qraw);
      float qs = qv.x * qv.x + qv.y * qv.y;
#pragma unroll
      for (int o2 = 16; o2 > 0; o2 >>= 1) qs += __shfl_xor_sync(0xffffffffu, qs, o2);
      float const qrn = rsqrtf(qs * (1.0f / V_D) + eps);
      float2 const tq_ = sm.tab[otok * 32 + lane];
      float const cq = tq_.x, sq = tq_.y;
      qv.x *= qrn; qv.y *= qrn;
      uint32_t const q8 = v_e4m3x2(qv.x * cq - qv.y * sq, qv.x * sq + qv.y * cq);
      float2 const qd = v_e4m3x2_dec(q8);
      sm.qdec[w * 65 + 2 * lane] = qd.x; sm.qdec[w * 65 + 2 * lane + 1] = qd.y;
    }
    Q_ST(16 + 5);
  }
  asm volatile("cp.async.wait_group 0;" ::: "memory");
  v_fence_async();
  __syncthreads();
  asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
  Q_ST(6 + 16 * wg);
  uint8_t* const mrb = sm.mrb;
  uint32_t const mbpush = v_u32(&sm.mbm);
  if (wg == 0) {

    float mref[2] = {V_MINIT, V_MINIT};
    float lA = 0.f, lB = 0.f;
    int const tokA = 2 * w, tokB = 2 * w + 1;
    int const lowA = seq + tokA - window + 1, lowB = seq + tokB - window + 1;
    if (ntiles > 0) {
      uint32_t sc[32];
      uint32_t const kb = v_u32(sm.kt) + 64 * hk;
      wg_fence();
      wg_qk16_0(sc, qf[0], v_desc(kb));
      wg_qk16_1(sc, qf[1], v_desc(kb + 32));
      wg_commit();
      wg_wait0();
      Q_ST(7);
      int const tbase = p0, tlim = p1;
      if (tbase < lowB) {
#pragma unroll
        for (int e = 0; e < 2; e++) {
          int const pos = tbase + 2 * t + e;
          uint32_t const keep = e ? 0x0000FFFFu : 0xFFFF0000u, neg = e ? 0xFC000000u : 0x0000FC00u;
          if (pos < lowA) sc[0] = (sc[0] & keep) | neg;
          if (pos < lowB) sc[1] = (sc[1] & keep) | neg;
        }
      }
      if (tbase + 128 > tlim) {
#pragma unroll
        for (int j8 = 0; j8 < 16; j8++)
#pragma unroll
          for (int e = 0; e < 2; e++) {
            uint32_t const keep = e ? 0x0000FFFFu : 0xFFFF0000u, neg = e ? 0xFC000000u : 0x0000FC00u;
            if (tbase + 8 * j8 + 2 * t + e >= tlim) {
              sc[2 * j8] = (sc[2 * j8] & keep) | neg;
              sc[2 * j8 + 1] = (sc[2 * j8 + 1] & keep) | neg;
            }
          }
      }
      __half2 mA2 = v_h2(sc[0]), mB2 = v_h2(sc[1]);
#pragma unroll
      for (int j8 = 1; j8 < 16; j8++) { mA2 = __hmax2(mA2, v_h2(sc[2 * j8])); mB2 = __hmax2(mB2, v_h2(sc[2 * j8 + 1])); }
      float mxA = fmaxf(__low2float(mA2), __high2float(mA2)), mxB = fmaxf(__low2float(mB2), __high2float(mB2));
      mxA = fmaxf(mxA, __shfl_xor_sync(0xffffffffu, mxA, 1)); mxA = fmaxf(mxA, __shfl_xor_sync(0xffffffffu, mxA, 2));
      mxB = fmaxf(mxB, __shfl_xor_sync(0xffffffffu, mxB, 1)); mxB = fmaxf(mxB, __shfl_xor_sync(0xffffffffu, mxB, 2));
      mxA *= c2; mxB *= c2;
      if (mxA > mref[0] + 8.5f) mref[0] = ceilf(mxA * 8.f) * 0.125f - 8.f;
      if (mxB > mref[1] + 8.5f) mref[1] = ceilf(mxB * 8.f) * 0.125f - 8.f;
#if ZH_SCH
      __half2 const c8h = __float2half2_rn(8.f * c2);
      __half2 const lo2 = __float2half2_rn(1024.f);
      __half2 const bA = __float2half2_rn(mref[0] > -1.0e29f ? 1080.f - 8.f * mref[0] : -60000.f);
      __half2 const bB = __float2half2_rn(mref[1] > -1.0e29f ? 1080.f - 8.f * mref[1] : -60000.f);
#pragma unroll
      for (int j8 = 0; j8 < 16; j8++) {
        sc[2 * j8] = v_u(__hmax2(__hfma2(v_h2(sc[2 * j8]), c8h, bA), lo2));
        sc[2 * j8 + 1] = v_u(__hmax2(__hfma2(v_h2(sc[2 * j8 + 1]), c8h, bB), lo2));
      }
#else
      // OKC: exact exp. P = 2^(s*c - mref) (at most 2^8.5) = ex2(fma(s, c, -mref)); masked scores (-inf) give 0
      __half2 const c1h = __float2half2_rn(c2);
      __half2 const bA = __float2half2_rn(mref[0] > -1.0e29f ? -mref[0] : -60000.f);
      __half2 const bB = __float2half2_rn(mref[1] > -1.0e29f ? -mref[1] : -60000.f);
#pragma unroll
      for (int j8 = 0; j8 < 16; j8++) {
        sc[2 * j8] = v_ex2h2(v_u(__hfma2(v_h2(sc[2 * j8]), c1h, bA)));
        sc[2 * j8 + 1] = v_ex2h2(v_u(__hfma2(v_h2(sc[2 * j8 + 1]), c1h, bB)));
      }
#endif
      {
        int const rA = 16 * w + g, rB = rA + 8;
#pragma unroll
        for (int jj = 0; jj < 8; jj++) {
#if ZH_SCH
          uint32_t const wA = v_prmt(sc[4 * jj], sc[4 * jj + 2], 0x6420), wB = v_prmt(sc[4 * jj + 1], sc[4 * jj + 3], 0x6420);
#else
          uint32_t const wA = v_cvt_e4m3x2h(sc[4 * jj]) | (v_cvt_e4m3x2h(sc[4 * jj + 2]) << 16);
          uint32_t const wB = v_cvt_e4m3x2h(sc[4 * jj + 1]) | (v_cvt_e4m3x2h(sc[4 * jj + 3]) << 16);
#endif
          *reinterpret_cast<uint32_t*>(sm.pt + rA * 128 + ((jj ^ (rA & 7)) << 4) + 4 * t) = wA;
          *reinterpret_cast<uint32_t*>(sm.pt + rB * 128 + ((jj ^ (rB & 7)) << 4) + 4 * t) = wB;
          {
            float2 const a0 = v_e4m3x2_dec(wA), a1 = v_e4m3x2_dec(wA >> 16), c0 = v_e4m3x2_dec(wB), c1 = v_e4m3x2_dec(wB >> 16);
            lA += (a0.x + a0.y) + (a1.x + a1.y);
            lB += (c0.x + c0.y) + (c1.x + c1.y);
          }
        }
      }
    }
    lA += __shfl_xor_sync(0xffffffffu, lA, 1); lA += __shfl_xor_sync(0xffffffffu, lA, 2);
    lB += __shfl_xor_sync(0xffffffffu, lB, 1); lB += __shfl_xor_sync(0xffffffffu, lB, 2);
    v_fence_async();
    __threadfence_block();
    asm volatile("bar.arrive 1, 256;" ::: "memory");
    Q_ST(8);
    if (t == 0) {
      uint32_t d0, d1, b0, b1;
      int const owA = 4 * w + (g >> 2), owB = owA + 2, rib = g & 3;
      uint32_t const mbsrc = mbpush;
      asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(d0) : "r"(v_u32(mrb + r * Q_BLK + 1088 + rib * 4)), "r"(owA));
      asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(b0) : "r"(mbsrc), "r"(owA));
      asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(d1) : "r"(v_u32(mrb + r * Q_BLK + 1088 + rib * 4)), "r"(owB));
      asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(b1) : "r"(mbsrc), "r"(owB));
      asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.f32 [%0], %1, [%2];" :: "r"(d0), "f"(mref[0]), "r"(b0) : "memory");
      asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.f32 [%0], %1, [%2];" :: "r"(d1), "f"(mref[1]), "r"(b1) : "memory");
      asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.f32 [%0], %1, [%2];" :: "r"(d0 + 16), "f"(lA), "r"(b0) : "memory");
      asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.f32 [%0], %1, [%2];" :: "r"(d1 + 16), "f"(lB), "r"(b1) : "memory");
    }

    float nk_m = V_MINIT, nk_l = 0.f, nk_o0 = 0.f, nk_o1 = 0.f;
    {
      int const j = lane & 7, dq = 16 * (lane >> 3);
      float d = 0.f;
#pragma unroll
      for (int i = 0; i < 16; i++) d += sm.qdec[w * 65 + dq + i] * sm.kdec[j * 65 + dq + i];
      d += __shfl_xor_sync(0xffffffffu, d, 8);
      d += __shfl_xor_sync(0xffffffffu, d, 16);
      float const sj = j <= otok ? c2 * d : V_MINIT;
      float mx = sj;
      mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 1));
      mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 2));
      mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 4));
      float const pj = j <= otok ? v_ex2(sj - mx) : 0.f;
      float sl = pj;
      sl += __shfl_xor_sync(0xffffffffu, sl, 1);
      sl += __shfl_xor_sync(0xffffffffu, sl, 2);
      sl += __shfl_xor_sync(0xffffffffu, sl, 4);
      nk_m = mx; nk_l = sl;
      if (lane < 8) sm.pp[w * 8 + lane] = pj;
      __syncwarp();
#pragma unroll
      for (int jj = 0; jj < V_L; jj++) {
        float const p = sm.pp[w * 8 + jj];
        float2 const v = *reinterpret_cast<const float2*>(sm.vdec + jj * 64 + 2 * lane);
        nk_o0 += p * v.x; nk_o1 += p * v.y;
      }
    }
    Q_ST(9);
    asm volatile("{\n.reg .pred p;\nPHW_%=:\nmbarrier.try_wait.parity.shared::cta.b64 p, [%0], 0;\n@!p bra PHW_%=;\n}\n"
                 :: "r"(v_u32(&sm.mbm)) : "memory");
    Q_ST(10);
    {
      int const sl = lane & 15;
      float const ms = *reinterpret_cast<const float*>(mrb + sl * Q_BLK + 1088 + w * 4);
      float const ls = *reinterpret_cast<const float*>(mrb + sl * Q_BLK + 1104 + w * 4);
      float M = fmaxf(ms, nk_m);
#pragma unroll
      for (int o2 = 1; o2 < 16; o2 <<= 1) M = fmaxf(M, __shfl_xor_sync(0xffffffffu, M, o2));
      float const ws = v_ex2(ms - M);
      float L2 = ls * ws;
#pragma unroll
      for (int o2 = 1; o2 < 16; o2 <<= 1) L2 += __shfl_xor_sync(0xffffffffu, L2, o2);
      float const wn = v_ex2(nk_m - M);
      L2 += nk_l * wn;
      float b0 = nk_o0 * wn, b1 = nk_o1 * wn;
#pragma unroll
      for (int src = 0; src < 16; src++) {
        float const wt = __shfl_sync(0xffffffffu, ws, src);
        float2 const ov = *reinterpret_cast<const float2*>(mrb + src * Q_BLK + w * 272 + 8 * lane);
        b0 += ov.x * wt; b1 += ov.y * wt;
      }
      // Preserve mean-before-square RMS: unnormalized numerators can overflow when squared.
      float const inv = __fdividef(v_scale, L2);
      b0 *= inv; b1 *= inv;
      float ss = b0 * b0 + b1 * b1;
#pragma unroll
      for (int o2 = 16; o2 > 0; o2 >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o2);
      float const rn = rsqrtf(ss * (1.0f / V_D) + eps);
      float const s0 = __fdividef(1.f, 1.f + __expf(-__bfloat162float(gpre.x)));
      float const s1 = __fdividef(1.f, 1.f + __expf(-__bfloat162float(gpre.y)));
      *reinterpret_cast<__nv_bfloat162*>(out + (long long)otok * (V_HQ * V_D) + (oh0 + w) * V_D + 2 * lane) =
          __floats2bfloat162_rn(b0 * rn * s0, b1 * rn * s1);
    }
    Q_ST(11);
  } else {

    float o[32], la[32];
#pragma unroll
    for (int i = 0; i < 32; i++) { o[i] = 0.f; la[i] = 0.f; }
    uint32_t va[4][4];
    if (ntiles > 0) {
      uint32_t const vaddr = v_u32(sm.vt);
#pragma unroll
      for (int ks = 0; ks < 4; ks++) {
        int const key = 32 * ks + 8 * (lane >> 3) + (lane & 7);
        uint32_t r0, r1, r2, r3;
        v_ldsm4t(r0, r1, r2, r3, vaddr + key * 128 + ((((4 * hk + w) ^ (key & 7))) << 4));
        va[ks][0] = v_prmt(r0, r1, 0x6420); va[ks][1] = v_prmt(r0, r1, 0x7531);
        va[ks][2] = v_prmt(r2, r3, 0x6420); va[ks][3] = v_prmt(r2, r3, 0x7531);
      }
    }
    Q_ST(16 + 7);
    asm volatile("bar.sync 1, 256;" ::: "memory");
    Q_ST(16 + 8);
    if (ntiles > 0) {
      uint32_t const ones[4] = {0x38383838u, 0x38383838u, 0x38383838u, 0x38383838u};
      uint32_t const pb = v_u32(sm.pt);
      wg_fence();
#pragma unroll
      for (int ks = 0; ks < 4; ks++) wg_pv(o, va[ks], v_desc(pb + 32 * ks));
      wg_commit();
      wg_wait0();
    }
    Q_ST(16 + 9);

    Q_ST(16 + 10);
#pragma unroll
    for (int jj = 0; jj < 8; jj++) {
      int const ow = 2 * jj + (t >> 1);
      uint32_t mbr;
      asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(mbr) : "r"(mbpush), "r"(ow));
#pragma unroll
      for (int e = 0; e < 2; e++) {
        int const rib = 2 * (t & 1) + e;
        float const v0 = o[4 * jj + e], v1 = o[4 * jj + 2 + e];
        uint32_t dst;
        asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(dst) : "r"(v_u32(mrb + r * Q_BLK + rib * 272 + (16 * w + 2 * g) * 4)), "r"(ow));
        asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v2.f32 [%0], {%1, %2}, [%3];"
                     :: "r"(dst), "f"(v0), "f"(v1), "r"(mbr) : "memory");
        if (!VP2_LWG0 && g == 0 && (jj >> 1) == w) {
          uint32_t dl;
          asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(dl) : "r"(v_u32(mrb + r * Q_BLK + 1104 + rib * 4)), "r"(ow));
          asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.f32 [%0], %1, [%2];" :: "r"(dl), "f"(la[4 * jj + e]), "r"(mbr) : "memory");
        }
      }
    }
    Q_ST(16 + 11);
  }
}

void launch(const Params& p, cudaStream_t stream, bool pdl) {
  static bool configured = false;
  const int shared = sizeof(VP2Smem) + 1024;
  if (!configured) {
    cudaFuncSetAttribute(vp2_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared);
    cudaFuncSetAttribute(vp2_kernel, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
    configured = true;
  }
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(32, p.B, 1);
  cfg.blockDim = dim3(256, 1, 1);
  cfg.dynamicSmemBytes = shared;
  cfg.stream = stream;
  cudaLaunchAttribute attrs[2] = {};
  attrs[0].id = cudaLaunchAttributeClusterDimension;
  attrs[0].val.clusterDim.x = 16;
  attrs[0].val.clusterDim.y = 1;
  attrs[0].val.clusterDim.z = 1;
  attrs[1].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[1].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = attrs;
  cfg.numAttrs = pdl ? 2 : 1;
  cudaLaunchKernelEx(&cfg, vp2_kernel, p);
}
#undef V_L
#undef V_HQ
#undef V_HKV
#undef V_D
#undef V_QKVW
#undef V_MINIT
#undef ZH_SCH
#undef V_QK16_OUT
#undef VP2_PUSH
#undef VP2_LATEQ
#undef VP2_ST
#undef VP2_QSM
#undef VP2_LWG0
#undef VP2_MLB
#undef VP2_P64
#undef VP2_NOTF
#undef VP2_STRM
#undef VP2_WG1A
#undef Q_BLK
#undef Q_TX
#undef Q_ST
}   // namespace keell_vp2
#endif

#if ATT_VERIFY_WST
// Keell 890809cee421500b00126d084e81b338ef9181ab / ZhimaoL/sglang, Apache-2.0.
// The measured FP32-partial WST build, adapted to Params, coherent QKV reads and the layer's PDL edge.
// Its per-shape workspace is isolated from the base kernel's zero-at-rest accumulators.
namespace keell_wst {
#define ZH_SCH 1
#define OKC_WST_ACC32 1
#define DINL __device__ __forceinline__

DINL void wgmma_fence()  { asm volatile("wgmma.fence.sync.aligned;" ::: "memory"); }
DINL void wgmma_commit() { asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory"); }
template <int N> DINL void wgmma_wait() { asm volatile("wgmma.wait_group.sync.aligned %0;" :: "n"(N)); }

DINL void mbar_init(unsigned bar, unsigned count) {
    asm volatile("mbarrier.init.shared.b64 [%0], %1;" :: "r"(bar), "r"(count));
}
DINL void mbar_arrive(unsigned bar) {
    asm volatile("mbarrier.arrive.shared.b64 _, [%0];" :: "r"(bar) : "memory");
}
DINL void mbar_wait(unsigned bar, unsigned phase) {
    asm volatile(
        "{\n.reg .pred p;\nWAIT_%=:\n"
        "mbarrier.try_wait.parity.shared.b64 p, [%0], %1;\n"
        "@!p bra WAIT_%=;\n}\n" :: "r"(bar), "r"(phase) : "memory");
}
DINL void fence_proxy_async() { asm volatile("fence.proxy.async.shared::cta;" ::: "memory"); }
DINL void fence_barrier_init() { asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory"); }
DINL void cpa_mbar_arrive_noinc(unsigned bar) {
    asm volatile("cp.async.mbarrier.arrive.noinc.shared.b64 [%0];" :: "r"(bar) : "memory");
}
DINL void fence_u(unsigned& x) { asm volatile("" : "+r"(x)); }
DINL void fence_f(float& x)    { asm volatile("" : "+f"(x)); }
template <int N> DINL void fence_ubank(unsigned (&b)[N]) {
#pragma unroll
    for (int i = 0; i < N; i++) fence_u(b[i]);
}
template <int N> DINL void fence_fbank(float (&b)[N]) {
#pragma unroll
    for (int i = 0; i < N; i++) fence_f(b[i]);
}

// no-swizzle (interleaved core matrices) descriptor - the V images
// 128B-swizzled K-major descriptor (TMA SWIZZLE_128B tiles, 128B rows, 8-row groups 1024B)
DINL unsigned long long make_desc_sw128(unsigned saddr) {
    return (unsigned long long)((saddr >> 4) & 0x3FFF)
         | (1ull << 16)
         | ((unsigned long long)(1024 >> 4) << 32)
         | (1ull << 62);
}

template <int ACC>
DINL void wgmma_qk(unsigned* d, unsigned* a, unsigned long long db) {
    asm volatile(
        "{\n.reg .pred p;\nsetp.ne.b32 p, %37, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n128k32.f16.e4m3.e4m3\n"
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
        "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31},"
        "{%32,%33,%34,%35}, %36, p, 1, 1;\n}\n"
        : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3]), "+r"(d[4]), "+r"(d[5]), "+r"(d[6]), "+r"(d[7]),
          "+r"(d[8]), "+r"(d[9]), "+r"(d[10]), "+r"(d[11]), "+r"(d[12]), "+r"(d[13]), "+r"(d[14]), "+r"(d[15]),
          "+r"(d[16]), "+r"(d[17]), "+r"(d[18]), "+r"(d[19]), "+r"(d[20]), "+r"(d[21]), "+r"(d[22]), "+r"(d[23]),
          "+r"(d[24]), "+r"(d[25]), "+r"(d[26]), "+r"(d[27]), "+r"(d[28]), "+r"(d[29]), "+r"(d[30]), "+r"(d[31]),
          "+r"(a[0]), "+r"(a[1]), "+r"(a[2]), "+r"(a[3])
        : "l"(db), "n"(ACC));
}

DINL __half2 u32h2(unsigned v) { return *reinterpret_cast<__half2*>(&v); }
DINL unsigned h2u32(__half2 v) { return *reinterpret_cast<unsigned*>(&v); }
DINL unsigned prmt(unsigned a, unsigned b, unsigned sel) {
    unsigned r;
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(sel));
    return r;
}
// OKC: exact exp2 on a packed f16 pair (one MUFU per 2 scores) and the f16x2 -> e4m3x2 conversion of the P codes
DINL unsigned ex2_f16x2(unsigned x) {
    unsigned r;
    asm("ex2.approx.f16x2 %0, %1;" : "=r"(r) : "r"(x));
    return r;
}
DINL unsigned short cvt_e4m3x2(unsigned h2) {
    unsigned short r;
    asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(r) : "r"(h2));
    return r;
}
DINL int pi_lane(int lam) {
    return (lam & 1) | (((lam >> 2) & 3) << 1) | (((lam >> 1) & 1) << 3) | (lam & 16);
}

constexpr int KIMG = 16384;             // K image: [128 keys][128B] SW128 (both kv heads)
constexpr int VIMGH = 8192;             // per-head V image (pi-permuted, interleaved)
constexpr int STG = KIMG + 2 * VIMGH;   // 32KB per stage
constexpr int RAWB = 16384;             // raw V slab: [128 keys][128B] SW128
constexpr unsigned NEG = 0xFC00;
#define M_INIT (-30000.0f)
#ifndef OKC_WST_ACC32
#define OKC_WST_ACC32 1
#endif
// serving: 4 dims of one split-KV O partial row as fp32 (fp32 partials, or the bank's bf16 ones)
DINL float4 ld_part4(const float* __restrict__ ws_acc, long row, int dq) {
#if OKC_WST_ACC32
    return __ldcg(reinterpret_cast<const float4*>(ws_acc + row * 64 + dq));
#else
    const uint2 x = __ldcg(reinterpret_cast<const uint2*>(reinterpret_cast<const __nv_bfloat16*>(ws_acc) + row * 64 + dq));
    const __nv_bfloat162 u0 = *reinterpret_cast<const __nv_bfloat162*>(&x.x), u1 = *reinterpret_cast<const __nv_bfloat162*>(&x.y);
    return make_float4(__bfloat162float(u0.x), __bfloat162float(u0.y), __bfloat162float(u1.x), __bfloat162float(u1.y));
#endif
}

DINL void mask_lohi(unsigned (&s)[32], int limA, int limB, int lowA, int lowB, int t) {
#pragma unroll
    for (int j = 0; j < 16; j++) {
        int k0 = 2 * t + 8 * j;
        if (k0 > limA || k0 < lowA)         s[2 * j]     = (s[2 * j]     & 0xFFFF0000u) | NEG;
        if (k0 + 1 > limA || k0 + 1 < lowA) s[2 * j]     = (s[2 * j]     & 0x0000FFFFu) | (NEG << 16);
        if (k0 > limB || k0 < lowB)         s[2 * j + 1] = (s[2 * j + 1] & 0xFFFF0000u) | NEG;
        if (k0 + 1 > limB || k0 + 1 < lowB) s[2 * j + 1] = (s[2 * j + 1] & 0x0000FFFFu) | (NEG << 16);
    }
}

// Schraudolph's 2^y in e4m3 (exponent = floor(y) + 7, mantissa = 8 * frac(y), linear in frac: +0..6%
// mid-octave, inside e4m3's own 6.25% rounding). t = fma(s, 8c, 1144 - 8m) lands in [1024, 2048) where
// the f16 ulp is 1, so the f16 bit pattern's low byte IS the code; max(t, 1024) clamps code >= 0
// (masked = -inf -> 0). The row sums come from the PV wgmma (ones column), so l matches the P used.
DINL void wgmma_pv72(float* d, unsigned* a, unsigned long long db) {
    asm volatile(
        "{\n.reg .pred p;\nsetp.ne.b32 p, %41, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n72k32.f32.e4m3.e4m3\n"
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
        "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35},"
        "{%36,%37,%38,%39}, %40, p, 1, 1;\n}\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]),
          "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]),
          "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]),
          "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]),
          "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]),
          "+r"(a[0]), "+r"(a[1]), "+r"(a[2]), "+r"(a[3])
        : "l"(db), "n"(1));
}

constexpr int VIMG9 = 9216;             // per-head V^T image with the ones row (72 rows x 128B, SW128)

constexpr int STG2 = KIMG + RAWB;
DINL void cpa16(unsigned dst, const void* src) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(dst), "l"(src) : "memory");
}
// ---------------------------------------------------------------- wst_ct10 helpers
// QK first k-step, write-only accumulators (a fresh S per key block)
DINL void wgmma_qkw(unsigned* d, unsigned* a, unsigned long long db) {
    asm volatile(
        "{\n.reg .pred p;\nsetp.ne.b32 p, 0, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n128k32.f16.e4m3.e4m3\n"
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
        "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31},"
        "{%32,%33,%34,%35}, %36, p, 1, 1;\n}\n"
        : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3]), "=r"(d[4]), "=r"(d[5]), "=r"(d[6]), "=r"(d[7]),
          "=r"(d[8]), "=r"(d[9]), "=r"(d[10]), "=r"(d[11]), "=r"(d[12]), "=r"(d[13]), "=r"(d[14]), "=r"(d[15]),
          "=r"(d[16]), "=r"(d[17]), "=r"(d[18]), "=r"(d[19]), "=r"(d[20]), "=r"(d[21]), "=r"(d[22]), "=r"(d[23]),
          "=r"(d[24]), "=r"(d[25]), "=r"(d[26]), "=r"(d[27]), "=r"(d[28]), "=r"(d[29]), "=r"(d[30]), "=r"(d[31])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(db));
}
// softmax_sch without the accumulator rescale and the packing (both run after PV(kt-1) retires): mask, max, lazy
// reference (m, kc, al), e4m3 codes in place. Returns whether this warp's rows moved their reference.
DINL bool softmax_sch_a(unsigned (&s)[32], float (&m)[2], float (&al)[2], __half2 (&kc)[2], float qk2f,
                        __half2 qk8h2, int diag, int limA, int limB, int t) {
    if (diag) {
#pragma unroll
        for (int j = 0; j < 16; j++) {
            int k0 = 2 * t + 8 * j;
            if (k0 > limA)     s[2 * j]     = (s[2 * j]     & 0xFFFF0000u) | NEG;
            if (k0 + 1 > limA) s[2 * j]     = (s[2 * j]     & 0x0000FFFFu) | (NEG << 16);
            if (k0 > limB)     s[2 * j + 1] = (s[2 * j + 1] & 0xFFFF0000u) | NEG;
            if (k0 + 1 > limB) s[2 * j + 1] = (s[2 * j + 1] & 0x0000FFFFu) | (NEG << 16);
        }
    }
    __half2 t0v[8], t1v[8];
#pragma unroll
    for (int j = 0; j < 8; j++) {
        t0v[j] = __hmax2(u32h2(s[4 * j]), u32h2(s[4 * j + 2]));
        t1v[j] = __hmax2(u32h2(s[4 * j + 1]), u32h2(s[4 * j + 3]));
    }
#pragma unroll
    for (int st = 4; st > 0; st >>= 1)
#pragma unroll
        for (int j = 0; j < 8; j++) if (j < st) {
            t0v[j] = __hmax2(t0v[j], t0v[j + st]);
            t1v[j] = __hmax2(t1v[j], t1v[j + st]);
        }
    const float f0 = fmaxf(__half2float(__low2half(t0v[0])), __half2float(__high2half(t0v[0])));
    const float f1 = fmaxf(__half2float(__low2half(t1v[0])), __half2float(__high2half(t1v[0])));
    al[0] = 1.0f; al[1] = 1.0f;
    const bool ok = (f0 * qk2f <= m[0] + 0.75f) & (f1 * qk2f <= m[1] + 0.75f);
    const bool moved = !__all_sync(0xFFFFFFFFu, ok);
    if (moved) {
        float r0 = f0, r1 = f1;
        r0 = fmaxf(r0, __shfl_xor_sync(0xFFFFFFFFu, r0, 1));
        r0 = fmaxf(r0, __shfl_xor_sync(0xFFFFFFFFu, r0, 2));
        r1 = fmaxf(r1, __shfl_xor_sync(0xFFFFFFFFu, r1, 1));
        r1 = fmaxf(r1, __shfl_xor_sync(0xFFFFFFFFu, r1, 2));
        const float mn0 = ceilf(fmaxf(m[0], r0 * qk2f) * 8.0f) * 0.125f;   // 1/8-octave reference: kc exact in f16
        const float mn1 = ceilf(fmaxf(m[1], r1 * qk2f) * 8.0f) * 0.125f;
        al[0] = exp2f(m[0] - mn0);
        al[1] = exp2f(m[1] - mn1);
        m[0] = mn0; m[1] = mn1;
#if ZH_SCH
        kc[0] = __float2half2_rn(1144.0f - 8.0f * mn0);
        kc[1] = __float2half2_rn(1144.0f - 8.0f * mn1);
#else
        // OKC: kc = 8 - m (P = 2^(s*c - m + 8), the Schraudolph codes' scale); a row that has not seen a key keeps a finite -60000
        // (8 - m would overflow f16 to +inf, and -inf * c + inf is NaN, which would reach the e4m3 conversion)
        kc[0] = __float2half2_rn(mn0 < -20000.0f ? -60000.0f : 8.0f - mn0);
        kc[1] = __float2half2_rn(mn1 < -20000.0f ? -60000.0f : 8.0f - mn1);
#endif
    }
#if ZH_SCH
    const __half2 h1024 = __float2half2_rn(1024.0f);
#pragma unroll
    for (int j = 0; j < 32; j++)
        s[j] = h2u32(__hmax2(__hfma2(u32h2(s[j]), qk8h2, kc[j & 1]), h1024));
#else
    // OKC: exact exp. s[j] = f16x2 of 2^(s*c - m + 8), at most 2^8.75 = 430 (e4m3 max 448); masked scores (-inf) give 0
#pragma unroll
    for (int j = 0; j < 32; j++)
        s[j] = ex2_f16x2(h2u32(__hfma2(u32h2(s[j]), qk8h2, kc[j & 1])));
#endif
    return moved;
}

// softmax_sch_a for q_len 1: only row A (q rows g of warp 0) is live; row B registers (token 1, q = 0) hold exact
// zeros, which pack to e4m3 code 0 (P = 0), so they are left untouched.

// ---- attn-engine bal: wst_ct14b = wst_ct14 (FQ, PBF, in-kernel merge) with the work balanced over the batch (stream-K over requests) ----
template <int CORE = 0>
__device__ __forceinline__ void eng_bal_row(const float* __restrict__ ws_acc, const float* __restrict__ ws_m, const float* __restrict__ ws_l,
                                            long recb, int pst, int r, int dq, int np, float vsc, uint4 graw, float eps,
                                            __nv_bfloat16* __restrict__ op, unsigned gmask) {
    // one row, 8 threads x 8 dims: fold np partials (16 per batch, loads in flight together, online merge across batches)
    float M = -1e30f, Ls = 0.f, a[8];
#pragma unroll
    for (int i = 0; i < 8; i++) a[i] = 0.f;
#pragma unroll 1
    for (int p0 = 0; p0 < np; p0 += 6) {   // attn-next (bank 16)
        float mv[6], lv[6];
        float4 xv[6][2];
#pragma unroll
        for (int p = 0; p < 6; p++) {
            if (p0 + p < np) {
                const long row = recb + (long)(p0 + p) * pst + r;
                mv[p] = __ldcg(ws_m + row);
                lv[p] = __ldcg(ws_l + row);
                xv[p][0] = ld_part4(ws_acc, row, dq);
                xv[p][1] = ld_part4(ws_acc, row, dq + 4);
            } else {
                mv[p] = -1e30f; lv[p] = 0.f; xv[p][0] = xv[p][1] = make_float4(0.f, 0.f, 0.f, 0.f);
            }
        }
        float Mc = M;
#pragma unroll
        for (int p = 0; p < 6; p++) Mc = fmaxf(Mc, mv[p]);
        const float sc = exp2f(M - Mc);
        Ls *= sc;
#pragma unroll
        for (int i = 0; i < 8; i++) a[i] *= sc;
        M = Mc;
#pragma unroll
        for (int p = 0; p < 6; p++) {
            const float w = exp2f(mv[p] - M);
            Ls += lv[p] * w;
#pragma unroll
            for (int i = 0; i < 2; i++) {
                a[4 * i] += w * xv[p][i].x; a[4 * i + 1] += w * xv[p][i].y;
                a[4 * i + 2] += w * xv[p][i].z; a[4 * i + 3] += w * xv[p][i].w;
            }
        }
    }
    const float inv = vsc / Ls;
    float ss = 0.f;
#pragma unroll
    for (int i = 0; i < 8; i++) { a[i] *= inv; ss += a[i] * a[i]; }
    if constexpr (CORE) {   // serving: the riders' pre-norm attention output (no RMSNorm, no gate)
        uint4 c;
        unsigned* cu = reinterpret_cast<unsigned*>(&c);
#pragma unroll
        for (int i = 0; i < 4; i++) {
            const __nv_bfloat162 rr = __floats2bfloat162_rn(a[2 * i], a[2 * i + 1]);
            cu[i] = *reinterpret_cast<const unsigned*>(&rr);
        }
        *reinterpret_cast<uint4*>(op) = c;
        return;
    }
#pragma unroll
    for (int o2 = 1; o2 < 8; o2 <<= 1) ss += __shfl_xor_sync(gmask, ss, o2);   // the row's 8 lanes only
    const float rn = rsqrtf(ss * (1.0f / 64.f) + eps);
    const __nv_bfloat162* g = reinterpret_cast<const __nv_bfloat162*>(&graw);
    uint4 o;
    unsigned* ou = reinterpret_cast<unsigned*>(&o);
#pragma unroll
    for (int i = 0; i < 4; i++) {
        const float s0 = 1.f / (1.f + __expf(-__bfloat162float(g[i].x))), s1 = 1.f / (1.f + __expf(-__bfloat162float(g[i].y)));
        const __nv_bfloat162 rr = __floats2bfloat162_rn(a[2 * i] * rn * s0, a[2 * i + 1] * rn * s1);
        ou[i] = *reinterpret_cast<const unsigned*>(&rr);
    }
    *reinterpret_cast<uint4*>(op) = o;
}
// one WG: wait until every piece of request fb arrived (its counter is a multiple of 256), then fold rows j + k * nb (16 rows per pass)
template <int CORE = 0>
__device__ __forceinline__ void eng_bal_fold(const int* __restrict__ ctr, const float* __restrict__ ws_acc, const float* __restrict__ ws_m,
                                             const float* __restrict__ ws_l, const __nv_bfloat16* __restrict__ qkv, int qkv_stride,
                                             int gate_col0, __nv_bfloat16* __restrict__ out, float vsc, float eps,
                                             int fb, int wg, int j, int nb, int np, long recb, int L, int t128) {
    const int LG = L * 8;
    const int nk = j < LG ? (LG - j + nb - 1) / nb : 0;   // rows j + k nb of this folder (WG-uniform); np partials
    if (nk <= 0) return;
    const int kk = t128 >> 3, dq = (t128 & 7) * 8;       // 16 rows per pass, 8 threads per row
    if (t128 == 0) {
        unsigned v;
        do { asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(ctr) : "memory"); } while (v & 255u);
    }
    asm volatile("bar.sync %0, 128;" :: "r"(4 + wg) : "memory");
    const unsigned gm = 0xffu << (t128 & 24);
    for (int k = kk; k < nk; k += 16) {
        const int r = j + k * nb;
        const long tok_r = (long)fb * L + (r >> 3);
        const int qcol = (wg * 8 + (r & 7)) * 64 + dq;
        uint4 graw = make_uint4(0u, 0u, 0u, 0u);
        if constexpr (!CORE) graw = __ldcg(reinterpret_cast<const uint4*>(qkv + tok_r * qkv_stride + gate_col0 + qcol));
        __nv_bfloat16* op = out + tok_r * (16 * 64) + qcol;
        eng_bal_row<CORE>(ws_acc, ws_m, ws_l, recb, 2 * LG, r, dq, np, vsc, graw, eps, op, gm);
    }
}

template <int NST, int PF, int NPW, int RR, int DEC, int CORE = 0>
__global__ void __launch_bounds__(256 + 128 * NPW, 1) wst_ct14b(const __grid_constant__ Params p)
{
    const unsigned char* q8 = nullptr;
    const unsigned char* vc_ = p.v_cache;
    const unsigned char* kc_ = p.k_cache;
    const int* pt = p.page_table;
    const int* lens = p.seq_lens;
    float* ws_acc = p.ws_acc;
    float* ws_m = p.ws_l;
    float* ws_l = p.ws_acc + (gridDim.x + p.B) * 128 * 64;
    const float* k_scale = p.k_scale;
    const float* v_scale = p.v_scale;
    const float smqk_log2e = p.scale_log2;
    const int B = p.B, L = 8, maxP = p.max_pages, spl = 3, S = gridDim.x, window = 0;
    const __nv_bfloat16* qkv = p.qkv;
    const int qkv_stride = p.ld_qkv, gate_col0 = (16 + 2 * 2) * 64;
    __nv_bfloat16* out = p.out;
    int* tickets = p.ws_cnt;
    const long long* cache_loc = reinterpret_cast<const long long*>(p.cache_loc);
    const float rcp_k = __fdividef(1.f, __ldcg(k_scale)), rcp_v = __fdividef(1.f, __ldcg(v_scale)), eps = 1e-5f;
    pdl_wait();
    pdl_launch_dependents();
    constexpr int PDL = 0, FUSE = 0, EX = 0, MODE = 0, PFL2 = 0, ABL = 0, FQ = CORE ? 0 : 1, PBF = 1;
    constexpr int NPT = 128 * NPW;               // producer threads
    constexpr int NWP = 4 * NPW;                 // producer warps
    constexpr int VP = (128 + 4 * NWP - 1) / (4 * NWP);   // cp.async passes per tile (4 rows per warp per pass; K + V)
    const int tid = threadIdx.x;
    const int c = blockIdx.x, G = gridDim.x;
    const int LG = L * 8;

    extern __shared__ __align__(1024) char smem[];
    const unsigned sbase = (unsigned)__cvta_generic_to_shared(smem);
    const unsigned vimg_base = sbase + NST * STG2;      // 2 bufs x 2 heads x 9KB (72-row V^T images)
    const unsigned mb = vimg_base + 2 * 2 * VIMG9;
    const unsigned mfull = mb, mempty = mb + NST * 8;
    int* sP = reinterpret_cast<int*>(smem + NST * STG2 + 4 * VIMG9 + 2 * NST * 8 + 64);   // tile prefix over the batch [B + 1]
    if (tid < NST) {
        mbar_init(mfull + tid * 8, NPT);        // every producer's cp.async noinc
        mbar_init(mempty + tid * 8, 256);       // both consumer WGs
    }
    if (tid < 256) {                             // ones row (64) + zero rows (65-71) of all 4 V^T images
        const int img = tid >> 6, o = (tid & 63) * 16;
        const unsigned v = o < 128 ? 0x38383838u : 0u;
        *reinterpret_cast<uint4*>(smem + NST * STG2 + img * VIMG9 + 8192 + o) = make_uint4(v, v, v, v);
        fence_proxy_async();
    }
    if (tid >= 256 && tid < 288) {               // one producer warp: tiles per request from the device-side lengths -> prefix
        const int ln = tid & 31;
        int run = 0;
#pragma unroll 1
        for (int cb = 0; cb < B; cb += 32) {
            const int bb = cb + ln;
            int tl = 0;
            if (bb < B) { const int sb = lens[bb]; tl = (sb + L - (window ? max(0, sb - window + 1) : 0) + 127) >> 7; }
            int x = tl;
#pragma unroll
            for (int o = 1; o < 32; o <<= 1) { const int y = __shfl_up_sync(0xffffffffu, x, o); if (ln >= o) x += y; }
            if (bb < B) sP[bb + 1] = run + x;
            run += __shfl_sync(0xffffffffu, x, 31);
        }
        if (ln == 0) sP[0] = 0;
    }
    fence_barrier_init();
    __syncthreads();
    const int T = sP[B];
    const int Ge = min(G, T);
    if (c >= Ge) return;                         // T < G: no tiles here
    // virtual map: request b = [V(b), V(b + 1)), V(b) = sP[b] + bw b: bw setup units (no tile; a piece start costs about bw tiles of
    // consumer time) then its tiles; CTA c owns virtual [c W / Ge, (c + 1) W / Ge), W = T + bw B
    const int bw = spl;
    const int W = T + bw * B;
    auto VV = [&](int bb) { return sP[bb] + bw * bb; };
    auto cof = [&](int v) { return (int)(((long long)(v + 1) * Ge - 1) / W); };   // the CTA holding virtual unit v
    const int v0 = (int)(((long long)c * W) / Ge), v1 = (int)(((long long)(c + 1) * W) / Ge);
    int b0, tbs = 0, ntl = 0;
    {
        int lo_ = 0, hi_ = B - 1;                // largest b with V(b) <= v0
        while (lo_ < hi_) { const int mid = (lo_ + hi_ + 1) >> 1; if (VV(mid) <= v0) lo_ = mid; else hi_ = mid - 1; }
        int bf = -1;
        for (int bb = lo_; bb < B && VV(bb) + bw < v1; bb++) {
            const int r0 = VV(bb) + bw, x0 = max(v0, r0), x1 = min(v1, VV(bb + 1));
            if (x1 > x0) { if (bf < 0) { bf = bb; tbs = x0 - r0; } ntl += x1 - x0; }
        }
        b0 = bf;
    }
    if (ntl == 0) return;                        // only setup units here

    if (tid >= 256) {                            // ---------- producer: K and V rows by cp.async, the CTA's tiles back to back
        if (NPW == 1) asm volatile("setmaxnreg.dec.sync.aligned.u32 56;" ::: "memory");
        if (NPW >= 2 && NPW <= 4) asm volatile("setmaxnreg.dec.sync.aligned.u32 40;" ::: "memory");
        if (NPW >= 5) asm volatile("setmaxnreg.dec.sync.aligned.u32 32;" ::: "memory");
        const int ptid = tid - 256;
        const int pw = ptid >> 5, pl = ptid & 31;
        const int vsub = pl >> 3, vch = pl & 7;     // 4 rows per instruction, 8 lanes per row
        if (!CORE && ptid < 32 * L) {   // q rows of the later pieces (their q build follows a drain) + gate rows of every piece (fold) -> L2
            const int isg = ptid >= 16 * L, tq = (isg ? ptid - 16 * L : ptid) >> 4, ln_ = ptid & 15;
#pragma unroll 1
            for (int pb = isg ? b0 : b0 + 1; pb < B && VV(pb) + bw < v1; pb++)
                asm volatile("prefetch.global.L2 [%0];" :: "l"(qkv + ((long)pb * L + tq) * qkv_stride + (isg ? gate_col0 : 0) + ln_ * 64));
        }
        int pvs[PF][VP];
        int sb = b0, st_ = tbs, snt = sP[b0 + 1] - sP[b0];                // stage cursor (request, tile of it)
        int sseq = lens[b0], sbs = window ? max(0, sseq - window + 1) : 0;
        int fb = sb, ft = st_, fnt = snt, fseq = sseq, fbase = sbs;        // fetch cursor (runs PF tiles ahead)
        const int* fprow = pt + (long)fb * maxP;
        auto fetch = [&](int (&pv)[VP]) {
            const int p0 = fbase + ft * 128;
            const int last = fseq + L - 1 - p0;       // >= 0: every tile holds a valid row
#pragma unroll
            for (int i = 0; i < VP; i++) {
                const int r = i * 4 * NWP + pw * 4 + vsub;
                if (r < 128) pv[i] = fprow[p0 + min(r, last)];
            }
            if (++ft == fnt) {
                ++fb; ft = 0;
                if (fb < B) { fnt = sP[fb + 1] - sP[fb]; fseq = lens[fb]; fbase = window ? max(0, fseq - window + 1) : 0; fprow = pt + (long)fb * maxP; }
            }
        };
        auto stage = [&](int kt, const int (&pv)[VP]) {
            const int sl = kt % NST;
            const int round = kt / NST;
            if (round > 0) mbar_wait(mempty + sl * 8, (round - 1) & 1);
            if (MODE == 2) {
                mbar_arrive(mfull + sl * 8);
            } else {
                const unsigned st = sbase + sl * STG2;
#pragma unroll
                for (int i = 0; i < VP; i++) {
                    const int r = i * 4 * NWP + pw * 4 + vsub;
                    if (r < 128) {
                        const long src = (long)pv[i] * 128 + vch * 16;
                        cpa16(st + r * 128 + ((vch ^ (r & 7)) << 4), kc_ + src);
                        cpa16(st + KIMG + r * 128 + ((vch ^ ((r >> 4) & 7)) << 4), vc_ + src);
                    }
                }
                cpa_mbar_arrive_noinc(mfull + sl * 8);
            }
        };
#pragma unroll
        for (int u = 0; u < PF; u++) {
#pragma unroll
            for (int i = 0; i < VP; i++) pvs[u][i] = 0;
            if (u < ntl) fetch(pvs[u]);
        }
        bool nrdone = false;
        auto newrows = [&](int nrb) {   // unit = (token, k|v, kv head): 8 lanes x 8 dims; k RMSNorm / scale -> e4m3 -> cache
            const int nu = ptid >> 3, nd0 = (ptid & 7) * 8;
            const int ntok = nu >> 2, nisv = (nu >> 1) & 1, nkh = nu & 1;
            const bool act = nu < 4 * L;
            float x[8];
            if (act) {
                const uint4 raw = __ldcg(reinterpret_cast<const uint4*>(qkv + ((long)nrb * L + ntok) * qkv_stride
                                                                  + (16 + 2 * nisv + nkh) * 64 + nd0));
                const __nv_bfloat162* p2 = reinterpret_cast<const __nv_bfloat162*>(&raw);
#pragma unroll
                for (int i = 0; i < 4; i++) { x[2 * i] = __bfloat162float(p2[i].x); x[2 * i + 1] = __bfloat162float(p2[i].y); }
            } else {
#pragma unroll
                for (int i = 0; i < 8; i++) x[i] = 0.f;
            }
            float ss = 0.f;
#pragma unroll
            for (int i = 0; i < 8; i++) ss += x[i] * x[i];
            ss += __shfl_xor_sync(0xffffffffu, ss, 1);
            ss += __shfl_xor_sync(0xffffffffu, ss, 2);
            ss += __shfl_xor_sync(0xffffffffu, ss, 4);
            const float sc = nisv ? rcp_v : rsqrtf(ss * (1.0f / 64) + eps) * rcp_k;
            if (act) {
                uint2 q8v;
                q8v.x = (unsigned)__nv_cvt_float2_to_fp8x2(make_float2(x[0] * sc, x[1] * sc), __NV_SATFINITE, __NV_E4M3)
                      | ((unsigned)__nv_cvt_float2_to_fp8x2(make_float2(x[2] * sc, x[3] * sc), __NV_SATFINITE, __NV_E4M3) << 16);
                q8v.y = (unsigned)__nv_cvt_float2_to_fp8x2(make_float2(x[4] * sc, x[5] * sc), __NV_SATFINITE, __NV_E4M3)
                      | ((unsigned)__nv_cvt_float2_to_fp8x2(make_float2(x[6] * sc, x[7] * sc), __NV_SATFINITE, __NV_E4M3) << 16);
                unsigned char* dst = const_cast<unsigned char*>(nisv ? vc_ : kc_) + cache_loc[(long)nrb * L + ntok] * 128 + nkh * 64 + nd0;
                *reinterpret_cast<uint2*>(dst) = q8v;
            }
            __threadfence();
            asm volatile("bar.sync 8, %0;" :: "r"(NPT) : "memory");   // every producer thread: rows in L2 before staging
        };
        for (int kt0 = 0; kt0 < ntl; kt0 += PF) {
#pragma unroll
            for (int u = 0; u < PF; u++) {
                const int kt = kt0 + u;
                if (kt < ntl) {
                    if (!CORE && !nrdone && sbs + st_ * 128 + 127 >= sseq) { newrows(sb); nrdone = true; }   // this piece holds new positions
                    stage(kt, pvs[u]);
                    if (kt + PF < ntl) fetch(pvs[u]);
                    if (++st_ == snt) {
                        ++sb; st_ = 0; nrdone = false;
                        if (sb < B) { snt = sP[sb + 1] - sP[sb]; sseq = lens[sb]; sbs = window ? max(0, sseq - window + 1) : 0; }
                    }
                }
            }
        }
        return;
    }

    // ---------- consumer warpgroups (WG w = kv head w)
    if (NPW == 1) asm volatile("setmaxnreg.inc.sync.aligned.u32 216;" ::: "memory");
    if (NPW == 2) asm volatile("setmaxnreg.inc.sync.aligned.u32 200;" ::: "memory");
    if (NPW == 3) asm volatile("setmaxnreg.inc.sync.aligned.u32 176;" ::: "memory");
    if (NPW == 4) asm volatile("setmaxnreg.inc.sync.aligned.u32 160;" ::: "memory");
    if (NPW == 5) asm volatile("setmaxnreg.inc.sync.aligned.u32 168;" ::: "memory");
    if (NPW == 6) asm volatile("setmaxnreg.inc.sync.aligned.u32 160;" ::: "memory");
    const int wg = tid >> 7;
    const int wiw = (tid >> 5) & 3;
    const int lane = tid & 31;
    const int t = lane & 3;
    const int h = lane >> 2;
    const int t128 = tid & 127;
    const int kcv = t128 & 7;                            // 16-key group: lanes 0-7 of each octet
    const int d0 = 16 * (t128 >> 5) + 4 * ((t128 >> 3) & 3);   // 4-dim window
    const int cbyte = 64 * wg + d0;

    const float qk2f = k_scale[0] * smqk_log2e;
    const __half2 qk2h2 = __float2half2_rn(qk2f);
    const int tok0 = 2 * wiw;
    int gb = 0;                                  // ring position (CTA-local tile index) of the piece's first tile
#pragma unroll 1
    for (int b = b0, tb0 = tbs; b < B && VV(b) + bw < v1; b++, tb0 = 0) {   // pieces: the requests with tiles in [v0, v1)
    const int tb1 = min(v1, VV(b + 1)) - (VV(b) + bw);
    if (tb1 <= tb0) continue;
    const int nst_total = tb1 - tb0;
    const int seqb = lens[b];
    const int tot = seqb + L;
    const int base = window ? max(0, seqb - window + 1) : 0;   // first position token 0 sees
    const int lo = base + tb0 * 128;
    const int hi = min(tot, base + tb1 * 128);
    unsigned qf[8];
    if constexpr (FQ) {   // dims 16 c4 + 4t .. +3 (c4 = 2 sl2 + hi) of q head 8 wg + h, tokens tok0, tok0 + 1
        float x[2][16];
#pragma unroll
        for (int rr = 0; rr < 2; rr++) {
            const int tok = tok0 + rr;
            const __nv_bfloat16* qp = qkv + ((long)b * L + tok) * qkv_stride + (wg * 8 + h) * 64 + 4 * t;
#pragma unroll
            for (int c4 = 0; c4 < 4; c4++) {
                uint2 raw = make_uint2(0u, 0u);
                if (tok < L) raw = __ldcg(reinterpret_cast<const uint2*>(qp + 16 * c4));
                const __nv_bfloat162 u0 = *reinterpret_cast<const __nv_bfloat162*>(&raw.x);
                const __nv_bfloat162 u1 = *reinterpret_cast<const __nv_bfloat162*>(&raw.y);
                x[rr][4 * c4 + 0] = __bfloat162float(u0.x); x[rr][4 * c4 + 1] = __bfloat162float(u0.y);
                x[rr][4 * c4 + 2] = __bfloat162float(u1.x); x[rr][4 * c4 + 3] = __bfloat162float(u1.y);
            }
        }
#pragma unroll
        for (int rr = 0; rr < 2; rr++) {
            float ss = 0.f;
#pragma unroll
            for (int i = 0; i < 16; i++) ss += x[rr][i] * x[rr][i];
            ss += __shfl_xor_sync(0xffffffffu, ss, 1);
            ss += __shfl_xor_sync(0xffffffffu, ss, 2);
            const float rn = rsqrtf(ss * (1.0f / 64) + eps);
#pragma unroll
            for (int sl2 = 0; sl2 < 2; sl2++)
#pragma unroll
                for (int hi2 = 0; hi2 < 2; hi2++) {
                    const int c4 = 2 * sl2 + hi2;
                    qf[4 * sl2 + rr + 2 * hi2] =
                        (unsigned)__nv_cvt_float2_to_fp8x2(make_float2(x[rr][4 * c4] * rn, x[rr][4 * c4 + 1] * rn), __NV_SATFINITE, __NV_E4M3)
                        | ((unsigned)__nv_cvt_float2_to_fp8x2(make_float2(x[rr][4 * c4 + 2] * rn, x[rr][4 * c4 + 3] * rn), __NV_SATFINITE, __NV_E4M3) << 16);
                }
        }
    } else {
#pragma unroll
    for (int sl2 = 0; sl2 < 2; sl2++)
#pragma unroll
        for (int rr = 0; rr < 2; rr++) {
            const int tok = tok0 + rr;
            const unsigned char* qp = q8 + (((long)b * L + tok) * 16 + wg * 8 + h) * 64
                                    + sl2 * 32 + 4 * t;
            qf[4 * sl2 + rr]     = (tok < L) ? *reinterpret_cast<const unsigned*>(qp) : 0u;
            qf[4 * sl2 + rr + 2] = (tok < L) ? *reinterpret_cast<const unsigned*>(qp + 16) : 0u;
        }
    }

    float m[2] = {M_INIT, M_INIT}, al[2];
    __half2 kc[2] = {__float2half2_rn(-60000.0f), __float2half2_rn(-60000.0f)};
    const __half2 qk8h2 = ZH_SCH ? __float2half2_rn(8.0f * qk2f) : qk2h2;   // OKC: exact exp multiplies by c itself
    float acc[36];
#pragma unroll
    for (int i = 0; i < 36; i++) acc[i] = 0.f;
    float alc[2] = {1.f, 1.f};   // serving: lazy rescales since the last fold (rows A, B)
    bool fl = false;
    float* const fsm = reinterpret_cast<float*>(smem + (NST * STG2 + 4 * VIMG9 + 2 * NST * 8 + 64 + ((4 * (B + 1) + 15) & ~15))) + wg * 128 + t128;
    unsigned sv[32] = {}, pk[16] = {};

    const int vA = min(hi - 1, seqb + tok0);
    const int vB = min(hi - 1, seqb + tok0 + 1);
    const int wA = window ? (seqb + tok0 - window + 1) : -0x40000000;      // sliding lower bounds
    const int wB = window ? (seqb + tok0 + 2 - window) : -0x40000000;
    if (nst_total > 0) {   // FA3 order (prefill owner's fmha4): nothing pending across a back edge or branch merge
        unsigned voff[16], vsto[4];
#pragma unroll
        for (int ww = 0; ww < 4; ww++)
#pragma unroll
            for (int q = 0; q < 4; q++) {
                const int lamb = kcv * 16 + ww * 4;
                const int kk = ((lamb + q) & ~31) | pi_lane((lamb + q) & 31);
                voff[4 * ww + q] = KIMG + kk * 128 + ((((cbyte >> 4) ^ ((kk >> 4) & 7)) << 4) | (cbyte & 15));
            }
#pragma unroll
        for (int rr = 0; rr < 4; rr++) vsto[rr] = (d0 + rr) * 128 + ((kcv ^ ((d0 + rr) & 7)) << 4);
        fence_ubank(voff);
        fence_ubank(vsto);
        const unsigned long long dq0 = make_desc_sw128(sbase + 64 * wg);
        const unsigned long long dv0 = make_desc_sw128(vimg_base + wg * VIMG9);
        auto qk_issue = [&](unsigned (&s_)[32], int kt) {
            if (ABL & 64) return;
            const unsigned long long dq = dq0 + (unsigned long long)(((gb + kt) % NST) * (STG2 >> 4));
            wgmma_fence();
            wgmma_qkw(s_, qf, dq);
            wgmma_qk<1>(s_, qf + 4, dq + 2);
            wgmma_commit();
        };
        auto pv_issue = [&](int kt) {
            if (ABL & 4) return;
            const unsigned long long dv = dv0 + (unsigned long long)(((gb + kt) & 1) * (2 * VIMG9 >> 4));
            fence_fbank(acc);
            fence_ubank(pk);
            wgmma_fence();
#pragma unroll
            for (int js = 0; js < 4; js++) wgmma_pv72(acc, pk + 4 * js, dv + 2 * js);
            wgmma_commit();
        };
        auto vtrans = [&](int kt) {   // own V half of tile kt: raw (SW128) -> V^T image (gb + kt) & 1
            if (ABL & 1) {
                if (!(ABL & 4096)) fence_proxy_async();
                if (!(ABL & 8192)) asm volatile("bar.sync %0, 128;" :: "r"(4 + wg) : "memory");
                return;
            }
            const char* rb = smem + ((gb + kt) % NST) * STG2;
            unsigned w[4][4];
#pragma unroll
            for (int ww = 0; ww < 4; ww++) {
                unsigned lv[4];
#pragma unroll
                for (int q = 0; q < 4; q++) lv[q] = *reinterpret_cast<const unsigned*>(rb + voff[4 * ww + q]);
                const unsigned t0 = prmt(lv[0], lv[1], 0x5140);
                const unsigned t1 = prmt(lv[0], lv[1], 0x7362);
                const unsigned t2 = prmt(lv[2], lv[3], 0x5140);
                const unsigned t3 = prmt(lv[2], lv[3], 0x7362);
                w[ww][0] = prmt(t0, t2, 0x5410);
                w[ww][1] = prmt(t0, t2, 0x7632);
                w[ww][2] = prmt(t1, t3, 0x5410);
                w[ww][3] = prmt(t1, t3, 0x7632);
            }
            char* vimg = smem + NST * STG2 + (((gb + kt) & 1) * 2 + wg) * VIMG9;
#pragma unroll
            for (int rr = 0; rr < 4; rr++)
                *reinterpret_cast<uint4*>(vimg + vsto[rr]) = make_uint4(w[0][rr], w[1][rr], w[2][rr], w[3][rr]);
            fence_proxy_async();
            asm volatile("bar.sync %0, 128;" :: "r"(4 + wg) : "memory");
        };
        bool resc = false;
        auto sm_a = [&](unsigned (&s_)[32], int kt) {   // mask, max, lazy reference, e4m3 codes in place
            const int p0 = lo + kt * 128;
            const int limA = vA - p0, limB = vB - p0;
            const int lowA = wA - p0, lowB = wB - p0;
            const int diag = (limA < 127) | (limB < 127);
            const int wdiag = (lowA > 0) | (lowB > 0);
            if (wdiag) mask_lohi(s_, limA, limB, lowA, lowB, t);
            resc = softmax_sch_a(s_, m, al, kc, qk2f, qk8h2, diag & !wdiag, limA, limB, t);
        };
        auto sm_b = [&](unsigned (&s_)[32]) {   // after PV(kt-1): rescale acc, pack the codes as the next A operand
            if (resc) {
#pragma unroll
                for (int j = 0; j < 9; j++) {
                    acc[4 * j]     *= al[0]; acc[4 * j + 1] *= al[0];
                    acc[4 * j + 2] *= al[1]; acc[4 * j + 3] *= al[1];
                    if (j < 1) { alc[0] *= al[0]; alc[1] *= al[1]; }
                }
            }
#pragma unroll
            for (int js = 0; js < 4; js++) {
#if ZH_SCH
                pk[4 * js]     = prmt(s_[8 * js],     s_[8 * js + 2], 0x6420);
                pk[4 * js + 1] = prmt(s_[8 * js + 1], s_[8 * js + 3], 0x6420);
                pk[4 * js + 2] = prmt(s_[8 * js + 4], s_[8 * js + 6], 0x6420);
                pk[4 * js + 3] = prmt(s_[8 * js + 5], s_[8 * js + 7], 0x6420);
#else   // OKC: the f16 exps -> e4m3 pairs (cvt.rn.satfinite), two per register half
                pk[4 * js]     = (unsigned)cvt_e4m3x2(s_[8 * js])     | ((unsigned)cvt_e4m3x2(s_[8 * js + 2]) << 16);
                pk[4 * js + 1] = (unsigned)cvt_e4m3x2(s_[8 * js + 1]) | ((unsigned)cvt_e4m3x2(s_[8 * js + 3]) << 16);
                pk[4 * js + 2] = (unsigned)cvt_e4m3x2(s_[8 * js + 4]) | ((unsigned)cvt_e4m3x2(s_[8 * js + 6]) << 16);
                pk[4 * js + 3] = (unsigned)cvt_e4m3x2(s_[8 * js + 5]) | ((unsigned)cvt_e4m3x2(s_[8 * js + 7]) << 16);
#endif
            }
        };
        const bool live = !DEC || tok0 < L;      // DEC (q_len 1): a warp whose 16 q rows are all tokens >= L skips
        {   // kt = 0                               // the softmax (its P stays 0)
            unsigned s0[32];
            mbar_wait(mfull + (gb % NST) * 8, (gb / NST) & 1);
            fence_proxy_async();                    // K landed by cp.async (generic proxy) -> wgmma reads
            qk_issue(s0, 0);
            vtrans(0);
            wgmma_wait<0>();
            fence_ubank(s0);
            mbar_arrive(mempty + (gb % NST) * 8);
            if (live) {
                sm_a(s0, 0);
                sm_b(s0);
            }
        }
        if (RR && nst_total > 1 && wg == 1) asm volatile("bar.arrive 6, 256;" ::: "memory");   // WG0 issues first
        for (int kt = 1; kt < nst_total; kt++) {
            const int sl_ = (gb + kt) % NST;
            unsigned s1[32];
            mbar_wait(mfull + sl_ * 8, ((gb + kt) / NST) & 1);
            if (!(ABL & 8)) fence_proxy_async();
            if (RR) asm volatile("bar.sync %0, 256;" :: "r"(6 + wg) : "memory");
            qk_issue(s1, kt);
            if ((kt & 15) == 0) {   // serving: fp32 fold, the wgmma accumulator restarts at 0
#pragma unroll
                for (int i = 0; i < 36; i++) {
                    float* const p_ = fsm + i * 256;
                    *p_ = fl ? fmaf(*p_, alc[(i >> 1) & 1], acc[i]) : acc[i];
                    acc[i] = 0.f;
                }
                alc[0] = 1.f; alc[1] = 1.f; fl = true;
            }
            pv_issue(kt - 1);
            if (RR) asm volatile("bar.arrive %0, 256;" :: "r"(6 + (wg ^ 1)) : "memory");
            vtrans(kt);
            wgmma_wait<1>();
            fence_ubank(s1);
            mbar_arrive(mempty + sl_ * 8);          // K read, raw V transposed
            if (live && !(ABL & 2)) sm_a(s1, kt);
            fence_ubank(s1);                        // codes done before the PV(kt-1) wait (overlap the PV wgmma)
            wgmma_wait<0>();
            fence_fbank(acc);
            fence_ubank(pk);
            if (live && !(ABL & 2)) sm_b(s1);
        }
        if (RR && nst_total > 1 && wg == 0) asm volatile("bar.sync 6, 256;" ::: "memory");   // WG1's last arrive
        pv_issue(nst_total - 1);
        wgmma_wait<0>();
        if (fl) {
#pragma unroll
            for (int i = 0; i < 36; i++) acc[i] = fmaf(fsm[i * 256], alc[(i >> 1) & 1], acc[i]);
        }
    }
    fence_ubank(pk);
    fence_fbank(acc);
    {   // attn-engine bal: this piece's bf16 O partial -> slot c + b (staged in this WG's idle V^T image), then arrive on (b, kv head)
        const long pbase = ((long)(c + b) * 2 + wg) * LG;
#if OKC_WST_ACC32   // serving: fp32 O partial, one 8-B store per column pair from the accumulators
#pragma unroll
        for (int rr = 0; rr < 2; rr++) {
            const int tok = tok0 + rr;
            if (tok >= L) continue;
            const long row = pbase + tok * 8 + h;
            if (t == 0) {
                ws_m[row] = m[rr];
                ws_l[row] = acc[32 + 2 * rr];
            }
            float* op = ws_acc + row * 64 + 2 * t;
#pragma unroll
            for (int j = 0; j < 8; j++)
                *reinterpret_cast<float2*>(op + 8 * j) = make_float2(acc[4 * j + 2 * rr], acc[4 * j + 2 * rr + 1]);
        }
#else
        char* stg = smem + NST * STG2 + wg * VIMG9;
#pragma unroll
        for (int rr = 0; rr < 2; rr++) {
            const int tok = tok0 + rr;
            if (tok >= L) continue;
            const int r = tok * 8 + h;
#pragma unroll
            for (int j = 0; j < 8; j++)
                *reinterpret_cast<__nv_bfloat162*>(stg + r * 144 + (8 * j + 2 * t) * 2) =
                    __floats2bfloat162_rn(acc[4 * j + 2 * rr], acc[4 * j + 2 * rr + 1]);
            if (t == 0) {
                const long row = pbase + r;
                ws_m[row] = m[rr];
                ws_l[row] = acc[32 + 2 * rr];
            }
        }
        asm volatile("bar.sync %0, 128;" :: "r"(4 + wg) : "memory");
        __nv_bfloat16* gdst = reinterpret_cast<__nv_bfloat16*>(ws_acc) + pbase * 64;
        for (int i = t128; i < LG * 8; i += 128) {
            const int r = i >> 3, cc = i & 7;
            *reinterpret_cast<uint4*>(gdst + r * 64 + cc * 8) = *reinterpret_cast<const uint4*>(stg + r * 144 + cc * 16);
        }
        asm volatile("bar.sync %0, 128;" :: "r"(4 + wg) : "memory");   // staging reads done (the arrive waits for the CTA's last piece)
        // the staging overwrote rows 64-71 of this image (ones row + zero rows): restore them for the next piece's PV
        *reinterpret_cast<uint2*>(stg + 8192 + t128 * 8) = t128 < 16 ? make_uint2(0x38383838u, 0x38383838u) : make_uint2(0u, 0u);
        fence_proxy_async();
#endif
    }
    gb += nst_total;
    }   // pieces
    asm volatile("bar.sync %0, 128;" :: "r"(4 + wg) : "memory");   // every partial + m / l store of this WG issued
    if (t128 == 0) {   // one release for all pieces, then the per-request arrivals (+1, last piece + 256 - n_b + 1)
        asm volatile("fence.acq_rel.gpu;" ::: "memory");
#pragma unroll 1
        for (int ab = b0; ab < B && VV(ab) + bw < v1; ab++) {
            if (min(v1, VV(ab + 1)) <= max(v0, VV(ab) + bw)) continue;   // no tiles of ab here
            const int cf = cof(VV(ab) + bw), cl = cof(VV(ab + 1) - 1);
            const unsigned inc = c == cl ? (unsigned)(256 - (cl - cf + 1) + 1) : 1u;
            asm volatile("red.relaxed.gpu.global.add.u32 [%0], %1;" :: "l"(tickets + ab * 2 + wg), "r"(inc) : "memory");
        }
    }
    const float vsc = v_scale[0];
#pragma unroll 1
    for (int fb = b0; fb < B && VV(fb) + bw < v1; fb++) {   // fold: the interior pieces of each request (1-piece CTAs) fold its rows
        if (min(v1, VV(fb + 1)) <= max(v0, VV(fb) + bw)) continue;
        const int cf = cof(VV(fb) + bw), cl = cof(VV(fb + 1) - 1);
        const int nb = cl - cf + 1, j = c - cf;
        const int jlo = nb >= 3 ? 1 : 0, jhi = nb >= 3 ? nb - 2 : nb - 1;
        if (j < jlo || j > jhi) continue;
        eng_bal_fold<CORE>(tickets + fb * 2 + wg, ws_acc, ws_m, ws_l, qkv, qkv_stride, gate_col0, out, vsc, eps, fb, wg, j - jlo, jhi - jlo + 1, nb,
                     ((long)(cf + fb) * 2 + wg) * LG, L, t128);
    }
}


void launch(const Params& p, cudaStream_t stream, bool pdl) {
  static int sms = 0;
  auto kernel = wst_ct14b<4, 2, 4, 1, 0, 0>;
  const int shared = 4 * STG2 + 4 * VIMG9 + 2 * 4 * 8 + 64 + ((4 * (p.B + 1) + 15) & ~15) + 36864;
  if (!sms) {
    int device;
    cudaGetDevice(&device);
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device);
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared);
  }
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(sms, 1, 1);
  cfg.blockDim = dim3(256 + 128 * 4, 1, 1);
  cfg.dynamicSmemBytes = shared;
  cfg.stream = stream;
  cudaLaunchAttribute attrs[2] = {};
  attrs[0].id = cudaLaunchAttributeCooperative;
  attrs[0].val.cooperative = 1;
  attrs[1].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[1].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = attrs;
  cfg.numAttrs = pdl ? 2 : 1;
  cudaLaunchKernelEx(&cfg, kernel, p);
}

#undef DINL
#undef M_INIT
#undef OKC_WST_ACC32
#undef ZH_SCH
}   // namespace keell_wst
#endif

}   // namespace

template <int L>
void launch(const Params& p, dim3 grid, cudaStream_t stream, bool pdl, int variant) {
#if ATT_VERIFY_WST
  if constexpr (L == 8) {
    if (p.B == 16 && !p.window && p.P == 1) {
      keell_wst::launch(p, stream, pdl);
      return;
    }
  }
#endif
#if ATT_VERIFY_CLUSTER == 1
  if constexpr (L == 8) {
    if (p.B == 1 && p.window == 2048 && p.P == 1 && p.off32) {
      launch_verify_cluster(p, stream, pdl);
      return;
    }
  }
#elif ATT_VERIFY_CLUSTER == 2
  if constexpr (L == 8) {
    if (p.B == 1 && p.window == 2048 && p.P == 1 && p.off32) {
      keell_vp2::launch(p, stream, pdl);
      return;
    }
  }
#endif
  typedef void (*KernFn)(const Params);
  // per-request kernels by launch mode: plain split-KV (MODE 1) or cooperative finalize (3); | 4: every split holds
  // at most one tile; | 8: one request, page size 1
  static const KernFn kn[4] = {nullptr, attn_kernel<L, false, 1>, nullptr, attn_kernel<L, false, 3>};
  static const KernFn kn1[4] = {nullptr, attn_kernel<L, false, 5>, nullptr, attn_kernel<L, false, 7>};
  static const KernFn ksp[4] = {nullptr, attn_kernel<L, false, 9>, nullptr, attn_kernel<L, false, 11>};
  static const KernFn ksp1[4] = {nullptr, attn_kernel<L, false, 13>, nullptr, attn_kernel<L, false, 15>};
  static bool init = false;
  static int occ_c = 0, occ_f = 0, nsm = 0, occ_c1 = 0, occ_sp = 0, occ_sp1 = 0;
  const int sm_f = (int)sizeof(Smem<L, true>) + 1024, sm_n = (int)sizeof(Smem<L, false>) + 1024;
  if (!init) {
    for (const KernFn* t : {kn, kn1, ksp, ksp1})
      for (int m = 1; m < 4; m += 2) cudaFuncSetAttribute(t[m], cudaFuncAttributeMaxDynamicSharedMemorySize, sm_n);
    cudaFuncSetAttribute(attn_kernel<L, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, sm_f);
    if constexpr (L == 1) {
      cudaFuncSetAttribute(attn_kernel<1, false, 13 | 16>, cudaFuncAttributeMaxDynamicSharedMemorySize, sm_n);
      cudaFuncSetAttribute(attn_kernel<1, false, 1 | 32>, cudaFuncAttributeMaxDynamicSharedMemorySize, sm_n);
    } else {
      cudaFuncSetAttribute(attn_kernel<8, true, 64>, cudaFuncAttributeMaxDynamicSharedMemorySize, sm_f);
      cudaFuncSetAttribute(attn_kernel<8, false, 15 | 128>, cudaFuncAttributeMaxDynamicSharedMemorySize, sm_n);
    }
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ_c, kn[3], Cfg<L, false>::NTX, sm_n);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ_c1, kn1[3], Cfg<L, false>::NTX, sm_n);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ_sp, ksp[3], Cfg<L, false>::NTX, sm_n);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ_sp1, ksp1[3], Cfg<L, false>::NTX, sm_n);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ_f, attn_kernel<L, true>, Cfg<L, true>::NTX, sm_f);
    int dev;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev);
    init = true;
  }
  // the cooperative finalize waits for every split of a request: only when the whole grid is resident
  Params q = p;
  if (q.coop && (long long)grid.x * grid.y * grid.z > (long long)(p.flat ? occ_f : occ_c) * nsm) q.coop = 0;
  q.resident = (p.flat && grid.y * grid.z == 1 && (int)grid.x == 2 * nsm && occ_f >= 2) ? 1 : 0;
  KernFn kern = p.flat ? (KernFn)attn_kernel<L, true> : kn[q.coop ? 3 : 1];
  if (!p.flat) {
    // every split holds at most one tile: whole tiles per split and at least as many splits as the longest
    // request's tiles (sliding windows: window + L - 1 keys)
    const long long kmax = p.window ? std::min<long long>((long long)p.window + L - 1, (long long)p.max_pages * p.P)
                                    : (long long)p.max_pages * p.P;
    if ((long long)p.n_splits * 64 >= kmax) kern = kn1[q.coop ? 3 : 1];
  }
  // one request, page size 1: the speculative-length builds (the kernel verifies and falls back)
  if (!p.flat && p.B == 1 && p.P == 1 && p.max_pages >= L && (int)grid.y == 1) {
    const bool one = kern != kn[q.coop ? 3 : 1];
    kern = one ? ksp1[q.coop ? 3 : 1] : ksp[q.coop ? 3 : 1];
  }
  if (q.coop && !p.flat) {
    // the instance actually launched must hold the whole grid, else its grid barrier would wait forever
    int occ_k = occ_c;
    KernFn plain = kn[1];
    if (kern == kn1[3]) {
      occ_k = occ_c1;
      plain = kn1[1];
    }
    if (kern == ksp[3]) {
      occ_k = occ_sp;
      plain = ksp[1];
    }
    if (kern == ksp1[3]) {
      occ_k = occ_sp1;
      plain = ksp1[1];
    }
    if ((long long)grid.x * grid.y * grid.z > (long long)occ_k * nsm) {
      q.coop = 0;
      kern = plain;
    }
  }
  // the per-shape-class builds (MODE bits 16..128), each of the one kernel its shape class launches
  if constexpr (L == 1) {
    if (variant == 16 && kern == ksp1[1]) kern = attn_kernel<1, false, 13 | 16>;
    if (variant == 32 && kern == kn[1]) kern = attn_kernel<1, false, 1 | 32>;
  } else {
    if (variant == 64 && kern == (KernFn)attn_kernel<8, true>) kern = attn_kernel<8, true, 64>;
    if (variant == 128 && kern == ksp1[3]) kern = attn_kernel<8, false, 15 | 128>;
  }
  const int threads = p.flat ? Cfg<L, true>::NTX : Cfg<L, false>::NTX, sm = p.flat ? sm_f : sm_n;
#if ATT_V8C_CLUSTER || ATT_V8_OVL
  if constexpr (L == 8) {
    // v32s's own build; ATT_V8C_CLUSTER: one cluster of n_splits CTAs per request (no grid-wide residency needed: no
    // counter, no grid barrier); ATT_V8_OVL alone keeps the cooperative finalize (1 CTA/SM: the host's residency test)
    if (variant == 256 && kern == kn[3] && grid.x <= 8 && grid.z == 1 &&
        (ATT_V8C_CLUSTER || (long long)grid.x * grid.y <= (long long)nsm)) {
      const KernFn kc = attn_kernel<8, false, 3 | 256>;
      static bool cl_init = false;
      if (!cl_init) {
        cudaFuncSetAttribute(kc, cudaFuncAttributeMaxDynamicSharedMemorySize, sm_n);
        cl_init = true;
      }
      cudaLaunchConfig_t cfg = {};
      cfg.gridDim = grid;
      cfg.blockDim = dim3(threads, 1, 1);
      cfg.dynamicSmemBytes = sm_n;
      cfg.stream = stream;
      cudaLaunchAttribute attr[2];
      int na = 0;
      if (ATT_V8C_CLUSTER) {
        attr[na].id = cudaLaunchAttributeClusterDimension;
        attr[na].val.clusterDim.x = grid.x;
        attr[na].val.clusterDim.y = 1;
        attr[na].val.clusterDim.z = 1;
        ++na;
      }
      if (pdl) {
        attr[na].id = cudaLaunchAttributeProgrammaticStreamSerialization;
        attr[na].val.programmaticStreamSerializationAllowed = 1;
        ++na;
      }
      cfg.attrs = attr;
      cfg.numAttrs = na;
      cudaLaunchKernelEx(&cfg, kc, q);
      return;
    }
  }
#endif
  if (pdl) {
    // programmatic stream serialization: this grid may start while the preceding kernel finishes; the kernel waits
    // with griddepcontrol.wait before reading qkv
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = grid;
    cfg.blockDim = dim3(threads, 1, 1);
    cfg.dynamicSmemBytes = sm;
    cfg.stream = stream;
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = attr;
    cfg.numAttrs = 1;
    cudaLaunchKernelEx(&cfg, kern, q);
  } else {
    kern<<<grid, threads, sm, stream>>>(q);
  }
}

void run(TV qkv, TV seq_lens, TV cache_loc, TV k_cache, TV v_cache, TV page_table, TV k_scale, TV v_scale, TV inv_freq,
         TV out, TV ws_acc, TV ws_l, TV ws_cnt, int64_t B, int64_t L, int64_t Hq, int64_t Hkv, int64_t P,
         int64_t window, int64_t rope, int64_t n_splits, int64_t flat, int64_t nc, double scale_log2, int64_t coop,
         int64_t w0, int64_t pdl, int64_t variant, int64_t pf_ptr, int64_t pf_chunk, int64_t pf_seg, int64_t pf_nseg) {
  Params p;
  memset(&p, 0, sizeof(p));
  p.qkv = static_cast<const __nv_bfloat16*>(qkv.data_ptr());
  p.seq_lens = static_cast<const int*>(seq_lens.data_ptr());
  p.cache_loc = static_cast<const int64_t*>(cache_loc.data_ptr());
  p.k_cache = static_cast<uint8_t*>(k_cache.data_ptr());
  p.v_cache = static_cast<uint8_t*>(v_cache.data_ptr());
  p.page_table = static_cast<const int*>(page_table.data_ptr());
  p.k_scale = static_cast<const float*>(k_scale.data_ptr());
  p.v_scale = static_cast<const float*>(v_scale.data_ptr());
  p.inv_freq = static_cast<const float*>(inv_freq.data_ptr());
  p.out = static_cast<__nv_bfloat16*>(out.data_ptr());
  p.ws_acc = static_cast<float*>(ws_acc.data_ptr());
  p.ws_l = static_cast<float*>(ws_l.data_ptr());
  p.ws_cnt = static_cast<int*>(ws_cnt.data_ptr());
  p.B = (int)B;
  p.Hq = (int)Hq;
  p.Hkv = (int)Hkv;
  p.P = (int)P;
  while ((1 << p.lp) < P) ++p.lp;
  p.max_pages = (int)page_table.size(1);
  p.window = (int)window;
  p.rope = (int)rope;
  p.n_splits = (int)n_splits;
  p.flat = (int)flat;
  p.nc = (int)nc;
#if ATT_V1_COPIES != 1
  if (B == 1 && L == 8 && window == 2048 && P == 1) p.nc = ATT_V1_COPIES;
#endif
  p.ns_magic = (n_splits >= 2 && n_splits <= 1024)
                   ? (unsigned)(((1ull << 32) + (uint64_t)n_splits - 1) / (uint64_t)n_splits)
                   : 0u;
  p.ld_qkv = (int)qkv.size(qkv.ndim() - 1);
  p.row_bytes = (int)(Hkv * 64);
  while ((1 << p.lrb) < p.row_bytes) ++p.lrb;
  const int64_t cache_bytes = k_cache.size(0) * k_cache.size(1) * p.row_bytes;
  p.off32 = cache_bytes < (1LL << 32) ? 1 : 0;   // larger caches: 64-bit offsets
  p.scale_log2 = (float)scale_log2;
  p.coop = (int)coop;
  p.w0 = (int)w0;
  p.pdlf = pdl ? 1 : 0;
  p.pf = reinterpret_cast<const uint8_t*>(pf_ptr);
  p.pf_chunk = (unsigned)pf_chunk;
  p.pf_seg = (unsigned)pf_seg;
  p.pf_nseg = (unsigned)pf_nseg;
  dim3 grid((unsigned)n_splits, flat ? 1u : (unsigned)B, (unsigned)(Hkv / 2));
  cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(qkv.device().device_type, qkv.device().device_id));
  if (L == 1)
    launch<1>(p, grid, stream, pdl != 0, (int)variant);
  else
    launch<8>(p, grid, stream, pdl != 0, (int)variant);
}
"""

MOE_CU = r"""
// muse_moe_w4a16 v2: one persistent kernel: routing prologue, GEMM1+SwiGLU, GEMM2+norm stats, final norm/sum.
#include <cuda_bf16.h>
#include <type_traits>
#include <cuda_runtime.h>
#include <stdint.h>
#include <stddef.h>
#define SILU_DIV(a, b) __fdividef((a), (b))
#ifndef MOE_ROLEUNI
#define MOE_ROLEUNI 0   // preserve warp-uniform role selection across the skinny producer/consumer split
#endif
#ifndef MOE_ROUTE1
#define MOE_ROUTE1 0   // T=1: bitmap routing for distinct local slots; retain generic routing otherwise
#endif
#ifndef MOE_FNSBAL
#define MOE_FNSBAL 0   // select the first two tiny-build experts cooperatively instead of scalar find-nth
#endif
#ifndef MOE_BF16RED   // the small MoE's cross-warp partials (S.red) as BF16 (the TWO62 pre-scale keeps
#define MOE_BF16RED 0   // them near 2^-58, inside BF16's range)
#endif
#ifndef MOE_TANHSILU   // the small MoE's SiLU as x/2 (1 + tanh(x/2)), one MUFU.TANH, x < -8 as x e^x / (1 + e^x)
#define MOE_TANHSILU 0
#endif
__device__ __forceinline__ float moe_silu_s(float x) {   // the small MoE's SiLU (phase-A epilogues)
#if MOE_TANHSILU
  const float h = 0.5f * x;
  float t;
  asm("tanh.approx.f32 %0, %1;" : "=f"(t) : "f"(h));
  float r = fmaf(h, t, h);
  if (x < -8.f) {   // the tanh form cancels below -8 (1 + tanh(x/2) -> 0: up to 13,810 BF16 ulp off, 0 for -1.8e-6 at
    float e, q;     // x = -16); x e^x / (1 + e^x) there: within 1 ulp outside x < -87 (silu_exhaustive.py, form E)
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(e) : "f"(1.4426950408889634f * x));
    asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(q) : "f"(1.f + e));
    r = x * e * q;
  }
  return r;
#else
  return SILU_DIV(x, 1.f + __expf(-x));
#endif
}
#if !MOE_WGS
#define WARP_IDX(t) ((t) >> 5)
#else
#define WARP_IDX(t) __shfl_sync(0xffffffffu, (t) >> 5, 0)
#endif

namespace okc {

// L2 policies (CUTLASS CacheHintSm90): weights stream through once (evict first); inter and ybuf are re-read later in
// the same call (evict last). Hints only: the math and the memory traffic's results are unchanged.
#define L2H_SUFFIX ".L2::cache_hint"
#define L2H_OPERAND ", %4"
#define L2H_ARG , "l"(0x12F0000000000000ull)
__device__ __forceinline__ void st_keep_bf16(__nv_bfloat16* a, __nv_bfloat16 v) {
  asm volatile("st.global.L2::cache_hint.b16 [%0], %1, %2;\n" ::"l"(a), "h"(*reinterpret_cast<unsigned short*>(&v)),
               "l"(0x14F0000000000000ull)
               : "memory");
}
__device__ __forceinline__ void st_keep_f32(float* a, float v) {
  asm volatile("st.global.L2::cache_hint.b32 [%0], %1, %2;\n" ::"l"(a), "f"(v), "l"(0x14F0000000000000ull) : "memory");
}

constexpr int HD = 4096;
constexpr int ID = 4096;
#if !MOE_WGS
constexpr int NW = 8;
#else
constexpr int NW = 16;
#endif
constexpr int NT = NW * 32;
#ifndef MOE_IWARP   // skinny builds: the ring refills issued by a dedicated issuer warp on the
#define MOE_IWARP 0   // hand-over barrier instead of by compute thread 0 (eat_mst12: the issue holds warp 0 ~0.2 us per unit)
#endif
#ifndef MOE_ITAIL
#define MOE_ITAIL 0   // the issuer stops observing handovers once no later refill is possible
#endif
#ifndef MOE_BREV   // d32 builds: phase B's jobs in reverse order (the fat jobs, first in phase A, last in phase B: their
#define MOE_BREV 0   // B lands on the CTAs whose skinny A ends early, not on the CTAs still finishing their A)
#endif
#ifndef MOE_XPRB   // timing probe only (wrong output): phase B's x copies skip the phase-A credit wait
#define MOE_XPRB 0
#endif
#ifndef MOE_MID16   // MID: one job of <= 16 tokens per expert (two n-tiles: the second tile's x in shared memory) instead
#define MOE_MID16 0   // of <= 8-token jobs, so an expert of 9-16 tokens streams its weights once, not twice
#endif
constexpr int NTL =
    NT +
#if !MOE_WGS
    (32 + 32 * MOE_MID16 + 32 * MOE_IWARP);   // (MID16: a second helper warp, the second tile's epilogues; IWARP: the issuer)   // + helper warp (WS) | weights and x producer warps (WGS)  // launched: NT compute threads (+ helper warp | WGS producer warp + epilogue warpgroup)
#define TIDX ((int)((threadIdx.x + NT) % NTL))
constexpr int NRB = 4;   // cross-warp partial-sum buffers
#else
    (64);   // + helper warp (WS) | weights and x producer warps (WGS)  // launched: NT compute threads (+ helper warp | WGS producer warp + epilogue warpgroup)
constexpr int NRB = 2;   // cross-warp partial-sum buffers
#endif
// CTA barrier over the NT compute threads only (the helper warp syncs with them through mbarriers and barrier 2)
#define CTA_SYNC() asm volatile("bar.sync 1, %0;\n" ::"n"(NT) : "memory")
#define WS_JOIN() asm volatile("bar.sync 2, %0;\n" ::"n"(NTL) : "memory")
constexpr int NCH = 64 / NW;   // 64-K chunks per warp in a skinny unit (4096 / NW K per warp)
#ifndef MOE_MID16W   // MID16: a two-tile unit's cost in the CTA split, against 8 for a one-tile unit (d16s stamps: 1.45 vs
#define MOE_MID16W 11   // 1.07 us per unit)
#endif
#ifndef MOE_MID16WB   // the same in phase B (the two-tile phase-B unit loop: 3143 static cycles against 1838)
#define MOE_MID16WB 13
#endif
constexpr int SK_T = MOE_MID16 ? 16 : 8;   // (MID16's split and preissue assume one job per touched expert: T <= 16)   // max tokens of a skinny job (8: 1 n-tile; 16: 2 n-tiles via smem x)
constexpr int RED_W = MOE_MID16 ? 8 : 4;   // (MID16: the second tile's partials at [4, 8))
constexpr int FT_T = 64;   // max tokens of a fat job (longer jobs are split into chunks)
constexpr int RB_A = 2 * ID / 16;
constexpr int RB_B = HD / 16;   // 16-row blocks
constexpr int UB_A = RB_A / 16;
constexpr int UB_B = RB_B / 16;
constexpr int RB_A_LOG = 9;
constexpr int RB_B_LOG = 8;
static_assert((1 << RB_B_LOG) == RB_B, "RB_B must be 256");
// dynamic tail: the last pool units of a phase's skinny work are claimed at run time (dynk per CTA)
__device__ __forceinline__ int dyn_pool(int nS, int G) {
#if !MOE_WGS
  return 0;
#else
  const int d = G * 4;
  return d < nS ? d : nS;
#endif
}
static_assert((1 << RB_A_LOG) == RB_A, "RB_A must be 512");
constexpr int MMAX = 384;
constexpr int EMAX = MMAX * 8;
constexpr int JMAX = 32 + EMAX / FT_T + 8;
   // skinny x row pitch (bf16)
constexpr float TWO62 = 4.611686018427388e18f;

struct Params {
  const uint4* wA;
  const uint4* sA;
  const float* gsA;
  const uint4* wB;
  const uint4* sB;
  const float* gsB;
  const uint32_t* lut;
  const __nv_bfloat16* normw;   // post_norm_w as given (bf16); 1 + w is formed in FP32 here
  const unsigned char* wAf;
  const unsigned char* wBf;   // fat-order weight copies (36 KB per k-stage)
  const unsigned char* wAu;
  const unsigned char* wBu;   // unit-contiguous copies (36 KB per 16-row block)
  const unsigned char* wAn;
  const unsigned char* wBn;   // natural-k fat copies (wgmma path)
  const unsigned char* wAs;
  const unsigned char* wBs;   // WGS copies: [256-row unit][stage][warpgroup][9 KB]
  __nv_bfloat16* xA;
  __nv_bfloat16* xB;   // WGS: per job, per stage x / inter in wgmma B layout [stage][n8][k8][8][8]
  int* xready;   // WGS: per job, rows of xA written
  int* doneAs;   // WGS: per (job, phase-B stage), phase-A warpgroup epilogues done (8 = ready)
#if MOE_WGS
  int* mfl;   // MFIN flag region (cE / exit counter / xready / doneAs, two parities)
#endif
  uint32_t zero;   // always 0 (opaque to the compiler)
  const __nv_bfloat16* x;
  const int* topk_ids;
  const float* topk_w;
  int M, lo, nloc;
  int* ctr;   // [8]: 0 ctrA, 1 ctrB, 2 barrier, 3 exit
  int* doneA;   // [JMAX]
  __nv_bfloat16* inter;   // [EMAX, ID]
  float* ybuf;   // [EMAX, HD]
  const uint8_t* upf;
  unsigned upf_bytes;   // unpooler stream L2 prefetch
  float* ss;   // [EMAX] copy 0 of the per-token sums of squares (copy 1: ss1); a launch uses copy gen & 1
  float* ss1;   // [EMAX]
  __nv_bfloat16* out;
  void* unused;   // (no use: keeps the parameter layout the kernels were tuned with)
#if MOE_WGS
  float* sspart;   // B4FIN: per-(route, 16-column tile) sum-of-squares partials (WGS only)
#endif
};

struct Smem {
  uint32_t lut[256];
#if !MOE_WGS
  float red[NRB][NW][32][RED_W];   // per-unit cross-warp partial sums
  uint64_t wse[8], wsrf[8], wsre[8];   // WS: ring slot released by all compute warps / partials ready / buffer free
  int wsr[4];   // WS: this CTA's phase A and phase B unit ranges
#if MOE_MID16
  int mwsr[4];   // MID16: the same, by cost, from the preissue warp
#endif
  uint64_t bars[12];   // 0..4 skinny ring (NS <= 5), 5..11 fat ring (NSF <= 7)
#else
  float red[1][1][1][RED_W];   // (the skinny paths that use it are not run in WGS builds)
  uint64_t bars[24];   // WGS: 0..3 weights full, 4..7 weights empty, 8..15 x full, 16..23 x empty
  int wglast[4];   // WGS: this warpgroup completed a split unit
  int wcp[JMAX + 1];   // WGS: phase-A cost prefix per job
  int wcpB[JMAX + 1];   // WGS: phase-B cost prefix per job
  int jrow[JMAX + 1];   // WGS: first padded row of each job in xA / xB (rows padded to 8 per job)
  uint32_t wtot;   // WGS: phase-A total cost
  uint32_t wtotB;   // WGS: phase-B total cost
  int wgr[4];   // WGS: this CTA's phase A and phase B unit ranges
#endif
  int seq, fseq, preissued, preissuedB;
  int fcnt[32];   // FAST1 per-expert entry counts (the preissue's job -> expert map)
  void* xbuf_ptr;
  int cnt[64], off[64];
  int wc[NW][32];
  int jE[JMAX], jS[JMAX], jC[JMAX], forder[JMAX];
  int njF, njS, unit;
  int gen;   // the grid barrier's generation at launch start (+1 per launch): picks this launch's ss copy
  float* ssp;   // this launch's ss copy
  int uslot[8];   // unit id per skinny ring slot (-1: none; BFLAG | v: phase-B unit prefetched in phase A)
  int dync[JMAX];   // phase A: pool units this CTA processed, per job (credited to doneA at the end of phase A)
  int unit_next, unit_next2;
  signed char eid[EMAX];
  short ent[EMAX];   // sorted entries (token*8+slot) by expert
  short pos[EMAX];   // entry -> pair position, -1 if not local
};

// floor(a * c / G) without an integer divide (per-launch code runs from a cold instruction cache, where the divide
// subroutines cost ~1 us): a * c < 2^24, rG ~ 1 / G, then an exact +-1 fix-up
__device__ __forceinline__ int muldiv_g(int a, int c, int G, float rG) {
  const int p = a * c;
  int q = __float2int_rz((float)p * rG);
  if (q * G > p)
    --q;
  else if (((q + 1) * G) <= p)
    ++q;
  return q;
}
__device__ __forceinline__ uint32_t bmul2(uint32_t a, uint32_t b) {
  __nv_bfloat162 r = __hmul2(*reinterpret_cast<__nv_bfloat162*>(&a), *reinterpret_cast<__nv_bfloat162*>(&b));
  return *reinterpret_cast<uint32_t*>(&r);
}
// a x (b.lo, b.lo) and a x (b.hi, b.hi): the broadcast folds into the HMUL2 operand (.H0_H0 / .H1_H1)
__device__ __forceinline__ uint32_t bmul2l(uint32_t a, uint32_t b) {
  __nv_bfloat162 r =
      __hmul2(*reinterpret_cast<__nv_bfloat162*>(&a), __low2bfloat162(*reinterpret_cast<__nv_bfloat162*>(&b)));
  return *reinterpret_cast<uint32_t*>(&r);
}
__device__ __forceinline__ uint32_t bmul2h(uint32_t a, uint32_t b) {
  __nv_bfloat162 r =
      __hmul2(*reinterpret_cast<__nv_bfloat162*>(&a), __high2bfloat162(*reinterpret_cast<__nv_bfloat162*>(&b)));
  return *reinterpret_cast<uint32_t*>(&r);
}

__device__ __forceinline__ void mma_bf16(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
                                         uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
extern __shared__ __align__(16) unsigned char smem_raw[];
__device__ __forceinline__ float* ss_cur();   // this launch's ss copy (defined after Smem users)
#ifndef MOE_PLD   // predecessor-written inputs (x, topk ids / weights; this grid never writes them): 0 ld.global.cg, 1 plain
#define MOE_PLD 0   // weak loads after griddepcontrol.wait (the wait makes the predecessor's writes visible; L1-cacheable)
#endif
template <typename T>
__device__ __forceinline__ T ldp(const T* a) {
#if MOE_PLD
  return *a;
#else
  return __ldcg(a);
#endif
}
__device__ __forceinline__ int ld_acquire(const int* p) {
  int v;
  asm volatile("ld.acquire.gpu.global.b32 %0, [%1];\n" : "=r"(v) : "l"(p));
  return v;
}

// Dequantize one 32-bit word (8 E2M1 codes) into the 4 bf16x2 A-fragment registers (values e2m1 * 2^-126).
// The DQM and DQ7 packed layouts match _pack_codes.
__device__ __forceinline__ void dq4(uint32_t q, uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3) {
  constexpr uint32_t M = 0x81C081C0u;
  r0 = q & M;
  r1 = __funnelshift_r(q, q, 6) & M;
  r2 = __funnelshift_r(q, q, 10) & M;
  // DQ7 bit layout (template DQ7 = True): 7 ALU + 2 IMAD.SHL per 8 weights. Register sources partition the word:
  // r0 bits as stored, r1 = word << 4, r2 = rotl 10, r3 = word << 6 (6 bits) | rotl 14 (the two e1 bits)
#if MOE_DQM
  r1 = __funnelshift_r(q, q, 3) & M;
  r2 = __funnelshift_r(q, q, 6) & M;
  r3 = ((q & 0x001b001bu) * 4228u) & M;
#else
  {
    const uint32_t R = __funnelshift_r(q, q, 6);
    r1 = (q ^ (R * 4u)) & M;
    r2 = (R ^ (q * 16u)) & M;
    r3 = (R ^ (q * 64u)) & M;
  }
#endif
}

// ---------------------------------------------------------------- routing prologue (every CTA, identical result)
__device__ void route(const Params& p, Smem& S) {
#if !MOE_WGS
  const int tid = TIDX;
#else
  const int tid = threadIdx.x;
#endif
  const int lane = tid & 31;
  const int warp = WARP_IDX(tid);
#if MOE_ROUTE1 && MOE_TINY && !MOE_WGS
  if (p.M == 1) {
    if (__shfl_sync(0xffffffffu, warp, 0) == 0) {
      const int e = lane < 8 ? ldp(p.topk_ids + lane) - p.lo : -1;
      const bool valid = lane < 8 && e >= 0 && e < p.nloc;
      const unsigned mask = __reduce_or_sync(0xffffffffu, valid ? (1u << e) : 0u);
      const int routes = __popc(__ballot_sync(0xffffffffu, valid));
      const bool distinct = __popc(mask) == routes;
      if (lane == 0) S.lut[255] = distinct;
      if (distinct) {
        const int count = (mask >> lane) & 1u;
        const int offset = __popc(mask & ((1u << lane) - 1u));
        S.cnt[lane] = count;
        S.off[lane] = offset;
        if (count) {
          S.jE[offset] = lane;
          S.jS[offset] = offset;
          S.jC[offset] = 1;
        }
        if (lane < 8) {
          S.eid[lane] = valid ? e : -1;
          const int position = valid ? __popc(mask & ((1u << e) - 1u)) : -1;
          S.pos[lane] = position;
          if (valid) S.ent[position] = lane;
        }
        if (lane == 0) {
          S.njF = 0;
          S.njS = routes;
        }
      }
    }
    CTA_SYNC();
    if (S.lut[255]) return;
  }
#endif
  const int E = p.M * 8;
  const int CH = ((E + NW * 32 - 1) / (NW * 32)) * 32;
  const int i0 = warp * CH;
  const int i1 = min(E, i0 + CH);
  if (tid < NW * 32) S.wc[tid >> 5][tid & 31] = 0;
#if !MOE_WGS
#pragma unroll 8
  for (int i = tid; i < E; i += NT) {
    int e = ldp(p.topk_ids + i) - p.lo;
    S.eid[i] = (signed char)((e >= 0 && e < p.nloc) ? e : -1);
#else
  {   // 16 B loads (E = 8 M; the launcher checks the alignment): one or two per thread, all issued before the stores
    constexpr int LPT = (EMAX / 4 + NT - 1) / NT;
    int4 ev[LPT];
    const int E4 = E >> 2;
#pragma unroll
    for (int k = 0; k < LPT; ++k) {
      const int i4 = tid + k * NT;
      ev[k] = i4 < E4 ? ldp(reinterpret_cast<const int4*>(p.topk_ids) + i4) : make_int4(0, 0, 0, 0);
    }
#pragma unroll
    for (int k = 0; k < LPT; ++k) {
      const int i4 = tid + k * NT;
      if (i4 < E4) {
        const int v[4] = {ev[k].x - p.lo, ev[k].y - p.lo, ev[k].z - p.lo, ev[k].w - p.lo};
        char4 c;
        c.x = (signed char)((v[0] >= 0 && v[0] < p.nloc) ? v[0] : -1);
        c.y = (signed char)((v[1] >= 0 && v[1] < p.nloc) ? v[1] : -1);
        c.z = (signed char)((v[2] >= 0 && v[2] < p.nloc) ? v[2] : -1);
        c.w = (signed char)((v[3] >= 0 && v[3] < p.nloc) ? v[3] : -1);
        *reinterpret_cast<char4*>(&S.eid[i4 * 4]) = c;
      }
    }
#endif
  }
  CTA_SYNC();
  for (int base = i0; base < i1; base += 32) {
    const int i = base + lane;
    const int e = (i < i1) ? (int)S.eid[i] : -1;
    const unsigned peers = __match_any_sync(0xffffffffu, e);
    if (e >= 0 && lane == (__ffs(peers) - 1)) S.wc[warp][e] += __popc(peers);
  }
  CTA_SYNC();
  if (warp == 0) {
    int c[NW];
    int tot = 0;
#pragma unroll
    for (int w = 0; w < NW; ++w) {
      c[w] = (lane < p.nloc) ? S.wc[w][lane] : 0;
      tot += c[w];
    }
    int incl = tot;
#pragma unroll
    for (int d = 1; d < 32; d <<= 1) {
      int v = __shfl_up_sync(0xffffffffu, incl, d);
      if (lane >= d) incl += v;
    }
    int off = incl - tot;
    S.cnt[lane] = tot;
    S.off[lane] = off;
    int o2 = off;
#pragma unroll
    for (int w = 0; w < NW; ++w) {
      S.wc[w][lane] = o2;
      o2 += c[w];
    }
    // jobs: fat (cnt > SK_T, chunks of FT_T) first, then skinny (0 < cnt <= SK_T); expert order within each
#if MOE_MID
    const int nsj =
        (tot + SK_T - 1) / SK_T;   // ceil(tot / 8) jobs of <= 8 tokens; a split expert re-streams its weights
    int fi = 0, si = nsj;
#else
    const int nfj = tot > SK_T ? (tot + FT_T - 1) / FT_T : 0;
    const int nsj = (tot > 0 && tot <= SK_T) ? 1 : 0;
    int fi = nfj, si = nsj;
#endif
#pragma unroll
    for (int d = 1; d < 32; d <<= 1) {
      int v1 = __shfl_up_sync(0xffffffffu, fi, d), v2 = __shfl_up_sync(0xffffffffu, si, d);
      if (lane >= d) {
        fi += v1;
        si += v2;
      }
    }
    const int nfTot = __shfl_sync(0xffffffffu, fi, 31);
    const int nsTot = __shfl_sync(0xffffffffu, si, 31);
#if MOE_MID
    int fj = fi - 0, sj = nfTot + si - nsj;
    for (int q = 0; q < 0; ++q) {
#else
    int fj = fi - nfj, sj = nfTot + si - nsj;
    for (int q = 0; q < nfj; ++q) {
#endif
      S.jE[fj + q] = lane;
      S.jS[fj + q] = off + q * FT_T;
      S.jC[fj + q] = min(FT_T, tot - q * FT_T);
    }
#if MOE_MID
    for (int q = 0; q < nsj; ++q) {
      S.jE[sj + q] = lane;
      S.jS[sj + q] = off + q * SK_T;
      S.jC[sj + q] = min(SK_T, tot - q * SK_T);
#else
    if (nsj) {
      S.jE[sj] = lane;
      S.jS[sj] = off;
      S.jC[sj] = tot;
#endif
    }
    if (lane == 0) {
      S.njF = nfTot;
      S.njS = nsTot;
    }
    __syncwarp();
#if !MOE_WGS
    // fat queue order: largest n-tile count first (LPT), stable by job index
    int pos = 0;
    for (int key = FT_T / 8; key >= 1; --key) {
      for (int f0 = 0; f0 < nfTot; f0 += 32) {
        const int f = f0 + lane;
        const bool m = f < nfTot && ((S.jC[f] + 7) >> 3) == key;
        const unsigned bal = __ballot_sync(0xffffffffu, m);
        if (m) S.forder[pos + __popc(bal & ((1u << lane) - 1u))] = f;
        pos += __popc(bal);
      }
    }
#endif
  }
  CTA_SYNC();
  for (int base = i0; base < i1; base += 32) {
    const int i = base + lane;
    const int e = (i < i1) ? (int)S.eid[i] : -1;
    const unsigned peers = __match_any_sync(0xffffffffu, e);
    if (e >= 0) {
      const int q = S.wc[warp][e] + __popc(peers & ((1u << lane) - 1u));
      S.ent[q] = (short)i;
      S.pos[i] = (short)q;
    } else if (i < i1) {
      S.pos[i] = -1;
    }
    __syncwarp();
    if (e >= 0 && lane == (__ffs(peers) - 1)) S.wc[warp][e] += __popc(peers);
    __syncwarp();
  }
  CTA_SYNC();
}

// ---------------------------------------------------------------- the persistent kernel

// ---------------------------------------------------------------- skinny path: shared-memory weight ring
// A unit = (job <= 16 tokens, 16-row block) x full K = 32 KB codes + 4 KB scales, contiguous in global memory.
// Units of the CTA's range stream through ns smem slots (TMA bulk copies on sm_90, cp.async elsewhere).
// All 8 warps work on the same unit, each on its own 512-K range, with x for that K range held in registers.
constexpr int UNIT_CODES = 32768;
#if !MOE_WGS
constexpr int UNIT_SC = 4096;
#else
constexpr int UNIT_SC = 4096;
#endif
constexpr int UNIT_BYTES = UNIT_CODES + UNIT_SC;
#ifndef MOE_SC6S   // skinny builds: each unit's 4096 scale bytes stored as three 1024-byte planes of six-bit values (every
#define MOE_SC6S 0   // scale byte is 0x40 + a six-bit value; lossless): 35840 bytes per unit in global memory (the smem
#endif               // slot keeps UNIT_BYTES). Plane w, lane word byte n = v[4w + n] | (bits 2w + 1 : 2w of v[12 + n]) << 6
#if MOE_SC6S && !MOE_WGS
constexpr int UNIT_GB = UNIT_CODES + 3072;   // a unit's global bytes
#else
constexpr int UNIT_GB = UNIT_BYTES;
#endif

__device__ __forceinline__ uint32_t smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
// e4m3 scale byte b of word -> bf16x2(scale * 2^64). The multiplier is opaque to ptxas (k16 = 16 + p.zero):
// an IMAD on the FMA pipe instead of a LEA on the ALU pipe, which the FP4 decode saturates.
#define SCDK(word, b, k16) (__byte_perm((word), 0u, 0x4040u | ((b) * 0x0101u)) * (k16) + 0x5C005C00u)
#ifndef MOE_STAMP   // timing probe only (t1 / skinny builds): per-CTA globaltimer stamps, u64 [cta][72], into the lut slot
#define MOE_STAMP 0   // (never read since the placeholder args): 0 entry, 1 routed, 2 A done, 3 B done, 4 helper joined,
#endif                // 5 barrier arrive, 6 barrier passed; B: 8 entered, 9 doneA met, 10 first unit landed, 11 last unit
                      // done; 12 first A unit landed; helper: 13 unit 0 handed over, 14 its epilogue done, 15 A credit sent;
                      // per unit q < 8 of the CTA: 16 + q landed (compute thread 0), 24 + q epilogue done (helper), 32 + q its
                      // partials picked up (helper), 40 + q handed over (compute thread 0)
#define MST(k) do { if (MOE_STAMP) { unsigned long long t_; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t_)); \
    reinterpret_cast<unsigned long long*>(const_cast<uint32_t*>(p.lut))[blockIdx.x * 72 + (k)] = t_; } } while (0)
#ifndef MOE_RAFN   // skinny builds (the raf cell, form b): the unit loop's ptxas frequency hint (no entry hint, so route()
#define MOE_RAFN 0   // keeps its converged collective paths), the no-fat-units assumption instead of the trap, and an opaque
#endif               // gridDim read (run_phase)
#ifndef MOE_PDLE   // 1: the PDL trigger at kernel entry (right after griddepcontrol.wait) instead of after phase B: the next
#define MOE_PDLE 0   // kernel's launch is processed early (its griddepcontrol.wait still waits for this whole grid)
#endif
#ifndef MOE_T1DF   // t1 builds: phase B reads the job's inter rows straight from L2 until every value is written (the DF
#define MOE_T1DF 0   // sentinels) instead of waiting for the A credit: no release fence on the A -> B edge
#endif
#ifndef MOE_HCRED   // first-helper skinny builds: bit 0 the A credit as one fence.acq_rel + relaxed adds (a CTA spanning two
#define MOE_HCRED 0   // jobs paid two release fences), bit 1 the partials handed to the helper before thread 0's refill issue
#endif
#ifndef MOE_XSLOT   // skinny builds with the first helper (d1s / d1f / v1f): a 4th ring slot used from the last phase-A unit
#define MOE_XSLOT 0   // on: 3 units in flight during phase A, 4 from the A -> B transition (when HBM idles on the credit), so
#endif                // the CTA's 4th phase-B unit is not fetched only after its first B unit is consumed. Needs nA >= 1
                      // per CTA (T <= 8: 512 A units per expert over 132 CTAs) and the 4th 36 KB slot (inside the
                      // fat-x region these builds reserve and never use)
#if (MOE_XSLOT == 1 && !MOE_WGS && !MOE_MID && !MOE_TUNED_V1) || (MOE_XSLOT == 2 && MOE_TINY)
constexpr int RS = 4;   // ring slots (XSLOT 2, t1 builds: the helper issues the extra unit right after its A credit, so
                        // the transition traffic does not delay the credit)
#else
constexpr int RS = 3;
#endif
#ifndef MOE_BXW   // MID / TUNED_V1 phase-B cost split: a job's x load weighed as this many units (the CTA whose range
#define MOE_BXW 4   // enters a job pays one demand-load round of the job's inter rows)
#endif
#ifndef MOE_AXW   // TINY2 TUNED_V1: phase A's CTA split by cost, a job's x load weighed as this many units (the CTAs whose
#define MOE_AXW 0   // ranges cross a job boundary otherwise gate both jobs' A credit by one x load); 0: the plain unit split
#endif
#if MOE_AXW && (MOE_MID || !MOE_TUNED_V1)
#error "MOE_AXW: the TINY2 TUNED_V1 build only"
#endif
#ifndef MOE_BREL   // skinny builds: the grid barrier's arrive as one atom.add.release.gpu (no fence.sc.gpu before it, none
#define MOE_BREL 0   // after the TINY helper's last epilogue)
#endif
#ifndef MOE_SDEF   // t1 builds (one token): deferred block scales. Each mma's k16 is one NVFP4 block (natural mma k order)
#define MOE_SDEF 0   // and B column n carries x (times 2^64) only for the blocks 8q + n of the warp's K range, so the 8
#endif               // mmas of chunk pair q leave 8 blocks' partial sums in one C quad, scaled once in FP32 afterwards
// e4m3 scale byte b of word -> its FP32 value (a normal e4m3: exponent + 120); k20 = 2^20, opaque as k16
#define SCDF(word, b, k20) __uint_as_float(__byte_perm((word), 0u, 0x4440u | (b)) * (k20) + 0x3C000000u)
__device__ __forceinline__ void mbar_init(uint64_t* bar, int count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(smem_u32(bar)), "r"(count));
}
__device__ __forceinline__ void mbar_arrive1(uint64_t* bar) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" ::"r"(smem_u32(bar)) : "memory");
}
__device__ __forceinline__ void mbar_wait(uint64_t* bar, uint32_t parity) {
  asm volatile(
      "{\n .reg .pred p;\n WAIT_%=:\n mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n @!p bra WAIT_%=;\n}\n" ::
          "r"(smem_u32(bar)),
      "r"(parity)
      : "memory");
}

template <bool PB>
__device__ __forceinline__ void ring_issue(const Params& p, const Smem& S, int v, unsigned char* slot, uint64_t* bar) {
  const int rbs = PB ? RB_B : RB_A;
  const int njF = S.njF;
  const int j = njF + v / rbs;
  const int rb = v - (v / rbs) * rbs;
  const size_t blk = (size_t)S.jE[j] * rbs + rb;
  const unsigned char* src = (PB ? p.wBu : p.wAu) + blk * (size_t)UNIT_GB;   // [32 KB codes | 4 (SC6S: 3) KB scales]
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(smem_u32(bar)), "r"(UNIT_GB)
               : "memory");
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes" L2H_SUFFIX
               " [%0], [%1], %2, [%3]" L2H_OPERAND ";\n" ::"r"(smem_u32(slot)),
               "l"(src), "r"(UNIT_GB), "r"(smem_u32(bar))L2H_ARG
               : "memory");
}

template <bool PB>
__device__ __forceinline__ void load_x_regs(const Params& p, const Smem& S, int j, int lane, int warp,
                                            uint4 (&x)[2 * NCH]) {
  const int g = lane >> 2, t = lane & 3;
  const int cnt = S.jC[j];
  const int st = S.jS[j];
#if MOE_SDEF
  {   // lane (g, t) is B column g: the 2 x words of block 8q + g of the warp's K range for each chunk pair q
    const int gc = g >> 2, gs = g & 3;
    uint32_t v[NCH / 2][2];
#pragma unroll
    for (int q = 0; q < NCH / 2; ++q) v[q][0] = v[q][1] = 0u;
    if (cnt > 0) {
      const __nv_bfloat16* row = PB ? p.inter + (size_t)st * ID : p.x + (size_t)(S.ent[st] >> 3) * HD;
      const unsigned int* src =
          reinterpret_cast<const unsigned int*>(row + warp * (64 * NCH) + gc * 64 + gs * 16 + 2 * t);
#pragma unroll
      for (int q = 0; q < NCH / 2; ++q) {
        v[q][0] = bmul2(PB ? __ldcg(src + q * 64) : ldp(src + q * 64), 0x5F805F80u);   // x 2^64 (exact)
        v[q][1] = bmul2(PB ? __ldcg(src + q * 64 + 4) : ldp(src + q * 64 + 4), 0x5F805F80u);
      }
    }
#pragma unroll
    for (int c = 0; c < NCH; ++c) {
      uint32_t r[8];
#pragma unroll
      for (int s = 0; s < 4; ++s) {
        const bool mine = gc == (c & 1) && gs == s;
        r[2 * s] = mine ? v[c >> 1][0] : 0u;
        r[2 * s + 1] = mine ? v[c >> 1][1] : 0u;
      }
      x[2 * c] = make_uint4(r[0], r[1], r[2], r[3]);
      x[2 * c + 1] = make_uint4(r[4], r[5], r[6], r[7]);
    }
  }
#else
  {
    const int r = g;
    if (r < cnt) {
      const __nv_bfloat16* row;
      if (PB)
        row = p.inter + (size_t)(st + r) * ID;
      else
        row = p.x + (size_t)(S.ent[st + r] >> 3) * HD;
      const uint4* src = reinterpret_cast<const uint4*>(row + warp * (64 * NCH) + 16 * t);
#pragma unroll
      for (int c = 0; c < NCH; ++c) {
        x[2 * c] = PB ? __ldcg(src + c * 8) : ldp(src + c * 8);
        x[2 * c + 1] = PB ? __ldcg(src + c * 8 + 1) : ldp(src + c * 8 + 1);
      }
    } else {
#pragma unroll
      for (int c = 0; c < 2 * NCH; ++c) x[c] = make_uint4(0, 0, 0, 0);
    }
  }
#endif
}

#ifndef MOE_DFPOLL   // port list dfpoll: the DF poll's sentinel test as packed halfword arithmetic (old study +0.33%)
#define MOE_DFPOLL 0
#endif
#ifndef MOE_SPOLL2   // port list spoll2 64,128: the untuned DF builds poll x every 64 ns and doneA every 128 ns (32 / 64)
#define MOE_SPOLL2 0
#endif
#if MOE_DF || MOE_T1DF
// DF: phase-B x rows (the job's inter) straight from L2 until every value is written (sentinel 0xFFFF = bf16 NaN)
__device__ __forceinline__ uint32_t df_bad(const uint4& v) {
  const uint32_t w[4] = {v.x, v.y, v.z, v.w};
  uint32_t b = 0;
#pragma unroll
  for (int i = 0; i < 4; ++i) b |= (uint32_t)(((w[i] & 0xFFFFu) == 0xFFFFu) | ((w[i] >> 16) == 0xFFFFu));
  return b;
}
__device__ __forceinline__ void load_x_inter_df(const Params& p, const Smem& S, int j, int lane, int warp,
                                                uint4 (&x)[2 * NCH]) {
  const int g = lane >> 2, t = lane & 3;
  const int cnt = S.jC[j];
  const int st = S.jS[j];
  const bool act = g < cnt;
  const uint4* src =
      reinterpret_cast<const uint4*>(p.inter + (size_t)(st + (act ? g : 0)) * ID + warp * (64 * NCH) + 16 * t);
#if MOE_DFPOLL
  // the sentinel test as packed halfword arithmetic: a halfword of w is 0xFFFF iff ~w's is 0, and
  // (v - 0x00010001) & ~v & 0x80008000 with v = ~w flags exactly "a halfword is zero"; the words are OR-accumulated and
  // masked once, and the inactive lanes keep the zeros set here (the base: 257 instructions per pass)
#pragma unroll
  for (int c = 0; c < 2 * NCH; ++c) x[c] = make_uint4(0, 0, 0, 0);
  for (int spin = 0;; ++spin) {
    uint32_t bad = 0;
    if (act) {
#pragma unroll
      for (int c = 0; c < NCH; ++c) {
        x[2 * c] = __ldcg(src + c * 8);
        x[2 * c + 1] = __ldcg(src + c * 8 + 1);
      }
#pragma unroll
      for (int c = 0; c < 2 * NCH; ++c) {
        const uint32_t w[4] = {x[c].x, x[c].y, x[c].z, x[c].w};
#pragma unroll
        for (int i = 0; i < 4; ++i) bad |= (~w[i] - 0x00010001u) & w[i];
      }
      bad &= 0x80008000u;
    }
#else
  for (int spin = 0;; ++spin) {
    uint32_t bad = 0;
#pragma unroll
    for (int c = 0; c < NCH; ++c) {
      x[2 * c] = act ? __ldcg(src + c * 8) : make_uint4(0, 0, 0, 0);
      x[2 * c + 1] = act ? __ldcg(src + c * 8 + 1) : make_uint4(0, 0, 0, 0);
      bad |= df_bad(x[2 * c]) | df_bad(x[2 * c + 1]);
    }
#endif
    if (!__any_sync(0xffffffffu, bad != 0) || spin > 200000) break;
#if MOE_TUNED_D16 || MOE_TUNED_V1
#if MOE_TUNED_V1
    __nanosleep(8);
#endif
#if MOE_TUNED_D16
    __nanosleep(16);
#endif
#endif
#if MOE_DF && !MOE_TUNED_D16 && !MOE_TUNED_V1
    __nanosleep(MOE_SPOLL2 ? 64 : 32);
#endif
#if MOE_T1DF
    __nanosleep(32);
#endif
  }
}

#endif
#if MOE_MID16
static_assert(MOE_DF, "MID16: phase B's second tile polls the DF sentinels");
// phase B, a two-tile job: both tiles' inter rows behind one DF poll (two polls in a row cost a second L2 round trip at
// the job's first unit); the first tile into x0 (as load_x_inter_df), the second into the warp's shared buffer
__device__ __forceinline__ void load_x01_df(const Params& p, const Smem& S, int j, int lane, int warp, uint4 (&x)[2 * NCH],
                                            __nv_bfloat16* x1s);

// MID16: phase B's CTA range by cost (warp-parallel, lane = job; one job per touched expert): a two-tile unit
// MOE_MID16WB against 8, each job 4 units more (its x load), floor(tot * b / G) exact without a divide routine
__device__ __forceinline__ void mid16_rangesB(Smem& S, int lane) {
  const int nj = S.njS;
  const uint32_t cl = lane < nj ? 32u + (S.jC[lane] > 8 ? (uint32_t)MOE_MID16WB : 8u) * RB_B : 0u;
  uint32_t ci = cl;
#pragma unroll
  for (int d = 1; d < 32; d <<= 1) {
    const uint32_t v = __shfl_up_sync(0xffffffffu, ci, d);
    if (lane >= d) ci += v;
  }
  const uint32_t tot = __shfl_sync(0xffffffffu, ci, 31);
  const float rG = __fdividef(1.f, (float)gridDim.x);
#pragma unroll
  for (int k = 0; k < 2; ++k) {
    const int b = (int)blockIdx.x + k;
    int res = nj * RB_B;
    if (b < (int)gridDim.x) {
      const uint32_t pc = tot * (uint32_t)b;
      uint32_t x = (uint32_t)__float2uint_rz((float)pc * rG);
      while (x * gridDim.x > pc) --x;
      while ((x + 1u) * gridDim.x <= pc) ++x;
      const int jj = __popc(__ballot_sync(0xffffffffu, ci <= x));
      const int L = jj;   // (every job has cost: the first lane past x is job jj)
      const uint32_t r = x - (__shfl_sync(0xffffffffu, ci, L) - __shfl_sync(0xffffffffu, cl, L));
      const bool two = __shfl_sync(0xffffffffu, lane < nj ? S.jC[lane] : 0, L) > 8;
      res = jj * RB_B + (r < 32u ? 0 : (int)(two ? (r - 32u) / (uint32_t)MOE_MID16WB : (r - 32u) >> 3));
    }
    if (lane == 0) S.wsr[2 + k] = res;
  }
}
// a unit's partial sums of warp w, lane l, n-tile t: [buffer][tile][warp][lane][4] (a lane stride of 16 B, as RED_W = 4)
__device__ __forceinline__ float* red16(Smem& S, int bb, int t, int w, int l) {
  return &S.red[bb][0][0][0] + (t * NW + w) * 128 + l * 4;
}
__device__ __forceinline__ const float* red16(const Smem& S, int bb, int t, int w, int l) {
  return &S.red[bb][0][0][0] + (t * NW + w) * 128 + l * 4;
}
constexpr int X1P = 512 + 8;   // a second-tile row of a warp's K range in shared memory (bf16, padded)
// rows 8..15 of job j (x rows in phase A, inter rows behind the DF sentinels in phase B), the warp's K range, into its
// own shared buffer: lane = 16 B column, two chunks per row
template <bool PB>
__device__ __forceinline__ void load_x1_smem(const Params& p, const Smem& S, int j, int lane, int warp,
                                             __nv_bfloat16* x1s) {
  const int cnt = S.jC[j], st = S.jS[j];
  uint4 v[16];
  for (int spin = 0;; ++spin) {
    uint32_t bad = 0;
#pragma unroll
    for (int k = 0; k < 16; ++k) {
      const int row = k >> 1, col = lane + ((k & 1) << 5);
      v[k] = make_uint4(0, 0, 0, 0);
      if (8 + row < cnt) {
        const __nv_bfloat16* in = PB ? p.inter + (size_t)(st + 8 + row) * ID : p.x + (size_t)(S.ent[st + 8 + row] >> 3) * HD;
        const uint4* a = reinterpret_cast<const uint4*>(in + warp * 512) + col;
        if (PB) {   // (volatile: the poll's loads stay inside the loop; a hoisted one spins on a stale value)
          asm volatile("ld.volatile.global.v4.u32 {%0, %1, %2, %3}, [%4];"
                       : "=r"(v[k].x), "=r"(v[k].y), "=r"(v[k].z), "=r"(v[k].w)
                       : "l"(a)
                       : "memory");
          bad |= df_bad(v[k]);
        } else {
          v[k] = ldp(a);
        }
      }
    }
    if (!PB || !__any_sync(0xffffffffu, bad != 0) || spin > 200000) break;
    __nanosleep(16);
  }
#pragma unroll
  for (int k = 0; k < 16; ++k) {
    const int row = k >> 1, col = lane + ((k & 1) << 5);
    *reinterpret_cast<uint4*>(x1s + row * X1P + col * 8) = v[k];
  }
  __syncwarp();
}
__device__ __forceinline__ void load_x01_df(const Params& p, const Smem& S, int j, int lane, int warp, uint4 (&x)[2 * NCH],
                                            __nv_bfloat16* x1s) {
  const int g = lane >> 2, t = lane & 3;
  const int cnt = S.jC[j], st = S.jS[j];
  const uint4* src0 = reinterpret_cast<const uint4*>(p.inter + (size_t)(st + g) * ID + warp * (64 * NCH) + 16 * t);
  uint4 v[16];
  auto ldv = [](const uint4* a, uint4& o) {
    asm volatile("ld.volatile.global.v4.u32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(o.x), "=r"(o.y), "=r"(o.z), "=r"(o.w)
                 : "l"(a)
                 : "memory");
  };
  for (int spin = 0;; ++spin) {
    uint32_t bad = 0;
#pragma unroll
    for (int c = 0; c < NCH; ++c) {   // (a two-tile job: every first-tile row exists)
      ldv(src0 + c * 8, x[2 * c]);
      ldv(src0 + c * 8 + 1, x[2 * c + 1]);
    }
#pragma unroll
    for (int k = 0; k < 16; ++k) {
      const int row = k >> 1, col = lane + ((k & 1) << 5);
      v[k] = make_uint4(0, 0, 0, 0);
      if (8 + row < cnt) ldv(reinterpret_cast<const uint4*>(p.inter + (size_t)(st + 8 + row) * ID + warp * 512) + col, v[k]);
    }
#pragma unroll
    for (int c = 0; c < 2 * NCH; ++c) bad |= df_bad(x[c]);
#pragma unroll
    for (int k = 0; k < 16; ++k) bad |= df_bad(v[k]);
    if (!__any_sync(0xffffffffu, bad != 0) || spin > 200000) break;
    __nanosleep(16);
  }
#pragma unroll
  for (int k = 0; k < 16; ++k) {
    const int row = k >> 1, col = lane + ((k & 1) << 5);
    *reinterpret_cast<uint4*>(x1s + row * X1P + col * 8) = v[k];
  }
  __syncwarp();
}
#endif
__device__ __forceinline__ uint64_t wg_desc(uint32_t saddr, uint32_t lbo, uint32_t sbo) {
  return (uint64_t)((saddr >> 4) & 0x3FFF) | ((uint64_t)((lbo >> 4) & 0x3FFF) << 16) |
         ((uint64_t)((sbo >> 4) & 0x3FFF) << 32);
}
__device__ __forceinline__ void wg_fence() { asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory"); }
template <int N>
__device__ __forceinline__ void wg_wait() {
  asm volatile("wgmma.wait_group.sync.aligned %0;\n" ::"n"(N) : "memory");
}
// skinny share [v0, v1) of CTA c when nF fat units go round robin first: every CTA ends at the same modeled cost
#if !MOE_WGS
// ---------------------------------------------------------------- warp-specialized skinny ring (tiny build)
// The NW compute warps only wait for data, compute, release the slot and hand their partial sums over; the helper
// warp (threads NT..NT+31) issues every TMA copy as soon as all compute warps released a slot and runs every unit's
// epilogue, so no CTA-wide barrier sits in the per-unit loop.
// the last WSBLK units keep their partial buffers until the compute warps' own epilogues: WSBLK > WSNR deadlocks the
// partial ring (tuner, Oct 26: every WSNR=2 / WSBLK=3 config hung)
#ifndef MOE_WSBLK   // phase-B units whose epilogues the compute warps keep (port list kwsblk: TINY2 2, MID 1)
#if MOE_TINY2
#define MOE_WSBLK 1
#else
#define MOE_WSBLK 2
#endif
#endif
__device__ __forceinline__ void ws_issue(const Params& p, Smem& S, int q, int nA, int a0, int b0, unsigned char* ring);
template <bool PB>
__device__ void skinny_ring_ws(const Params& p, Smem& S, int v0, int v1, unsigned char* ring, uint64_t* bars,
                               int& seq) {
  const int tid = TIDX;
  const int lane = tid & 31;
  const int warp = WARP_IDX(tid);
  const uint32_t k16 = 16u + p.zero;   // opaque 16 (scale decode IMAD)
  (void)k16;
#if MOE_SDEF
  const uint32_t k20 = (1u << 20) + p.zero;
#endif
  const int rbs = PB ? RB_B : RB_A;
  const int njF = S.njF;
  const int nU = v1 - v0;
  const int seq0 = seq;
  uint4 x0[2 * NCH];
  int curj = -1;
#if MOE_MID16
  static_assert(!MOE_BF16RED && !MOE_SDEF, "MID16: FP32 partials, scaled B fragments");
  __nv_bfloat16* const x1s = reinterpret_cast<__nv_bfloat16*>(ring + 3 * UNIT_BYTES) + warp * 8 * X1P;
  int curj16 = -1;
#endif
  float lgs = 0.f;   // the last phase-B unit's epilogue constants (warp 0)
  uint32_t lnw0 = 0u, lnw1 = 0u;   // (norm weights as raw bf16 bits)
#if MOE_MID16
  // one loop per n-tile count (the count changes only at a job boundary): the one-tile units run the MID loop's code,
  // not a merged loop with the two-tile blocks branched around (+36% instructions)
  auto iter = [&](int i, auto two_c) {
    constexpr bool TWO = decltype(two_c)::value;
#else
  for (int i = 0; i < nU; ++i) {
#endif
#if MOE_RAFN   // ptxas block-frequency hint (PTX 9.0 .pragma "frequency"): the unit loop is the kernel's hot path
    asm volatile(".pragma \"frequency 1000\";");
#endif
    const int v = v0 + i;
    const int j = njF + v / rbs;
#if MOE_TINY2
    if (PB && warp < MOE_WSBLK && i == ((nU - 1) - warp)) {
#endif
#if !MOE_WGS && !MOE_TINY2
    if (PB && warp < MOE_WSBLK && i == ((nU - 1) - warp)) {
#endif
      const int rbl = v - (v / RB_B) * RB_B;
      const int n0l = rbl * 16 + (lane >> 2);
      lgs = __ldg(p.gsB + S.jE[j]);
      lnw0 = (uint32_t)__ldg(reinterpret_cast<const unsigned short*>(p.normw) + n0l);
      lnw1 = (uint32_t)__ldg(reinterpret_cast<const unsigned short*>(p.normw) + n0l + 8);
    }
#if MOE_TINY && MOE_T1DF
    if (j != curj) {
      curj = j;
      if (PB)
        load_x_inter_df(p, S, j, lane, warp, x0);
      else
        load_x_regs<false>(p, S, j, lane, warp, x0);
    }
#else
    if (j != curj) {
      curj = j;
      if (PB) {
#if MOE_DF && MOE_MID16
        if constexpr (TWO)
          load_x01_df(p, S, j, lane, warp, x0, x1s);
        else
          load_x_inter_df(p, S, j, lane, warp, x0);
      } else {
        if (PB) {   // (never: the DF builds' phase B loads above; closes below as in the MID build)
#elif MOE_DF
        load_x_inter_df(p, S, j, lane, warp, x0);
      } else {
        if (PB) {
          if (lane == 0) {
#if MOE_TUNED_D16 || MOE_TUNED_V1
#if MOE_TUNED_V1
            while (ld_acquire(p.doneA + j) < RB_A) __nanosleep(16);
#endif
#if MOE_TUNED_D16
            while (ld_acquire(p.doneA + j) < RB_A) __nanosleep(32);
#endif
#endif
#if MOE_DF && !MOE_TUNED_D16 && !MOE_TUNED_V1
            while (ld_acquire(p.doneA + j) < RB_A) __nanosleep(MOE_SPOLL2 ? 128 : 64);
#endif
          }
          __syncwarp();
#endif
#if MOE_TINY
        if (warp == 0 && lane == 0) {
          if (MOE_STAMP && i == 0) MST(8);
          while (ld_acquire(p.doneA + j) < RB_A) __nanosleep(32);
          if (MOE_STAMP && i == 0) MST(9);
#endif
        }
#if MOE_DF
        load_x_regs<PB>(p, S, j, lane, warp, x0);
#endif
#if MOE_TINY
        CTA_SYNC();
#endif
      }
#if MOE_TINY
      load_x_regs<PB>(p, S, j, lane, warp, x0);
#endif
    }
#endif
#if MOE_MID16
    if constexpr (TWO && !PB) {   // (phase B: load_x01_df above)
      if (j != curj16) {
        curj16 = j;
        load_x1_smem<false>(p, S, j, lane, warp, x1s);
      }
    }
#endif
    const int q = seq0 + i;
    const int sl = q % RS;
    mbar_wait(bars + sl, (uint32_t)((q / RS) & 1));
    if (MOE_STAMP && tid == 0 && i == 0) MST(PB ? 10 : 12);
    if (MOE_STAMP && tid == 0 && q < 8) MST(16 + q);
    if (MOE_STAMP && tid == 0 && PB && i < 8) MST(48 + i);   // (phase B unit i landed)
    const unsigned char* slot = ring + sl * UNIT_BYTES;
#if MOE_SC6S   // the three planes -> the lane's 16 scale bytes (as the base layout's uint4)
    uint4 sc;
    {
      const uint32_t* pl = reinterpret_cast<const uint32_t*>(slot + UNIT_CODES) + warp * 32 + lane;
      const uint32_t w0 = pl[0], w1 = pl[256], w2 = pl[512];
      sc.x = (w0 & 0x3F3F3F3Fu) | 0x40404040u;
      sc.y = (w1 & 0x3F3F3F3Fu) | 0x40404040u;
      sc.z = (w2 & 0x3F3F3F3Fu) | 0x40404040u;
      sc.w = ((w0 >> 6) & 0x03030303u) | ((w1 >> 4) & 0x0C0C0C0Cu) | ((w2 >> 2) & 0x30303030u) | 0x40404040u;
    }
#else
    const uint4 sc = *reinterpret_cast<const uint4*>(slot + UNIT_CODES + warp * 512 + lane * 16);
#endif
#if MOE_SDEF
    float acc[4];
    {
      float cq[NCH / 2][4];
#pragma unroll
      for (int q = 0; q < NCH / 2; ++q) cq[q][0] = cq[q][1] = cq[q][2] = cq[q][3] = 0.f;
#pragma unroll
      for (int c = 0; c < NCH; ++c) {
        const uint4 w = *reinterpret_cast<const uint4*>(slot + (warp * NCH + c) * 512 + lane * 16);
        const uint32_t ww[4] = {w.x, w.y, w.z, w.w};
        const uint32_t b[8] = {x0[2 * c].x,     x0[2 * c].y,     x0[2 * c].z,     x0[2 * c].w,
                               x0[2 * c + 1].x, x0[2 * c + 1].y, x0[2 * c + 1].z, x0[2 * c + 1].w};
#pragma unroll
        for (int st = 0; st < 4; ++st) {
          uint32_t r0, r1, r2, r3;
          dq4(ww[st], r0, r1, r2, r3);
          mma_bf16(cq[c >> 1], r0, r1, r2, r3, b[2 * st], b[2 * st + 1]);
        }
      }
      // C column 2t + h of pair q is block 8q + 2t + h; scale bytes per lane and pair: (row g, h 0 / 1), (row g + 8, ...)
      float a0 = 0.f, a1 = 0.f;
#pragma unroll
      for (int q = 0; q < NCH / 2; ++q) {
        const uint32_t sw = q == 0 ? sc.x : q == 1 ? sc.y : q == 2 ? sc.z : sc.w;
        a0 = fmaf(SCDF(sw, 0, k20), cq[q][0], a0);
        a0 = fmaf(SCDF(sw, 1, k20), cq[q][1], a0);
        a1 = fmaf(SCDF(sw, 2, k20), cq[q][2], a1);
        a1 = fmaf(SCDF(sw, 3, k20), cq[q][3], a1);
      }
      a0 += __shfl_xor_sync(0xffffffffu, a0, 1);
      a1 += __shfl_xor_sync(0xffffffffu, a1, 1);
      a0 += __shfl_xor_sync(0xffffffffu, a0, 2);
      a1 += __shfl_xor_sync(0xffffffffu, a1, 2);
      const bool t0 = (lane & 3) == 0;   // the token's column 0 (the epilogue reads columns < cnt = 1)
      acc[0] = t0 ? a0 : 0.f;
      acc[1] = 0.f;
      acc[2] = t0 ? a1 : 0.f;
      acc[3] = 0.f;
    }
#else
    float acc[4] = {0.f, 0.f, 0.f, 0.f}, acc2[4] = {0.f, 0.f, 0.f, 0.f};
    float acc3[4] = {0.f, 0.f, 0.f, 0.f}, acc4[4] = {0.f, 0.f, 0.f, 0.f};
#if !MOE_MID16
#pragma unroll
    for (int c = 0; c < NCH; ++c) {
      const uint4 w = *reinterpret_cast<const uint4*>(slot + (warp * NCH + c) * 512 + lane * 16);
      const uint32_t swd = c < 4 ? (c < 2 ? sc.x : sc.y) : (c < 6 ? sc.z : sc.w);
      const uint32_t sg = SCDK(swd, (c & 1) * 2, k16);
      const uint32_t sg8 = SCDK(swd, (c & 1) * 2 + 1, k16);
      const uint32_t ww[4] = {w.x, w.y, w.z, w.w};
      const uint32_t b[8] = {x0[2 * c].x,     x0[2 * c].y,     x0[2 * c].z,     x0[2 * c].w,
                             x0[2 * c + 1].x, x0[2 * c + 1].y, x0[2 * c + 1].z, x0[2 * c + 1].w};
#pragma unroll
      for (int st = 0; st < 4; ++st) {
        uint32_t r0, r1, r2, r3;
        dq4(ww[st], r0, r1, r2, r3);
        mma_bf16(st == 0   ? acc
                 : st == 1 ? acc2
                 : st == 2 ? acc3
                           : acc4,
                 bmul2(r0, sg), bmul2(r1, sg8), bmul2(r2, sg), bmul2(r3, sg8), b[2 * st], b[2 * st + 1]);
      }
    }
#else
    // one decode, one or two n-tiles (the second tile's B fragments from shared memory, two chains of its own)
    float bcc[4] = {0.f, 0.f, 0.f, 0.f}, bcc2[4] = {0.f, 0.f, 0.f, 0.f};
    auto unit = [&](auto two_c) {
      constexpr bool TWO = decltype(two_c)::value;
      const __nv_bfloat16* x1r = x1s + (lane >> 2) * X1P + 16 * (lane & 3);
#pragma unroll
      for (int c = 0; c < NCH; ++c) {
        const uint4 w = *reinterpret_cast<const uint4*>(slot + (warp * NCH + c) * 512 + lane * 16);
        const uint32_t swd = c < 4 ? (c < 2 ? sc.x : sc.y) : (c < 6 ? sc.z : sc.w);
        const uint32_t sg = SCDK(swd, (c & 1) * 2, k16);
        const uint32_t sg8 = SCDK(swd, (c & 1) * 2 + 1, k16);
        const uint32_t ww[4] = {w.x, w.y, w.z, w.w};
        const uint32_t b[8] = {x0[2 * c].x,     x0[2 * c].y,     x0[2 * c].z,     x0[2 * c].w,
                               x0[2 * c + 1].x, x0[2 * c + 1].y, x0[2 * c + 1].z, x0[2 * c + 1].w};
        uint32_t b1[8];
        if constexpr (TWO) {
          const uint4 u0 = *reinterpret_cast<const uint4*>(x1r + c * 64), u1 = *reinterpret_cast<const uint4*>(x1r + c * 64 + 8);
          b1[0] = u0.x, b1[1] = u0.y, b1[2] = u0.z, b1[3] = u0.w, b1[4] = u1.x, b1[5] = u1.y, b1[6] = u1.z, b1[7] = u1.w;
        }
#pragma unroll
        for (int st = 0; st < 4; ++st) {
          uint32_t r0, r1, r2, r3;
          dq4(ww[st], r0, r1, r2, r3);
          r0 = bmul2(r0, sg), r1 = bmul2(r1, sg8), r2 = bmul2(r2, sg), r3 = bmul2(r3, sg8);
          mma_bf16(st == 0   ? acc
                   : st == 1 ? acc2
                   : st == 2 ? acc3
                             : acc4,
                   r0, r1, r2, r3, b[2 * st], b[2 * st + 1]);
          if constexpr (TWO) mma_bf16((st & 1) ? bcc2 : bcc, r0, r1, r2, r3, b1[2 * st], b1[2 * st + 1]);
        }
      }
    };
    unit(std::integral_constant<bool, TWO>{});
#pragma unroll
    for (int k = 0; k < 4; ++k) bcc[k] += bcc2[k];
#endif
#pragma unroll
    for (int k = 0; k < 4; ++k) acc[k] += acc2[k] + acc3[k] + acc4[k];
#endif
    // WSB: the compute warps keep the base loop's CTA barrier per unit (slot consumed + partials visible), thread 0
    // refills the slot and hands the partial buffer to the helper warp, which runs every epilogue off this loop
    const int bb = q % NRB;
    mbar_wait(S.wsre + bb, (uint32_t)(((q / NRB) & 1) ^ 1));   // the helper has consumed this buffer's previous unit
#if MOE_BF16RED   // 8 B per lane at an 8 B stride in the buffer's first half
    {
      __nv_bfloat162 lo = __floats2bfloat162_rn(acc[0], acc[1]), hi = __floats2bfloat162_rn(acc[2], acc[3]);
      reinterpret_cast<uint2*>(&S.red[bb][warp][0][0])[lane] =
          make_uint2(*reinterpret_cast<uint32_t*>(&lo), *reinterpret_cast<uint32_t*>(&hi));
    }
#elif !MOE_MID16
    *reinterpret_cast<float4*>(&S.red[bb][warp][lane][0]) = make_float4(acc[0], acc[1], acc[2], acc[3]);
#else
    *reinterpret_cast<float4*>(red16(S, bb, 0, warp, lane)) = make_float4(acc[0], acc[1], acc[2], acc[3]);
    if constexpr (TWO) *reinterpret_cast<float4*>(red16(S, bb, 1, warp, lane)) = make_float4(bcc[0], bcc[1], bcc[2], bcc[3]);
#endif
    CTA_SYNC();
    if (MOE_STAMP && PB && tid == 0 && i == nU - 1) MST(11);
    if (MOE_STAMP && tid == 0 && q < 8) MST(40 + q);
    if (MOE_STAMP && tid == 0 && PB && i < 8) MST(56 + i);   // (phase B unit i handed over)
    if (tid == 0) {
      const int nA = S.wsr[1] - S.wsr[0];
      const int N = nA + (S.wsr[3] - S.wsr[2]);
#if MOE_HCRED & 2   // the partials to the helper before the refill's issue (the TMA issue can stall ~0.2 us)
      if (!(PB && i >= (nU - MOE_WSBLK))) mbar_arrive1(S.wsrf + bb);
#endif
      if (MOE_IWARP) {
      } else if (RS == 3) {
        if ((q + 3) < N) ws_issue(p, S, q + 3, nA, S.wsr[0], S.wsr[2], ring);
      } else if (MOE_XSLOT == 1) {   // 3 ahead through phase A, 4 ahead from the last A unit on (both at that unit)
        if (q < nA && (q + 3) < N) ws_issue(p, S, q + 3, nA, S.wsr[0], S.wsr[2], ring);
        if (q >= nA - 1 && (q + 4) < N) ws_issue(p, S, q + 4, nA, S.wsr[0], S.wsr[2], ring);
      } else {   // 3 ahead in phase A, 4 ahead in phase B; unit nA + 3 is the helper's
        const int nq = q + 3 + (q >= nA);
        if (nq < N) ws_issue(p, S, nq, nA, S.wsr[0], S.wsr[2], ring);
      }
      if (MOE_STAMP && q < 8) MST(64 + q);   // (thread 0's refill issued)
#if MOE_TINY2 && !(MOE_HCRED & 2)
      if (!(PB && i >= (nU - MOE_WSBLK))) mbar_arrive1(S.wsrf + bb);   // WSBL: the last units stay with warps 0..K-1
#endif
#if !MOE_WGS && !MOE_TINY2 && !(MOE_HCRED & 2)
      if (!(PB && i >= (nU - MOE_WSBLK))) mbar_arrive1(S.wsrf + bb);   // WSBL: the last units stay with warps 0..K-1
#endif
    }
#if MOE_MID16
  };
  for (int i = 0; i < nU;) {
    const int jl = (v0 + i) / rbs;
    const int e = min(nU, (jl + 1) * rbs - v0);
    if (S.jC[njF + jl] > 8)
      for (; i < e; ++i) iter(i, std::true_type{});
    else
      for (; i < e; ++i) iter(i, std::false_type{});
  }
#else
  }
#endif
#if MOE_TINY2
  if (PB && warp < MOE_WSBLK && warp < nU) {   // the CTA's last units: epilogues here, not behind the helper's queue
#endif
#if !MOE_WGS && !MOE_TINY2
  if (PB && warp < MOE_WSBLK && warp < nU) {   // the CTA's last units: epilogues here, not behind the helper's queue
#endif
    const int ul = nU - 1 - warp;
    const int q = seq0 + ul;
    const int bb = q % NRB;
    const int v = v0 + ul;
    const int j = njF + v / RB_B;
    const int rb = v - (v / RB_B) * RB_B;
    const int gg = lane >> 2, tt = lane & 3;
    float4 sm = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
    for (int w = 0; w < NW; ++w) {
#if MOE_BF16RED
      const uint2 u = reinterpret_cast<const uint2*>(&S.red[bb][w][0][0])[lane];
      const float4 t = make_float4(__uint_as_float(u.x << 16), __uint_as_float(u.x & 0xffff0000u),
                                   __uint_as_float(u.y << 16), __uint_as_float(u.y & 0xffff0000u));
#elif !MOE_MID16
      const float4 t = *reinterpret_cast<const float4*>(&S.red[bb][w][lane][0]);
#else
      const float4 t = *reinterpret_cast<const float4*>(red16(S, bb, 0, w, lane));
#endif
      sm.x += t.x;
      sm.y += t.y;
      sm.z += t.z;
      sm.w += t.w;
    }
    const int cnt = S.jC[j];
    const int st = S.jS[j];
    const int ce = S.jE[j];
    const int n0 = rb * 16 + gg;
    const float gsc = lgs * TWO62;
    const float nw0 = 1.f + __uint_as_float(lnw0 << 16), nw1 = 1.f + __uint_as_float(lnw1 << 16);
    auto epi = [&](const float4& sm, int t0) {
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int tk = t0 + 2 * tt + h;
        const float v0f = (h ? sm.y : sm.x) * gsc * nw0, v1f = (h ? sm.w : sm.z) * gsc * nw1;
        float sq = v0f * v0f + v1f * v1f;
        if (tk < cnt) {
          st_keep_f32(p.ybuf + ((size_t)(st + tk) * HD + n0), v0f);
          st_keep_f32(p.ybuf + ((size_t)(st + tk) * HD + n0 + 8), v1f);
        }
        sq += __shfl_xor_sync(0xffffffffu, sq, 4);
        sq += __shfl_xor_sync(0xffffffffu, sq, 8);
        sq += __shfl_xor_sync(0xffffffffu, sq, 16);
        if (gg == 0 && tk < cnt) atomicAdd(&ss_cur()[st + tk], sq);
      }
    };
    epi(sm, 0);
#if MOE_MID16
    if (__shfl_sync(0xffffffffu, cnt, 0) > 8) {   // the second tile: tokens 8..15 (shfl'd: warp-uniform for ptxas)
      float4 s1 = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
      for (int w = 0; w < NW; ++w) {
        const float4 t = *reinterpret_cast<const float4*>(red16(S, bb, 1, w, lane));
        s1.x += t.x;
        s1.y += t.y;
        s1.z += t.z;
        s1.w += t.w;
      }
      epi(s1, 8);
    }
#endif
    (void)ce;
  }
  seq = seq0 + nU;
}
#if !MOE_WGS && !MOE_MID && !MOE_TUNED_V1

__device__ __forceinline__ void ws_issue(const Params& p, Smem& S, int q, int nA, int a0, int b0, unsigned char* ring) {
  const int sl = q % RS;
  if (q < nA)
    ring_issue<false>(p, S, a0 + q, ring + sl * UNIT_BYTES, S.bars + sl);
  else
    ring_issue<true>(p, S, b0 + (q - nA), ring + sl * UNIT_BYTES, S.bars + sl);
}

__device__ void ws_helper(const Params& p, Smem& S, unsigned char* ring) {
  const int lane = TIDX & 31;
  WS_JOIN();   // routing tables and this CTA's ranges are ready
  const int a0 = S.wsr[0];
  const int a1 = S.wsr[1];
  const int b0 = S.wsr[2];
  const int b1 = S.wsr[3];
  const int nA = a1 - a0;
  const int N = nA + (b1 - b0);
  const int njF = S.njF;
  if (lane == 0)
    for (int q = S.preissued; q < 3 && q < N; ++q) ws_issue(p, S, q, nA, a0, b0, ring);
  const int gg = lane >> 2, tt = lane & 3;
#endif
#if MOE_TINY2 && !MOE_TUNED_V1
  const int Nh = (true) ? N - min(MOE_WSBLK, b1 - b0) : N;
#endif
#if MOE_TINY
  const int Nh = (true) ? N - min(MOE_WSBLK, b1 - b0) : N;
#endif
#if !MOE_WGS && !MOE_MID && !MOE_TUNED_V1
  for (int q = 0; q < Nh; ++q) {
    const bool PB = q >= nA;
    const int v = PB ? b0 + (q - nA) : a0 + q;
    const int rbs = PB ? RB_B : RB_A;
    const int j = njF + v / rbs;
    const int rb = v - (v / rbs) * rbs;
    const int bb = q % NRB;
    mbar_wait(S.wsrf + bb, (uint32_t)((q / NRB) & 1));
    if (MOE_STAMP && lane == 0 && q == 0) MST(13);
    if (MOE_STAMP && lane == 0 && q < 8) MST(32 + q);
    float4 sm = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
    for (int w = 0; w < NW; ++w) {
#if MOE_BF16RED
      const uint2 u = reinterpret_cast<const uint2*>(&S.red[bb][w][0][0])[lane];
      const float4 t = make_float4(__uint_as_float(u.x << 16), __uint_as_float(u.x & 0xffff0000u),
                                   __uint_as_float(u.y << 16), __uint_as_float(u.y & 0xffff0000u));
#else
      const float4 t = *reinterpret_cast<const float4*>(&S.red[bb][w][lane][0]);
#endif
      sm.x += t.x;
      sm.y += t.y;
      sm.z += t.z;
      sm.w += t.w;
    }
    __syncwarp();
    if (lane == 0) mbar_arrive1(S.wsre + bb);   // buffer free again
    // sm: (row gg, token 2tt), (row gg, token 2tt+1), (row gg+8, token 2tt), (row gg+8, token 2tt+1)
    const int cnt = S.jC[j];
    const int st = S.jS[j];
    const int ce = S.jE[j];
    if (!PB) {
      const float gsc = p.gsA[ce] * TWO62;
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int tk = 2 * tt + h;
        if (tk < cnt) {
          const float gt = (h ? sm.y : sm.x) * gsc, up = (h ? sm.w : sm.z) * gsc;
          st_keep_bf16(p.inter + ((size_t)(st + tk) * ID + rb * 8 + gg),
                       __float2bfloat16((moe_silu_s(gt) * up)));
        }
      }
#endif
#if MOE_TINY2 && !MOE_TUNED_V1

#endif
#if MOE_TINY
      if (q == (nA - 1)) {   // this CTA's phase A is done: credit its units to the jobs' counters
        __syncwarp();
        if (lane == 0) {
          const int jf = njF + a0 / RB_A;
          const int jl = njF + (a1 - 1) / RB_A;
          if (MOE_STAMP) MST(7);   // (the last A epilogue's stores issued; the credit's fence next)
          if ((MOE_HCRED & 1) && !MOE_T1DF) asm volatile("fence.acq_rel.gpu;\n" ::: "memory");   // one fence for every job's credit
          for (int jj = jf; jj <= jl && !MOE_T1DF; ++jj) {   // (T1DF: phase B polls the data itself)
            const int a = max(a0, (jj - njF) * RB_A);
            const int b = min(a1, (jj - njF + 1) * RB_A);
            if (MOE_HCRED & 1)
              asm volatile("red.relaxed.gpu.global.add.s32 [%0], %1;\n" ::"l"(p.doneA + jj), "r"(b - a) : "memory");
            else
              asm volatile("red.release.gpu.global.add.s32 [%0], %1;\n" ::"l"(p.doneA + jj), "r"(b - a) : "memory");
          }
          if (MOE_STAMP) MST(15);
          if (MOE_XSLOT == 2 && nA + 3 < N) ws_issue(p, S, nA + 3, nA, a0, b0, ring);   // (its slot: the last A unit's)
        }
      }
#endif
#if !MOE_WGS && !MOE_MID && !MOE_TUNED_V1
    } else {
      const float gsc = p.gsB[ce] * TWO62;
      const int n0 = rb * 16 + gg;
      const float nw0 = 1.f + __bfloat162float(p.normw[n0]), nw1 = 1.f + __bfloat162float(p.normw[n0 + 8]);
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int tk = 2 * tt + h;
        const float v0f = (h ? sm.y : sm.x) * gsc * nw0, v1f = (h ? sm.w : sm.z) * gsc * nw1;
        float sq = v0f * v0f + v1f * v1f;
        if (tk < cnt) {
          st_keep_f32(p.ybuf + ((size_t)(st + tk) * HD + n0), v0f);
          st_keep_f32(p.ybuf + ((size_t)(st + tk) * HD + n0 + 8), v1f);
        }
        sq += __shfl_xor_sync(0xffffffffu, sq, 4);
        sq += __shfl_xor_sync(0xffffffffu, sq, 8);
        sq += __shfl_xor_sync(0xffffffffu, sq, 16);
        if (gg == 0 && tk < cnt) atomicAdd(&ss_cur()[st + tk], sq);
      }
    }
    if (MOE_STAMP && lane == 0 && q == 0) MST(14);
    if (MOE_STAMP && lane == 0 && q < 8) MST(24 + q);
  }
  if (!MOE_BREL) __threadfence();   // (BREL: thread 0's release arrive at the grid barrier covers these stores)
  WS_JOIN();   // every epilogue store of this CTA is done before the grid barrier
}

#endif
#endif

#if MOE_MID || MOE_TUNED_V1
__device__ __forceinline__ void ws_issue(const Params& p, Smem& S, int q, int nA, int a0, int b0, unsigned char* ring) {
  const int sl = q % 3;
  if (q < nA)
    ring_issue<false>(p, S, a0 + q, ring + sl * UNIT_BYTES, S.bars + sl);
  else
    ring_issue<true>(p, S, b0 + (q - nA), ring + sl * UNIT_BYTES, S.bars + sl);
}

__device__ void ws_helper(const Params& p, Smem& S, unsigned char* ring) {
  const int lane = TIDX & 31;
#if MOE_MID16
  // helper warp 0: every unit's first n-tile; 1: two-tile units' second (shfl'd: provably warp-uniform, else ptxas
  // guards the epilogue's shuffles with BRA.DIV / WARPSYNC)
  const int tile = __shfl_sync(0xffffffffu, (TIDX - NT) >> 5, 0);
#endif
  WS_JOIN();   // routing tables and this CTA's ranges are ready
  const int a0 = S.wsr[0];
  const int a1 = S.wsr[1];
  const int b0 = S.wsr[2];
  const int b1 = S.wsr[3];
  const int nA = a1 - a0;
  const int N = nA + (b1 - b0);
  const int njF = S.njF;
#if MOE_MID16
  if (lane == 0 && tile == 0)
#else
  if (lane == 0)
#endif
    for (int q = S.preissued; q < 3 && q < N; ++q) ws_issue(p, S, q, nA, a0, b0, ring);
  const int gg = lane >> 2, tt = lane & 3;
#if MOE_TUNED_V1
  const int Nh = (true) ? N - min(MOE_WSBLK, b1 - b0) : N;
#endif
#if MOE_MID
  const int Nh = (true) ? N - min(MOE_WSBLK, b1 - b0) : N;
#endif
  for (int q = 0; q < Nh; ++q) {
    const bool PB = q >= nA;
    const int v = PB ? b0 + (q - nA) : a0 + q;
    const int rbs = PB ? RB_B : RB_A;
    const int j = njF + v / rbs;
    const int rb = v - (v / rbs) * rbs;
    const int bb = q % NRB;
    mbar_wait(S.wsrf + bb, (uint32_t)((q / NRB) & 1));
    float4 sm = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
    for (int w = 0; w < NW; ++w) {
#if MOE_BF16RED
      const uint2 u = reinterpret_cast<const uint2*>(&S.red[bb][w][0][0])[lane];
      const float4 t = make_float4(__uint_as_float(u.x << 16), __uint_as_float(u.x & 0xffff0000u),
                                   __uint_as_float(u.y << 16), __uint_as_float(u.y & 0xffff0000u));
#elif !MOE_MID16
      const float4 t = *reinterpret_cast<const float4*>(&S.red[bb][w][lane][0]);
#else
      const float4 t = *reinterpret_cast<const float4*>(red16(S, bb, tile, w, lane));
#endif
      sm.x += t.x;
      sm.y += t.y;
      sm.z += t.z;
      sm.w += t.w;
    }
    __syncwarp();
    if (lane == 0) mbar_arrive1(S.wsre + bb);   // buffer free again (MID16: both helper warps arrive)
    // sm: (row gg, token 2tt), (row gg, token 2tt+1), (row gg+8, token 2tt), (row gg+8, token 2tt+1)
    const int cnt = S.jC[j];
    const int st = S.jS[j];
    const int ce = S.jE[j];
    if (!PB) {
      const float gsc = p.gsA[ce] * TWO62;
      auto epi = [&](const float4& sm, int t0) {
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          const int tk = t0 + 2 * tt + h;
          if (tk < cnt) {
            const float gt = (h ? sm.y : sm.x) * gsc, up = (h ? sm.w : sm.z) * gsc;
            st_keep_bf16(p.inter + ((size_t)(st + tk) * ID + rb * 8 + gg),
                         __float2bfloat16((moe_silu_s(gt) * up)));
          }
        }
      };
#if MOE_MID16
      if (tile == 0 || __shfl_sync(0xffffffffu, cnt, 0) > 8) epi(sm, 8 * tile);
#else
      epi(sm, 0);
#endif

    } else {
      const float gsc = p.gsB[ce] * TWO62;
      const int n0 = rb * 16 + gg;
      const float nw0 = 1.f + __bfloat162float(p.normw[n0]), nw1 = 1.f + __bfloat162float(p.normw[n0 + 8]);
      auto epi = [&](const float4& sm, int t0) {
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          const int tk = t0 + 2 * tt + h;
          const float v0f = (h ? sm.y : sm.x) * gsc * nw0, v1f = (h ? sm.w : sm.z) * gsc * nw1;
          float sq = v0f * v0f + v1f * v1f;
          if (tk < cnt) {
            st_keep_f32(p.ybuf + ((size_t)(st + tk) * HD + n0), v0f);
            st_keep_f32(p.ybuf + ((size_t)(st + tk) * HD + n0 + 8), v1f);
          }
          sq += __shfl_xor_sync(0xffffffffu, sq, 4);
          sq += __shfl_xor_sync(0xffffffffu, sq, 8);
          sq += __shfl_xor_sync(0xffffffffu, sq, 16);
          if (gg == 0 && tk < cnt) atomicAdd(&ss_cur()[st + tk], sq);
        }
      };
#if MOE_MID16
      if (tile == 0 || __shfl_sync(0xffffffffu, cnt, 0) > 8) epi(sm, 8 * tile);
#else
      epi(sm, 0);
#endif
    }
  }
  if (!MOE_BREL) __threadfence();
  WS_JOIN();   // every epilogue store of this CTA is done before the grid barrier
}

__device__ __forceinline__ int fresh_b_cost_position(int position) {
  constexpr int cost = RB_B + MOE_BXW;
  const int job = position / cost;
  return job * RB_B + max(0, position - job * cost - MOE_BXW);
}
#if MOE_AXW
__device__ __forceinline__ int fresh_a_cost_position(int position) {
  constexpr int cost = RB_A + MOE_AXW;
  const int job = position / cost;
  return job * RB_A + max(0, position - job * cost - MOE_AXW);
}
#endif
#endif
__device__ __forceinline__ void skinny_split(int nS, int nF, int G, int c, int& v0, int& v1) {
  const float rG = __fdividef(1.f, (float)G);
  const int q = muldiv_g(nF, 1, G, rG);
  const int r = nF - q * G;
  auto fcost = [](long long k) -> long long {
    return k > 0 ? 40ll * k : 0;
  };
  const long long T =
      (long long)nS + (long long)r * fcost(q + 1) + (long long)(G - r) * fcost(q);   // total cost = G x level
  long long wA = T - (long long)G * fcost(q + 1);
  if (wA < 0) wA = 0;
  long long wB = T - (long long)G * fcost(q);
  if (wB < 0) wB = 0;
  auto W = [&](int cc) -> long long {
    return cc <= r ? (long long)cc * wA : (long long)r * wA + (long long)(cc - r) * wB;
  };
  const long long WG = W(G);
  if (nF > 0 && WG > 0) {
    // split point of CTA x: floor(nS * W(x) / WG) in float (one formula for both neighbours, so the shares tile [0, nS))
    const float sc = __fdividef((float)nS, (float)WG);
    auto split = [&](int x) -> int {
      if (x >= G) return nS;
      const int v = __float2int_rz((float)W(x) * sc);
      return v < nS ? v : nS;
    };
    v0 = split(c);
    v1 = split(c + 1);
  } else {
    v0 = muldiv_g(nS, c, G, rG);
    v1 = muldiv_g(nS, c + 1, G, rG);
  }
}

#if MOE_WGS
// ---------------------------------------------------------------- WGS: RS-wgmma for every job (row split, stream-K)
// Unit = (job, 256-row block): phase A 32 per job, phase B 16, each 16 stages of 256 K. Stage = 36 KB of weights,
// for each warpgroup q its 64 rows: 8 KB codes [grp][warp][lane][4 words] (natural mma k order) + 1 KB scales
// [grp][warp][g][h][j]; the stage's x slice (the job's token rows, rounded up to 8, x 256 K as wgmma B core matrices
// [n8][32 k8][8 rows][16 B]) sits in its own ring. Warpgroup q accumulates its 64 rows over K with wgmma m64nNk16
// (A = dequantized codes in registers, N = 8 * n-tiles) and runs its own epilogue (rows g / g + 8 are gate / up).
// Work is split by cost-weighted stage ranges (stream-K): a unit cut between two CTAs adds its partial sums into a
// global buffer (indexed by the CTA boundary) and the second contributor runs the epilogue.
constexpr int WGS_NS = 3;   // weight ring slots (36 KB; freed as soon as the codes are in registers)
constexpr int WGS_NX = 3;   // x ring slots (freed when the stage's last wgmma retired)
constexpr int WGS_NT = FT_T / 8;   // max n-tiles of a job
constexpr int WGS_PIECE = 8960;   // one warpgroup's weights in a stage (SC6: 8 KB codes + 768 B scales)
constexpr int WGS_STG = 35840;   // bytes of a stage in global memory (the smem slot keeps UNIT_BYTES)
constexpr int WGS_XS = WGS_NT * 4096;   // an x slot: up to FT_T rows x 256 K
constexpr int BW_F = 0;
constexpr int BW_E = 3;
constexpr int BX_F = 6;
constexpr int BX_E = 6 + WGS_NX;   // barrier indices
static_assert(BX_E + WGS_NX <= 24, "WGS barriers");
constexpr int WGS_SLOTS = 160;   // fixup slots per phase (>= grid + 1)
constexpr int WGS_FIXW = WGS_NT * 512;   // floats per (slot, warpgroup): [n-tile][128 threads][4]
   // + doneAs
__device__ __forceinline__ float* wgs_fix(const Params& p, int pb) {
  return p.ybuf + (size_t)EMAX * HD + (size_t)pb * WGS_SLOTS * 4 * WGS_FIXW;
}
__device__ __forceinline__ int* wgs_cnt(const Params& p, int pb) {
  return reinterpret_cast<int*>(p.ybuf + (size_t)EMAX * HD + (size_t)2 * WGS_SLOTS * 4 * WGS_FIXW) + pb * WGS_SLOTS * 4;
}
constexpr int MF_CE = 0;
constexpr int MF_EXIT = 64;
constexpr int MF_XR = 128;
constexpr int MF_XRS = 96;
constexpr int MF_DA = 384;
constexpr int MF_INTS = 4096;
static_assert(JMAX <= MF_XRS && MF_DA + 2 * JMAX * 16 <= MF_INTS, "MFIN flag region");
// phase-A hand-off flags: one set per launch parity under MFIN (the old code zeroed them after the grid barrier)
// (volatile shared load: a plain S.gen load is loop-invariant and would stay live in a register across the stage loops)
__device__ __forceinline__ int mf_gen(const Smem& S) {
  int v;
  asm volatile("ld.volatile.shared.s32 %0, [%1];\n" : "=r"(v) : "r"(smem_u32(&S.gen)));
  return v;
}
__device__ __forceinline__ int* mf_xr(const Params& p, const Smem& S) {
  return p.xready + ((1 && (47 & 2)) ? mf_gen(S) * MF_XRS : 0);
}
__device__ __forceinline__ int* mf_da(const Params& p, const Smem& S) {
  return p.doneAs + ((1 && (47 & 1)) ? mf_gen(S) * (JMAX * 16) : 0);
}
// per-stage cost by n-tiles (1..8), phase A and phase B (relative units, <= 255; fit on per-CTA phase times at
// bs 96..256, t_fit2.py): tuner knobs wgsa / wgsb = 8 comma-separated values each
#ifndef MOE_CA   // per-shape phase-A table (d32s: the 2- / 4-tile stages measured against the 1-tile ones)
#define MOE_CA 0
#endif
#if MOE_CA == 1
#define WGS_COST_A 112, 135, 140, 166, 173, 185, 200, 207
#elif MOE_CA == 2
#define WGS_COST_A 112, 135, 140, 174, 173, 185, 200, 207
#elif MOE_CA == 3
#define WGS_COST_A 112, 128, 140, 166, 173, 185, 200, 207
#elif MOE_CA == 4
#define WGS_COST_A 112, 135, 140, 182, 173, 185, 200, 207
#elif MOE_CA == 5
#define WGS_COST_A 112, 142, 140, 174, 173, 185, 200, 207
#elif MOE_CA == 6
#define WGS_COST_A 112, 150, 140, 174, 173, 185, 200, 207
#elif MOE_CA == 7
#define WGS_COST_A 112, 135, 140, 190, 173, 185, 200, 207
#elif MOE_CA == 8
#define WGS_COST_A 112, 135, 140, 198, 173, 185, 200, 207
#elif MOE_CA == 9
#define WGS_COST_A 112, 135, 140, 214, 173, 185, 200, 207
#elif MOE_CA == 10
#define WGS_COST_A 112, 135, 140, 230, 173, 185, 200, 207
#elif MOE_CA == 11
#define WGS_COST_A 112, 135, 140, 250, 173, 185, 200, 207
#elif MOE_CA == 12
#define WGS_COST_A 112, 160, 140, 214, 173, 185, 200, 207
#elif MOE_CA == 13
#define WGS_COST_A 112, 185, 140, 214, 173, 185, 200, 207
#else
#define WGS_COST_A \
  112, 135, 140, 158, 173, 185, 200, 207   // tuner combo_m798 on v182x (Oct 26: bs32 +0.45, bs128 +0.94, bs256 +0.17%)
#endif
#define WGS_COST_B 100, 104, 115, 123, 132, 144, 156, 174
constexpr unsigned long long wgs_pack8(const int (&c)[8]) {
  unsigned long long r = 0;
  for (int i = 0; i < 8; ++i) r |= (unsigned long long)(c[i] & 255) << (8 * i);
  return r;
}
constexpr int WGS_CA[8] = {WGS_COST_A};
constexpr int WGS_CB[8] = {WGS_COST_B};
constexpr unsigned long long WGS_PA = wgs_pack8(WGS_CA), WGS_PB = wgs_pack8(WGS_CB);
static_assert(WGS_CA[0] > 0 && WGS_CA[7] < 256 && WGS_CB[0] > 0 && WGS_CB[7] < 256, "WGS cost tables: 1..255");
#if MOE_TUNED_D32
__device__ __forceinline__ int fresh_bucket_width(int n);
#endif
__device__ __forceinline__ int wgs_cost(int pb, int cnt) {
#if MOE_TUNED_D32
  // Charge the tile width the math kernel executes, including rounded-up jobs.
  return (int)(((pb ? WGS_PB : WGS_PA) >> (8 * (fresh_bucket_width((cnt + 7) >> 3) - 1))) & 255ull);
#else
  return (int)(((pb ? WGS_PB : WGS_PA) >> (8 * (((cnt + 7) >> 3) - 1))) & 255ull);
#endif
}
// the job at position p of phase pb's order
__device__ __forceinline__ int wgs_job(const Smem& S, int pb, int p) {
  return (MOE_BREV && pb) ? max(S.njF + S.njS - 1 - p, 0) : p;   // (no jobs: stage_at's position 0 stays in range)
}
__device__ __forceinline__ void wgs_wait(uint64_t* bar, uint32_t parity, int tag) {
  (void)tag;
  mbar_wait(bar, parity);
}
template <int NTT>
struct WGSMMA;
template <>
struct WGSMMA<1> {
  static __device__ __forceinline__ void mma0(float (&d)[WGS_NT][4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                              uint64_t desc) {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 {%0, %1, %2, %3}, {%4, %5, %6, %7}, %8, 0, 1, 1, 0;\n"
        : "=f"(d[0][0]), "=f"(d[0][1]), "=f"(d[0][2]), "=f"(d[0][3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc));
  }
  static __device__ __forceinline__ void mma(float (&d)[WGS_NT][4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                             uint64_t desc) {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 {%0, %1, %2, %3}, {%4, %5, %6, %7}, %8, 1, 1, 1, 0;\n"
        : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc));
  }
};
template <>
struct WGSMMA<2> {
  static __device__ __forceinline__ void mma0(float (&d)[WGS_NT][4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                              uint64_t desc) {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7}, {%8, %9, %10, %11}, "
        "%12, 0, 1, 1, 0;\n"
        : "=f"(d[0][0]), "=f"(d[0][1]), "=f"(d[0][2]), "=f"(d[0][3]), "=f"(d[1][0]), "=f"(d[1][1]), "=f"(d[1][2]),
          "=f"(d[1][3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc));
  }
  static __device__ __forceinline__ void mma(float (&d)[WGS_NT][4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                             uint64_t desc) {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7}, {%8, %9, %10, %11}, "
        "%12, 1, 1, 1, 0;\n"
        : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]),
          "+f"(d[1][3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc));
  }
};
template <>
struct WGSMMA<3> {
  static __device__ __forceinline__ void mma0(float (&d)[WGS_NT][4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                              uint64_t desc) {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n24k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11}, "
        "{%12, %13, %14, %15}, %16, 0, 1, 1, 0;\n"
        : "=f"(d[0][0]), "=f"(d[0][1]), "=f"(d[0][2]), "=f"(d[0][3]), "=f"(d[1][0]), "=f"(d[1][1]), "=f"(d[1][2]),
          "=f"(d[1][3]), "=f"(d[2][0]), "=f"(d[2][1]), "=f"(d[2][2]), "=f"(d[2][3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc));
  }
  static __device__ __forceinline__ void mma(float (&d)[WGS_NT][4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                             uint64_t desc) {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n24k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11}, "
        "{%12, %13, %14, %15}, %16, 1, 1, 1, 0;\n"
        : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]),
          "+f"(d[1][3]), "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc));
  }
};
template <>
struct WGSMMA<4> {
  static __device__ __forceinline__ void mma0(float (&d)[WGS_NT][4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                              uint64_t desc) {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, "
        "%13, %14, %15}, {%16, %17, %18, %19}, %20, 0, 1, 1, 0;\n"
        : "=f"(d[0][0]), "=f"(d[0][1]), "=f"(d[0][2]), "=f"(d[0][3]), "=f"(d[1][0]), "=f"(d[1][1]), "=f"(d[1][2]),
          "=f"(d[1][3]), "=f"(d[2][0]), "=f"(d[2][1]), "=f"(d[2][2]), "=f"(d[2][3]), "=f"(d[3][0]), "=f"(d[3][1]),
          "=f"(d[3][2]), "=f"(d[3][3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc));
  }
  static __device__ __forceinline__ void mma(float (&d)[WGS_NT][4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                             uint64_t desc) {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, "
        "%13, %14, %15}, {%16, %17, %18, %19}, %20, 1, 1, 1, 0;\n"
        : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]),
          "+f"(d[1][3]), "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]), "+f"(d[3][0]), "+f"(d[3][1]),
          "+f"(d[3][2]), "+f"(d[3][3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc));
  }
};
template <>
struct WGSMMA<5> {
  static __device__ __forceinline__ void mma0(float (&d)[WGS_NT][4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                              uint64_t desc) {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n40k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, "
        "%13, %14, %15, %16, %17, %18, %19}, {%20, %21, %22, %23}, %24, 0, 1, 1, 0;\n"
        : "=f"(d[0][0]), "=f"(d[0][1]), "=f"(d[0][2]), "=f"(d[0][3]), "=f"(d[1][0]), "=f"(d[1][1]), "=f"(d[1][2]),
          "=f"(d[1][3]), "=f"(d[2][0]), "=f"(d[2][1]), "=f"(d[2][2]), "=f"(d[2][3]), "=f"(d[3][0]), "=f"(d[3][1]),
          "=f"(d[3][2]), "=f"(d[3][3]), "=f"(d[4][0]), "=f"(d[4][1]), "=f"(d[4][2]), "=f"(d[4][3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc));
  }
  static __device__ __forceinline__ void mma(float (&d)[WGS_NT][4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                             uint64_t desc) {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n40k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, "
        "%13, %14, %15, %16, %17, %18, %19}, {%20, %21, %22, %23}, %24, 1, 1, 1, 0;\n"
        : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]),
          "+f"(d[1][3]), "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]), "+f"(d[3][0]), "+f"(d[3][1]),
          "+f"(d[3][2]), "+f"(d[3][3]), "+f"(d[4][0]), "+f"(d[4][1]), "+f"(d[4][2]), "+f"(d[4][3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc));
  }
};
template <>
struct WGSMMA<6> {
  static __device__ __forceinline__ void mma0(float (&d)[WGS_NT][4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                              uint64_t desc) {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n48k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, "
        "%13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23}, {%24, %25, %26, %27}, %28, 0, 1, 1, 0;\n"
        : "=f"(d[0][0]), "=f"(d[0][1]), "=f"(d[0][2]), "=f"(d[0][3]), "=f"(d[1][0]), "=f"(d[1][1]), "=f"(d[1][2]),
          "=f"(d[1][3]), "=f"(d[2][0]), "=f"(d[2][1]), "=f"(d[2][2]), "=f"(d[2][3]), "=f"(d[3][0]), "=f"(d[3][1]),
          "=f"(d[3][2]), "=f"(d[3][3]), "=f"(d[4][0]), "=f"(d[4][1]), "=f"(d[4][2]), "=f"(d[4][3]), "=f"(d[5][0]),
          "=f"(d[5][1]), "=f"(d[5][2]), "=f"(d[5][3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc));
  }
  static __device__ __forceinline__ void mma(float (&d)[WGS_NT][4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                             uint64_t desc) {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n48k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, "
        "%13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23}, {%24, %25, %26, %27}, %28, 1, 1, 1, 0;\n"
        : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]),
          "+f"(d[1][3]), "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]), "+f"(d[3][0]), "+f"(d[3][1]),
          "+f"(d[3][2]), "+f"(d[3][3]), "+f"(d[4][0]), "+f"(d[4][1]), "+f"(d[4][2]), "+f"(d[4][3]), "+f"(d[5][0]),
          "+f"(d[5][1]), "+f"(d[5][2]), "+f"(d[5][3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc));
  }
};
template <>
struct WGSMMA<7> {
  static __device__ __forceinline__ void mma0(float (&d)[WGS_NT][4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                              uint64_t desc) {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n56k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, "
        "%13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27}, {%28, %29, %30, %31}, %32, 0, 1, "
        "1, 0;\n"
        : "=f"(d[0][0]), "=f"(d[0][1]), "=f"(d[0][2]), "=f"(d[0][3]), "=f"(d[1][0]), "=f"(d[1][1]), "=f"(d[1][2]),
          "=f"(d[1][3]), "=f"(d[2][0]), "=f"(d[2][1]), "=f"(d[2][2]), "=f"(d[2][3]), "=f"(d[3][0]), "=f"(d[3][1]),
          "=f"(d[3][2]), "=f"(d[3][3]), "=f"(d[4][0]), "=f"(d[4][1]), "=f"(d[4][2]), "=f"(d[4][3]), "=f"(d[5][0]),
          "=f"(d[5][1]), "=f"(d[5][2]), "=f"(d[5][3]), "=f"(d[6][0]), "=f"(d[6][1]), "=f"(d[6][2]), "=f"(d[6][3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc));
  }
  static __device__ __forceinline__ void mma(float (&d)[WGS_NT][4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                             uint64_t desc) {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n56k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, "
        "%13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27}, {%28, %29, %30, %31}, %32, 1, 1, "
        "1, 0;\n"
        : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]),
          "+f"(d[1][3]), "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]), "+f"(d[3][0]), "+f"(d[3][1]),
          "+f"(d[3][2]), "+f"(d[3][3]), "+f"(d[4][0]), "+f"(d[4][1]), "+f"(d[4][2]), "+f"(d[4][3]), "+f"(d[5][0]),
          "+f"(d[5][1]), "+f"(d[5][2]), "+f"(d[5][3]), "+f"(d[6][0]), "+f"(d[6][1]), "+f"(d[6][2]), "+f"(d[6][3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc));
  }
};
template <>
struct WGSMMA<8> {
  static __device__ __forceinline__ void mma0(float (&d)[WGS_NT][4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                              uint64_t desc) {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, "
        "%13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, {%32, %33, "
        "%34, %35}, %36, 0, 1, 1, 0;\n"
        : "=f"(d[0][0]), "=f"(d[0][1]), "=f"(d[0][2]), "=f"(d[0][3]), "=f"(d[1][0]), "=f"(d[1][1]), "=f"(d[1][2]),
          "=f"(d[1][3]), "=f"(d[2][0]), "=f"(d[2][1]), "=f"(d[2][2]), "=f"(d[2][3]), "=f"(d[3][0]), "=f"(d[3][1]),
          "=f"(d[3][2]), "=f"(d[3][3]), "=f"(d[4][0]), "=f"(d[4][1]), "=f"(d[4][2]), "=f"(d[4][3]), "=f"(d[5][0]),
          "=f"(d[5][1]), "=f"(d[5][2]), "=f"(d[5][3]), "=f"(d[6][0]), "=f"(d[6][1]), "=f"(d[6][2]), "=f"(d[6][3]),
          "=f"(d[7][0]), "=f"(d[7][1]), "=f"(d[7][2]), "=f"(d[7][3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc));
  }
  static __device__ __forceinline__ void mma(float (&d)[WGS_NT][4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                             uint64_t desc) {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, "
        "%13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, {%32, %33, "
        "%34, %35}, %36, 1, 1, 1, 0;\n"
        : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]),
          "+f"(d[1][3]), "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]), "+f"(d[3][0]), "+f"(d[3][1]),
          "+f"(d[3][2]), "+f"(d[3][3]), "+f"(d[4][0]), "+f"(d[4][1]), "+f"(d[4][2]), "+f"(d[4][3]), "+f"(d[5][0]),
          "+f"(d[5][1]), "+f"(d[5][2]), "+f"(d[5][3]), "+f"(d[6][0]), "+f"(d[6][1]), "+f"(d[6][2]), "+f"(d[6][3]),
          "+f"(d[7][0]), "+f"(d[7][1]), "+f"(d[7][2]), "+f"(d[7][3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc));
  }
};
// two producer warps: the weights warp streams every stage's 36 KB through the weight ring (TMA, lane 0), the x warp
// gathers every stage's x slice into the x ring (cp.async) and, in phase B, first waits for the job's phase-A units;
// so the weights run ahead by the whole ring even while x waits
// Phase A runs a CTA's trailing partial unit FIRST, then the rest of its range in order: the CTA that holds a unit's
// first stages otherwise reaches them last, finishes the unit (read back + epilogue) and pays it exposed at the phase
// end. Returns where the first sub-range starts: [sp, g1) then [g0, sp) (sp == g0: no reordering). Phase B keeps the
// order (its first stages wait for phase A's units in order).
__device__ __forceinline__ int wgs_split(int pb, int g0, int g1) {
  const int ue = ((g1 - 1) >> 4) << 4;   // first stage of the unit holding the range's last stage
  return (0 && pb == 0 && (g1 & 15) != 0 && ue > g0) ? ue : g0;
}
__device__ void wgs_ranges(const Params& p, Smem& S);   // below (needs the cost tables)
#ifndef MOE_YKEEP   // big build: the final pass keeps each token's 8 y rows in registers from pass 1 (sums of squares) for
#define MOE_YKEEP 0   // pass 2 (the weighted sum) instead of re-reading them from L2 after the reduction
#endif
#ifndef MOE_SGX   // WGS builds: the consumers load the x of their CTA's first WGS_NX phase-A stages from x straight into the
#define MOE_SGX 0   // x slots (no wait for the job's slowest gatherer, no TMA); the x producer starts at stage WGS_NX
#endif
#if MOE_SGX
__device__ __forceinline__ int wgs_sgx_n(const Smem& S) { return min(WGS_NX, S.wgr[1] - S.wgr[0]); }
#endif
__device__ void wgs_wproducer(const Params& p, Smem& S, unsigned char* ring) {
#endif
#if MOE_TUNED_D32
  asm volatile("bar.sync 8, 64;\n" ::: "memory");   // routing tables and this CTA's stage ranges are ready
#endif
#if MOE_WGS && !MOE_TUNED_D32
  WS_JOIN();   // routing tables and this CTA's stage ranges are ready
#endif
#if MOE_WGS
  if ((threadIdx.x & 31) != 0) return;
  int ws = 0, n = 0;
  uint32_t wph = 0;
#pragma unroll 1
  for (int pb = 0; pb < 2; ++pb) {
    const int g0 = S.wgr[2 * pb];
    const int g1 = S.wgr[2 * pb + 1];
    const int sp = wgs_split(pb, g0, g1);
    const int ubl = pb ? 4 : 5;   // log2 of 256-row units per job
    const unsigned char* base = pb ? p.wBs : p.wAs;
#pragma unroll 1
    for (int i = 0, gs = sp; i < g1 - g0; ++i, ++gs, ++n) {
      if (gs == g1) gs = g0;   // [sp, g1) done: then [g0, sp)
      const int u = gs >> 4, st = gs & 15;
      const int j = wgs_job(S, pb, u >> ubl), ub = u & ((1 << ubl) - 1);
      if (n >= WGS_NS) wgs_wait(S.bars + BW_E + ws, wph ^ 1, 1000 + gs);
      // st1 (wp2s): at M <= 64 stage n is issued only once stage n - 2 has landed (no 3-stage burst; E371 / E372)
      if (n >= 2 && p.M <= 64) {
        const int pn = n - 2;
        wgs_wait(S.bars + BW_F + pn % WGS_NS, (uint32_t)(pn / WGS_NS) & 1u, 5000 + gs);
      }
      uint64_t* bar = S.bars + BW_F + ws;
      const unsigned char* src = base + ((size_t)(((S.jE[j] << ubl) + ub) * 16 + st)) * (size_t)WGS_STG;
      asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(smem_u32(bar)), "r"(WGS_STG)
                   : "memory");
      asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes" L2H_SUFFIX
                   " [%0], [%1], %2, [%3]" L2H_OPERAND ";\n" ::"r"(smem_u32(ring + ws * UNIT_BYTES)),
                   "l"(src), "r"(WGS_STG), "r"(smem_u32(bar))L2H_ARG
                   : "memory");
      if (++ws == WGS_NS) {
        ws = 0;
        wph ^= 1;
      }
    }
  }
}
// x producer: one bulk copy per stage (lane 0): the job's x slice for the stage is contiguous in xA (phase A, written
// by every CTA's gather pass at the start) or xB (phase B, written by the phase-A epilogues), N_pad x 512 B
constexpr int WGS_XROWS = EMAX + 8 * JMAX;   // padded rows of all jobs, upper bound
__device__ void wgs_xproducer(const Params& p, Smem& S, unsigned char* ring) {
  // the routing tables are ready (the consumers arrive on barrier 7 right after routing): this idle warp computes the
  // stage ranges while the consumers gather phase A's x
  asm volatile("bar.sync 7, %0;\n" ::"n"(NT + 32) : "memory");
  wgs_ranges(p, S);   // all 32 lanes (warp-parallel)
  __syncwarp();
#endif
#if MOE_TUNED_D32
  asm volatile("bar.arrive 8, 64;\n" ::: "memory");   // the weight producer may start
  asm volatile("bar.sync 2, %0;\n" ::"n"(NT + 32) : "memory");   // the gather is done
#endif
#if MOE_WGS && !MOE_TUNED_D32
  WS_JOIN();   // routing tables and this CTA's stage ranges are ready
#endif
#if MOE_WGS
  if ((threadIdx.x & 31) != 0) return;
  unsigned char* xring = ring + WGS_NS * UNIT_BYTES;
  int* const xr_flags = mf_xr(p, S);
  int* const da_flags = mf_da(p, S);
#if MOE_SGX
  const int nsg = wgs_sgx_n(S);   // stages whose x the consumers load (slots 0 .. nsg - 1 of the first round)
  int xs = nsg % WGS_NX, n = nsg;
  uint32_t xph = nsg == WGS_NX;
#else
  int xs = 0, n = 0;
  uint32_t xph = 0;
#endif
#pragma unroll 1
  for (int pb = 0; pb < 2; ++pb) {
    const int g0 = S.wgr[2 * pb];
    const int g1 = S.wgr[2 * pb + 1];
    const int sp = wgs_split(pb, g0, g1);
    const int ubl = pb ? 4 : 5;
    const __nv_bfloat16* xsrc = pb ? p.xB : p.xA;
    int curj = -1, npad = 8;
    const __nv_bfloat16* jb = xsrc;
#if MOE_SGX
#pragma unroll 1
    for (int i = pb ? 0 : nsg, gs = sp + i; i < g1 - g0; ++i, ++gs, ++n) {
#else
#pragma unroll 1
    for (int i = 0, gs = sp; i < g1 - g0; ++i, ++gs, ++n) {
#endif
      if (gs == g1) gs = g0;   // [sp, g1) done: then [g0, sp)
      const int u = gs >> 4, st = gs & 15;
      const int j = wgs_job(S, pb, u >> ubl);
      if (n >= WGS_NX) wgs_wait(S.bars + BX_E + xs, xph ^ 1, 2000000 + gs);
      if (j != curj) {
        curj = j;
        npad = ((S.jC[j] + 7) >> 3) << 3;
        jb = xsrc + (size_t)S.jrow[j] * HD;
        if (!pb) {
          while (ld_acquire(xr_flags + j) < npad) __nanosleep(32);
        }   // every row gathered
      }
      // phase B stage st reads inter columns 256 st .. +255: the 2 phase-A units (x 4 warpgroups) that wrote them
      if (pb && !MOE_XPRB) {
        while (ld_acquire(da_flags + j * 16 + st) < 8) __nanosleep(32);
      }
      uint64_t* bar = S.bars + BX_F + xs;
      const uint32_t bytes = (uint32_t)npad * 512u;
      asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(smem_u32(bar)), "r"(bytes)
                   : "memory");
      asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n" ::"r"(
                       smem_u32(xring + xs * WGS_XS)),
                   "l"(jb + (size_t)st * npad * 256), "r"(bytes), "r"(smem_u32(bar))
                   : "memory");
      if (++xs == WGS_NX) {
        xs = 0;
        xph ^= 1;
      }
    }
  }
}
// phase A's x gather (all consumer threads of every CTA, once, at the start): this CTA's share of the jobs' padded rows
// copied into xA's per-stage wgmma layout, then counted into xready (padding rows are counted, not written)
__device__ void wgs_gather(const Params& p, Smem& S) {
  const int tid = threadIdx.x;
  const int nj = S.njF + S.njS;
  const int R = S.jrow[nj];   // R * G < 2^24: exact float muldiv, no divide routine
  const float rG = __fdividef(1.f, (float)gridDim.x);
  const int q0 = muldiv_g(R, blockIdx.x, gridDim.x, rG);
  const int q1 = muldiv_g(R, blockIdx.x + 1, gridDim.x, rG);
  int j = 0;
  // all of a batch's row loads first, then the stores: one x row at a time was one dependent HBM round trip per row
  // (the next row's load could not move above the previous row's store: they may alias), 1.3-7 us at bs256
  static_assert(NT * 8 == HD, "one 16 B chunk per thread per row");
  for (int qb = q0; qb < q1; qb += 8) {
    uint4 v[8];
    uint4* d[8];
#pragma unroll
    for (int k = 0; k < 8; ++k) {
      const int q = qb + k;
      d[k] = nullptr;
      if (q < q1) {
        while (S.jrow[j + 1] <= q) ++j;
        const int r = q - S.jrow[j];
        const int cnt = S.jC[j];
        if (r < cnt) {
          const int ntt = (cnt + 7) >> 3;
          d[k] = reinterpret_cast<uint4*>(p.xA + (size_t)S.jrow[j] * HD) +
                 ((((tid >> 5) * ntt + (r >> 3)) * 32 + (tid & 31)) * 8 + (r & 7));
          {   // KPERM3: k8 group G = tid & 31 of stage tid >> 5 holds the bf16 pairs at stage offsets 2 B(G) + 64 t,
            // B(G) = 8 ((G >> 1) & 3) + 2 (G >> 3) + (G & 1)
            const int G = tid & 31;
            const unsigned* xs32 =
                reinterpret_cast<const unsigned*>(p.x + (size_t)(S.ent[S.jS[j] + r] >> 3) * HD + (tid >> 5) * 256) +
                (8 * ((G >> 1) & 3) + 2 * (G >> 3) + (G & 1));
            v[k] = make_uint4(ldp(xs32), ldp(xs32 + 32), ldp(xs32 + 64), ldp(xs32 + 96));
          }
        }
      }
    }
#pragma unroll
    for (int k = 0; k < 8; ++k)
      if (d[k]) *d[k] = v[k];
  }
  CTA_SYNC();
  if (tid == 0) {   // count this CTA's rows per job (release: the CTA's stores are ordered before by the barrier)
    int jj = 0;
    for (int q = q0; q < q1;) {
      while (S.jrow[jj + 1] <= q) ++jj;
      const int e = min(q1, S.jrow[jj + 1]);
      asm volatile("red.release.gpu.global.add.s32 [%0], %1;\n" ::"l"(mf_xr(p, S) + jj), "r"(e - q) : "memory");
      q = e;
    }
  }
}
#if MOE_SGX
// the x of this CTA's first nsg phase-A stages, from x straight into x slots 0 .. nsg - 1 in the producer's slot layout
// ([n8][32 k8][8 rows][16 B], KPERM3 as above; padding rows unwritten as in xA): all loads first, then the stores
__device__ void wgs_selfx(const Params& p, Smem& S, unsigned char* ring) {
  static_assert(NW * 4 >= FT_T, "four rows per warp cover a job");
  const int tid = threadIdx.x, warp = tid >> 5, G = tid & 31;
  const int nsg = wgs_sgx_n(S);
  const int g0 = S.wgr[0], g1 = S.wgr[1];
  const uint32_t xring_s = smem_u32(ring + WGS_NS * UNIT_BYTES);
  uint4 v[WGS_NX][4];
  uint32_t a[WGS_NX][4];
#pragma unroll
  for (int k = 0; k < WGS_NX; ++k) {
    int gs = wgs_split(0, g0, g1) + k;
    if (gs >= g1) gs -= g1 - g0;
    const int st = gs & 15;
    const int j = wgs_job(S, 0, gs >> 9);
    const int cnt = S.jC[j];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int r = warp + i * NW;
      a[k][i] = 0;
      if (k < nsg && r < cnt) {
        a[k][i] = xring_s + k * WGS_XS + (((r >> 3) * 32 + G) * 8 + (r & 7)) * 16;
        const unsigned* xs32 =
            reinterpret_cast<const unsigned*>(p.x + (size_t)(S.ent[S.jS[j] + r] >> 3) * HD + st * 256) +
            (8 * ((G >> 1) & 3) + 2 * (G >> 3) + (G & 1));
        v[k][i] = make_uint4(ldp(xs32), ldp(xs32 + 32), ldp(xs32 + 64), ldp(xs32 + 96));
      }
    }
  }
#pragma unroll
  for (int k = 0; k < WGS_NX; ++k)
#pragma unroll
    for (int i = 0; i < 4; ++i)
      if (a[k][i])
        asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};\n" ::"r"(a[k][i]), "r"(v[k][i].x), "r"(v[k][i].y),
                     "r"(v[k][i].z), "r"(v[k][i].w)
                     : "memory");
  asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");   // the wgmmas read the slots through the async proxy
  CTA_SYNC();
  if (tid == 0)
    for (int k = 0; k < nsg; ++k) mbar_arrive1(S.bars + BX_F + k);
}
#endif
// Work left from the previous segment, done after the next segment's first wgmma group (so the stores and atomics it
// waits for have long completed): kind 1 = release doneA for job j (phase A), kind 2 = add the stage count of a split
// unit's share and, if that completes the unit, read the summed partials back and run its epilogue.
#endif
#if MOE_TUNED_D32

__device__ __forceinline__ float astra_wgs_silu(float gate, float up) {
  const float half = gate * 0.5f;
  float value;
  asm("tanh.approx.f32 %0, %1;" : "=f"(value) : "f"(half));
  float silu = half * (1.f + value);
  if (gate < -8.f) {   // the tanh form cancels below -8: gate e^gate / (1 + e^gate) there (as moe_silu_s)
    float e, q;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(e) : "f"(1.4426950408889634f * gate));
    asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(q) : "f"(1.f + e));
    silu = gate * e * q;
  }
  return silu * up;
}
#endif
#if MOE_WGS
struct WgsPend {
  int kind, u, slot, nst;
};
template <bool PB>
__device__ __forceinline__ void wgs_epi_tile(const Params& p, const Smem& S, int j, int r16, int nt, const float* d4) {
  const int lane = threadIdx.x & 31;
  const int g = lane >> 2, t = lane & 3;
  const int cnt = S.jC[j];
  const int jst = S.jS[j];
  const int e = S.jE[j];
  if (!PB) {
    // SwiGLU straight into xB (phase B's per-stage wgmma B layout) instead of row-major inter
    const int col = r16 * 8 + g;
    const int ntt = (cnt + 7) >> 3;
    const float gsc = p.gsA[e] * TWO62;
    __nv_bfloat16* xb = p.xB + (size_t)S.jrow[j] * HD + (col & 7);
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int tk = nt * 8 + 2 * t + h;
      if (tk < cnt) {
        const float gt = d4[h] * gsc, up = d4[2 + h] * gsc;
#endif
#if MOE_TUNED_D32
        const float v = astra_wgs_silu(gt, up);
#endif
#if MOE_WGS && !MOE_TUNED_D32
        const float v = SILU_DIV(gt, 1.f + __expf(-gt)) * up;
#endif
#if MOE_WGS
        st_keep_bf16(xb + (size_t)((((col >> 8) * ntt + (tk >> 3)) * 32 + ((col & 255) >> 3)) * 64 + (tk & 7) * 8),
                     __float2bfloat16((v)));
      }
    }
  } else {
    // y rows only: the per-route sum of squares is formed by the final pass from these rows (no shuffles / atomics)
    const int n0 = r16 * 16 + g;
    const int tok0 = nt * 8 + 2 * t;
    const float gsc = p.gsB[e] * TWO62;
    const float w0 = (1.f + __bfloat162float(__ldg(p.normw + n0))) * gsc,
                w1 = (1.f + __bfloat162float(__ldg(p.normw + n0 + 8))) * gsc;
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int tk = tok0 + h;
      if (tk < cnt) {
#endif
#if MOE_B4FIN
        st_keep_f32(p.ybuf + ((size_t)(jst + tk) * HD + n0), d4[h] * w0);
        st_keep_f32(p.ybuf + ((size_t)(jst + tk) * HD + n0 + 8), d4[2 + h] * w1);
      }
    }
    {   // B4FIN: this tile's 16-column sum of squares per token, reduced over the 8 column lanes (fixed tree)
      float q0 = 0.f, q1 = 0.f;
      if (tok0 < cnt) {
        const float a = d4[0] * w0, b = d4[2] * w1;
        q0 = a * a + b * b;
      }
      if ((tok0 + 1) < cnt) {
        const float a = d4[1] * w0, b = d4[3] * w1;
        q1 = a * a + b * b;
      }
#pragma unroll
      for (int o = 4; o < 32; o <<= 1) {
        q0 += __shfl_xor_sync(0xffffffffu, q0, o);
        q1 += __shfl_xor_sync(0xffffffffu, q1, o);
      }
      if (g == 0) {
        if (tok0 < cnt) p.sspart[(size_t)(jst + tok0) * 256 + r16] = q0;
        if ((tok0 + 1) < cnt) p.sspart[(size_t)(jst + tok0 + 1) * 256 + r16] = q1;
#endif
#if MOE_WGS && !MOE_B4FIN
        st_keep_bf16(reinterpret_cast<__nv_bfloat16*>(p.ybuf) + ((size_t)(jst + tk) * HD + n0),
                     __float2bfloat16(d4[h] * w0));
        st_keep_bf16(reinterpret_cast<__nv_bfloat16*>(p.ybuf) + ((size_t)(jst + tk) * HD + n0 + 8),
                     __float2bfloat16(d4[2 + h] * w1));
#endif
#if MOE_WGS
      }
    }
  }
}
#endif
#if MOE_TUNED_D32
__device__ __forceinline__ int fresh_bucket_width(int n) { return n <= 2 ? n : n <= 4 ? 4 : 8; }
#endif
#if MOE_WGS
template <bool PB>
__device__ __forceinline__ void wgs_flush(const Params& p, Smem& S, WgsPend& pd) {
  if (pd.kind == 0) return;
  const int tid = threadIdx.x;
  const int warp = WARP_IDX(tid);
  const int wg = warp >> 2, wp = warp & 3, wt = tid & 127;
  constexpr int UBL = PB ? 4 : 5;
  const int j = pd.u >> UBL, ub = pd.u & ((1 << UBL) - 1);
  asm volatile("bar.sync %0, 128;\n" ::"r"(3 + wg) : "memory");
  if (pd.kind == 4) {
    if (wt == 0)
      asm volatile("red.release.gpu.global.add.s32 [%0], %1;\n" ::"l"(p.mfl + MF_CE + (pd.slot)), "r"(1) : "memory");
    pd.kind = 0;
    return;
  }
  if (pd.kind == 1) {
    // release (cumulative over the warpgroup's epilogue stores ordered by the barrier); pairs with the producer's
    // acquire. Unit ub wrote inter columns ub*128 .. +127: half of phase B stage ub / 2
    if (wt == 0)
      asm volatile("red.release.gpu.global.add.s32 [%0], %1;\n" ::"l"(mf_da(p, S) + j * 16 + (ub >> 1)), "r"(1)
                   : "memory");
    pd.kind = 0;
    return;
  }
  int* fc = wgs_cnt(p, PB) + pd.slot * 4 + wg;
#endif
#if MOE_TUNED_D32
  constexpr bool DF = (3 >> (PB ? 1 : 0)) & 1;   // designated finisher in this phase
#endif
#if MOE_WGS && !MOE_TUNED_D32
  constexpr bool DF = (2 >> (PB ? 1 : 0)) & 1;   // designated finisher in this phase
#endif
#if MOE_WGS
  if constexpr (DF) {
    // designated finisher (kind 3): the share processed last in time by construction (phase A: the one holding the
    // unit's last stage, run early by the reorder; phase B: the one holding stage 0, at its CTA's end). The others just
    // count with a fire-and-forget release (no atomic round trip, no drain wait); the finisher waits for their counts
    if (pd.kind == 2) {
      if (wt == 0) asm volatile("red.release.gpu.global.add.s32 [%0], %1;\n" ::"l"(fc), "r"(pd.nst) : "memory");
      pd.kind = 0;
      return;
    }
    if (wt == 0) {
      while (ld_acquire(fc) < 16 - pd.nst) __nanosleep(32);
    }
    asm volatile("bar.sync %0, 128;\n" ::"r"(3 + wg) : "memory");
    pd.kind = 0;
  } else {
    if (wt == 0) {
      int old;
      asm volatile("atom.acq_rel.gpu.global.add.s32 %0, [%1], %2;\n" : "=r"(old) : "l"(fc), "r"(pd.nst) : "memory");
      S.wglast[wg] = old + pd.nst == 16;
    }
    asm volatile("bar.sync %0, 128;\n" ::"r"(3 + wg) : "memory");
    const bool last = S.wglast[wg];
    pd.kind = 0;
    if (!last) return;
  }
  float* fb = wgs_fix(p, PB) + (size_t)(pd.slot * 4 + wg) * WGS_FIXW + wt * 4;
#endif
#if MOE_TUNED_D32
  const int ntt = fresh_bucket_width((S.jC[j] + 7) >> 3), r16 = ub * 16 + wg * 4 + wp;
#endif
#if MOE_WGS && !MOE_TUNED_D32
  const int ntt = (S.jC[j] + 7) >> 3, r16 = ub * 16 + wg * 4 + wp;
#endif
#if MOE_WGS
  // two n-tiles' loads in flight before their zeroing stores and epilogues (the scout): one at a time, each load
  // waits behind the previous tile's stores (they may alias), ntt serialized L2 round trips under full HBM load
  for (int nt0 = 0; nt0 < ntt; nt0 += 2) {
    float4 v[2];
#pragma unroll
    for (int k = 0; k < 2; ++k)
      if (nt0 + k < ntt) v[k] = __ldcg(reinterpret_cast<const float4*>(fb + (nt0 + k) * 512));
#pragma unroll
    for (int k = 0; k < 2; ++k)
      if (nt0 + k < ntt) {
        *reinterpret_cast<float4*>(fb + (nt0 + k) * 512) = make_float4(0.f, 0.f, 0.f, 0.f);
        const float d4[4] = {v[k].x, v[k].y, v[k].z, v[k].w};
        wgs_epi_tile<PB>(p, S, j, r16, nt0 + k, d4);
      }
  }
  if (wt == 0) *fc = 0;
  if (PB) {
    asm volatile("bar.sync %0, 128;\n" ::"r"(3 + wg) : "memory");
    if (wt == 0)
      asm volatile("red.release.gpu.global.add.s32 [%0], %1;\n" ::"l"(p.mfl + MF_CE + mf_gen(S) * 32 + S.jE[j]), "r"(1)
                   : "memory");
  }
  if (!PB) {
    asm volatile("bar.sync %0, 128;\n" ::"r"(3 + wg) : "memory");
    if (wt == 0)
      asm volatile("red.release.gpu.global.add.s32 [%0], %1;\n" ::"l"(mf_da(p, S) + j * 16 + (ub >> 1)), "r"(1)
                   : "memory");
  }
}
// one segment (stages [s0, s1) of unit u) on NTT n-tiles: accumulate, then finish (whole unit) or hand off (split)
template <bool PB, int NTT>
__device__ __forceinline__ void wgs_seg(const Params& p, Smem& S, unsigned char* ring, int u, int s0, int s1, int slot,
                                        int& seq, WgsPend& pd) {
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = WARP_IDX(tid);
  const int wg = warp >> 2, wp = warp & 3, g = lane >> 2, wt = tid & 127;
  constexpr int UBL = PB ? 4 : 5;
  const uint32_t k16c =
      (4u) + p.zero;   // opaque 16 (4 with SC6W): the scale decode's multiply stays an IMAD (FMA pipe)
  const uint32_t xring_s = smem_u32(ring + WGS_NS * UNIT_BYTES);
  float d[WGS_NT][4];   // written first by a scale-d = 0 wgmma (no zeroing: ptxas would serialize the wgmmas, C7513)
#endif
#if MOE_TUNED_D32
  float d2[WGS_NT][4];   // odd commit groups' chain
#endif
#if MOE_WGS
  uint32_t af[2][2][4];
  int prev = -1;   // x slot the last committed group may still read
  // slot-0 descriptor and code pointer, once per segment: a stage adds its slot's offset (the address field cannot
  // carry: shared addresses >> 4 stay below 2^14) instead of rebuilding the descriptor (~8 uniform ops per stage)
  const uint64_t xd0 = wg_desc(xring_s, 128, 4096);
  const unsigned char* pc0 = ring + wg * WGS_PIECE;
  auto stage = [&](auto first) {
    constexpr bool FIRST = decltype(first)::value;
    const int ws = seq % WGS_NS;
    const int xs = seq % WGS_NX;
    wgs_wait(S.bars + BW_F + ws, (uint32_t)(seq / WGS_NS) & 1u, 3000000 + seq);
    const unsigned char* pc = pc0 + ws * UNIT_BYTES;
    wgs_wait(S.bars + BX_F + xs, (uint32_t)(seq / WGS_NX) & 1u, 4000000 + seq);
    const uint64_t xd = xd0 + (uint64_t)(uint32_t)(xs * (WGS_XS >> 4));
    // codes and scales one 4-step group at a time (fewer live registers)
    uint4 cwn = *reinterpret_cast<const uint4*>(pc + (wp * 32 + lane) * 16);
    // KPERM3: K is permuted inside the stage so that step j of every load group is scale block 4 t + j for this lane:
    // the 4 scale pairs ([j][h] bytes) come from one 8-byte load per stage and are built once per stage
    // SC6: 768 B of scales per piece: plane W (uint32 per thread) | plane H (uint16 per thread); the 4 scale pairs of the stage
    // (bytes of 4 v, see harness/make_sc6.py) are built here once and stay in registers for the 4 load groups
    uint32_t sgs[4];
    {
      const int tt = wp * 32 + lane;
      const uint32_t W = *reinterpret_cast<const uint32_t*>(pc + 8192 + tt * 4);
      const uint32_t H = static_cast<uint32_t>(*reinterpret_cast<const unsigned short*>(pc + 8192 + 512 + tt * 2));
      const uint32_t X = W & 0xFCFCFCFCu;
      const uint32_t Hm = H & 0xFCFCu;
      const uint32_t p6 = (W & 0x00030303u) * 0x04104000u;
      const uint32_t p7 = ((H & 0x0303u) | (W & 0x03000000u)) * 0x10400004u;
      sgs[0] = __byte_perm(X, 0u, 0x4140u) * k16c + 0x60006000u;
      sgs[1] = __byte_perm(X, 0u, 0x4342u) * k16c + 0x60006000u;
      sgs[2] = __byte_perm(Hm, 0u, 0x4140u) * k16c + 0x60006000u;
      sgs[3] = __byte_perm(p6, p7, 0x0703u) * k16c + 0x60006000u;
    }
#pragma unroll
    for (int lg = 0; lg < 4; ++lg) {
      const uint4 cwg = cwn;
      if (lg < 3) {
        cwn = *reinterpret_cast<const uint4*>(pc + (((lg + 1) * 4 + wp) * 32 + lane) * 16);
      } else if (lane == 0) {
        mbar_arrive1(S.bars + BW_E + ws);   // every code and scale of this stage is in registers: weights slot free
      }
#pragma unroll
      for (int h = 0; h < 4 / 2; ++h) {
        const int ci = lg * (4 / 2) + h;   // commit group index in the stage
#pragma unroll
        for (int jj = 0; jj < 2; ++jj) {
          const int js = h * 2 + jj;   // k16 step within the 4-step load group
          const uint32_t q = js == 0 ? cwg.x : js == 1 ? cwg.y : js == 2 ? cwg.z : cwg.w;
          uint32_t r0, r1, r2, r3;
          dq4(q, r0, r1, r2, r3);
          // scales stored [j][h] (rows g, g+8 of step j side by side): one PRMT + one IMAD build both rows' scales in
          // one bf16x2, and the multiplies broadcast its halves (HMUL2 .H0_H0 / .H1_H1): 2 instructions per k16 fewer
          const uint32_t sgp = sgs[js];
          af[ci & 1][jj][0] = bmul2l(r0, sgp);
          af[ci & 1][jj][1] = bmul2h(r1, sgp);
          af[ci & 1][jj][2] = bmul2l(r2, sgp);
          af[ci & 1][jj][3] = bmul2h(r3, sgp);
        }
        wg_fence();
#pragma unroll
        for (int jj = 0; jj < 2; ++jj) {
          if (FIRST && ci == 0 && jj == 0)
            WGSMMA<NTT>::mma0(d, af[0][0][0], af[0][0][1], af[0][0][2], af[0][0][3], xd);
#endif
#if MOE_TUNED_D32
          else if (NTT == 1 && FIRST && ci == 1 && jj == 0)
            WGSMMA<NTT>::mma0(d2, af[1][0][0], af[1][0][1], af[1][0][2], af[1][0][3],
                              xd + (uint64_t)((lg * 4 + h * 2) * 16));
          else if (NTT == 1 && (ci & 1))
            WGSMMA<NTT>::mma(d2, af[ci & 1][jj][0], af[ci & 1][jj][1], af[ci & 1][jj][2], af[ci & 1][jj][3],
                             xd + (uint64_t)((lg * 4 + h * 2 + jj) * 16));
#endif
#if MOE_WGS
          else
            WGSMMA<NTT>::mma(d, af[ci & 1][jj][0], af[ci & 1][jj][1], af[ci & 1][jj][2], af[ci & 1][jj][3],
                             xd + (uint64_t)((lg * 4 + h * 2 + jj) * 16));
        }
        wg_commit();
        if (!FIRST || ci != 0) wg_wait<1>();
        if (ci == 0) {
          if (prev >= 0) {   // the previous stage's last group just retired: its x slot is free
            if (lane == 0)
              asm volatile(
                  "mbarrier.arrive.relaxed.cta.shared::cta.b64 _, [%0];\n" ::"r"(smem_u32(S.bars + BX_E + prev))
                  : "memory");
            prev = -1;
          }
        }
        // the previous segment's deferred work, behind group wgsfci of the first stage (scout: CI 3 lets this WG's
        // own red.adds / epilogue stores drain a little longer before the count / release atomics: +0.5-1%)
        if (FIRST && ci == 4 && pd.kind) wgs_flush<PB>(p, S, pd);
      }
    }
    prev = xs;
    ++seq;
  };
  stage(std::true_type{});
#pragma unroll 1
  for (int st = s0 + 1; st < s1; ++st) stage(std::false_type{});
  wg_wait<0>();
#endif
#if MOE_TUNED_D32
  if (NTT == 1) {
#pragma unroll
    for (int e = 0; e < 4; ++e) d[0][e] += d2[0][e];
  }
#endif
#if MOE_WGS
  if (lane == 0)
    asm volatile("mbarrier.arrive.relaxed.cta.shared::cta.b64 _, [%0];\n" ::"r"(smem_u32(S.bars + BX_E + prev))
                 : "memory");

  if (slot >= 0) {   // a share of a unit split between CTAs: add it now, count it later (deferred)
    float* fb = wgs_fix(p, PB) + (size_t)(slot * 4 + wg) * WGS_FIXW + wt * 4;
#pragma unroll
    for (int nt = 0; nt < NTT; ++nt)
      asm volatile("red.global.add.v4.f32 [%0], {%1, %2, %3, %4};\n" ::"l"(fb + nt * 512), "f"(d[nt][0]), "f"(d[nt][1]),
                   "f"(d[nt][2]), "f"(d[nt][3])
                   : "memory");
    // kind 3: this share finishes the unit (designated finisher, wgsdfin): phase A the share holding stage 15,
    // phase B the one holding stage 0
#endif
#if MOE_TUNED_D32
    pd = WgsPend{(((3 >> (PB ? 1 : 0)) & 1) && (PB ? s0 == 0 : (s0 == 0))) ? 3 : 2, u, slot, s1 - s0};
#endif
#if MOE_WGS && !MOE_TUNED_D32
    pd = WgsPend{(((2 >> (PB ? 1 : 0)) & 1) && (PB ? s0 == 0 : s1 == 16)) ? 3 : 2, u, slot, s1 - s0};
#endif
#if MOE_WGS
    return;
  }
  const int j = u >> UBL, r16 = (u & ((1 << UBL) - 1)) * 16 + wg * 4 + wp;
#pragma unroll
  for (int nt = 0; nt < NTT; ++nt) wgs_epi_tile<PB>(p, S, j, r16, nt, d[nt]);
  if (!PB)
    pd = WgsPend{1, u, -1, 0};   // the doneA release waits until the stores are done: deferred
  else
    pd = WgsPend{4, u, (15 & 8) ? (int)(mf_gen(S) * 32 + S.jE[j]) : -1,
                 0};   // phase B: the expert counter, same deferral
}
template <bool PB>
__device__ __forceinline__ void wgs_phase(const Params& p, Smem& S, unsigned char* ring, int g0, int g1, int& seq) {
  // (__forceinline__: outlined by NVVM when it grows, a CALL that makes ptxas serialize every wgmma, C7510)
  constexpr int UBL = PB ? 4 : 5;
  WgsPend pd{0, 0, 0, 0};
#pragma unroll 1
  for (int gs = wgs_split(PB, g0, g1), ge = g1;;) {
    if (gs >= ge) {   // [sp, g1) done: then [g0, sp); the bounds come back from smem (no extra live state)
      const int a0 = S.wgr[2 * PB];
      const int a1 = S.wgr[2 * PB + 1];
      const int sp = wgs_split(PB, a0, a1);
      if (ge == a1 && sp > a0) {
        gs = a0;
        ge = sp;
      } else
        break;
    }
#if MOE_BREV
    const int pu = gs >> 4, s0 = gs & 15, s1 = min(16, s0 + (ge - gs));
    const int u = (wgs_job(S, PB, pu >> UBL) << UBL) | (pu & ((1 << UBL) - 1));
#else
    const int pu = gs >> 4, u = pu, s0 = gs & 15, s1 = min(16, s0 + (ge - gs));
#endif
    // a unit split between CTAs: its fixup slot is the CTA holding its first stage
    const int slot = (s1 - s0 == 16) ? -1 : (s0 == 0 ? (int)blockIdx.x : wgs_cta_of(S, PB, pu << 4, gridDim.x));
    const int ntt = (S.jC[u >> UBL] + 7) >> 3;
    switch (ntt) {
      case 1:
        wgs_seg<PB, 1>(p, S, ring, u, s0, s1, slot, seq, pd);
        break;
      case 2:
        wgs_seg<PB, 2>(p, S, ring, u, s0, s1, slot, seq, pd);
        break;
      case 3:
#endif
#if MOE_WGS && !MOE_TUNED_D32
        wgs_seg<PB, 3>(p, S, ring, u, s0, s1, slot, seq, pd);
        break;
#endif
#if MOE_WGS
      case 4:
        wgs_seg<PB, 4>(p, S, ring, u, s0, s1, slot, seq, pd);
        break;
      case 5:
#endif
#if MOE_WGS && !MOE_TUNED_D32
        wgs_seg<PB, 5>(p, S, ring, u, s0, s1, slot, seq, pd);
        break;
#endif
#if MOE_WGS
      case 6:
#endif
#if MOE_WGS && !MOE_TUNED_D32
        wgs_seg<PB, 6>(p, S, ring, u, s0, s1, slot, seq, pd);
        break;
#endif
#if MOE_WGS
      case 7:
#endif
#if MOE_WGS && !MOE_TUNED_D32
        wgs_seg<PB, 7>(p, S, ring, u, s0, s1, slot, seq, pd);
        break;
#endif
#if MOE_WGS
      default:
        wgs_seg<PB, 8>(p, S, ring, u, s0, s1, slot, seq, pd);
        break;
    }
    gs += s1 - s0;
  }
  wgs_flush<PB>(p, S, pd);
}
// cost-weighted stage ranges: per-stage cost by the job's n-tiles, a prefix per job and phase (S.wcp, S.wcpB)
// all 32-bit (tot * c < 2^32: tot <= 136 jobs x 512 stages x 255 x 132): no 64-bit divide subroutine calls, which run
// from a cold instruction cache at the start of every launch
__device__ __forceinline__ int wgs_unitc(int pb) { return pb ? 140 : 160; }
// stage at cost position x (phase pb), the stage whose cost interval holds x
__device__ __forceinline__ int wgs_stage_at(const Smem& S, int nj, int pb, uint32_t x) {
  const int* pre = pb ? S.wcpB : S.wcp;
  int lo = 0, hi = nj;   // job: last j with prefix(j) <= x
  while (hi - lo > 1) {
    const int mid = (lo + hi) >> 1;
    if ((uint32_t)pre[mid] <= x)
      lo = mid;
    else
      hi = mid;
  }
  const uint32_t cs = (uint32_t)wgs_cost(pb, S.jC[wgs_job(S, pb, lo)]);
  const uint32_t U = (uint32_t)wgs_unitc(pb);
  const uint32_t cu = 16u * cs + U;
  const uint32_t r = x - (uint32_t)pre[lo];
  const uint32_t un = (uint32_t)muldiv_g((int)r, 1, (int)cu, __fdividef(1.f, (float)cu));
  const uint32_t rem = r - un * cu;
  const uint32_t st =
      rem < U
          ? 0u
          : min(15u, (uint32_t)muldiv_g((int)(rem - U), 1, (int)cs,
                                        __fdividef(1.f, (float)cs)));   // cepi: the unit cost sits before its stage 0
  return (lo << (9 - pb)) + (int)(un * 16u + st);
}
// this CTA's stage ranges (stream-K over 256-row units x 16 stages), one thread: the producers stream the same
// this CTA's stage ranges, warp-parallel (all 32 lanes of the x-producer warp; bs256-hot E457): the job-cost prefix as a
// shuffle scan over 32-job chunks, then the four stage_at searches on lanes 0..3 (lane = 2 pb + k). The serial one-lane version
// was the prologue's critical path (~1500 instructions before the weight stream can start)
__device__ __noinline__ void wgs_ranges(const Params& p, Smem& S) {
  const int lane = threadIdx.x & 31;
  const int nj = S.njF + S.njS;
  const int G = gridDim.x;
  const int c = blockIdx.x;
  uint32_t carryA = 0, carryB = 0;
  for (int j0 = 0; j0 < nj; j0 += 32) {
    const int j = j0 + lane;
    uint32_t ia = 0, ib = 0;
    if (j < nj) {
      ia = ((uint32_t)wgs_cost(0, S.jC[j]) * 16u + 160) << 5;
      ib = ((uint32_t)wgs_cost(1, S.jC[wgs_job(S, 1, j)]) * 16u + 140) << 4;
    }
    uint32_t sa = ia, sb = ib;
#pragma unroll
    for (int d = 1; d < 32; d <<= 1) {
      const uint32_t va = __shfl_up_sync(0xffffffffu, sa, d);
      const uint32_t vb = __shfl_up_sync(0xffffffffu, sb, d);
      if (lane >= d) {
        sa += va;
        sb += vb;
      }
    }
    if (j < nj) {
      S.wcp[j] = (int)(carryA + sa - ia);
      S.wcpB[j] = (int)(carryB + sb - ib);
    }
    carryA += __shfl_sync(0xffffffffu, sa, 31);
    carryB += __shfl_sync(0xffffffffu, sb, 31);
  }
  if (lane == 0) {
    S.wcp[nj] = (int)carryA;
    S.wcpB[nj] = (int)carryB;
    S.wtot = carryA;
    S.wtotB = carryB;
  }
  __syncwarp();
  const int l4 = lane & 3;
  const int pbl = l4 >> 1, kk = l4 & 1;
  const uint32_t tot = pbl ? carryB : carryA;
  int res;
  if (kk == 1 && c == (G - 1))
    res = nj << (9 - pbl);
  else
    res = wgs_stage_at(S, nj, pbl, tot * (uint32_t)(c + kk) / (uint32_t)G);
  if (lane < 4) S.wgr[lane] = res;
  __syncwarp();
}
// the CTA whose range holds stage s (the fixup slot of a split unit is the CTA holding its first stage)
__device__ __forceinline__ int wgs_cta_of(const Smem& S, int pb, int s, int G) {
  const uint32_t tot = pb ? S.wtotB : S.wtot;
  const int j = s >> (9 - pb);
  const uint32_t sj = (uint32_t)(s & ((1 << (9 - pb)) - 1)), cs = (uint32_t)wgs_cost(pb, S.jC[wgs_job(S, pb, j)]),
                 U = (uint32_t)wgs_unitc(pb);
  const uint32_t xe =
      (uint32_t)(pb ? S.wcpB : S.wcp)[j] + (sj >> 4) * (16u * cs + U) + U + ((sj & 15u) + 1u) * cs;   // cepi
  // largest c with floor(tot * c / G) < xe
  return min(G - 1, (int)((xe * (uint32_t)G - 1u) / tot));
}

#else
template <bool PB>
__device__ __forceinline__ void run_phase(const Params& p, Smem& S, __nv_bfloat16* xbuf) {
  const int njF = S.njF;
  const int njS = S.njS;
  const int ubs = PB ? UB_B : UB_A;
  const int rbs = PB ? RB_B : RB_A;
  const int nS = njS * rbs;
  const int nF = njF * ubs;
  // the skinny builds' jobs hold at most 16 tokens, so there are no fat units; this (never-taken) trap lets the
  // compiler drop their share, and keeps the skinny loop's code as tuned (an assumption instead compiles differently)
#if MOE_RAFN
  __builtin_assume(nF <= 0);
  int G;   // gridDim read opaquely: a merged read's phi would make the ring's bulk-copy operands divergent to ptxas
  asm volatile("mov.u32 %0, %%nctaid.x;" : "=r"(G));
#else
  if (nF > 0) __trap();
  const int G = gridDim.x;
#endif
  const int c = blockIdx.x;
  int v0, v1;
  const int nSs = nS - dyn_pool(nS, G);   // static skinny units; [nSs, nS) is the run-time pool
#if MOE_MID || MOE_TUNED_V1
  if constexpr (PB) {
    skinny_split(nSs + (nSs / RB_B) * MOE_BXW, nF, G, c, v0, v1);
    v0 = fresh_b_cost_position(v0);
    v1 = fresh_b_cost_position(v1);
  } else {
#if MOE_AXW
    skinny_split(nSs + (nSs / RB_A) * MOE_AXW, nF, G, c, v0, v1);
    v0 = fresh_a_cost_position(v0);
    v1 = fresh_a_cost_position(v1);
#else
    skinny_split(nSs, nF, G, c, v0, v1);
#endif
  }
#endif
#if !MOE_MID && !MOE_TUNED_V1
  skinny_split(nSs, nF, G, c, v0, v1);
#endif
#if MOE_MID16
  v0 = S.wsr[2 * PB];   // (the prologue's cost split: the helper reads the same ranges)
  v1 = S.wsr[2 * PB + 1];
#endif
  skinny_ring_ws<PB>(p, S, v0, v1, reinterpret_cast<unsigned char*>(xbuf), S.bars, S.seq);
}
#endif

// a volatile shared load at every use: hoisted, the pointer would stay live across the accumulate loops (96-register
// cap: spills)
__device__ __forceinline__ float* ss_cur() {
#if !MOE_WGS
  return reinterpret_cast<const Smem*>(smem_raw)
      ->ssp;   // the skinny / fat builds have registers to spare: let it hoist
#else
  unsigned long long v;
  asm volatile("ld.volatile.shared.u64 %0, [%1];\n"
               : "=l"(v)
               : "r"(smem_u32(&reinterpret_cast<const Smem*>(smem_raw)->ssp)));
  return reinterpret_cast<float*>(v);
#endif
}
#if MOE_IWARP && !MOE_WGS
__device__ void ws_issuer(const Params& p, Smem& S, unsigned char* ring) {   // IWARP: the refills, on each hand-over
  const int lane = TIDX & 31;
  WS_JOIN();   // routing tables and this CTA's ranges are ready
  const int a0 = S.wsr[0], b0 = S.wsr[2];
  const int nA = S.wsr[1] - a0, N = nA + (S.wsr[3] - b0);
#if MOE_ITAIL
  const int nB = S.wsr[3] - b0;
  // RS=4 advances one extra unit after A. All later q have an issue target >= N.
  const int Nh = min(N - min(MOE_WSBLK, nB), N - 3 - (RS == 4 && nB > 3));
#else
  const int Nh = N - min(MOE_WSBLK, S.wsr[3] - b0);   // (the last units are not handed over: their refills are >= N)
#endif
  for (int q = 0; q < Nh; ++q) {
    mbar_wait(S.wsrf + q % NRB, (uint32_t)((q / NRB) & 1));   // unit q consumed: its slot is free
    if (lane == 0) {
      if (RS == 3) {
        if ((q + 3) < N) ws_issue(p, S, q + 3, nA, a0, b0, ring);
      } else if (MOE_XSLOT == 1) {
        if (q < nA && (q + 3) < N) ws_issue(p, S, q + 3, nA, a0, b0, ring);
        if (q >= nA - 1 && (q + 4) < N) ws_issue(p, S, q + 4, nA, a0, b0, ring);
      } else {
        const int nq = q + 3 + (q >= nA);
        if (nq < N) ws_issue(p, S, nq, nA, a0, b0, ring);
      }
    }
    __syncwarp();
  }
  WS_JOIN();
}
#endif
__global__ void __launch_bounds__(NTL, 1) moe_fused(Params p) {
  Smem& S = *reinterpret_cast<Smem*>(smem_raw);
  __nv_bfloat16* xbuf = reinterpret_cast<__nv_bfloat16*>(smem_raw + ((sizeof(Smem) + 127) / 128) * 128);
#if !MOE_WGS
  const int tid = TIDX;
#else
  const int tid = threadIdx.x;
#endif
  // programmatic dependent launch: everything below reads the previous kernel's outputs (routing, x) or writes
  // scratch, so wait for it here; without the launch attribute this returns at once
  asm volatile("griddepcontrol.wait;\n" ::: "memory");
  if (MOE_PDLE) asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory");
  if (MOE_STAMP && tid == 0) MST(0);
#if MOE_ROLEUNI && !MOE_WGS
  const int role = __shfl_sync(0xffffffffu, tid >> 5, 0);
  if (role >= NW) {
#else
  if (tid >= NT) {
#endif
#if !MOE_WGS
#if MOE_IWARP
#if MOE_ROLEUNI
    if (role >= NTL / 32 - 1)
#else
    if (tid >= NTL - 32)
#endif
      ws_issuer(p, S, reinterpret_cast<unsigned char*>(xbuf));
    else
#endif
    ws_helper(p, S, reinterpret_cast<unsigned char*>(xbuf));
#else
    if (tid < NT + 32)
      wgs_wproducer(p, S, reinterpret_cast<unsigned char*>(xbuf));
    else
      wgs_xproducer(p, S, reinterpret_cast<unsigned char*>(xbuf));
#endif
    return;
#if !MOE_WGS
  }
  // tiny build (M <= 8): every touched expert is one skinny job, in expert order, so this CTA's first units follow
  // from the touched-expert mask alone; the last warp issues them before routing, off the cold-start critical path
#if MOE_ROLEUNI
  if (role == NW - 1) {
#else
  if (tid >= NT - 32) {
#endif
    const int ln = tid & 31;
#if !MOE_WGS && !MOE_MID
    // the topk loads go out first (E <= 64 in the tiny builds: two per lane at most) and lane 0 initializes the
    // barriers while they are in flight (the inits used to wait behind the cold loads)
    uint32_t m = 0;
    const int E8 = p.M * 8;
    const int ev0 = ln < E8 ? ldp(p.topk_ids + ln) : 0, ev1 = ln + 32 < E8 ? ldp(p.topk_ids + ln + 32) : 0;
#endif
#if MOE_MID
    // per-expert entry counts -> jobs per expert (ceil(c / 8)) -> inclusive prefix over experts (lane = expert)
    S.fcnt[ln] = 0;
    __syncwarp();
    for (int i0 = 0; i0 < p.M * 8; i0 += 32) {
      const int i = i0 + ln;
      int e = -1;
      if (i < p.M * 8) {
        e = ldp(p.topk_ids + i) - p.lo;
        if (e < 0 || e >= p.nloc) e = -1;
      }
      const unsigned peers = __match_any_sync(0xffffffffu, e);
      if (e >= 0 && ln == (__ffs(peers) - 1)) atomicAdd(&S.fcnt[e], __popc(peers));
    }
    __syncwarp();
    int jincl = (S.fcnt[ln] + SK_T - 1) / SK_T;
#pragma unroll
    for (int d = 1; d < 32; d <<= 1) {
      const int v = __shfl_up_sync(0xffffffffu, jincl, d);
      if (ln >= d) jincl += v;
    }
    const int njob = __shfl_sync(0xffffffffu, jincl, 31);
#if !MOE_MID16
    const int nS1 = njob * RB_A - dyn_pool(njob * RB_A, gridDim.x);
    const float rG1 = __fdividef(1.f, (float)gridDim.x);
    const int v01 = muldiv_g(nS1, blockIdx.x, gridDim.x, rG1);
    const int v11 = muldiv_g(nS1, blockIdx.x + 1, gridDim.x, rG1);
#else
    // the prologue's cost split (mid16_ranges) from the counts: one job per touched expert, a two-tile unit MOE_MID16W
    const uint32_t cl = S.fcnt[ln] > 0 ? (S.fcnt[ln] > 8 ? (uint32_t)MOE_MID16W : 8u) * RB_A : 0u;
    uint32_t ci = cl;
#pragma unroll
    for (int d = 1; d < 32; d <<= 1) {
      const uint32_t v = __shfl_up_sync(0xffffffffu, ci, d);
      if (ln >= d) ci += v;
    }
    const uint32_t tot1 = __shfl_sync(0xffffffffu, ci, 31);
    const float rG1 = __fdividef(1.f, (float)gridDim.x);
    // (warp-uniform) the unit at cost position floor(tot * b / G) of phase A (ci, cl, tot1) or B (ciB, clB, totB)
    auto at1 = [&](int b, uint32_t c_i, uint32_t c_l, uint32_t tt, int rbs, uint32_t ov) -> int {
      if (b == (int)gridDim.x) return njob * rbs;
      const uint32_t pc = tt * (uint32_t)b;
      uint32_t x = (uint32_t)__float2uint_rz((float)pc * rG1);
      while (x * gridDim.x > pc) --x;
      while ((x + 1u) * gridDim.x <= pc) ++x;
      const int jj = __popc(__ballot_sync(0xffffffffu, c_l > 0 && c_i <= x));
      const int L = __ffs(__ballot_sync(0xffffffffu, c_i > x)) - 1;
      const uint32_t r = x - (__shfl_sync(0xffffffffu, c_i, L) - __shfl_sync(0xffffffffu, c_l, L));
      const bool two = __shfl_sync(0xffffffffu, S.fcnt[ln], L) > 8;
      return jj * rbs + (r < ov ? 0 : (int)(two ? (r - ov) / (uint32_t)MOE_MID16W : (r - ov) >> 3));
    };
    const int v01 = at1(blockIdx.x, ci, cl, tot1, RB_A, 0u);
    const int v11 = at1(blockIdx.x + 1, ci, cl, tot1, RB_A, 0u);
    if (ln == 0) {   // the prologue takes phase A's range from here, and computes phase B's after routing
      S.mwsr[0] = v01;
      S.mwsr[1] = v11;
    }
#endif
    const int n1 = min(2, v11 - v01);
    int fex[2];
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      const int jj = (v01 + i) >> RB_A_LOG;   // (n1 is warp-uniform: every lane runs the same ballots)
      fex[i] = i < n1 ? __ffs(__ballot_sync(0xffffffffu, jincl > jj)) - 1 : 0;
    }
#endif
    if (ln == 0) {
#if !MOE_MID16
      for (int k = 0; k < 12; ++k) mbar_init(S.bars + k, 1);
      for (int k = 0; k < 3; ++k) mbar_init(S.wse + k, NW);
      for (int k = 0; k < NRB; ++k) {
        mbar_init(S.wsrf + k, 1);
        mbar_init(S.wsre + k, 1);
      }
#else
      {   // the addresses from a locally rematerialized shared base: from S (a kernel-entry vector value), ptxas made
        // each init a SYNCS waterfall loop in this build
        uint32_t sb_;
        asm volatile("mov.u32 %0, _ZN3okc8smem_rawE;" : "=r"(sb_));
        auto init = [&](size_t off, int count) {
          asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(sb_ + (uint32_t)off), "r"(count));
        };
        for (int k = 0; k < 12; ++k) init(offsetof(Smem, bars) + 8 * k, 1);
        for (int k = 0; k < 3; ++k) init(offsetof(Smem, wse) + 8 * k, NW);
        for (int k = 0; k < NRB; ++k) {
          init(offsetof(Smem, wsrf) + 8 * k, 1);
          init(offsetof(Smem, wsre) + 8 * k, 2);   // (the two helper warps)
        }
      }
#endif
      asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
#if !MOE_WGS && !MOE_MID
    }
    if (ln < E8) {
      const int e = ev0 - p.lo;
      if (e >= 0 && e < p.nloc) m |= 1u << e;
    }
    if (ln + 32 < E8) {
      const int e = ev1 - p.lo;
      if (e >= 0 && e < p.nloc) m |= 1u << e;
    }
    for (int i = ln + 64; i < E8; i += 32) {   // (never in the tiny builds: M <= 8)
      const int e = ldp(p.topk_ids + i) - p.lo;
      if (e >= 0 && e < p.nloc) m |= 1u << e;
    }
    m = __reduce_or_sync(0xffffffffu, m);
#if MOE_FNSBAL
    int first_grid;
    asm volatile("mov.u32 %0, %%nctaid.x;" : "=r"(first_grid));
    const int first_units = __popc(m) * RB_A;
    const float first_inverse = __fdividef(1.f, (float)first_grid);
#if MOE_AXW
    const int v0 = fresh_a_cost_position(
        muldiv_g(first_units + __popc(m) * MOE_AXW, blockIdx.x, first_grid, first_inverse));
    const int v1 = fresh_a_cost_position(
        muldiv_g(first_units + __popc(m) * MOE_AXW, blockIdx.x + 1, first_grid, first_inverse));
#else
    const int v0 = muldiv_g(first_units, blockIdx.x, first_grid, first_inverse);
    const int v1 = muldiv_g(first_units, blockIdx.x + 1, first_grid, first_inverse);
#endif
    const int n = min(2, v1 - v0);
    const int rank = __popc(m & ((1u << ln) - 1u));
    const bool present = (m >> ln) & 1u;
    const int first_expert = __ffs(__ballot_sync(0xffffffffu, present && rank == (v0 >> RB_A_LOG))) - 1;
    const int next_expert = __ffs(__ballot_sync(0xffffffffu, present && rank == ((v0 + 1) >> RB_A_LOG))) - 1;
#endif
    if (ln == 0) {
#if !MOE_FNSBAL
      const int nS =
          __popc(m) * RB_A - dyn_pool(__popc(m) * RB_A, gridDim.x);   // static part (pool tail claimed at run time)
      const float rG = __fdividef(1.f, (float)gridDim.x);
#if MOE_AXW
      const int v0 = fresh_a_cost_position(muldiv_g(nS + __popc(m) * MOE_AXW, blockIdx.x, gridDim.x, rG));
      const int v1 = fresh_a_cost_position(muldiv_g(nS + __popc(m) * MOE_AXW, blockIdx.x + 1, gridDim.x, rG));
#else
      const int v0 = muldiv_g(nS, blockIdx.x, gridDim.x, rG);
      const int v1 = muldiv_g(nS, blockIdx.x + 1, gridDim.x, rG);
#endif
      const int n = min(2, v1 - v0);
#endif
#endif
#if MOE_MID
      const int v0 = v01;
      const int n = n1;
#endif
      asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
#if !MOE_WGS && !MOE_MID
      for (int i = 0; i < n; ++i) {
#endif
#if MOE_MID
#pragma unroll
      for (int i = 0; i < 2; ++i) {
        if (i >= n) break;
#endif
        const int v = v0 + i;
#if !MOE_WGS && !MOE_MID
#if MOE_FNSBAL
        const int e = i == 0 ? first_expert : next_expert;
#else
        const int e = (int)__fns(m, 0, (v >> RB_A_LOG) + 1);
#endif
#endif
#if MOE_MID
        const int e = fex[i];
#endif
        const unsigned char* src = p.wAu + ((size_t)e * RB_A + (v & (RB_A - 1))) * (size_t)UNIT_GB;
        unsigned char* slot = reinterpret_cast<unsigned char*>(xbuf) + i * UNIT_BYTES;
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(smem_u32(S.bars + i)),
                     "r"(UNIT_GB)
                     : "memory");
        asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes" L2H_SUFFIX
                     " [%0], [%1], %2, [%3]" L2H_OPERAND ";\n" ::"r"(smem_u32(slot)),
                     "l"(src), "r"(UNIT_GB), "r"(smem_u32(S.bars + i)) L2H_ARG
                     : "memory");
      }
      S.preissued = n > 0 ? n : 0;
    }
#endif
  }
  // pull every token row of x into L2 now (one 128 B line per thread-iteration); the skinny jobs read them later
  for (int li = blockIdx.x * NT + tid; li < p.M * (HD * 2 / 128); li += gridDim.x * NT)
    asm volatile("prefetch.global.L2 [%0];\n" ::"l"(reinterpret_cast<const char*>(p.x) + (size_t)li * 128));
  for (int i = tid; i < (0); i += NT) {   // fp8 e4m3 code -> bf16x2(value * 2^64), computed (no global load)
    const int sgn = i >> 7, ex = (i >> 3) & 15, mt = i & 7;
    uint32_t b;
    if (ex == 15 && mt == 7) {
      b = 0xFFFFu;
    } else {
      float v = ex == 0 ? (float)mt * 0x1p-9f : (1.f + (float)mt * 0.125f) * __int_as_float((ex - 7 + 127) << 23);
      v *= 0x1p64f;
      b = __float_as_uint(sgn ? -v : v) >> 16;
    }
    S.lut[i] = b | (b << 16);
  }
  if (tid == 0) {
#if !MOE_WGS
    S.seq = 0;
    S.fseq = 0;
    S.preissuedB = 0;
    S.xbuf_ptr = xbuf;
#else
    // full slots (TMA tx), empty slots (one arrival per consumer warp), partials ready (every consumer thread),
    // partial buffer free (the epilogue warpgroup)
    for (int k = 0; k < BX_E + WGS_NX; ++k) mbar_init(S.bars + k, k < BW_E ? 1 : k < BX_F ? NW : k < BX_E ? 1 : NW);
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
#endif
    // the generation can only change at this launch's grid barrier, which every CTA reaches after this point
#if !MOE_WGS
    const int gen = *reinterpret_cast<volatile const int*>(p.ctr + 3);   // launch generation (bumped at the barrier)
#else
    // launch parity: the exit counter wraps at 2 G and every CTA bumps it once at its start, so it reads G * n (mod 2 G) plus
    // the CTAs of this launch that got here first (< G): >= G exactly in odd launches
    const int gen = atomicInc(reinterpret_cast<unsigned*>(p.mfl + MF_EXIT), 2u * gridDim.x - 1u) >= gridDim.x ? 1 : 0;
#endif
    S.gen = gen;
    S.ssp = (gen & 1) ? p.ss1 : p.ss;
  }
  route(p, S);
  if (MOE_STAMP && tid == 0) MST(1);
#if !MOE_WGS
  {   // the previous launch's ss copy and barrier counter are the next launch's: zero them now (this CTA's share)
    float* other = (S.gen & 1) ? p.ss : p.ss1;
    for (int i = blockIdx.x * NT + tid; i < EMAX; i += gridDim.x * NT) other[i] = 0.f;
    if (blockIdx.x == 0 && tid == 0) p.ctr[4 + ((S.gen + 1) & 1)] = 0;
#if !MOE_WGS && !MOE_TUNED_D16
  }
  if (tid == 0) {   // this CTA's unit ranges for the helper warp (the same exact split run_phase uses: no fat jobs)
    const int G = gridDim.x;
    const int c = blockIdx.x;
    const float rG = __fdividef(1.f, (float)G);
    const int nA = S.njS * RB_A;
    const int nB = S.njS * RB_B;
#if MOE_AXW
    S.wsr[0] = fresh_a_cost_position(muldiv_g(nA + S.njS * MOE_AXW, c, G, rG));
    S.wsr[1] = fresh_a_cost_position(muldiv_g(nA + S.njS * MOE_AXW, c + 1, G, rG));
#else
    S.wsr[0] = muldiv_g(nA, c, G, rG);
    S.wsr[1] = muldiv_g(nA, c + 1, G, rG);
#endif
#endif
#if !MOE_WGS && !MOE_MID && !MOE_TUNED_V1
    S.wsr[2] = muldiv_g(nB, c, G, rG);
    S.wsr[3] = muldiv_g(nB, c + 1, G, rG);
#endif
#if (MOE_MID && !MOE_TUNED_D16) || MOE_TUNED_V1
    const int costB = nB + S.njS * MOE_BXW;
    S.wsr[2] = fresh_b_cost_position(muldiv_g(costB, c, G, rG));
    S.wsr[3] = fresh_b_cost_position(muldiv_g(costB, c + 1, G, rG));
#endif
#if MOE_MID16 && !MOE_TUNED_D16
    S.wsr[0] = S.mwsr[0];   // the preissue warp's cost split (phase B's: warp 0 below)
    S.wsr[1] = S.mwsr[1];
#endif
#if !MOE_WGS && !MOE_TUNED_D16
  }
#if MOE_MID16
  if (tid < 32) mid16_rangesB(S, tid);
#endif
  WS_JOIN();
  run_phase<false>(p, S, xbuf);
  if (MOE_STAMP && tid == 0) MST(2);
  run_phase<true>(p, S, xbuf);
  if (MOE_STAMP && tid == 0) MST(3);
  if (threadIdx.x == 0) {   // (zero bytes: the loop is empty)
    const unsigned ch = ((p.upf_bytes / gridDim.x) + 15u) & ~15u;
    const unsigned b0 = blockIdx.x * ch;
    for (unsigned o = b0; o < b0 + ch && o < p.upf_bytes; o += 32768u) {
      const unsigned n = min(min(32768u, b0 + ch - o), p.upf_bytes - o);
      asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(p.upf + o), "r"(n) : "memory");
#endif
#else
  __threadfence_block();
  asm volatile("bar.arrive 7, %0;\n" ::"n"(NT + 32) : "memory");   // the x producer computes the stage ranges
  {   // the other parity's counters and hand-off flags belong to the next launch (the previous launch is over): zero them
    const int o = 1 - S.gen;
    for (int i = blockIdx.x * NT + tid; i < 32 + MF_XRS + JMAX * 16; i += gridDim.x * NT) {
      if (i < 32)
        p.mfl[MF_CE + o * 32 + i] = 0;
      else if (i < 32 + MF_XRS)
        p.mfl[MF_XR + o * MF_XRS + (i - 32)] = 0;
      else
        p.mfl[MF_DA + o * (JMAX * 16) + (i - 32 - MF_XRS)] = 0;
#endif
#if !MOE_TUNED_D16
    }
#endif
  }
#if !MOE_WGS
#if !MOE_WGS && !MOE_TUNED_D16
  WS_JOIN();   // the helper's last epilogue stores are visible
  if (MOE_STAMP && tid == 0) MST(4);
#endif
#if MOE_TUNED_D16
  if (tid == 0) {   // this CTA's unit ranges for the helper warp (the same exact split run_phase uses: no fat jobs)
    const int G = gridDim.x;
    const int c = blockIdx.x;
    const float rG = __fdividef(1.f, (float)G);
    const int nA = S.njS * RB_A;
    const int nB = S.njS * RB_B;
    S.wsr[0] = muldiv_g(nA, c, G, rG);
    S.wsr[1] = muldiv_g(nA, c + 1, G, rG);
    const int costB = nB + S.njS * MOE_BXW;
    S.wsr[2] = fresh_b_cost_position(muldiv_g(costB, c, G, rG));
    S.wsr[3] = fresh_b_cost_position(muldiv_g(costB, c + 1, G, rG));
#if MOE_MID16
    S.wsr[0] = S.mwsr[0];   // the preissue warp's cost split (phase B's: warp 0 below)
    S.wsr[1] = S.mwsr[1];
#endif
#endif
#else
  if (tid < 32) {   // padded-row base of every job in xA / xB (warp scan)
    const int nj = S.njF + S.njS;
    int carry = 0;
    for (int j0 = 0; j0 < nj; j0 += 32) {
      const int j = j0 + tid;
      const int v = j < nj ? (((S.jC[j] + 7) >> 3) << 3) : 0;
      int incl = v;
#pragma unroll
      for (int d = 1; d < 32; d <<= 1) {
        const int t = __shfl_up_sync(0xffffffffu, incl, d);
        if (tid >= d) incl += t;
      }
      if (j < nj) S.jrow[j] = carry + incl - v;
      carry += __shfl_sync(0xffffffffu, incl, 31);
    }
    if (tid == 0) S.jrow[nj] = carry;
#endif
#if MOE_WGS || MOE_TUNED_D16
  }
#endif
#if MOE_WGS
  CTA_SYNC();
  // phase A's x into xA (this CTA's share of the rows), before the producers start: its x reads do not queue behind
  // every CTA's first weight stages
  wgs_gather(p, S);
#endif
#if MOE_TUNED_D32
  asm volatile("bar.sync 2, %0;\n" ::"n"(NT + 32) : "memory");
#endif
#if MOE_MID16 && MOE_TUNED_D16
  if (tid < 32) mid16_rangesB(S, tid);
#endif
#if (MOE_WGS || MOE_TUNED_D16) && !MOE_TUNED_D32
  WS_JOIN();
#endif
#if MOE_TUNED_D16
  run_phase<false>(p, S, xbuf);
  run_phase<true>(p, S, xbuf);

  WS_JOIN();   // the helper's last epilogue stores are visible
#endif
#if MOE_WGS
  {
    unsigned char* ring = reinterpret_cast<unsigned char*>(xbuf);
    int seq = 0;
#if MOE_SGX
    wgs_selfx(p, S, ring);
#endif
    wgs_phase<false>(p, S, ring, S.wgr[0], S.wgr[1], seq);
    wgs_phase<true>(p, S, ring, S.wgr[2], S.wgr[3], seq);
  }
#endif
  // programmatic dependent launch: the next kernel may start its own prologue now (its griddepcontrol.wait still
  // waits for this whole grid to finish and its writes to be visible)
  if (!MOE_PDLE) asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory");
  // grid barrier on this launch's counter (two, by generation: the other one was zeroed by this launch for the next);
  // the last arriver bumps the generation, which no CTA of this launch reads again
#if !MOE_WGS
  // the CTA's first final item: its routing (pos) and route weights are inputs, so they load before the barrier
  constexpr int FSEGP = HD / (NT * 4);
  const int itp = blockIdx.x;
  const bool hasp = itp < p.M * FSEGP;
  const int tkp = hasp ? itp / FSEGP : 0;
  int qvp[8];
  float wvp[8];
#pragma unroll
  for (int s = 0; s < 8; ++s) {
    qvp[s] = hasp ? (int)S.pos[tkp * 8 + s] : -1;
    wvp[s] = (qvp[s] >= 0) ? ldp(p.topk_w + tkp * 8 + s) : 0.f;
  }
  CTA_SYNC();
  if (tid == 0) {
    int* bc = p.ctr + 4 + (S.gen & 1);
    if (!MOE_BREL) __threadfence();
    if (MOE_STAMP) MST(5);
    int old;
    if (MOE_BREL)   // release arrive: cumulative over the CTA's stores ordered before it by the CTA barrier
      asm volatile("atom.add.release.gpu.global.s32 %0, [%1], 1;\n" : "=r"(old) : "l"(bc) : "memory");
    else
      old = atomicAdd(bc, 1);
    if (old == (int)gridDim.x - 1) atomicAdd(p.ctr + 3, 1);
    while (ld_acquire(bc) < (int)gridDim.x) __nanosleep(64);
    if (MOE_STAMP) MST(6);
  }
  CTA_SYNC();
  // phases A and B are over everywhere: zero the per-launch flags for the next launch (this CTA's share)
  {
    const int G = gridDim.x;
    const int i0 = blockIdx.x * NT + tid;
    if (i0 < 2) p.ctr[i0] = 0;   // dynamic pool counters
    for (int i = i0; i < JMAX; i += G * NT) p.doneA[i] = 0;
  }
#if MOE_DF || MOE_T1DF
  {   // every CTA is past phase B: inter is dead until the next launch; give it back its "not yet written" sentinel
    const int rows = min(128, S.off[31] + S.cnt[31]);   // this launch's local pairs: the rows it wrote
    uint4* ip = reinterpret_cast<uint4*>(p.inter);
    for (int i = blockIdx.x * NT + tid; i < rows * (ID / 8); i += gridDim.x * NT)
      ip[i] = make_uint4(0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu);
  }
#endif
#endif
  // final: out[t] = sum_s w * rsqrt(ss/HD + eps) * ybuf[pos]
#if !MOE_WGS || MOE_B4FIN
#if MOE_B4FIN
  // B4FIN: item = (token, 1024-column segment), thread = 2 columns: 4 x M items over all CTAs. Each route's sum
  // of squares = fixed-order sum of its 256 epilogue partials (warp w: route w / 2, half w & 1; xor tree; 2 halves)
#endif
#if !MOE_WGS
  // column-parallel: item = (token, 1024-column segment); thread = 4 columns
  constexpr int FSEG = HD / (NT * 4);   // column segments per token (thread = 4 columns)
  int it0 = blockIdx.x;
  if (hasp) {   // first item: weights / positions from before the barrier; all 16 loads issued before the math
    const int col = (itp % FSEG) * (NT * 4) + tid * 4;
    float ssv[8];
    float4 yv[8];
#pragma unroll
    for (int s = 0; s < 8; ++s) {
      ssv[s] = qvp[s] >= 0 ? __ldcg(ss_cur() + qvp[s]) : 0.f;
      yv[s] = qvp[s] >= 0 ? __ldcg(reinterpret_cast<const float4*>(p.ybuf + (size_t)qvp[s] * HD + col))
                          : make_float4(0.f, 0.f, 0.f, 0.f);
    }
    float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
    for (int s = 0; s < 8; ++s) {
      if (qvp[s] >= 0) {
        const float r = wvp[s] * rsqrtf(ssv[s] * (1.f / HD) + 1e-8f);
        acc.x += r * yv[s].x;
        acc.y += r * yv[s].y;
        acc.z += r * yv[s].z;
        acc.w += r * yv[s].w;
#endif
#endif
#if MOE_WGS && !MOE_B4FIN
  // item = one token's whole row (thread = 8 columns): pass 1 sums each local route's y^2 over the row (warp
  // shuffles, then the 16 warps' partials through smem: the ring is free now), pass 2 re-reads the rows (L2 hits)
#endif
#if MOE_WGS
  {
#endif
#if MOE_B4FIN
    static_assert(NT == 512 && NW == 16, "B4FIN: 16 warps = 8 routes x 2 halves, 512 threads x 2 columns");
#endif
#if MOE_WGS && !MOE_B4FIN
    static_assert(HD == NT * 8, "8 columns per thread");
#endif
#if MOE_WGS
    float* red = reinterpret_cast<float*>(xbuf);
    const int lane = tid & 31;
    const int warp = WARP_IDX(tid);
#endif
#if MOE_B4FIN
    constexpr int NSEG = HD / 1024;
    for (int it = blockIdx.x; it < p.M * NSEG; it += gridDim.x) {
      const int tk = it / NSEG;
      const int col = (it % NSEG) * 1024 + tid * 2;
#endif
#if MOE_WGS && !MOE_B4FIN
    for (int tk = blockIdx.x; tk < p.M; tk += gridDim.x) {
#endif
#if MOE_WGS
      if (tid <
          8) {   // the token's local experts are complete when all their jobs' units were counted by all 4 warpgroups
        if (S.pos[tk * 8 + tid] >= 0) {
          const int e = S.eid[tk * 8 + tid];
          const int tot = S.cnt[e];
          const int target = 64 * ((tot + FT_T - 1) / FT_T);
          const int* cp = p.mfl + MF_CE + S.gen * 32 + e;
          while (ld_acquire(cp) < target) {
          }
        }
#endif
      }
#if !MOE_WGS
    }
    __nv_bfloat162 o0 = __floats2bfloat162_rn(acc.x, acc.y), o1 = __floats2bfloat162_rn(acc.z, acc.w);
    uint2 o;
    o.x = *reinterpret_cast<uint32_t*>(&o0);
    o.y = *reinterpret_cast<uint32_t*>(&o1);
    *reinterpret_cast<uint2*>(p.out + (size_t)tkp * HD + col) = o;
  }
  it0 += gridDim.x;
  for (int it = it0; it < p.M * FSEG; it += gridDim.x) {
    const int tk = it / FSEG;
    const int col = (it % FSEG) * (NT * 4) + tid * 4;
    float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
#else
      CTA_SYNC();
#endif
#if MOE_B4FIN
      float2 yv[8];   // A2: the 8 routes' y loads go out before the ss partial loads and the reduction
#endif
#if MOE_WGS && !MOE_B4FIN
      int qv[8];
#endif
#if !MOE_B4FIN
#pragma unroll
#endif
#if !MOE_WGS
    for (int s = 0; s < 8; ++s) {
      const int q = S.pos[tk * 8 + s];
      if (q >= 0) {
        const float r = ldp(p.topk_w + tk * 8 + s) * rsqrtf(__ldcg(ss_cur() + q) * (1.f / HD) + 1e-8f);
        const float4 v = __ldcg(reinterpret_cast<const float4*>(p.ybuf + (size_t)q * HD + col));
        acc.x += r * v.x;
        acc.y += r * v.y;
        acc.z += r * v.z;
        acc.w += r * v.w;
#endif
#if MOE_WGS && !MOE_B4FIN
      for (int sl = 0; sl < 8; ++sl) qv[sl] = S.pos[tk * 8 + sl];
      float ps[8];
#endif
#if MOE_WGS && !MOE_B4FIN && MOE_YKEEP
      uint4 yk[8];   // pass 1's rows, kept for pass 2 (no second L2 round trip on the tail)
      float wk[8];   // and the route weights, loaded with them
#pragma unroll
      for (int sl = 0; sl < 8; ++sl) wk[sl] = qv[sl] >= 0 ? ldp(p.topk_w + tk * 8 + sl) : 0.f;
#endif
#if MOE_WGS
#pragma unroll
      for (int sl = 0; sl < 8; ++sl) {
#endif
#if MOE_B4FIN
        const int q = S.pos[tk * 8 + sl];
        yv[sl] =
            q >= 0 ? __ldcg(reinterpret_cast<const float2*>(p.ybuf + (size_t)q * HD + col)) : make_float2(0.f, 0.f);
      }
      {
        const int sl = warp >> 1, hh = warp & 1;
        const int q = S.pos[tk * 8 + sl];
        float v = 0.f;
        if (q >= 0) {
          const float4 a = __ldcg(reinterpret_cast<const float4*>(p.sspart + (size_t)q * 256 + hh * 128 + lane * 4));
          v = (a.x + a.y) + (a.z + a.w);
        }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
        if (lane == 0) red[warp] = v;
      }
      CTA_SYNC();
      float2 acc = make_float2(0.f, 0.f);
#pragma unroll
      for (int sl = 0; sl < 8; ++sl) {
        const int q = S.pos[tk * 8 + sl];
        if (q >= 0) {
          const float r = ldp(p.topk_w + tk * 8 + sl) * rsqrtf((red[2 * sl] + red[2 * sl + 1]) * (1.f / HD) + 1e-8f);
          const float2 v = yv[sl];
          acc.x += r * v.x;
          acc.y += r * v.y;
#endif
#if MOE_WGS && !MOE_B4FIN
        ps[sl] = 0.f;
        if (qv[sl] >= 0) {
          const uint4 yr = __ldcg(
              reinterpret_cast<const uint4*>(reinterpret_cast<const __nv_bfloat16*>(p.ybuf) + (size_t)qv[sl] * HD) +
              tid);
#if MOE_YKEEP
          yk[sl] = yr;
#endif
          const float2 y0 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&yr.x)),
                       y1 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&yr.y));
          const float2 y2 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&yr.z)),
                       y3 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&yr.w));
          const float4 a = make_float4(y0.x, y0.y, y1.x, y1.y), b = make_float4(y2.x, y2.y, y3.x, y3.y);
          ps[sl] = a.x * a.x + a.y * a.y + a.z * a.z + a.w * a.w + b.x * b.x + b.y * b.y + b.z * b.z + b.w * b.w;
#endif
#if MOE_WGS
        }
#endif
      }
#if MOE_B4FIN
      *reinterpret_cast<__nv_bfloat162*>(p.out + (size_t)tk * HD + col) = __floats2bfloat162_rn(acc.x, acc.y);
      if (it + (int)gridDim.x < p.M * NSEG) CTA_SYNC();   // red is reused by the next item
#endif
#if MOE_WGS && !MOE_B4FIN
#pragma unroll
      for (int sl = 0; sl < 8; ++sl)
        if (qv[sl] >= 0) {   // uniform over the CTA (one token)
#pragma unroll
          for (int o = 16; o > 0; o >>= 1) ps[sl] += __shfl_xor_sync(0xffffffffu, ps[sl], o);
        }
      if (lane == 0) {
#pragma unroll
        for (int sl = 0; sl < 8; ++sl) red[warp * 8 + sl] = ps[sl];
      }
      CTA_SYNC();
      if (tid < 8) {
        float tsum = 0.f;
        for (int w = 0; w < NW; ++w) tsum += red[w * 8 + tid];
        red[NW * 8 + tid] = tsum;
      }
      CTA_SYNC();
      float acc[8];
#pragma unroll
      for (int i = 0; i < 8; ++i) acc[i] = 0.f;
#pragma unroll
      for (int sl = 0; sl < 8; ++sl)
        if (qv[sl] >= 0) {
#if MOE_YKEEP
          const float r = wk[sl] * rsqrtf(red[NW * 8 + sl] * (1.f / HD) + 1e-8f);
          const uint4 yr = yk[sl];
#else
          const float r = ldp(p.topk_w + tk * 8 + sl) * rsqrtf(red[NW * 8 + sl] * (1.f / HD) + 1e-8f);
          const uint4 yr = __ldcg(
              reinterpret_cast<const uint4*>(reinterpret_cast<const __nv_bfloat16*>(p.ybuf) + (size_t)qv[sl] * HD) +
              tid);
#endif
          const float2 y0 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&yr.x)),
                       y1 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&yr.y));
          const float2 y2 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&yr.z)),
                       y3 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&yr.w));
          const float4 a = make_float4(y0.x, y0.y, y1.x, y1.y), b = make_float4(y2.x, y2.y, y3.x, y3.y);
          acc[0] += r * a.x;
          acc[1] += r * a.y;
          acc[2] += r * a.z;
          acc[3] += r * a.w;
          acc[4] += r * b.x;
          acc[5] += r * b.y;
          acc[6] += r * b.z;
          acc[7] += r * b.w;
        }
      uint4 o;
      uint32_t* ow = reinterpret_cast<uint32_t*>(&o);
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        __nv_bfloat162 v = __floats2bfloat162_rn(acc[2 * i], acc[2 * i + 1]);
        ow[i] = *reinterpret_cast<uint32_t*>(&v);
      }
      *reinterpret_cast<uint4*>(p.out + (size_t)tk * HD + tid * 8) = o;
      if (tk + (int)gridDim.x < p.M) CTA_SYNC();   // red is reused by the next token
#endif
    }
#if !MOE_WGS
    __nv_bfloat162 o0 = __floats2bfloat162_rn(acc.x, acc.y), o1 = __floats2bfloat162_rn(acc.z, acc.w);
    uint2 o;
    o.x = *reinterpret_cast<uint32_t*>(&o0);
    o.y = *reinterpret_cast<uint32_t*>(&o1);
    *reinterpret_cast<uint2*>(p.out + (size_t)tk * HD + col) = o;
#endif
  }
  // Nothing to reset at exit: the flags were zeroed after the barrier, ss is double buffered.
}

}   // namespace okc

#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/extra/c_env_api.h>

using TV = tvm::ffi::TensorView;

static size_t okc_smem_bytes() {
  size_t base = ((sizeof(okc::Smem) + 127) / 128) * 128;
#if MOE_WGS
  return base + (size_t)3 * okc::UNIT_BYTES + (size_t)3 * okc::WGS_XS;
#else
  size_t xa = (size_t)3 * okc::UNIT_BYTES;
  size_t xb = (size_t)3 * okc::UNIT_BYTES + 2 * (size_t)64 * 264 * 2;
  return base + (xa > xb ? xa : xb);
#endif
}

// One expert-parallel MoE step over this GPU's experts [lo, lo + nloc): x [M, 4096] bf16 rows, their top-8 ids and
// weights -> out [M, 4096]. Weights: the prepared layouts (codes, scales, global scales, fat / unit / WGS orders);
// scratch: counters + flags (int32), sums + y rows (fp32), inter + x copies (bf16). upf: the unpooler stream's L2
// prefetch (small builds).
void run(TV wA, TV sA, TV gsA, TV wB, TV sB, TV gsB, TV lut, TV normw, TV wAf, TV wBf, TV wAu, TV wBu, TV wAn, TV wBn,
         TV ints, TV floats, TV inter, TV x, TV ids, TV tw, TV out, int64_t lo, int64_t nloc, int64_t grid, int64_t pdl,
         int64_t upf_ptr, int64_t upf_bytes) {
  okc::Params p;
  p.wA = (const uint4*)wA.data_ptr(); p.sA = (const uint4*)sA.data_ptr(); p.gsA = (const float*)gsA.data_ptr();
  p.wB = (const uint4*)wB.data_ptr(); p.sB = (const uint4*)sB.data_ptr(); p.gsB = (const float*)gsB.data_ptr();
  p.lut = (const uint32_t*)lut.data_ptr(); p.normw = (const __nv_bfloat16*)normw.data_ptr();
  p.wAf = (const unsigned char*)wAf.data_ptr(); p.wBf = (const unsigned char*)wBf.data_ptr();
  p.wAu = (const unsigned char*)wAu.data_ptr(); p.wBu = (const unsigned char*)wBu.data_ptr();
  p.wAn = (const unsigned char*)wAn.data_ptr(); p.wBn = (const unsigned char*)wBn.data_ptr();
#if MOE_WGS
  p.wAs = p.wAn; p.wBs = p.wBn;
#else
  p.wAs = nullptr; p.wBs = nullptr;
#endif
  p.zero = 0;
  p.x = (const __nv_bfloat16*)x.data_ptr(); p.topk_ids = (const int*)ids.data_ptr(); p.topk_w = (const float*)tw.data_ptr();
  p.M = (int)x.size(0); p.lo = (int)lo; p.nloc = (int)nloc;
  int* ib = (int*)ints.data_ptr();
  p.ctr = ib; p.doneA = ib + 8;
  p.ss = (float*)floats.data_ptr();
  p.ss1 = reinterpret_cast<float*>(ib + 8 + 1024);
  p.ybuf = p.ss + okc::EMAX;
  p.upf = reinterpret_cast<const uint8_t*>(upf_ptr);
  p.upf_bytes = (unsigned)upf_bytes;
  p.inter = (__nv_bfloat16*)inter.data_ptr();
#if MOE_WGS
  p.mfl = ib + 8 + 1024 + okc::EMAX;
#if MOE_B4FIN
  // B4FIN runs only at M <= 32: ybuf rows (sorted routes) stay below M * 8 <= 256, so its last 64 rows (1024 routes x 256 partials) are free
  p.sspart = p.ybuf + (size_t)(okc::EMAX - 64) * okc::HD;
#else
  p.sspart = nullptr;
#endif
  p.xA = p.inter + (size_t)okc::EMAX * okc::HD;
  p.xB = p.xA + (size_t)okc::WGS_XROWS * okc::HD;
  p.xready = p.mfl + okc::MF_XR;
  p.doneAs = p.mfl + okc::MF_DA;
#else
#if MOE_DF || MOE_T1DF
  p.inter += (size_t)128 * 4096;   // DF sentinel rows [128, 256) of the scratch: only DF launches use them
#endif
  p.xA = p.xB = nullptr;
  p.xready = nullptr;
  p.doneAs = nullptr;
#endif
  p.out = (__nv_bfloat16*)out.data_ptr();
  auto kern = okc::moe_fused;
  static bool attr = false;
  const size_t sm = okc_smem_bytes();
  if (!attr) {
    cudaFuncSetAttribute(okc::moe_fused, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm);
    attr = true;
  }
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3((unsigned)grid);
  cfg.blockDim = dim3(okc::NTL);
  cfg.dynamicSmemBytes = sm;
  cfg.stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(x.device().device_type, x.device().device_id));
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  at[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = at;
  cfg.numAttrs = pdl ? 1 : 0;
  cudaLaunchKernelEx(&cfg, kern, p);
}
"""

SK2_CU = r"""
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <cudaTypedefs.h>
#include <cuda_bf16.h>
#include <stdint.h>

__device__ __forceinline__ void ldsm_x4(uint32_t addr, uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
               : "r"(addr));
}

__device__ __forceinline__ void mma_bf16(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
                                         uint32_t b1) {
  asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint32_t prmt(uint32_t a, uint32_t b, uint32_t sel) {
  uint32_t d;
  asm("prmt.b32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(sel));
  return d;
}

__device__ __forceinline__ uint32_t lop3_sel(uint32_t a, uint32_t b, uint32_t c) {
  // c ? b : a, bitwise
  uint32_t d;
  asm("lop3.b32 %0, %1, %2, %3, 0xD8;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
  return d;
}

// Decode 8 weights: low bytes of weights 0..3 in lw0, 4..7 in lw1, codes in nw
// (low nibble of byte b -> weight b, high nibble -> weight 4+b; code = s<<3 | o).
// Output: 4 bf16x2 registers r[0..3] = (w0,w1) (w2,w3) (w4,w5) (w6,w7).
#ifndef SK2_C7   // the 7-op codec (eat c7): each code byte holds o_b (bits 0-2), s_b (bit 7), o_{4+b} (bits 4-6) and
#define SK2_C7 0   // s_{4+((b-1)&3)} (bit 3); the matrix is packed times 2^sv so its window sits on the aligned [56, 64)
#endif             // (the base ORs in), sv rides in base4's bit 7s and every output is scaled by 2^-sv where it is stored
__device__ __forceinline__ float c7_dsc(uint32_t base4) {   // 2^-sv
  const int s = (int)(((base4 >> 7) & 1u) | ((base4 >> 14) & 2u) | ((base4 >> 21) & 4u) | ((base4 >> 28) & 8u));
  return __int_as_float((127 - (s >= 8 ? s - 16 : s)) << 23);
}
__device__ __forceinline__ void decode8(uint32_t lw0, uint32_t lw1, uint32_t nw, uint32_t base4, uint32_t* r) {
#if SK2_C7
  const uint32_t B = base4 & 0x7F7F7F7Fu;
  const uint32_t ha = (nw & 0x87878787u) | B, hb = (__funnelshift_r(nw, nw, 4) & 0x87878787u) | B;
#else
  uint32_t ha = lop3_sel(nw & 0x07070707u, nw << 4, 0x80808080u) + base4;
  uint32_t hb = lop3_sel((nw >> 4) & 0x07070707u, nw, 0x80808080u) + base4;
#endif
  r[0] = prmt(lw0, ha, 0x5140);
  r[1] = prmt(lw0, ha, 0x7362);
  r[2] = prmt(lw1, hb, 0x5140);
  r[3] = prmt(lw1, hb, 0x7362);
}

template <typename T>
__device__ __forceinline__ T cvt_out(float v);
template <>
__device__ __forceinline__ float cvt_out<float>(float v) {
  return v;
}
template <>
__device__ __forceinline__ __nv_bfloat16 cvt_out<__nv_bfloat16>(float v) {
  return __float2bfloat16_rn(v);
}

// Weights, uncompressed (COMP=0): [num_rg][S][32 lanes][16 B] A fragments, S = Kp / 16.
// Weights, compressed (COMP=1): [num_rg][S/4 units][3 parts][32 lanes][16 B]: parts 0,1 = low
//   bytes of the lane's 32 weights of 4 k-steps, part 2 = their 32 codes.
// x: [M][Kx] bf16.  Weight column k uses x column (k % Kx) (router hi|lo concat).
// CTA (rb, sp): row groups rb*RG .. +RG, k range [sp*Kc, (sp+1)*Kc).
// Warp w: row group rb*RG + w / WS, k sub-range (w % WS).
// D: prefetch depth in loads of 16 B per lane (k-steps if !COMP, unit parts if COMP).

// ------------------------------------------------------------------ mbarrier / bulk copy helpers
__device__ __forceinline__ void mbar_init(uint32_t bar, int count) {
  asm volatile("mbarrier.init.shared.b64 [%0], %1;" ::"r"(bar), "r"(count) : "memory");
}
__device__ __forceinline__ bool mbar_test(uint32_t bar, uint32_t parity) {
  uint32_t ok;
  asm volatile("{\n .reg .pred p;\n mbarrier.try_wait.parity.shared.b64 p, [%1], %2;\n selp.u32 %0, 1, 0, p;\n}\n"
               : "=r"(ok)
               : "r"(bar), "r"(parity)
               : "memory");
  return ok != 0;
}
__device__ __forceinline__ void mbar_wait(uint32_t bar, uint32_t parity) {
  while (!mbar_test(bar, parity)) {
  }
}
// wait with a suspend-time hint: the warp sleeps in hardware until the phase completes instead of re-polling
__device__ __forceinline__ void mbar_wait_sleep(uint32_t bar, uint32_t parity) {
  uint32_t ok = 0;
  do {
    asm volatile(
        "{\n .reg .pred p;\n mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2, %3;\n selp.u32 %0, 1, 0, p;\n}\n"
        : "=r"(ok)
        : "r"(bar), "r"(parity), "r"(0x989680u)
        : "memory");
  } while (!ok);
}
__device__ __forceinline__ void mbar_arrive(uint32_t bar) {
  asm volatile("{\n .reg .b64 st;\n mbarrier.arrive.shared.b64 st, [%0];\n}\n" ::"r"(bar) : "memory");
}
__device__ __forceinline__ void named_bar(int id, int n) {
  asm volatile("bar.sync %0, %1;" ::"r"(id), "r"(n) : "memory");
}
#ifndef SK2_BF16P   // the k-warp partials through shared memory (and DSMEM, k-warps > 0) as BF16: half the bytes; the
#define SK2_BF16P 0   // escape corrections stay FP32 (corr, and k-warp 0's FP32 peer partial that carries them)
#endif
__device__ __forceinline__ void sk2_pst(float* b, int i, float a, float c) {   // partial pair (i, i + 1)
  if constexpr (SK2_BF16P)
    *reinterpret_cast<__nv_bfloat162*>(reinterpret_cast<__nv_bfloat16*>(b) + i) = __floats2bfloat162_rn(a, c);
  else
    *reinterpret_cast<float2*>(b + i) = make_float2(a, c);
}
__device__ __forceinline__ float4 sk2_hld(const void* b, int i) {   // 4 BF16 partials (i .. i + 3) as FP32
  const uint2 u = *reinterpret_cast<const uint2*>(reinterpret_cast<const __nv_bfloat16*>(b) + i);
  return make_float4(__uint_as_float(u.x << 16), __uint_as_float(u.x & 0xffff0000u), __uint_as_float(u.y << 16),
                     __uint_as_float(u.y & 0xffff0000u));
}
__device__ __forceinline__ float4 sk2_pld(const float* b, int i) {
  if constexpr (SK2_BF16P)
    return sk2_hld(b, i);
  else
    return *reinterpret_cast<const float4*>(b + i);
}
__device__ __forceinline__ void mbar_expect_tx(uint32_t bar, uint32_t bytes) {
  asm volatile("{\n .reg .b64 st;\n mbarrier.arrive.expect_tx.shared.b64 st, [%0], %1;\n}\n" ::"r"(bar), "r"(bytes)
               : "memory");
}
// L2 policies (v86y = v86r + bf16_proj v131's hints): weight streams are read once per call -> evict_first; x tiles
// are re-read by every CTA -> evict_last
__device__ __forceinline__ uint64_t l2_pol_ef() { return 0x12F0000000000000ull; }
__device__ __forceinline__ uint64_t l2_pol_el() { return 0x14F0000000000000ull; }
__device__ __forceinline__ void bulk_g2s(uint32_t dst, const void* src, uint32_t bytes, uint32_t bar) {
  asm volatile(
      "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;" ::
          "r"(dst),
      "l"(src), "r"(bytes), "r"(bar), "l"(l2_pol_ef())
      : "memory");
}
// wait on a local mbarrier whose phase another CTA of the cluster completes (acquire at cluster scope)
__device__ __forceinline__ void mbar_wait_cluster(uint32_t bar, uint32_t parity) {
  uint32_t ok = 0;
  do {
    asm volatile(
        "{\n .reg .pred p;\n mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [%1], %2;\n"
        " selp.u32 %0, 1, 0, p;\n}\n"
        : "=r"(ok)
        : "r"(bar), "r"(parity)
        : "memory");
  } while (!ok);
}


#include <cudaTypedefs.h>
#include <cstring>
__device__ __forceinline__ void tma_load_3d(uint32_t dst, const CUtensorMap* map, uint32_t bar, int c0, int c1,
                                            int c2) {
  asm volatile(
      "cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1, {%3, "
      "%4, %5}], [%2], %6;" ::"r"(dst),
      "l"(map), "r"(bar), "r"(c0), "r"(c1), "r"(c2), "l"(l2_pol_el())
      : "memory");
}
static PFN_cuTensorMapEncodeTiled_v12000 get_encode() {
  static PFN_cuTensorMapEncodeTiled_v12000 fn = nullptr;
  if (!fn) {
    cudaDriverEntryPointQueryResult q;
    void* p = nullptr;
    cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", &p, cudaEnableDefault, &q);
    fn = reinterpret_cast<PFN_cuTensorMapEncodeTiled_v12000>(p);
  }
  return fn;
}

// x [M, K] bf16 as 3-D {64 (k in unit), M (token), K/64 (unit)}: one TMA brings `units` tiles of [rows x 128 B]
static CUtensorMap make_map_x3d(const void* ptr, uint64_t m, uint64_t k, uint32_t box_rows, uint32_t units) {
  CUtensorMap mp;
  cuuint64_t dims[3] = {64, m, k / 64};
  cuuint64_t strides[2] = {k * 2, 128};
  cuuint32_t box[3] = {64, box_rows, units};
  cuuint32_t estr[3] = {1, 1, 1};
  CUresult r = get_encode()(&mp, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, const_cast<void*>(ptr), dims, strides, box, estr,
                            CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                            CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  return mp;
}


#define SK2_WAIT(bar, par) mbar_wait_sleep((bar), (par))
#ifndef SK2_GLE16   // the GLE pooler + FP32 router also for 9..16 rows (gle_pr at T = 8: the u / v rows of 8 tokens):
#define SK2_GLE16 0   // TOKR 16 and eight 256-column FP32 router stages (8 rows x 1 KB weights + 16 rows x 1 KB x per stage)
#endif
#ifndef SK2_ESPL   // M > 8: every escape warp works on every stage, on its share of the stage's escapes (entry index mod
#define SK2_ESPL 0   // NESC), instead of whole stages in turn: the last stage's corrections finish ~1/NESC of a stage's
#endif               // escape work after it lands (d16s: escapes done 1.2 us after the main loop)
#ifndef SK2_RRED   // fused pooler + router (sk2f, 8 < M <= 32): the router's k-warp reduction in a dedicated 4..8 KB region,
#define SK2_RRED 0   // not the ring, so it does not wait for the escape warps (d16s: router done 2.25 us after the loop)
#endif
#ifndef SK2_RSUM   // fused pooler + router: each CTA leaves its quarter-K router partial in part and stops there; the route
#define SK2_RSUM 0   // kernel adds the four quarters (same order) instead of a counter-elected last CTA (no release atomic,
#endif               // fence and partial reads at the end of every pooler CTA)
#ifndef SK2_ELECT   // 1: the whole producer warp runs the refill loop (warp index made uniform with a shuffle, the empty
#define SK2_ELECT 0   // wait spins inside its asm) and the copies issue under elect.sync: ptxas then emits each bulk /
#endif                // tensor copy once, not in an ELECT ... R2UR ... BRA.U.ANY waterfall
#ifndef SK2_XR   // M = 32: the x tiles in their own SK2_XR-slot ring beside the weight ring, which then takes SK2_XRB bytes
#define SK2_XR 0   // less the x slots (deeper: 4 -> 6 qkv, 3 -> 4 pooler at XR 3). The producer issues stage j's x LAG =
#endif             // weight slots - x slots iterations after its weights, so one empty barrier frees both slots
#ifndef SK2_XRM   // which d32 kernels: 1 qkv, 2 the fused pooler + router
#define SK2_XRM 3
#endif
#ifndef SK2_XRB
#define SK2_XRB 208896
#endif
template <int TOKR, int NT, int NWR, bool FUSE, int CL>
__host__ __device__ constexpr bool sk2_xr() {
  return !SK2_NARROW && !SK2_GLE && SK2_XR > 0 && CL == 1 && TOKR == 32 &&
         ((FUSE && (SK2_XRM & 2)) || (!FUSE && NT == 3 && NWR == 1 && (SK2_XRM & 1)));
}
template <int NSTAGE, int TOKR, int NT, int NWR, bool FUSE, int CL>
__host__ __device__ constexpr int sk2_nesc() {   // escape warps (SK2_XR: one per x slot, the register budget of before)
  return sk2_xr<TOKR, NT, NWR, FUSE, CL>() ? (SK2_XR >= 4 ? 4 : SK2_XR) : (NSTAGE >= 4 ? 4 : NSTAGE);
}
__device__ __forceinline__ void mbar_wait_spin(uint32_t bar, uint32_t parity) {
  asm volatile(
      "{\n .reg .pred p;\n SK2W_%=:\n mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1, %2;\n @!p bra SK2W_%=;\n}\n" ::"r"(
          bar),
      "r"(parity), "r"(0x989680u)
      : "memory");
}
__device__ __forceinline__ bool sk2_elect() {
  uint32_t e;
  asm volatile("{\n .reg .pred e;\n elect.sync _|e, 0xffffffff;\n selp.u32 %0, 1, 0, e;\n}\n" : "=r"(e));
  return e;
}
// PDL (programmatic dependent launch) for the sk2 launches: 0 = off (plain launches); N > 0 = launch with programmatic
// stream serialization and issue the first N weight stages before griddepcontrol.wait (set by sk2_set_pdl)
using TV = tvm::ffi::TensorView;
#if SK2_GLE
struct GleEpi {
  const float *s, *a, *b, *a2;
  const __nv_bfloat16* p;
  __nv_bfloat16* uv;
  float* part;
  float* uvf;
  const float *x32, *w32;
  int fpr, ilv;
};
#endif
// ============================================================================ sk2 (m <= 32): swapped operands
// A = x tile (16 tokens x 16 k; ldmatrix from the TMA-loaded, 128B-swizzled tile), B = weights (8 rows x 16 k),
// decoded to registers from the 12-bit stream (BF16 hi + lo for the router).  Each CTA owns R rows over the whole
// K: no cross-CTA reduction.  Warps split the CTA's rows (NWR x NT n8 tiles) and, inside every stage, K (WSK).
// A stage = UPS 64-k units: x tiles [UPS][TOKR rows x 128 B] (TMA, rows >= M zero-filled) + this CTA's weights.
template <int MT, int TOKR, int NT, int NWR, int WSK, int UPW, int NSTAGE, bool COMP, bool HL, int NACC, int CL,
          typename OutT, bool PK, bool FUSE>
#if SK2_NARROW
__global__ void __launch_bounds__((NWR * WSK + 1 + (COMP ? 4 : 1)) * 32, 1)
#else
__global__ void __launch_bounds__((NWR * WSK + 1 + (COMP ? sk2_nesc<NSTAGE, TOKR, NT, NWR, FUSE, CL>() : 1)) * 32, 1)
#endif
    // (the unnamed int and pointer are unused: they keep the parameter layout the kernels were tuned with)
    sk2_kernel(const __grid_constant__ CUtensorMap xmap, const uint8_t* __restrict__ Ws,
               const __nv_bfloat16* __restrict__ x, OutT* __restrict__ out, const int2* __restrict__ ptab, int pmax,
               uint32_t base4, int M, int N, int K, int, const void*, const uint8_t* __restrict__ Wsr, float* __restrict__ outr,
#if SK2_GLE
               float* __restrict__ part, int* __restrict__ rcnt, int pdl, const GleEpi ep) {
#else
               float* __restrict__ part, int* __restrict__ rcnt, int pdl) {
#endif
  constexpr int NW = NWR * WSK;
  constexpr int NC = NW * 32;
  constexpr int R = NWR * NT * 8;
  constexpr int UPS = WSK * UPW;
  constexpr int TU = COMP ? 768 : (HL ? 2048 : 1024);
  constexpr int WBY = NW * UPW * NT * TU;
  constexpr int XT = TOKR * 128;
  constexpr int XBY = UPS * XT;
  constexpr int SST = ((XBY + WBY) + 1023) & ~1023;
  // SK2_XR: weight slots [NSTAGE][WST], then x slots [XS][XBY]
  constexpr bool XRN = sk2_xr<TOKR, NT, NWR, FUSE, CL>();
  constexpr int XS = XRN ? SK2_XR : NSTAGE;
  constexpr int WST = (WBY + 1023) & ~1023;
  constexpr int RING = XRN ? NSTAGE * WST + XS * XBY : NSTAGE * SST;
  static_assert(!XRN || (XS <= NSTAGE && NSTAGE * WST + XS * XBY <= SK2_XRB), "sk2: x ring larger than the weight ring");
  // partial sums [warp][token][RP] (rows fastest, padded): 64-bit stores in, 128-bit loads out
  constexpr int TOKS = 16 * MT;
  constexpr int RP = R + ((40 - (R % 32)) % 32);   // >= R, = 8 (mod 32): conflict-free 64-bit stores
  constexpr int CP = R + 1;   // corr[token][CP]: odd stride, conflict-free per-token updates
  extern __shared__ __align__(1024) unsigned char smem_raw[];
  const uint32_t sraw = (uint32_t)__cvta_generic_to_shared(smem_raw);
  const uint32_t sbase = (sraw + 1023u) & ~1023u;
  unsigned char* gbase = smem_raw + (sbase - sraw);
  const uint32_t bars = sbase + RING;
  const uint32_t full0 = bars;
  const uint32_t empty0 = bars + 8 * NSTAGE;
  const uint32_t pbar = bars + 16 * NSTAGE;
  const uint32_t cbar = pbar + 8;
  const uint32_t qbar = pbar + 16;
  // corrections, one copy per escape warp: [NESC][TOKS tokens][CP]; escape warps take turns on the stages
#if SK2_NARROW
  constexpr int NESC = COMP ? 4 : 1;

#else
  constexpr int NESC = COMP ? sk2_nesc<NSTAGE, TOKR, NT, NWR, FUSE, CL>() : 1;
  static_assert(NESC <= NSTAGE, "sk2: escape warps need NESC <= NSTAGE");
#endif
  float* corr = reinterpret_cast<float*>(gbase + RING + 16 * NSTAGE + 32);
  constexpr int CSZ = (TOKS * CP + 3) & ~3;
  float* peer = corr + NESC * CSZ;   // [CL-1][WSK][TOKS][RP]: ranks > 0 per-warp sums
  int2* ptab_s = reinterpret_cast<int2*>(peer + (CL > 1 ? (CL - 1) * WSK * TOKS * RP : 0));   // [pmax]: escapes
  const uint32_t zrow = (uint32_t)__cvta_generic_to_shared(ptab_s + pmax);   // 128 B of zeros (x rows >= M)
  // m <= 8: the x box has exactly M rows per 64-k unit (packed, M*128 B per unit; no zero rows moved)
  // PK (M < TOKR, or TOKR == 8): packed x, exactly M rows per 64-k unit (no out-of-bounds TMA rows; rows >= M
  // read zrow); else the full TOKR-row tile with compile-time addressing
  const int Mb = PK ? M : TOKR;
  const uint32_t XU = (uint32_t)Mb * 128u;
  const uint32_t XBYr = UPS * XU;
  float* red = reinterpret_cast<float*>(gbase);   // reuses the ring once every stage is consumed

  const int tid = threadIdx.x;
  const int warp = SK2_ELECT ? __shfl_sync(0xffffffffu, tid >> 5, 0) : tid >> 5, lane = tid & 31;
  // CL = 2: a cluster of two CTAs shares a row block; each streams one half of K, rank 1 hands its sums to rank 0
  const int cta = blockIdx.x / CL;
  const int crank = CL > 1 ? (int)(blockIdx.x % CL) : 0;
  const int tst = K / (UPS * 64);   // stages over all of K
  const int gs0 = (crank * tst) / CL;   // this CTA's first stage (uneven split for CL = 3)
  const int nst = ((crank + 1) * tst) / CL - gs0;   // and its stage count
  // stage-major weight stream: [stage][row block][...] (at any moment all CTAs read one contiguous region)
  const uint8_t* wsrc = Ws + (size_t)cta * WBY;
  const size_t wstride = (size_t)(gridDim.x / CL) * WBY;
  // FUSE (pooler + router in one launch): after its pooler stages every CTA streams one router piece: 8 router rows
  // (tile rt = cta / 4) over one quarter of K (rq = cta % 4), from the router's own stage-major hi|lo stream
  // ([16 stages][32 tiles][8 warps][2048 B]); 4 more stages through the same ring
  constexpr bool R16 = SK2_GLE && FUSE && TOKR > 8;   // (GLE16: 256-column FP32 router stages for 16 rows)
  constexpr int NSTR = FUSE ? (R16 ? 8 : 4) : 0;
  constexpr int RCOL = R16 ? 256 : 512;   // FP32 router columns per stage
  constexpr int WBYR = NW * 2048;
  static_assert(!FUSE || (COMP && CL == 1 && UPW == 1 && UPS == 8 && WBYR <= WBY),
                "sk2 fused: pooler-like config only");
  const int rt = cta >> 2, rq = cta & 3;
  const uint8_t* wsr = FUSE ? Wsr + (size_t)rt * WBYR : nullptr;
  const int nstt = nst + NSTR;
  // PCX: pdl bits 0-3 = P (pre-wait stages), 4-8 = d0 and 9-13 = g (pacing, units of 448 cycles)
  const long long pcx_d0 = (long long)((pdl >> 4) & 31) * 448, pcx_g = (long long)((pdl >> 9) & 31) * 448;
  const int pcx_e = (pdl >> 14) & 7;
  // stages issued by the prologue: 1, or under PDL up to pdl (CL == 1 only; never past this CTA's own stages)
  const int P = (CL == 1 && pdl > 0) ? min(pdl & 15, min(NSTAGE, nst)) : 1;
#if SK2_GLE
  const bool ilv = FUSE && ep.fpr && ep.ilv && (nstt % NSTR) == 0 && nstt / NSTR > P;
  const int PER = ilv ? nstt / NSTR : (1 << 30);
  const int OFF = ilv ? (ep.ilv == 2 ? P : PER - 1) : 0;   // router stage slot inside each period
#endif

  if (tid == 0) {
    // stage 0 first of all (the first bytes' latency is the ramp of the whole kernel): the weight copy needs only
    // its barrier; the tensor map prefetch goes out before it so the x box follows right behind. In a real cluster
    // (CL > 1) a cluster-scope release issued AFTER a copy is in flight waits for it (~0.7 us): there every barrier
    // is initialized and fenced before the first copy.
    asm volatile("prefetch.tensormap [%0];" ::"l"(&xmap) : "memory");
    mbar_init(full0, 1);
    if constexpr (CL == 1)   // PDL: the first P stages' weights go out before the wait; their barriers first
      for (int i = 1; i < P; ++i) mbar_init(full0 + 8 * i, 1);
    if constexpr (CL > 1) {
#if SK2_NARROW || SK2_GLE
#if SK2_GLE
      mbar_init(empty0, NW + ((COMP && (M <= 2)) ? NESC : 0));
#endif
#if SK2_NARROW
      mbar_init(empty0, NW);
#endif
#endif
#if !SK2_NARROW && !SK2_GLE
      mbar_init(empty0, NW + ((COMP && (M > 8)) ? NESC : 0));
#endif
      for (int i = 1; i < NSTAGE; ++i) {
        mbar_init(full0 + 8 * i, 1);
#if SK2_NARROW || SK2_GLE
#if SK2_GLE
        mbar_init(empty0 + 8 * i, NW + ((COMP && (M <= 2)) ? NESC : 0));
#endif
#if SK2_NARROW
        mbar_init(empty0 + 8 * i, NW);
#endif
#endif
#if !SK2_NARROW && !SK2_GLE
        mbar_init(empty0 + 8 * i, NW + ((COMP && (M > 8)) ? NESC : 0));
#endif
      }
      mbar_init(pbar, 1);
      mbar_init(cbar, 32 * NESC);
      mbar_init(qbar, 1);
      if (crank == 0) mbar_expect_tx(qbar, SK2_BF16P ? (CL - 1) * MT * NT * (NC / WSK) * (16 + 8 * (WSK - 1))
                                                     : (CL - 1) * NC * MT * NT * 16);
    }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    // weights never depend on the previous kernel: under PDL (pdl > 0) they stream while it finishes; x (its output)
    // only after griddepcontrol.wait (a no-op when the launch has no programmatic dependency)
    for (int st = 0; st < P; ++st) {
      mbar_expect_tx(full0 + 8 * st, (XBYr) + WBY);
      bulk_g2s(XRN ? sbase + st * WST : sbase + st * SST + XBY, wsrc + (size_t)(gs0 + st) * wstride, WBY, full0 + 8 * st);
    }
    if constexpr (CL == 1) {
      // PCX hoist: every barrier init and the escape-table copy go out BEFORE the wait (they need nothing from the
      // previous kernel); after the wait thread 0 only issues the x tiles, the producer lane the refills
      if (COMP) {
        mbar_init(pbar, 1);
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        mbar_expect_tx(pbar, pmax * 8);
        bulk_g2s((uint32_t)__cvta_generic_to_shared(ptab_s), ptab + (size_t)cta * pmax, pmax * 8, pbar);
      }
#if SK2_NARROW || SK2_GLE
#if SK2_GLE
      mbar_init(empty0, NW + ((COMP && (M <= 2)) ? NESC : 0));
#endif
#if SK2_NARROW
      mbar_init(empty0, NW);
#endif
#endif
#if !SK2_NARROW && !SK2_GLE
      mbar_init(empty0, NW + ((COMP && (M > 8)) ? NESC : 0));
#endif
      for (int i = 1; i < NSTAGE; ++i) {
        if (i >= P) mbar_init(full0 + 8 * i, 1);
#if SK2_NARROW || SK2_GLE
#if SK2_GLE
        mbar_init(empty0 + 8 * i, NW + ((COMP && (M <= 2)) ? NESC : 0));
#endif
#if SK2_NARROW
        mbar_init(empty0 + 8 * i, NW);
#endif
#endif
#if !SK2_NARROW && !SK2_GLE
        mbar_init(empty0 + 8 * i, NW + ((COMP && (M > 8)) ? NESC : 0));
#endif
      }
      if (!COMP) mbar_init(pbar, 1);
      mbar_init(cbar, 32 * NESC);
      asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    } else {
      if (pdl) asm volatile("griddepcontrol.wait;" ::: "memory");
      for (int st = 0; st < P; ++st) tma_load_3d(sbase + st * SST, &xmap, full0 + 8 * st, 0, 0, (gs0 + st) * UPS);
      if (COMP) {
        mbar_expect_tx(pbar, pmax * 8);
        bulk_g2s((uint32_t)__cvta_generic_to_shared(ptab_s), ptab + (size_t)cta * pmax, pmax * 8, pbar);
      }
    }
  }
  if (PK && M < TOKR && tid >= 32 && tid < 64) reinterpret_cast<uint32_t*>(ptab_s + pmax)[tid - 32] = 0u;
  // local barriers are all this CTA needs to start streaming; the cluster barrier (rank 0's qbar initialized before
  // ranks > 0 send into it) is only arrived at here and waited on right before the remote stores
#if SK2_GLE
  __shared__ __align__(16) float4 gx_s[64], gx_a[64], gx_b[64], gx_a2[64];
  __shared__ __align__(16) uint2 gx_p[64];
  // the epilogue's input-side vectors (residual stream row, alpha / beta / alpha2 / pre) go to shared memory by
  // cp.async before the wait and are awaited only in the epilogue: no registers held across the main loop (2 CTAs / SM
  // next to attention) and no stall behind the pre-wait weight stages
  // ep.s is data['stream'], the layer's input residual stream: nothing in the call writes it, which is what makes this
  // pre-wait copy safe (keep it that way: any in-call writer of data['stream'] must move this copy after the wait)
  if (!FUSE && tid < 8 * M) {   // o only (this source also holds the fused pooler)
    const size_t col = (size_t)cta * R + (tid & 7) * 4;
    const uint32_t d0 = (uint32_t)__cvta_generic_to_shared(&gx_s[tid]);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(d0), "l"(ep.s + (size_t)(tid >> 3) * N + col)
                 : "memory");
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"((uint32_t)__cvta_generic_to_shared(&gx_a[tid])),
                 "l"(ep.a + col)
                 : "memory");
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"((uint32_t)__cvta_generic_to_shared(&gx_b[tid])),
                 "l"(ep.b + col)
                 : "memory");
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"((uint32_t)__cvta_generic_to_shared(&gx_a2[tid])),
                 "l"(ep.a2 + col)
                 : "memory");
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8;" ::"r"((uint32_t)__cvta_generic_to_shared(&gx_p[tid])),
                 "l"(ep.p + col)
                 : "memory");
    asm volatile("cp.async.commit_group;" ::: "memory");
  }
#endif
  __syncthreads();
  if (pdl) asm volatile("griddepcontrol.wait;" ::: "memory");   // every thread: x and the outputs are ours from here
  if constexpr (CL == 1) {   // PCX hoist: the x tiles of the pre-issued stages, right after the wait
    if (tid == 0)
      for (int st = 0; st < (XRN ? min(P, XS) : P); ++st)
        tma_load_3d(XRN ? sbase + NSTAGE * WST + st * XBY : sbase + st * SST, &xmap, full0 + 8 * st, 0, 0,
                    (gs0 + st) * UPS);
  }
  if constexpr (CL > 1) asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");   // inits fenced above

  if (warp == NW) {
    // producer warp: lane 0 refills the ring (SK2_ELECT: the whole warp, one elected lane issuing)
    if (XRN && (SK2_ELECT || lane == 0)) {
      // SK2_XR: iteration st issues stage st's weights and stage j = st - LAG's x. Weight slot st % NSTAGE and x slot
      // j % XS both last held stage st - NSTAGE, so its empty barrier frees the two. Stage j's x completes its full
      // barrier, armed with the weights (x bytes included) at iteration j; stages < min(P, XS) went out after the wait
      constexpr int LAG = NSTAGE - XS;
      const int px = min(P, XS);
      for (int st = P; st < nstt + LAG; ++st) {
        const int slot = st % NSTAGE;
        if (st >= NSTAGE) {
          if (SK2_ELECT)
            mbar_wait_spin(empty0 + 8 * slot, ((st / NSTAGE) - 1) & 1);
          else
            SK2_WAIT(empty0 + 8 * slot, ((st / NSTAGE) - 1) & 1);
        }
        if (SK2_ELECT && !sk2_elect()) continue;
        if (st < nstt) {
          const bool rs = FUSE && st >= nst;
          mbar_expect_tx(full0 + 8 * slot, XBYr + (rs ? WBYR : WBY));
          if (rs)
            bulk_g2s(sbase + slot * WST, wsr + (size_t)(rq * NSTR + (st - nst)) * (32 * WBYR), WBYR, full0 + 8 * slot);
          else
            bulk_g2s(sbase + slot * WST, wsrc + (size_t)(gs0 + st) * wstride, WBY, full0 + 8 * slot);
          if (pdl && st == nstt - 1) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");   // last weight stage
        }
        const int j = st - LAG;
        if (j >= px)
          tma_load_3d(sbase + NSTAGE * WST + (j % XS) * XBY, &xmap, full0 + 8 * (j % NSTAGE), 0, 0,
                      (FUSE && j >= nst) ? (rq * NSTR + (j - nst)) * UPS : (gs0 + j) * UPS);
      }
    } else if (SK2_ELECT || lane == 0) {
      for (int st = P; st < nstt; ++st) {
        const int slot = st % NSTAGE;
        if (st >= NSTAGE) {
          if (SK2_ELECT)
            mbar_wait_spin(empty0 + 8 * slot, ((st / NSTAGE) - 1) & 1);
          else
            SK2_WAIT(empty0 + 8 * slot, ((st / NSTAGE) - 1) & 1);
        }
        if (SK2_ELECT && !sk2_elect()) continue;

        const uint32_t s0 = sbase + slot * SST;
#if SK2_GLE
        const bool isr = ilv ? (st % PER == OFF) : (FUSE && st >= nst);
        const int ps = ilv ? st - (st / PER + (st % PER > OFF ? 1 : 0)) : st;
        if (!isr) {
#else
        if (!FUSE || st < nst) {
#endif
          mbar_expect_tx(full0 + 8 * slot, (XBYr) + WBY);
#if SK2_GLE
          bulk_g2s(s0 + XBY, wsrc + (size_t)(gs0 + ps) * wstride, WBY, full0 + 8 * slot);
          tma_load_3d(s0, &xmap, full0 + 8 * slot, 0, 0, (gs0 + ps) * UPS);
        } else if (ep.fpr) {   // FP32 router stage j = st - nst: columns rq * 2048 + RCOL j .. + RCOL - 1
          const int kb = rq * 2048 + (ilv ? st / PER : st - nst) * RCOL;
          mbar_expect_tx(full0 + 8 * slot, 8 * RCOL * 4 + M * RCOL * 4);
          for (int r = 0; r < 8; ++r)
            bulk_g2s(s0 + r * RCOL * 4, reinterpret_cast<const uint8_t*>(ep.w32 + (size_t)(rt * 8 + r) * 8192 + kb),
                     RCOL * 4, full0 + 8 * slot);
          for (int t = 0; t < M; ++t)
            bulk_g2s(s0 + 8 * RCOL * 4 + t * RCOL * 4, reinterpret_cast<const uint8_t*>(ep.x32 + (size_t)t * 8192 + kb),
                     RCOL * 4, full0 + 8 * slot);
#else
          bulk_g2s(s0 + XBY, wsrc + (size_t)(gs0 + st) * wstride, WBY, full0 + 8 * slot);
          tma_load_3d(s0, &xmap, full0 + 8 * slot, 0, 0, (gs0 + st) * UPS);
#endif
        } else {   // router piece: global router stage rq * 4 + j of tile rt
          const int gsr = rq * NSTR + (st - nst);
          mbar_expect_tx(full0 + 8 * slot, XBYr + WBYR);
          bulk_g2s(s0 + XBY, wsr + (size_t)gsr * (32 * WBYR), WBYR, full0 + 8 * slot);
          tma_load_3d(s0, &xmap, full0 + 8 * slot, 0, 0, gsr * UPS);
        }
      }
      if (pdl && (!SK2_ELECT || lane == 0)) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");   // last weight stage issued
    }
    __syncwarp();
    if constexpr (CL > 1) asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
    return;
  }
  if (warp >= (NW + 1)) {
    // NESC escape warps take turns on the stages (each has NESC stage-times per stage): exact corrections for the
    // escapes, read from the x tiles of the ring as each stage lands (the table is in k order); lane t owns token t,
    // so corr[t][row] accumulates with no atomics; each warp has its own copy
    const int ew = warp - (NW + 1);
    if (COMP) {
      float* cr = corr + ew * CSZ;
      for (int i = lane; i < (CL == 1 ? M : TOKS) * CP; i += 32) cr[i] = 0.f;
      mbar_wait_sleep(pbar, 0);
      const int cnt = ptab_s[0].x;
      constexpr int KST = UPS * 64;
#if SK2_NARROW || SK2_GLE
#if SK2_GLE
      if (!(M <= 2)) {
#endif
#if SK2_NARROW

      for (int pe = 1 + ew; pe <= cnt; pe += NESC) {
        const int2 e = ptab_s[pe];
        const int kk = e.x & 65535;
        if (lane < M && kk >= gs0 * KST && kk < (gs0 + nst) * KST)
          cr[lane * CP + (e.x >> 16)] += __int_as_float(e.y) * __bfloat162float(__ldcg(x + (size_t)lane * K + kk));
#endif
#endif
#if !SK2_NARROW && !SK2_GLE
      if (!(M > 8)) {
#endif
#if !SK2_NARROW
        for (int pe = 1 + ew; pe <= cnt; pe += NESC) {
          const int2 e = ptab_s[pe];
          const int kk = e.x & 65535;
#endif
#if SK2_GLE
          if (lane < M && kk >= gs0 * KST && kk < (gs0 + nst) * KST)
#endif
#if !SK2_NARROW && !SK2_GLE
          if (lane < M && (CL == 1 || kk >= (gs0 * KST) && kk < ((gs0 + nst) * KST)))
#endif
#if !SK2_NARROW
            cr[lane * CP + (e.x >> 16)] += __int_as_float(e.y) * __bfloat162float(__ldcg(x + (size_t)lane * K + kk));
        }
      } else {
        int pe = 1;
        for (int st = 0; st < nstt; ++st) {
          const int slot = st % NSTAGE;
#endif
#if SK2_GLE
          if (ilv ? (st % PER == OFF)
                  : (FUSE &&
                     st >= nst)) {   // router stages: nothing to correct, only the slot hand-back (in order, as below)
#endif
#if !SK2_NARROW && !SK2_GLE
          if (FUSE && st >= nst) {   // router stages: nothing to correct, only the slot hand-back (in order, as below)
#endif
#if !SK2_NARROW
            SK2_WAIT(full0 + 8 * slot, (st / NSTAGE) & 1);
            __syncwarp();
            if (lane == 0) mbar_arrive(empty0 + 8 * slot);   // every escape warp hands every stage back
            continue;
          }
#endif
#if SK2_GLE
          const int psq = ilv ? st - (st / PER + (st % PER > OFF ? 1 : 0)) : st;
          const int kbeg = (gs0 + psq) * KST;
#endif
#if !SK2_NARROW && !SK2_GLE
          const int kbeg = (gs0 + st) * KST;
#endif
#if !SK2_NARROW
          const int kend = kbeg + KST;
          // Every escape warp observes every stage, in order, works only on its own (st % NESC == ew), and hands EVERY
          // stage back (the empty barriers count NW + NESC arrivals). The producer therefore never refills a slot
          // before every escape warp has seen the stage it held, so no warp can find its slot two phases ahead (a
          // parity wait that aliases and wedges the ring: reproduced with a delayed escape warp).
          SK2_WAIT(full0 + 8 * slot, (st / NSTAGE) & 1);
#endif
#if SK2_GLE
          if ((psq % NESC) != ew) {   // another warp's stage: skip its escapes, hand the slot back
#endif
#if !SK2_NARROW && !SK2_GLE
          if (!SK2_ESPL && (st % NESC) != ew) {   // another warp's stage: skip its escapes, hand the slot back
#endif
#if !SK2_NARROW
            while (pe <= cnt && (ptab_s[pe].x & 0xFFFF) < kend) ++pe;
            __syncwarp();
            if (lane == 0) mbar_arrive(empty0 + 8 * slot);
            continue;
          }
          const unsigned char* xs = XRN ? gbase + NSTAGE * WST + (st % XS) * XBY : gbase + slot * SST;
          while (pe <= cnt) {
            const int2 e = ptab_s[pe];
            const int kk = e.x & 0xFFFF;
            if (kk >= kend) break;
            if (lane < M && kk >= kbeg && (!SK2_ESPL || SK2_GLE || (pe % NESC) == ew)) {
              const int kl = kk - kbeg;
              const int u = kl >> 6, kc = kl & 63;
              const __nv_bfloat16 v = *reinterpret_cast<const __nv_bfloat16*>(
                  xs + u * XU + lane * 128 + (((kc >> 3) ^ ((u * Mb + lane) & 7)) << 4) + (kc & 7) * 2);
              cr[lane * CP + (e.x >> 16)] += __int_as_float(e.y) * __bfloat162float(v);
            }
            ++pe;
          }
          __syncwarp();
          if (lane == 0) mbar_arrive(empty0 + 8 * slot);
        }
#endif
      }
    }
    mbar_arrive(cbar);
    __syncwarp();
    if constexpr (CL > 1) asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
    return;
  }

  // ---------------- consumers
  const int wr = warp / WSK;
  const int wk = warp % WSK;
  const int g = lane >> 2, t4 = lane & 3;
  const int q = lane >> 3;
  const int chi = q >> 1;
  int trow[MT];
#pragma unroll
  for (int mt = 0; mt < MT; ++mt) trow[mt] = TOKR >= 16 ? mt * 16 + (lane & 7) + 8 * (q & 1) : (lane & 7);
  float acc[NACC][MT][NT][4];
#pragma unroll
  for (int c = 0; c < NACC; ++c)
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
      for (int nt = 0; nt < NT; ++nt)
        acc[c][mt][nt][0] = acc[c][mt][nt][1] = acc[c][mt][nt][2] = acc[c][mt][nt][3] = 0.f;

#if SK2_GLE
  constexpr int RT = R16 ? 16 : 8;   // (FP32 router tokens per thread)
  float afi[RT];   // ilv: FP32 router accumulators (warp = router row)
#pragma unroll
  for (int t = 0; t < RT; ++t) afi[t] = 0.f;
  for (int st = 0; st < (ilv ? nstt : nst); ++st) {
#else
  for (int st = 0; st < nst; ++st) {
#endif
    const int slot = st % NSTAGE;
    mbar_wait(full0 + 8 * slot, (st / NSTAGE) & 1);
#if SK2_GLE
    if constexpr (FUSE)
      if (ilv && st % PER == OFF) {   // an interleaved FP32 router stage
        const float* ws = reinterpret_cast<const float*>(gbase + slot * SST) + warp * RCOL;
        const float* xs = reinterpret_cast<const float*>(gbase + slot * SST + 8 * RCOL * 4);
#pragma unroll
        for (int i = 0; i < RCOL / 128; ++i) {
          const int k = lane * 4 + i * 128;
          const float4 bw = *reinterpret_cast<const float4*>(ws + k);
#pragma unroll
          for (int t = 0; t < RT; ++t)
            if (t < M) {
              const float4 a = *reinterpret_cast<const float4*>(xs + t * RCOL + k);
              afi[t] = fmaf(a.x, bw.x, afi[t]);
              afi[t] = fmaf(a.y, bw.y, afi[t]);
              afi[t] = fmaf(a.z, bw.z, afi[t]);
              afi[t] = fmaf(a.w, bw.w, afi[t]);
            }
        }
        __syncwarp();
        if (lane == 0) mbar_arrive(empty0 + 8 * slot);
        continue;
      }
#endif

    const uint32_t s0 = sbase + slot * SST;
    const uint32_t sx = XRN ? sbase + NSTAGE * WST + (st % XS) * XBY : s0;
    const uint32_t sw = XRN ? sbase + slot * WST : s0 + XBY;

#pragma unroll
    for (int uu = 0; uu < UPW; ++uu) {
      const int ui = wk * UPW + uu;
      const uint32_t xt = sx + ui * XU;
      const uint32_t wb = sw + (warp * UPW + uu) * NT * TU;
      // 1) every shared-memory load of this unit first (weights of all tiles, x fragments of all 4 k-steps)
      uint4 lw[NT];
      uint2 cw[NT];
      uint32_t b[NT][8];
      uint32_t bl[HL ? NT : 1][8];
#pragma unroll
      for (int nt = 0; nt < NT; ++nt) {
        if constexpr (COMP) {
          asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];"
                       : "=r"(lw[nt].x), "=r"(lw[nt].y), "=r"(lw[nt].z), "=r"(lw[nt].w)
                       : "r"(wb + nt * TU + lane * 16));
          asm volatile("ld.shared.v2.u32 {%0,%1}, [%2];"
                       : "=r"(cw[nt].x), "=r"(cw[nt].y)
                       : "r"(wb + nt * TU + 512 + lane * 8));
        } else {
          const uint32_t a0 = wb + nt * TU + lane * 32;
          asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];"
                       : "=r"(b[nt][0]), "=r"(b[nt][1]), "=r"(b[nt][2]), "=r"(b[nt][3])
                       : "r"(a0));
          asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];"
                       : "=r"(b[nt][4]), "=r"(b[nt][5]), "=r"(b[nt][6]), "=r"(b[nt][7])
                       : "r"(a0 + 16));
          if constexpr (HL) {
            asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];"
                         : "=r"(bl[nt][0]), "=r"(bl[nt][1]), "=r"(bl[nt][2]), "=r"(bl[nt][3])
                         : "r"(a0 + 1024));
            asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];"
                         : "=r"(bl[nt][4]), "=r"(bl[nt][5]), "=r"(bl[nt][6]), "=r"(bl[nt][7])
                         : "r"(a0 + 1040));
          }
        }
      }
      uint32_t a[4][MT][4];
      {
#pragma unroll
        for (int j = 0; j < 4; ++j)
#pragma unroll
          for (int mt = 0; mt < MT; ++mt)
            ldsm_x4((PK && trow[mt] >= M) ? zrow
                                          : xt + trow[mt] * 128 + (((2 * j + chi) ^ ((ui * Mb + trow[mt]) & 7)) << 4),
                    a[j][mt][0], a[j][mt][1], a[j][mt][2], a[j][mt][3]);
      }
      // the stage's shared memory is in registers now: hand the slot back before decoding and multiplying
      if (UPW == 1 && uu == 0) {
        __syncwarp();
        if (lane == 0) mbar_arrive(empty0 + 8 * slot);
      }
      // 2) decode
      if constexpr (COMP) {
#pragma unroll
        for (int nt = 0; nt < NT; ++nt) {
          decode8(lw[nt].x, lw[nt].y, cw[nt].x, base4, &b[nt][0]);
          decode8(lw[nt].z, lw[nt].w, cw[nt].y, base4, &b[nt][4]);
        }
      }

#pragma unroll
      for (int j = 0; j < 4; ++j)
#pragma unroll
        for (int mt = 0; mt < MT; ++mt)
#pragma unroll
          for (int nt = 0; nt < NT; ++nt) {
            constexpr int HM = HL ? 2 : 1;
            mma_bf16(acc[((uu * 4 + j) * HM) % NACC][mt][nt], a[j][mt][0], a[j][mt][1], a[j][mt][2], a[j][mt][3],
                     b[nt][2 * j], b[nt][2 * j + 1]);
            if constexpr (HL)
              mma_bf16(acc[((uu * 4 + j) * HM + 1) % NACC][mt][nt], a[j][mt][0], a[j][mt][1], a[j][mt][2], a[j][mt][3],
                       bl[nt][2 * j], bl[nt][2 * j + 1]);
          }
    }
    if (UPW > 1) {
      __syncwarp();
      if (lane == 0) mbar_arrive(empty0 + 8 * slot);
    }
  }
  if (pdl) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");   // the next kernel may start its prologue
  if constexpr (FUSE) {
#if SK2_GLE
    if (ilv) {   // interleaved FP32 router: warp butterflies of afi, then the 4-quarter reduction below
#else
    // router piece: 8 rows (one tile) over 4 stages of 8 units; hi and lo BF16 tiles into two FP32 accumulators
    float accr[2][MT][4];
#endif
#pragma unroll
#if SK2_GLE
      for (int t = 0; t < RT; ++t)
#else
    for (int h = 0; h < 2; ++h)
#endif
#pragma unroll
#if SK2_GLE
        for (int d = 16; d; d >>= 1) afi[t] += __shfl_xor_sync(0xffffffffu, afi[t], d);
      float mine = afi[0];
#else
      for (int mt = 0; mt < MT; ++mt) accr[h][mt][0] = accr[h][mt][1] = accr[h][mt][2] = accr[h][mt][3] = 0.f;
    for (int st = nst; st < nstt; ++st) {
      const int slot = st % NSTAGE;
      mbar_wait(full0 + 8 * slot, (st / NSTAGE) & 1);
      const uint32_t s0 = sbase + slot * SST;
      const int ui = wk;
      const uint32_t xt = (XRN ? sbase + NSTAGE * WST + (st % XS) * XBY : s0) + ui * XU;
      const uint32_t a0 = (XRN ? sbase + slot * WST : s0 + XBY) + warp * 2048 + lane * 32;
      uint32_t bh[8], bo[8];
      asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];"
                   : "=r"(bh[0]), "=r"(bh[1]), "=r"(bh[2]), "=r"(bh[3])
                   : "r"(a0));
      asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];"
                   : "=r"(bh[4]), "=r"(bh[5]), "=r"(bh[6]), "=r"(bh[7])
                   : "r"(a0 + 16));
      asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];"
                   : "=r"(bo[0]), "=r"(bo[1]), "=r"(bo[2]), "=r"(bo[3])
                   : "r"(a0 + 1024));
      asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];"
                   : "=r"(bo[4]), "=r"(bo[5]), "=r"(bo[6]), "=r"(bo[7])
                   : "r"(a0 + 1040));
      uint32_t a[4][MT][4];
#endif
#pragma unroll
#if SK2_GLE
      for (int t = 1; t < RT; ++t) mine = lane == t ? afi[t] : mine;
      if (lane < M) __stcg(part + ((size_t)rt * 4 + rq) * 32 * 8 + lane * 8 + warp, mine);
#if !SK2_RSUM
      if constexpr (COMP) mbar_wait(cbar, 0);
      named_bar(1, NC);
#endif
    } else if (ep.fpr) {
      static_assert(!FUSE || (NW == 8 && SST >= 32768), "fp32 router stages: 8 consumer warps, 32 KB slots");
      // thread: column group cg = tid & 127 (4 of the stage's 512 columns) x all 8 router rows x tokens 4h .. 4h + 3
      // (h = tid >> 7): every shared-memory float4 is read by 2 threads, not 8 warps; FP32 FMAs
      const int cg = tid & 127;
      const int th = tid >> 7;
      float af[32];
#else
      for (int j = 0; j < 4; ++j)
#endif
#pragma unroll
#if SK2_GLE
      for (int i = 0; i < 32; ++i) af[i] = 0.f;
      for (int st = nst; st < nstt; ++st) {
        const int slot = st % NSTAGE;
        mbar_wait(full0 + 8 * slot, (st / NSTAGE) & 1);
        const float* ws = reinterpret_cast<const float*>(gbase + slot * SST) + cg * 4;
        const float* xs = reinterpret_cast<const float*>(gbase + slot * SST + 16384) + cg * 4;
        float4 xa[4];
#else
        for (int mt = 0; mt < MT; ++mt)
          ldsm_x4(
              (PK && trow[mt] >= M) ? zrow : xt + trow[mt] * 128 + (((2 * j + chi) ^ ((ui * Mb + trow[mt]) & 7)) << 4),
              a[j][mt][0], a[j][mt][1], a[j][mt][2], a[j][mt][3]);
      __syncwarp();
      if (lane == 0) mbar_arrive(empty0 + 8 * slot);
#endif
#pragma unroll
#if SK2_GLE
        for (int u = 0; u < 4; ++u)
          xa[u] = (th * 4 + u < M) ? *reinterpret_cast<const float4*>(xs + (th * 4 + u) * 512)
                                   : make_float4(0.f, 0.f, 0.f, 0.f);
#else
      for (int j = 0; j < 4; ++j)
#endif
#pragma unroll
#if SK2_GLE
        for (int r = 0; r < 8; ++r) {
          const float4 bw = *reinterpret_cast<const float4*>(ws + r * 512);
#pragma unroll
          for (int u = 0; u < 4; ++u) {
            float v = af[r * 4 + u];
            v = fmaf(xa[u].x, bw.x, v);
            v = fmaf(xa[u].y, bw.y, v);
            v = fmaf(xa[u].z, bw.z, v);
            v = fmaf(xa[u].w, bw.w, v);
            af[r * 4 + u] = v;
          }
#else
        for (int mt = 0; mt < MT; ++mt) {
          mma_bf16(accr[0][mt], a[j][mt][0], a[j][mt][1], a[j][mt][2], a[j][mt][3], bh[2 * j], bh[2 * j + 1]);
          mma_bf16(accr[1][mt], a[j][mt][0], a[j][mt][1], a[j][mt][2], a[j][mt][3], bo[2 * j], bo[2 * j + 1]);
#endif
        }
#if SK2_GLE
        __syncwarp();
        if (lane == 0) mbar_arrive(empty0 + 8 * slot);
      }
      // reduction over the 128 column groups through the (now idle) ring: red[th][cg][33] (static indices only)
      if constexpr (COMP) mbar_wait(cbar, 0);   // as the BF16 router: the escape warps are done with the ring
      named_bar(1, NC);   // and every consumer is done reading its last stage
      float* red32 = reinterpret_cast<float*>(gbase);
#pragma unroll
      for (int i = 0; i < 32; ++i) red32[(th * 128 + cg) * 33 + i] = af[i];
      named_bar(1, NC);
      float* red2 = red32 + 256 * 33;
      {
        const int o = tid & 63;
        const int pq = tid >> 6, h = o >> 5, e = o & 31;
        float v = 0.f;
        for (int c = pq * 32; c < pq * 32 + 32; ++c) v += red32[(h * 128 + c) * 33 + e];
        red2[pq * 64 + o] = v;
      }
      named_bar(1, NC);
      if (tid < 64) {
        const int h = tid >> 5, e = tid & 31, r = e >> 2, t = h * 4 + (e & 3);
        const float v = red2[tid] + red2[64 + tid] + red2[128 + tid] + red2[192 + tid];
        if (t < M) __stcg(part + ((size_t)rt * 4 + rq) * 32 * 8 + t * 8 + r, v);
      }
      named_bar(1, NC);
    } else {
      // router piece: 8 rows (one tile) over 4 stages of 8 units; hi and lo BF16 tiles into two FP32 accumulators
      float accr[2][MT][4];
#pragma unroll
      for (int h = 0; h < 2; ++h)
#pragma unroll
        for (int mt = 0; mt < MT; ++mt) accr[h][mt][0] = accr[h][mt][1] = accr[h][mt][2] = accr[h][mt][3] = 0.f;
      for (int st = nst; st < nstt; ++st) {
        const int slot = st % NSTAGE;
        mbar_wait(full0 + 8 * slot, (st / NSTAGE) & 1);
        const uint32_t s0 = sbase + slot * SST;
        const int ui = wk;
        const uint32_t xt = s0 + ui * XU;
        const uint32_t a0 = s0 + XBY + warp * 2048 + lane * 32;
        uint32_t bh[8], bo[8];
        asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(bh[0]), "=r"(bh[1]), "=r"(bh[2]), "=r"(bh[3])
                     : "r"(a0));
        asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(bh[4]), "=r"(bh[5]), "=r"(bh[6]), "=r"(bh[7])
                     : "r"(a0 + 16));
        asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(bo[0]), "=r"(bo[1]), "=r"(bo[2]), "=r"(bo[3])
                     : "r"(a0 + 1024));
        asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(bo[4]), "=r"(bo[5]), "=r"(bo[6]), "=r"(bo[7])
                     : "r"(a0 + 1040));
        uint32_t a[4][MT][4];
#pragma unroll
        for (int j = 0; j < 4; ++j)
#pragma unroll
          for (int mt = 0; mt < MT; ++mt)
            ldsm_x4((PK && trow[mt] >= M) ? zrow
                                          : xt + trow[mt] * 128 + (((2 * j + chi) ^ ((ui * Mb + trow[mt]) & 7)) << 4),
                    a[j][mt][0], a[j][mt][1], a[j][mt][2], a[j][mt][3]);
        __syncwarp();
        if (lane == 0) mbar_arrive(empty0 + 8 * slot);
#pragma unroll
        for (int j = 0; j < 4; ++j)
#pragma unroll
          for (int mt = 0; mt < MT; ++mt) {
            mma_bf16(accr[0][mt], a[j][mt][0], a[j][mt][1], a[j][mt][2], a[j][mt][3], bh[2 * j], bh[2 * j + 1]);
            mma_bf16(accr[1][mt], a[j][mt][0], a[j][mt][1], a[j][mt][2], a[j][mt][3], bo[2 * j], bo[2 * j + 1]);
          }
      }
      // k-warp reduction of the 8 x TOKS piece through the (now idle) ring: redr[wk][tok][8]
      if constexpr (COMP) mbar_wait(cbar, 0);   // the escape warps are done with the ring too
      named_bar(1, NC);
      float* redr = reinterpret_cast<float*>(gbase);
#pragma unroll
      for (int mt = 0; mt < MT; ++mt) {
        const int row = 2 * t4;
        const int tok = mt * 16 + g;
        redr[(wk * TOKS + tok) * 8 + row] = accr[0][mt][0] + accr[1][mt][0];
        redr[(wk * TOKS + tok) * 8 + row + 1] = accr[0][mt][1] + accr[1][mt][1];
        redr[(wk * TOKS + tok + 8) * 8 + row] = accr[0][mt][2] + accr[1][mt][2];
        redr[(wk * TOKS + tok + 8) * 8 + row + 1] = accr[0][mt][3] + accr[1][mt][3];
      }
      named_bar(1, NC);
      // this CTA's quarter-K partial -> part[rt][rq][TOKS][8]; the last of the 4 quarters (release/acquire counter)
      // adds them in quarter order (deterministic) and writes the router output
      float* pp = part + ((size_t)rt * 4 + rq) * 32 * 8;
      for (int i = tid; i < TOKS * 8; i += NC) {
        float v = 0.f;
#pragma unroll
        for (int w = 0; w < WSK; ++w) v += redr[(w * TOKS + (i >> 3)) * 8 + (i & 7)];
        __stcg(pp + i, v);
      }
      named_bar(1, NC);
#endif
    }
#if !SK2_GLE
    // k-warp reduction of the 8 x TOKS piece through the (now idle) ring: redr[wk][tok][8] (SK2_RRED: its own region
    // past the escape table, so it need not wait for the escape warps, which read x tiles from the ring)
    if constexpr (COMP && !SK2_RRED) mbar_wait(cbar, 0);   // the escape warps are done with the ring too
    named_bar(1, NC);
    float* redr = SK2_RRED ? reinterpret_cast<float*>(reinterpret_cast<unsigned char*>(ptab_s + pmax) + 128)
                           : reinterpret_cast<float*>(gbase);
#pragma unroll
    for (int mt = 0; mt < MT; ++mt) {
      const int row = 2 * t4;
      const int tok = mt * 16 + g;
      redr[(wk * TOKS + tok) * 8 + row] = accr[0][mt][0] + accr[1][mt][0];
      redr[(wk * TOKS + tok) * 8 + row + 1] = accr[0][mt][1] + accr[1][mt][1];
      redr[(wk * TOKS + tok + 8) * 8 + row] = accr[0][mt][2] + accr[1][mt][2];
      redr[(wk * TOKS + tok + 8) * 8 + row + 1] = accr[0][mt][3] + accr[1][mt][3];
    }
    named_bar(1, NC);
    // this CTA's quarter-K partial -> part[rt][rq][TOKS][8]; the last of the 4 quarters (release/acquire counter)
    // adds them in quarter order (deterministic) and writes the router output
    float* pp = part + ((size_t)rt * 4 + rq) * 32 * 8;
    for (int i = tid; i < TOKS * 8; i += NC) {
      float v = 0.f;
#pragma unroll
      for (int w = 0; w < WSK; ++w) v += redr[(w * TOKS + (i >> 3)) * 8 + (i & 7)];
      __stcg(pp + i, v);
    }
    named_bar(1, NC);
#endif
#if !SK2_RSUM
    __shared__ int s_rlast;
    if (tid == 0) {
      int old;
      asm volatile("atom.add.release.gpu.global.s32 %0, [%1], 1;" : "=r"(old) : "l"(rcnt + rt) : "memory");
      s_rlast = (old & 3) == 3;
      if (s_rlast) asm volatile("fence.acq_rel.gpu;" ::: "memory");
    }
    named_bar(1, NC);
    if (s_rlast) {
      const float* pr = part + (size_t)rt * 4 * 32 * 8;
      for (int i = tid; i < M * 8; i += NC) {
        const float v = __ldcg(pr + i) + __ldcg(pr + 256 + i) + __ldcg(pr + 512 + i) + __ldcg(pr + 768 + i);
        outr[(size_t)(i >> 3) * 256 + rt * 8 + (i & 7)] = v;
      }
    }
    named_bar(1, NC);   // the ring (redr) is reused by the pooler epilogue below
#endif
  }
  // per consumer warp: main loop done

  if constexpr (COMP) mbar_wait(cbar, 0);   // the escape warp reads the ring's x tiles until it is done
  if (CL > 1 && crank > 0) {
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");   // rank 0's qbar is initialized
    // ranks > 0: every accumulator goes straight into rank 0's peer buffer (slot crank - 1) by st.async
    // (completes rank 0's qbar transaction count; no fence, no wait), escape corrections folded into k-warp 0's
    const uint32_t peer_l = (uint32_t)__cvta_generic_to_shared(peer + (crank - 1) * (SK2_BF16P ? 1 : WSK) * TOKS * RP);
    const uint32_t peerh_l = (uint32_t)__cvta_generic_to_shared(reinterpret_cast<__nv_bfloat16*>(peer + (CL - 1) * TOKS * RP) +
                                                                 (crank - 1) * (WSK - 1) * TOKS * RP);
    uint32_t peer_r, qbar_r;
    asm volatile("mapa.shared::cluster.u32 %0, %1, 0;" : "=r"(peer_r) : "r"(peer_l));
    asm volatile("mapa.shared::cluster.u32 %0, %1, 0;" : "=r"(qbar_r) : "r"(qbar));
    uint32_t peerh_r = 0;
    if constexpr (SK2_BF16P) asm volatile("mapa.shared::cluster.u32 %0, %1, 0;" : "=r"(peerh_r) : "r"(peerh_l));
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
      for (int nt = 0; nt < NT; ++nt) {
        const int row = (wr * NT + nt) * 8 + 2 * t4;
        const int tok = mt * 16 + g;
        float v[4];
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          v[e] = acc[0][mt][nt][e];
#pragma unroll
          for (int c = 1; c < NACC; ++c) v[e] += acc[c][mt][nt][e];
        }
        if (COMP && wk == 0) {
          v[0] += corr[tok * CP + row];
          v[1] += corr[tok * CP + row + 1];
          if (tok + 8 < 32) {
            v[2] += corr[(tok + 8) * CP + row];
            v[3] += corr[(tok + 8) * CP + row + 1];
          }
        }
        if (SK2_BF16P && wk > 0) {   // BF16 pairs; k-warp 0 (with the corrections) stays FP32 below
          const uint32_t h0 = peerh_r + (uint32_t)((((wk - 1) * TOKS + tok) * RP + row) * 2);
          __nv_bfloat162 lo = __floats2bfloat162_rn(v[0], v[1]), hi = __floats2bfloat162_rn(v[2], v[3]);
          asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.b32 [%0], %1, [%2];" ::"r"(h0),
                       "r"(*reinterpret_cast<uint32_t*>(&lo)), "r"(qbar_r)
                       : "memory");
          asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.b32 [%0], %1, [%2];" ::"r"(h0 + 8 * RP * 2),
                       "r"(*reinterpret_cast<uint32_t*>(&hi)), "r"(qbar_r)
                       : "memory");
          continue;
        }
        const uint32_t a0 = peer_r + (uint32_t)((((SK2_BF16P ? 0 : wk) * TOKS + tok) * RP + row) * 4);
        asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v2.b32 [%0], {%1, %2}, [%3];" ::"r"(a0),
                     "r"(__float_as_uint(v[0])), "r"(__float_as_uint(v[1])), "r"(qbar_r)
                     : "memory");
        asm volatile(
            "st.async.shared::cluster.mbarrier::complete_tx::bytes.v2.b32 [%0], {%1, %2}, [%3];" ::"r"(a0 + 8 * RP * 4),
            "r"(__float_as_uint(v[2])), "r"(__float_as_uint(v[3])), "r"(qbar_r)
            : "memory");
      }

    return;
  }
  named_bar(1, NC);

#pragma unroll
  for (int mt = 0; mt < MT; ++mt)
#pragma unroll
    for (int nt = 0; nt < NT; ++nt) {
      const int row = (wr * NT + nt) * 8 + 2 * t4;
      const int tok = mt * 16 + g;
      const int i0 = (wk * TOKS + tok) * RP + row;
      float v[4];
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        v[e] = acc[0][mt][nt][e];
#pragma unroll
        for (int c = 1; c < NACC; ++c) v[e] += acc[c][mt][nt][e];
      }

      sk2_pst(red, i0, v[0], v[1]);
      sk2_pst(red, i0 + 8 * RP, v[2], v[3]);
    }

  named_bar(1, NC);

  if constexpr (CL > 1) {
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
    mbar_wait_cluster(qbar, 0);
  }

  constexpr int R4 = R / 4;
#if SK2_GLE
  if constexpr (!FUSE) {   // o: the gle epilogue; the fused pooler keeps the stock epilogue loop below
    static_assert(R4 == 8 && CL == 1 && sizeof(OutT) == 2, "gle epilogue: o8h geometry, BF16 out");
    const int tt = tid >> 3, row = (tid & 7) * 4;
    const bool on = tid < R4 * M;
    float q[7] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    if (on) {
      float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
      if constexpr (COMP) {
#else
  for (int idx = tid; idx < R4 * M; idx += NC) {
    const int tt = idx / R4;
    const int row = (idx - tt * R4) * 4;
    float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
    if constexpr (COMP) {
#endif
#pragma unroll
#if SK2_GLE
        for (int c = 0; c < NESC; ++c) {
          v.x += corr[c * CSZ + tt * CP + row];
          v.y += corr[c * CSZ + tt * CP + row + 1];
          v.z += corr[c * CSZ + tt * CP + row + 2];
          v.w += corr[c * CSZ + tt * CP + row + 3];
        }
#else
      for (int c = 0; c < NESC; ++c) {
        v.x += corr[c * CSZ + tt * CP + row];
        v.y += corr[c * CSZ + tt * CP + row + 1];
        v.z += corr[c * CSZ + tt * CP + row + 2];
        v.w += corr[c * CSZ + tt * CP + row + 3];
#endif
      }
#if !SK2_GLE
    }
#endif
#pragma unroll
#if SK2_GLE
      for (int w = 0; w < WSK; ++w) {
        const float4 p = sk2_pld(red, (w * TOKS + tt) * RP + row);
        v.x += p.x;
        v.y += p.y;
        v.z += p.z;
        v.w += p.w;
#else
    for (int w = 0; w < WSK; ++w) {
      const float4 p = sk2_pld(red, (w * TOKS + tt) * RP + row);
      v.x += p.x;
      v.y += p.y;
      v.z += p.z;
      v.w += p.w;
#pragma unroll
      for (int c = 0; c < CL - 1; ++c) {
        const float4 q = (SK2_BF16P && w > 0) ? sk2_hld(peer + (CL - 1) * TOKS * RP, ((c * (WSK - 1) + w - 1) * TOKS + tt) * RP + row)
                                                  : *reinterpret_cast<const float4*>(peer + ((SK2_BF16P ? c : c * WSK + w) * TOKS + tt) * RP + row);
        v.x += q.x;
        v.y += q.y;
        v.z += q.z;
        v.w += q.w;
#endif
      }
#if SK2_GLE
      const size_t col = (size_t)cta * R + row;
      if constexpr (SK2_C7 && COMP) { const float dsc = c7_dsc(base4); v.x *= dsc; v.y *= dsc; v.z *= dsc; v.w *= dsc; }
      OutT* o = out + (size_t)tt * N + col;
#else
    }
    if constexpr (SK2_C7 && COMP) { const float dsc = c7_dsc(base4); v.x *= dsc; v.y *= dsc; v.z *= dsc; v.w *= dsc; }
    OutT* o = out + (size_t)tt * N + cta * R + row;
    if constexpr (sizeof(OutT) == 4) {
      *reinterpret_cast<float4*>(o) = v;
    } else {
#endif
      __nv_bfloat162 lo = __floats2bfloat162_rn(v.x, v.y), hi = __floats2bfloat162_rn(v.z, v.w);
      uint2 pk;
      pk.x = *reinterpret_cast<uint32_t*>(&lo);
      pk.y = *reinterpret_cast<uint32_t*>(&hi);
      *reinterpret_cast<uint2*>(o) = pk;
#if SK2_GLE
      const float pf[4] = {__low2float(lo), __high2float(lo), __low2float(hi), __high2float(hi)};
      asm volatile("cp.async.wait_all;" ::: "memory");   // this thread's own copies
      const uint2 xp = gx_p[tid];
      const float4 xs = gx_s[tid], xa = gx_a[tid], xb = gx_b[tid], xa2 = gx_a2[tid];
      const float4 xas =
          make_float4(__fmul_rn(xa.x, xs.x), __fmul_rn(xa.y, xs.y), __fmul_rn(xa.z, xs.z), __fmul_rn(xa.w, xs.w));
      // pre = 1 + pre_ffn_norm in FP32 (the reference's (1 + w), not a BF16-rounded copy)
      const float pp[4] = {
          __fadd_rn(1.f, __uint_as_float(xp.x << 16)), __fadd_rn(1.f, __uint_as_float(xp.x & 0xffff0000u)),
          __fadd_rn(1.f, __uint_as_float(xp.y << 16)), __fadd_rn(1.f, __uint_as_float(xp.y & 0xffff0000u))};
      const float ex_as[4] = {xas.x, xas.y, xas.z, xas.w}, ex_b[4] = {xb.x, xb.y, xb.z, xb.w};
      const float ex_a2[4] = {xa2.x, xa2.y, xa2.z, xa2.w};
      float uu[4], vv[4];
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const float as = ex_as[j], bp = __fmul_rn(ex_b[j], pf[j]);
        uu[j] = __fmul_rn(as, pp[j]);
        vv[j] = __fmul_rn(bp, pp[j]);
        q[0] = __fmaf_rn(pf[j], pf[j], q[0]);
        q[1] = __fmaf_rn(as, as, q[1]);
        q[2] = __fmaf_rn(as, bp, q[2]);
        q[3] = __fmaf_rn(bp, bp, q[3]);
        if (col >= 1024) {
          const float x = __fmul_rn(ex_a2[j], as), y = __fmul_rn(ex_a2[j], bp);
          q[4] = __fmaf_rn(x, x, q[4]);
          q[5] = __fmaf_rn(x, y, q[5]);
          q[6] = __fmaf_rn(y, y, q[6]);
        }
      }
      // u, v: FP32 rows for the router (uvf) and BF16 rows for the pooler (uv); rows tt (u) and M + tt (v)
      auto put = [&](const float* f, size_t r) {
        __nv_bfloat162 h0 = __floats2bfloat162_rn(f[0], f[1]), h1 = __floats2bfloat162_rn(f[2], f[3]);
        uint2 hk;
        hk.x = *reinterpret_cast<uint32_t*>(&h0);
        hk.y = *reinterpret_cast<uint32_t*>(&h1);
        *reinterpret_cast<uint2*>(ep.uv + r * N + col) = hk;
        *reinterpret_cast<float4*>(ep.uvf + r * N + col) = make_float4(f[0], f[1], f[2], f[3]);
      };
      put(uu, (size_t)tt);
      put(vv, (size_t)(M + tt));
#endif
    }
#if SK2_GLE
    // the 8 lanes of one token (one aligned lane group) sum their columns: one partial row per (CTA, token)
#pragma unroll
    for (int k = 0; k < 7; ++k) {
      q[k] += __shfl_xor_sync(0xffffffffu, q[k], 4);
      q[k] += __shfl_xor_sync(0xffffffffu, q[k], 2);
      q[k] += __shfl_xor_sync(0xffffffffu, q[k], 1);
    }
    if (on && (tid & 7) == 0) {
      float4* pq = reinterpret_cast<float4*>(ep.part + ((size_t)cta * M + tt) * 8);
      pq[0] = make_float4(q[0], q[1], q[2], q[3]);
      pq[1] = make_float4(q[4], q[5], q[6], 0.f);
    }
  } else
    for (int idx = tid; idx < R4 * M; idx += NC) {
      const int tt = idx / R4;
      const int row = (idx - tt * R4) * 4;
      float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
      if constexpr (COMP) {
#pragma unroll
        for (int c = 0; c < NESC; ++c) {
          v.x += corr[c * CSZ + tt * CP + row];
          v.y += corr[c * CSZ + tt * CP + row + 1];
          v.z += corr[c * CSZ + tt * CP + row + 2];
          v.w += corr[c * CSZ + tt * CP + row + 3];
        }
      }
#pragma unroll
      for (int w = 0; w < WSK; ++w) {
        const float4 p = sk2_pld(red, (w * TOKS + tt) * RP + row);
        v.x += p.x;
        v.y += p.y;
        v.z += p.z;
        v.w += p.w;
#pragma unroll
        for (int c = 0; c < CL - 1; ++c) {
          const float4 q = (SK2_BF16P && w > 0) ? sk2_hld(peer + (CL - 1) * TOKS * RP, ((c * (WSK - 1) + w - 1) * TOKS + tt) * RP + row)
                                                  : *reinterpret_cast<const float4*>(peer + ((SK2_BF16P ? c : c * WSK + w) * TOKS + tt) * RP + row);
          v.x += q.x;
          v.y += q.y;
          v.z += q.z;
          v.w += q.w;
        }
      }
      if constexpr (SK2_C7 && COMP) { const float dsc = c7_dsc(base4); v.x *= dsc; v.y *= dsc; v.z *= dsc; v.w *= dsc; }
      OutT* o = out + (size_t)tt * N + cta * R + row;
      if constexpr (sizeof(OutT) == 4) {
        *reinterpret_cast<float4*>(o) = v;
      } else {
        __nv_bfloat162 lo = __floats2bfloat162_rn(v.x, v.y), hi = __floats2bfloat162_rn(v.z, v.w);
        uint2 pk;
        pk.x = *reinterpret_cast<uint32_t*>(&lo);
        pk.y = *reinterpret_cast<uint32_t*>(&hi);
        *reinterpret_cast<uint2*>(o) = pk;
      }
    }
#else
  }
#endif
}

template <int MT, int TOKR, int NT, int NWR, int WSK, int UPW, int NSTAGE, bool COMP, bool HL, int NACC, int CL, typename OutT,
          bool FUSE = false>
static void launch_sk2(TV Ws, TV x, TV out, TV ptab, int64_t pmax, int64_t base4, int M, int N, int K, int pdl,
#if SK2_GLE
                       const GleEpi& ep,
#endif
                       const uint8_t* wsr = nullptr, float* outr = nullptr, float* part = nullptr, int* rcnt = nullptr) {
  constexpr int NW = NWR * WSK;
  constexpr int R = NWR * NT * 8;
  constexpr int UPS = WSK * UPW;
  constexpr int TU = COMP ? 768 : (HL ? 2048 : 1024);
  constexpr int WBY = NW * UPW * NT * TU;
  constexpr int XBY = UPS * TOKR * 128;
  constexpr int SST = ((XBY + WBY) + 1023) & ~1023;
  // SK2_XR: the weight ring fills SK2_XRB bytes less the x slots (the kernel's NSTAGE is then the weight depth)
  constexpr bool XRN = sk2_xr<TOKR, NT, NWR, FUSE, CL>();
  constexpr int WST = (WBY + 1023) & ~1023;
  constexpr int NS = XRN ? (SK2_XRB - SK2_XR * XBY) / WST : NSTAGE;
  constexpr int RING = XRN ? NS * WST + SK2_XR * XBY : NSTAGE * SST;
  constexpr int TOKS = 16 * MT;
  constexpr int RP = R + ((40 - (R % 32)) % 32);
  constexpr int CP = R + 1;
#if SK2_NARROW
  constexpr int NESC = COMP ? 4 : 1;
#else
  constexpr int NESC = COMP ? sk2_nesc<NS, TOKR, NT, NWR, FUSE, CL>() : 1;
#endif
  const size_t smem = 1024 + (size_t)RING + 16 * NS + 32 + (size_t)NESC * ((TOKS * CP + 3) & ~3) * 4 +
                      (CL > 1 ? (size_t)(CL - 1) * WSK * TOKS * RP * 4 : 0) + (size_t)pmax * 8 + 128 +
                      (SK2_RRED && FUSE ? (size_t)WSK * TOKS * 8 * 4 : 0);
  auto go = [&](auto pkc) {
    constexpr bool PK = decltype(pkc)::value;
    auto kern = sk2_kernel<MT, TOKR, NT, NWR, WSK, UPW, NS, COMP, HL, NACC, CL, OutT, PK, FUSE>;
    static bool attr_done = false;
    if (!attr_done) {
      int dev = 0, optin = 0;
      cudaGetDevice(&dev);
      cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
      cudaFuncAttributes fa;
      cudaFuncGetAttributes(&fa, kern);
      cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, optin - (int)fa.sharedSizeBytes);
      attr_done = true;
    }
    CUtensorMap xm = make_map_x3d(x.data_ptr(), (uint64_t)M, (uint64_t)K, PK ? (uint32_t)M : (uint32_t)TOKR, UPS);
    cudaLaunchConfig_t lc = {};
    lc.gridDim = dim3(N / R * CL);
    lc.blockDim = dim3((NW + 1 + NESC) * 32);
    lc.dynamicSmemBytes = smem;
    lc.stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(x.device().device_type, x.device().device_id));
    cudaLaunchAttribute at_[2];
    int na = 0;
    if (CL > 1) {
      at_[na].id = cudaLaunchAttributeClusterDimension;
      at_[na].val.clusterDim.x = CL;
      at_[na].val.clusterDim.y = 1;
      at_[na].val.clusterDim.z = 1;
      ++na;
    }
    if (pdl > 0) {   // programmatic dependent launch: may start while the previous kernel in the stream finishes
      at_[na].id = cudaLaunchAttributeProgrammaticStreamSerialization;
      at_[na].val.programmaticStreamSerializationAllowed = 1;
      ++na;
    }
    lc.attrs = at_;
    lc.numAttrs = na;
    cudaLaunchKernelEx(&lc, kern, xm, (const uint8_t*)Ws.data_ptr(), (const __nv_bfloat16*)x.data_ptr(), (OutT*)out.data_ptr(),
                       (const int2*)ptab.data_ptr(), (int)pmax, (uint32_t)base4, M, N, K, 0, nullptr,
#if SK2_GLE
                       wsr, outr, part, rcnt, pdl, ep);
#else
                       wsr, outr, part, rcnt, pdl);
#endif
  };
  if (TOKR == 8 || M < TOKR) go(std::true_type{}); else go(std::false_type{});
}

// pdl: N > 0 launches with programmatic stream serialization and issues the first N weight stages before
// griddepcontrol.wait (low bits of the host's PDL word), 0 a plain launch
#if SK2_GLE
// the o projection at M <= 8 with the first residual boundary in its epilogue: u = rows of P scaled by the boundary's
// first norm, v = the stream part, both for the pooler (bf16 uv) and the FP32 router copy (uvf); part: per-CTA sums
void gle_o(TV Ws, TV x, TV out, TV ptab, int64_t pmax, int64_t base4, TV s, TV a, TV b, TV a2, TV p, TV uv, TV part,
           TV uvf, int64_t pdl) {
  const GleEpi ep{(const float*)s.data_ptr(), (const float*)a.data_ptr(), (const float*)b.data_ptr(), (const float*)a2.data_ptr(),
                  (const __nv_bfloat16*)p.data_ptr(), (__nv_bfloat16*)uv.data_ptr(), (float*)part.data_ptr(),
                  (float*)uvf.data_ptr()};
  launch_sk2<1, 8, 4, 1, 4, 1, 4, true, false, 2, 1, __nv_bfloat16>(Ws, x, out, ptab, pmax, base4, (int)x.size(0), 8192, 1024,
                                                                     (int)pdl, ep);
}
// the pooler (bf16 x rows) with the FP32 router (FP32 rows x32, FP32 weights w32) at M <= 8; ilv: the router's four
// stages interleave the pooler stream (1: the last slot of each period)
void gle_pr(TV Ws, TV Wsr, TV x, TV out, TV outr, TV ptab, int64_t pmax, int64_t base4, TV part, TV rcnt, TV x32, TV w32,
            int64_t ilv, int64_t pdl) {
  GleEpi ep{};
  ep.x32 = (const float*)x32.data_ptr(); ep.w32 = (const float*)w32.data_ptr(); ep.fpr = 1; ep.ilv = (int)ilv;
#if SK2_GLE16
  if (x.size(0) > 8)   // (T = 8 with the boundary in o's epilogue: the u / v rows of 8 tokens)
    launch_sk2<1, 16, 4, 1, 8, 1, 5, true, false, 2, 1, __nv_bfloat16, true>(
        Ws, x, out, ptab, pmax, base4, (int)x.size(0), 4096, 8192, (int)pdl, ep, (const uint8_t*)Wsr.data_ptr(),
        (float*)outr.data_ptr(), (float*)part.data_ptr(), (int*)rcnt.data_ptr());
  else
#endif
  launch_sk2<1, 8, 4, 1, 8, 1, 6, true, false, 2, 1, __nv_bfloat16, true>(
      Ws, x, out, ptab, pmax, base4, (int)x.size(0), 4096, 8192, (int)pdl, ep, (const uint8_t*)Wsr.data_ptr(),
      (float*)outr.data_ptr(), (float*)part.data_ptr(), (int*)rcnt.data_ptr());
}
#elif SK2_NARROW
// the qkv projection at M <= 8
void sk2(TV Ws, TV x, TV out, TV ptab, int64_t pmax, int64_t base4, int64_t pdl) {
  launch_sk2<1, 8, 3, 1, 4, 2, 5, true, false, 2, 1, __nv_bfloat16>(Ws, x, out, ptab, pmax, base4, (int)x.size(0),
                                                                     (int)out.size(1), (int)x.size(1), (int)pdl);
}
#else
#ifndef SK2_P05   // retimed ring depths (Paradigm's okc_tune ring caps; port list p05): qkv T = 1 6 -> 5 stages,
#define SK2_P05 0   // o / unpooler / pooler T = 16: 4 -> 3, 5 -> 3, 5 -> 4
#endif
// the projections of the M <= 32 shape classes; assemble.py expands layer.py's SK2_GEOMETRY below
void sk2(TV Ws, TV x, TV out, TV ptab, int64_t pmax, int64_t base4, int64_t cfg, int64_t pdl) {
  const int M = (int)x.size(0), N = (int)out.size(1), K = (int)x.size(1), P = (int)pdl;
  switch (cfg) {
    case 0: launch_sk2<1, 8, 3, 1, 4, 2, SK2_P05 ? 5 : 6, true, false, 1, 1, __nv_bfloat16>(Ws, x, out, ptab, pmax, base4, M, N, K, P); break;
    case 1: launch_sk2<1, 16, 3, 1, 4, 2, 5, true, false, 1, 1, __nv_bfloat16>(Ws, x, out, ptab, pmax, base4, M, N, K, P); break;
    case 2: launch_sk2<2, 32, 3, 1, 4, 2, 4, true, false, 1, 1, __nv_bfloat16>(Ws, x, out, ptab, pmax, base4, M, N, K, P); break;
    case 3: launch_sk2<1, 8, 4, 1, 4, 1, 4, true, false, 2, 1, __nv_bfloat16>(Ws, x, out, ptab, pmax, base4, M, N, K, P); break;
    case 4: launch_sk2<1, 16, 4, 1, 4, 1, SK2_P05 ? 3 : 4, true, false, 2, 1, __nv_bfloat16>(Ws, x, out, ptab, pmax, base4, M, N, K, P); break;
    case 5: launch_sk2<2, 32, 2, 2, 2, 2, 3, true, false, 1, 1, __nv_bfloat16>(Ws, x, out, ptab, pmax, base4, M, N, K, P); break;
    case 6: launch_sk2<1, 8, 1, 1, 8, 2, 4, true, false, 2, 1, __nv_bfloat16>(Ws, x, out, ptab, pmax, base4, M, N, K, P); break;
    case 7: launch_sk2<1, 16, 4, 1, 4, 1, SK2_P05 ? 3 : 5, true, false, 2, 1, __nv_bfloat16>(Ws, x, out, ptab, pmax, base4, M, N, K, P); break;
    case 8: launch_sk2<2, 32, 4, 2, 2, 2, 3, true, false, 1, 1, __nv_bfloat16>(Ws, x, out, ptab, pmax, base4, M, N, K, P); break;
  }
}
// the pooler with the router (BF16 hi + lo rows, FP32 accumulation) in one launch, 8 < M <= 32
void sk2f(TV Ws, TV Wsr, TV x, TV out, TV outr, TV ptab, int64_t pmax, int64_t base4, TV part, TV rcnt, int64_t pdl) {
  const int M = (int)x.size(0);
  if (M <= 16)
    launch_sk2<1, 16, 4, 1, 8, 1, SK2_P05 ? 4 : 5, true, false, 2, 1, __nv_bfloat16, true>(
        Ws, x, out, ptab, pmax, base4, M, 4096, 8192, (int)pdl, (const uint8_t*)Wsr.data_ptr(), (float*)outr.data_ptr(),
        (float*)part.data_ptr(), (int*)rcnt.data_ptr());
  else
    launch_sk2<2, 32, 4, 1, 8, 1, 3, true, false, 2, 1, __nv_bfloat16, true>(
        Ws, x, out, ptab, pmax, base4, M, 4096, 8192, (int)pdl, (const uint8_t*)Wsr.data_ptr(), (float*)outr.data_ptr(),
        (float*)part.data_ptr(), (int*)rcnt.data_ptr());
}
#endif
"""

RQ_CU = r"""
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <type_traits>
struct alignas(64) RqTmap {
  unsigned long long v[16];
};
using TV = tvm::ffi::TensorView;
struct RqP {
  const uint8_t* W;   // packed [NB][KC][8 tiles][1536]
  const uint32_t* esc;   // escape entries, [KS][NB][8][ecap] sentinel-terminated lists (+32 sentinels)
  const uint32_t* esc2;   // per-(warp,lane) sub-lists [KS][NB][8][32][4], sentinel-terminated
  int ecap;   // entries per (ks, nb, warp) list slot: fixed stride, so no offset load
  __nv_bfloat16* out;   // [m][N] (in-kernel zeroing only; the reduce-adds go through the out tensor map)
  int* cnt;   // [2][NB] zero / poll counters, all zero between launches
  void* layout0;   // (unused: keeps the parameter layout the kernels were tuned with)
  uint32_t* sink;
  int m, N, K, KC, NB, walias, ncnt;
  uint32_t Tb;
  uint32_t ks_magic;
};
__device__ __forceinline__ void rq_wait(uint32_t b, uint32_t ph) {
  asm volatile(
      "{\n.reg .pred p;\nRQ_W_%=:\nmbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n@!p bra RQ_W_%=;\n}\n" ::"r"(
          b),
      "r"(ph)
      : "memory");
}
__device__ __forceinline__ void rq_arrive(uint32_t b) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" ::"r"(b) : "memory");
}
// relaxed (non-release) arrive. The x/W slots are only ever *read* by the consumer (LDS + wgmma, the latter
// retired by wait_group), so the release ordering of the default arrive -- which makes the arrive wait for the LDS just
// issued  -- is not needed.
__device__ __forceinline__ void rq_arrive_r(uint32_t b) {
  asm volatile("mbarrier.arrive.relaxed.cta.shared::cta.b64 _, [%0];\n" ::"r"(b) : "memory");
}
__device__ __forceinline__ void rq_expect(uint32_t b, uint32_t tx) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(b), "r"(tx) : "memory");
}
__device__ __forceinline__ void rq_bulkh(uint32_t dst, const void* src, uint32_t bytes, uint32_t bar, uint64_t pol) {
  asm volatile(
      "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;\n" ::
          "r"(dst),
      "l"(src), "r"(bytes), "r"(bar), "l"(pol)
      : "memory");
}
__device__ __forceinline__ void rq_tileh(uint32_t dst, const RqTmap* tm, int k0, int r0, uint32_t bar, uint64_t pol) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1, {%2, "
      "%3}], [%4], %5;\n" ::"r"(dst),
      "l"((unsigned long long)tm), "r"(k0), "r"(r0), "r"(bar), "l"(pol)
      : "memory");
}
__device__ __forceinline__ uint4 rq_lds(uint32_t a) {
  uint4 v;
  asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "r"(a)
               : "memory");
  return v;
}
__device__ __forceinline__ uint32_t rq_prmt(uint32_t a, uint32_t b, uint32_t s) {
  uint32_t d;
  asm("prmt.b32 %0, %1, %2, %3;\n" : "=r"(d) : "r"(a), "r"(b), "r"(s));
  return d;
}
#ifndef RQ_PROBE   // timing probes only (wrong output, never built into a candidate): 1 = every CTA streams N block 0's
#define RQ_PROBE 0   // weights (L2 hits), 2 = no x tile copies, 3 = both; 4 = plain TMA stores instead of the reduce-adds
#endif
#ifndef RQ_STAMP   // timing probe only: per-CTA globaltimer stamps into the sink buffer (u64 [cta][16]; 8 = %smid): 0 entry,
#define RQ_STAMP 0   // 1 past the PDL wait, 2 first stage loaded (math warp 0), 3 main loop done, 4 output reduce done
#endif              // (rq128: 5 both WGs done, 6 slab written; 7 the zero poller saw the block zeroed)
#define RQ_ST(k) do { if (RQ_STAMP) { unsigned long long t_; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t_)); \
    reinterpret_cast<unsigned long long*>(p.sink)[((size_t)blockIdx.y * gridDim.x + blockIdx.x) * 16 + (k)] = t_; } } while (0)
#ifndef RQ_WPRE   // 1: thread 0 issues the W stream's first ST ring slots right after the barrier init, before the PDL
#define RQ_WPRE 0    // wait (W is a weight, independent of the previous kernel): the fill overlaps the previous kernel's tail
#endif
#ifndef RQ_REDV   // rq128 epilogue: 1 = posted red.global.add.v4.bf16x2 from the stmatrix slab instead of the TMA reduce-add
#define RQ_REDV 0    // (the TMA's bulk-group read wait is ~1 us per CTA tail); 2 = also each WG's own slab (no WG join)
#endif               // and no async-proxy fence (+32 KB smem)
#ifndef RQ_C7   // the aligned exponent window (as sk2's SK2_C7): W is packed times 2^sv so its window base is a multiple
#define RQ_C7 0   // of 8, the base ORs into the masked code bits (one LOP3 per high-byte word, no IADD), sv rides in the
#endif            // base word's bit 7s and the outputs are scaled by 2^-sv before rounding: bit-identical results
#if RQ_C7
#define RQ_HW(c) (((c) & 0x87878787u) | Tb)
#else
#define RQ_HW(c) (((c) & 0x87878787u) + Tb)
#endif
// r6 c12 decode (variant 1), pw10 pw_dec / pw_unit unchanged
__device__ __forceinline__ void rq_dec(uint32_t L0, uint32_t L1, uint32_t C, uint32_t Tb, uint32_t* a) {
  const uint32_t hl = RQ_HW(C);
  const uint32_t hh = RQ_HW(__funnelshift_r(C, C, 4));
  a[0] = rq_prmt(L0, hl, 0x5140);
  a[1] = rq_prmt(L0, hl, 0x7362);
  a[2] = rq_prmt(L1, hh, 0x5140);
  a[3] = rq_prmt(L1, hh, 0x7362);
}
__device__ __forceinline__ void rq_unit(const uint4& q0, const uint4& q1, const uint4& q2, uint32_t Tb, uint32_t* A) {
  rq_dec(q0.x, q0.y, q2.x, Tb, A);
  rq_dec(q0.z, q0.w, q2.y, Tb, A + 4);
  rq_dec(q1.x, q1.y, q2.z, Tb, A + 8);
  rq_dec(q1.z, q1.w, q2.w, Tb, A + 12);
}
// escape fix : entry = seq << 17 | lane << 12 | reg << 8 | half << 7 | xor7, seq = stage within
// the CTA's range. The warp's list is cached in registers, 32 entries per lane-spread window (entry i in lane i % 32), read
// by shfl: no dependent global load per hit .
__device__ __forceinline__ void rq_fix(uint32_t (&A)[16], int seq, uint32_t& nx, uint32_t& nxs, int& ei, uint32_t& ec,
                                       const uint32_t* el, int lane) {
  while (nxs == (uint32_t)seq) {
    if (lane == (int)((nx >> 12) & 31)) {
      const int r = (int)((nx >> 8) & 15);
      const uint32_t xr = (nx & 0x7fu) << (8 + ((nx >> 7) & 1) * 16);
#pragma unroll
      for (int i = 0; i < 16; i++)
        if (i == r) A[i] ^= xr;
    }
    ++ei;
    if ((ei & 31) == 0) ec = __ldg(el + ei + lane);
    nx = __shfl_sync(0xffffffffu, ec, ei & 31);
    nxs = nx >> 17;
  }
}
__device__ __forceinline__ void rq_adv(uint4& E, uint32_t& hs) {
  E.x = E.y;
  E.y = E.z;
  E.z = E.w;
  E.w = ~0u;
  hs = E.x >> 17;
}
// : patch the 8 hi-byte registers instead of the 16 A registers.
// rq_unit_ilp builds A[4g+0]=PRMT(q0,hl_g,0x5140) A[4g+1]=PRMT(q0,hl_g,0x7362) A[4g+2]=PRMT(q0,hh_g,0x5140)
// A[4g+3]=PRMT(q0,hh_g,0x7362), so entry (reg, half) -> H[reg>>1], byte (reg&1)*2 + half => 8 predicated XORs
// instead of 16 (half the apply block, so the taken branch over it is cheap).
__device__ __forceinline__ void rq_hi8(const uint4& q0, const uint4& q1, const uint4& q2, uint32_t Tb, uint32_t* H) {
  H[0] = RQ_HW(q2.x);
  H[1] = RQ_HW(__funnelshift_r(q2.x, q2.x, 4));
  H[2] = RQ_HW(q2.y);
  H[3] = RQ_HW(__funnelshift_r(q2.y, q2.y, 4));
  H[4] = RQ_HW(q2.z);
  H[5] = RQ_HW(__funnelshift_r(q2.z, q2.z, 4));
  H[6] = RQ_HW(q2.w);
  H[7] = RQ_HW(__funnelshift_r(q2.w, q2.w, 4));
}
__device__ __forceinline__ void rq_prmtA(const uint4& q0, const uint4& q1, const uint32_t* H, uint32_t* A) {
  A[0] = rq_prmt(q0.x, H[0], 0x5140);
  A[1] = rq_prmt(q0.x, H[0], 0x7362);
  A[2] = rq_prmt(q0.y, H[1], 0x5140);
  A[3] = rq_prmt(q0.y, H[1], 0x7362);
  A[4] = rq_prmt(q0.z, H[2], 0x5140);
  A[5] = rq_prmt(q0.z, H[2], 0x7362);
  A[6] = rq_prmt(q0.w, H[3], 0x5140);
  A[7] = rq_prmt(q0.w, H[3], 0x7362);
  A[8] = rq_prmt(q1.x, H[4], 0x5140);
  A[9] = rq_prmt(q1.x, H[4], 0x7362);
  A[10] = rq_prmt(q1.y, H[5], 0x5140);
  A[11] = rq_prmt(q1.y, H[5], 0x7362);
  A[12] = rq_prmt(q1.z, H[6], 0x5140);
  A[13] = rq_prmt(q1.z, H[6], 0x7362);
  A[14] = rq_prmt(q1.w, H[7], 0x5140);
  A[15] = rq_prmt(q1.w, H[7], 0x7362);
}
__device__ __forceinline__ void rq_xor8(uint32_t* H, uint32_t nx) {
  const uint32_t r = (nx >> 8) & 15u;
  const uint32_t xr = (nx & 0x7fu) << (16 * (r & 1u) + 8 * ((nx >> 7) & 1u));
  const uint32_t ri = r >> 1;
  asm volatile(
      "{\n\t.reg .pred p;\n\t"
      "setp.eq.u32 p, %8, 0;\n\t@p xor.b32 %0, %0, %9;\n\t"
      "setp.eq.u32 p, %8, 1;\n\t@p xor.b32 %1, %1, %9;\n\t"
      "setp.eq.u32 p, %8, 2;\n\t@p xor.b32 %2, %2, %9;\n\t"
      "setp.eq.u32 p, %8, 3;\n\t@p xor.b32 %3, %3, %9;\n\t"
      "setp.eq.u32 p, %8, 4;\n\t@p xor.b32 %4, %4, %9;\n\t"
      "setp.eq.u32 p, %8, 5;\n\t@p xor.b32 %5, %5, %9;\n\t"
      "setp.eq.u32 p, %8, 6;\n\t@p xor.b32 %6, %6, %9;\n\t"
      "setp.eq.u32 p, %8, 7;\n\t@p xor.b32 %7, %7, %9;\n\t}"
      : "+r"(H[0]), "+r"(H[1]), "+r"(H[2]), "+r"(H[3]), "+r"(H[4]), "+r"(H[5]), "+r"(H[6]), "+r"(H[7])
      : "r"(ri), "r"(xr));
}
__device__ __forceinline__ void rq_fixp8(uint32_t* H, int seq, uint4& E, uint32_t& hs) {
  while (__any_sync(0xffffffffu, hs == (uint32_t)seq)) {
    if (hs == (uint32_t)seq) {
      rq_xor8(H, E.x);
      rq_adv(E, hs);
    }
  }
}
__device__ __forceinline__ uint64_t rq_desc(uint32_t saddr) {   // K-major SW128: LBO 1, SBO 1024 B, base offset 0
  return (uint64_t)((saddr & 0x3FFFFu) >> 4) | ((uint64_t)1 << 16) | ((uint64_t)(1024 >> 4) << 32) |
         ((uint64_t)1 << 62);
}
template <int ND>
__device__ __forceinline__ void rq_facc(float (&D)[ND]) {
#pragma unroll
  for (int i = 0; i < ND; i++) asm volatile("" : "+f"(D[i])::"memory");
}
__device__ __forceinline__ void rq_fa(uint32_t (&A)[16]) {
#pragma unroll
  for (int i = 0; i < 16; i++) asm volatile("" : "+r"(A[i])::"memory");
}
__device__ __forceinline__ uint32_t rq_pk(float lo,
                                          float hi) {   // inline cvt: __float2bfloat16 serialises the RS pipeline 
  uint32_t d;
  asm("cvt.rn.bf16x2.f32 %0, %1, %2;\n" : "=r"(d) : "f"(hi), "f"(lo));
  return d;
}
template <int NT>
__device__ __forceinline__ void rq_wgmma(float (&D)[NT / 2], const uint32_t* a, uint64_t desc, int scd);
template <>
__device__ __forceinline__ void rq_wgmma<16>(float (&D)[8], const uint32_t* a, uint64_t desc, int scd) {
  asm volatile(
      "{\n.reg .pred p;\nsetp.ne.b32 p, %13, 0;\nwgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%0, %1, %2, %3, "
      "%4, %5, %6, %7}, {%8, %9, %10, %11}, %12, p, 1, 1, 0;\n}\n"
      : "+f"(D[0]), "+f"(D[1]), "+f"(D[2]), "+f"(D[3]), "+f"(D[4]), "+f"(D[5]), "+f"(D[6]), "+f"(D[7])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "r"(scd));
}
template <>
__device__ __forceinline__ void rq_wgmma<32>(float (&D)[16], const uint32_t* a, uint64_t desc, int scd) {
  asm volatile(
      "{\n.reg .pred p;\nsetp.ne.b32 p, %21, 0;\nwgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 {%0, %1, %2, %3, "
      "%4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, {%16, %17, %18, %19}, %20, p, 1, 1, 0;\n}\n"
      : "+f"(D[0]), "+f"(D[1]), "+f"(D[2]), "+f"(D[3]), "+f"(D[4]), "+f"(D[5]), "+f"(D[6]), "+f"(D[7]), "+f"(D[8]),
        "+f"(D[9]), "+f"(D[10]), "+f"(D[11]), "+f"(D[12]), "+f"(D[13]), "+f"(D[14]), "+f"(D[15])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "r"(scd));
}
template <>
__device__ __forceinline__ void rq_wgmma<128>(float (&D)[64], const uint32_t* a, uint64_t desc, int scd) {
  asm volatile(
      "{\n.reg .pred p;\nsetp.ne.b32 p, %69, 0;\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, "
      "%3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, "
      "%26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, "
      "%48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, p, "
      "1, 1, 0;\n}\n"
      : "+f"(D[0]), "+f"(D[1]), "+f"(D[2]), "+f"(D[3]), "+f"(D[4]), "+f"(D[5]), "+f"(D[6]), "+f"(D[7]), "+f"(D[8]),
        "+f"(D[9]), "+f"(D[10]), "+f"(D[11]), "+f"(D[12]), "+f"(D[13]), "+f"(D[14]), "+f"(D[15]), "+f"(D[16]),
        "+f"(D[17]), "+f"(D[18]), "+f"(D[19]), "+f"(D[20]), "+f"(D[21]), "+f"(D[22]), "+f"(D[23]), "+f"(D[24]),
        "+f"(D[25]), "+f"(D[26]), "+f"(D[27]), "+f"(D[28]), "+f"(D[29]), "+f"(D[30]), "+f"(D[31]), "+f"(D[32]),
        "+f"(D[33]), "+f"(D[34]), "+f"(D[35]), "+f"(D[36]), "+f"(D[37]), "+f"(D[38]), "+f"(D[39]), "+f"(D[40]),
        "+f"(D[41]), "+f"(D[42]), "+f"(D[43]), "+f"(D[44]), "+f"(D[45]), "+f"(D[46]), "+f"(D[47]), "+f"(D[48]),
        "+f"(D[49]), "+f"(D[50]), "+f"(D[51]), "+f"(D[52]), "+f"(D[53]), "+f"(D[54]), "+f"(D[55]), "+f"(D[56]),
        "+f"(D[57]), "+f"(D[58]), "+f"(D[59]), "+f"(D[60]), "+f"(D[61]), "+f"(D[62]), "+f"(D[63])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "r"(scd));
}
template <>
__device__ __forceinline__ void rq_wgmma<256>(float (&D)[128], const uint32_t* a, uint64_t desc, int scd) {
  asm volatile(
      "{\n.reg .pred p;\nsetp.ne.b32 p, %133, 0;\nwgmma.mma_async.sync.aligned.m64n256k16.f32.bf16.bf16 {%0, %1, %2, "
      "%3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, "
      "%26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, "
      "%48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, %64, %65, %66, %67, %68, %69, "
      "%70, %71, %72, %73, %74, %75, %76, %77, %78, %79, %80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, "
      "%92, %93, %94, %95, %96, %97, %98, %99, %100, %101, %102, %103, %104, %105, %106, %107, %108, %109, %110, %111, "
      "%112, %113, %114, %115, %116, %117, %118, %119, %120, %121, %122, %123, %124, %125, %126, %127}, {%128, %129, "
      "%130, %131}, %132, p, 1, 1, 0;\n}\n"
      : "+f"(D[0]), "+f"(D[1]), "+f"(D[2]), "+f"(D[3]), "+f"(D[4]), "+f"(D[5]), "+f"(D[6]), "+f"(D[7]), "+f"(D[8]),
        "+f"(D[9]), "+f"(D[10]), "+f"(D[11]), "+f"(D[12]), "+f"(D[13]), "+f"(D[14]), "+f"(D[15]), "+f"(D[16]),
        "+f"(D[17]), "+f"(D[18]), "+f"(D[19]), "+f"(D[20]), "+f"(D[21]), "+f"(D[22]), "+f"(D[23]), "+f"(D[24]),
        "+f"(D[25]), "+f"(D[26]), "+f"(D[27]), "+f"(D[28]), "+f"(D[29]), "+f"(D[30]), "+f"(D[31]), "+f"(D[32]),
        "+f"(D[33]), "+f"(D[34]), "+f"(D[35]), "+f"(D[36]), "+f"(D[37]), "+f"(D[38]), "+f"(D[39]), "+f"(D[40]),
        "+f"(D[41]), "+f"(D[42]), "+f"(D[43]), "+f"(D[44]), "+f"(D[45]), "+f"(D[46]), "+f"(D[47]), "+f"(D[48]),
        "+f"(D[49]), "+f"(D[50]), "+f"(D[51]), "+f"(D[52]), "+f"(D[53]), "+f"(D[54]), "+f"(D[55]), "+f"(D[56]),
        "+f"(D[57]), "+f"(D[58]), "+f"(D[59]), "+f"(D[60]), "+f"(D[61]), "+f"(D[62]), "+f"(D[63]), "+f"(D[64]),
        "+f"(D[65]), "+f"(D[66]), "+f"(D[67]), "+f"(D[68]), "+f"(D[69]), "+f"(D[70]), "+f"(D[71]), "+f"(D[72]),
        "+f"(D[73]), "+f"(D[74]), "+f"(D[75]), "+f"(D[76]), "+f"(D[77]), "+f"(D[78]), "+f"(D[79]), "+f"(D[80]),
        "+f"(D[81]), "+f"(D[82]), "+f"(D[83]), "+f"(D[84]), "+f"(D[85]), "+f"(D[86]), "+f"(D[87]), "+f"(D[88]),
        "+f"(D[89]), "+f"(D[90]), "+f"(D[91]), "+f"(D[92]), "+f"(D[93]), "+f"(D[94]), "+f"(D[95]), "+f"(D[96]),
        "+f"(D[97]), "+f"(D[98]), "+f"(D[99]), "+f"(D[100]), "+f"(D[101]), "+f"(D[102]), "+f"(D[103]), "+f"(D[104]),
        "+f"(D[105]), "+f"(D[106]), "+f"(D[107]), "+f"(D[108]), "+f"(D[109]), "+f"(D[110]), "+f"(D[111]), "+f"(D[112]),
        "+f"(D[113]), "+f"(D[114]), "+f"(D[115]), "+f"(D[116]), "+f"(D[117]), "+f"(D[118]), "+f"(D[119]), "+f"(D[120]),
        "+f"(D[121]), "+f"(D[122]), "+f"(D[123]), "+f"(D[124]), "+f"(D[125]), "+f"(D[126]), "+f"(D[127])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "r"(scd));
}

#if RQ_256
constexpr int NT = 256;
#else
constexpr int NT = 128;
#endif
#ifndef RQ_STAGES   // ring slots (rq128 fits up to 8 in the 227 KB: 1 KB + 8 * 28 KB + barriers)
#define RQ_STAGES 5
#endif
constexpr int ST = RQ_STAGES;
#ifndef RQ_MS   // k64 stages per ring slot (one W bulk of RQ_MS * 12 KB, RQ_MS x tiles) and per consumer round (GS == MS)
#define RQ_MS 1
#endif
constexpr int GS = RQ_MS;
#ifndef RQ_WM   // the consumer round: 0 decode on the wait -> issue chain, 1 decode of round G + 1 under round G's wgmma
#define RQ_WM RQ_256
#endif
constexpr int WM = RQ_WM;
constexpr int MS = RQ_MS;

struct RqC {
  static constexpr int XB = NT * 128;
#if RQ_256
  static constexpr int WB = (65603 & 16) ? 6144 : 12288;
#else
  static constexpr int WB = (42999939 & 16) ? 6144 : 12288;
#endif
  static constexpr int ND = NT / 2;   // : HALF, 64 W rows per CTA
  static constexpr int XBS = MS * XB;
  static constexpr int WBS = MS * WB;   // MS k64 stages per ring slot (MS = 2: merged k128 slot)
  static constexpr int SMEM = 1024 + ST * (XBS + WBS) + 24 * ST + 16 + (RQ_REDV == 2 ? 128 + 2 * XB : 0);
};

// Ring: ST k64 slots, slot i = x tile (XB, 1024-aligned) in the x ring + 12 KB W chunk in the W ring. ONE full barrier
// per slot , a W empty barrier
// (released by every math warp right after its LDS: W is dead once in registers) and an x empty barrier (released after
// the wait_group that retires the slot's wgmmas). The W lane may refill a slot before the x lane: its next arrive on the
// slot's full barrier can only happen after the consumers saw the current phase complete (they LDS after the full wait),
// so phases never mix. Consumers run GS k64 stages per round: GS full waits, one wgmma.fence, 4*GS wgmma, one commit,
// one wait_group 1 .
__global__ void __launch_bounds__(384, 1)
    rq(const RqP p, const __grid_constant__ RqTmap tx, const __grid_constant__ RqTmap to) {
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
  using C = RqC;
  constexpr int XB = C::XB;
  constexpr int WB = C::WB;
  constexpr int ND = C::ND;
  constexpr int XBS = C::XBS;
  constexpr int WBS = C::WBS;
  constexpr int NW = 8;
  constexpr int CW = 128;
  constexpr int PW = 8;   // first producer warp (W lane = PW, x lane = PW+1, zero = PW+2..)
  constexpr int MREG = 232;   // math-warp setmaxnreg target (H2 pool = 256*128, F2 = 384*80)
  // MS (k64 stages per ring slot): one full/empty barrier pair, one MS*12 KB W bulk and MS x tiles per slot; the consumer
  // round must cover a whole slot (GS == MS) so a slot is released exactly once per round
  extern __shared__ __align__(1024) uint8_t smem[];
  asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory");
  const uint32_t sraw = (uint32_t)__cvta_generic_to_shared(smem);
  const uint32_t sbase = (sraw + 1023u) & ~1023u;
  const uint32_t xring = sbase;
  const uint32_t wring = sbase + ST * XBS;
  const uint32_t bars = wring + ST * WBS;
  const uint32_t full = bars;
  const uint32_t wempty = bars + 8 * ST;
  const uint32_t xempty = bars + 16 * ST;
  const uint32_t zflag = bars + 24 * ST;
  const int tid = threadIdx.x;
  const int warp = tid >> 5, lane = tid & 31;
  const int nb = (int)blockIdx.x, ks = blockIdx.y, KS = gridDim.y;
  const int cb = nb * 128;   // first out column of this CTA
  const int s0 = KS == 1 ? 0 : (int)__umulhi((uint32_t)ks * (uint32_t)p.KC, p.ks_magic);
  const int KCS = (KS == 1 ? p.KC : (int)__umulhi((uint32_t)(ks + 1) * (uint32_t)p.KC, p.ks_magic)) - s0;
  const uint32_t* el = p.esc + (size_t)((ks * p.NB + nb) * 8 + ((warp & 7))) * p.ecap;
#if RQ_256
  uint32_t ec = (warp < NW) ? __ldg(el + lane) : 0u;
#else
  uint32_t ec = 0u;
#endif
  uint4 E4 = make_uint4(~0u, ~0u, ~0u, ~0u);
#if RQ_256

#else
  if (warp < NW)   // 16 B per thread, 512 B per warp, coalesced; latency hidden behind the first TMA wait
    E4 = *(const uint4*)(p.esc2 + (size_t)(((ks * p.NB + nb) * 8 + ((warp & 7))) * 32 + lane) * 4);
#endif
  uint32_t hs = E4.x >> 17;
  if (tid == 0) {
    for (int i = 0; i < ST; i++) {
      asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(full + 8 * i), "r"(2));
      asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(wempty + 8 * i), "r"(NW));
      asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(xempty + 8 * i), "r"(NW));
    }
    asm volatile("st.shared.u32 [%0], 0;\n" ::"r"(zflag) : "memory");
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
#if RQ_WPRE
    {
      const uint8_t* src = p.W + ((size_t)((RQ_PROBE & 1) ? 0 : nb) * p.KC + s0) * 12288;
      uint64_t pol = 0;
      asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;\n" : "=l"(pol));
      for (int s = 0, slot = 0; s < KCS && slot < ST; s += MS, slot++) {
        const int n = (KCS - s < MS) ? KCS - s : MS;
        rq_expect(full + 8 * slot, n * WB);
        rq_bulkh(wring + slot * WBS, src + (size_t)s * 12288, n * WB, full + 8 * slot, pol);
      }
    }
#endif
  }
  if (tid == ((PW + 2) * 32)) asm volatile("prefetch.tensormap [%0];\n" ::"l"((unsigned long long)&tx) : "memory");
  if (RQ_STAMP && tid == 0) {
    RQ_ST(0);
    uint32_t sm_;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(sm_));
    reinterpret_cast<unsigned long long*>(p.sink)[((size_t)blockIdx.y * gridDim.x + blockIdx.x) * 16 + 8] = sm_;
  }
  if (tid == 0) asm volatile("griddepcontrol.wait;\n" ::: "memory");
  __syncthreads();
  if (RQ_STAMP && tid == 0) RQ_ST(1);
  const unsigned long long cta = (unsigned long long)blockIdx.y * gridDim.x + blockIdx.x;
  if (warp >= PW) {   // ---------------- producer warpgroup: W lane (warp PW), x lane (warp PW+1)
    asm volatile("setmaxnreg.dec.sync.aligned.u32 40;\n");
    if (warp == PW && lane == 0) {
      const int nbw = (RQ_PROBE & 1) ? 0 : nb;
      const uint8_t* src = p.W + ((size_t)nbw * p.KC + s0) * 12288;
      uint64_t pol = 0;
      asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;\n" : "=l"(pol));
      int slot = 0, r = RQ_WPRE;
      for (int s = RQ_WPRE ? MS * ST : 0; s < KCS; s += MS) {
        const int n = (KCS - s < MS) ? KCS - s : MS;
        if (s >= MS * ST) {
          rq_wait((wempty) + 8 * slot, (r & 1) ^ 1);
        }
        rq_expect(full + 8 * slot, n * WB);
        rq_bulkh(wring + slot * WBS, src + (size_t)s * 12288, n * WB, full + 8 * slot, pol);
        if (++slot == ST) {
          slot = 0;
          r++;
        }
      }

    } else if (warp == (PW + 1) && lane == 0) {
      uint64_t pol = 0;
      asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;\n" : "=l"(pol));
      int slot = 0, r = 0;
      for (int s = 0; s < KCS; s += MS) {
        const int n = (KCS - s < MS) ? KCS - s : MS;
        if (s >= MS * ST) {
#if RQ_256
          rq_wait(((wempty)) + 8 * slot, (r & 1) ^ 1);
#else
          rq_wait(((xempty)) + 8 * slot, (r & 1) ^ 1);
#endif
        }
        rq_expect(full + 8 * slot, (RQ_PROBE & 2) ? 0 : n * XB);
        for (int j = 0; j < n && !(RQ_PROBE & 2); j++) {
          rq_tileh(xring + slot * XBS + j * XB, &tx, (s0 + s + j) * 64, 0, full + 8 * slot, pol);
        }
        if (++slot == ST) {
          slot = 0;
          r++;
        }
      }

    } else if (warp >=
               (PW + 2)) {   // in-kernel zeroing: this CTA's 1/KS token slice of its row block's 128 out columns,
      // then a release increment of the block's counter; every reduce-add into the block first acquires counter == KS
      const int t = tid - (PW + 2) * 32;
      const int r0 = KS == 1 ? 0 : (int)__umulhi((uint32_t)ks * (uint32_t)p.m, p.ks_magic);
      const int r1 = KS == 1 ? p.m : (int)__umulhi((uint32_t)(ks + 1) * (uint32_t)p.m, p.ks_magic);
      for (int e = t; e < (r1 - r0) * (CW / 8); e += 64) {
        __nv_bfloat16* d = p.out + (size_t)(r0 + e / (CW / 8)) * p.N + cb + (e % (CW / 8)) * 8;
        asm volatile("st.global.v4.u32 [%0], {%1, %1, %1, %1};\n" ::"l"(d), "r"(0) : "memory");
      }
      asm volatile("fence.proxy.async.global;\n" ::: "memory");   // generic zero stores -> the TMA (async proxy) adds
      asm volatile("bar.sync 4, 64;\n" ::: "memory");
      if (t == 0) {
        asm volatile("red.release.gpu.global.add.u32 [%0], 1;\n" ::"l"(p.cnt + blockIdx.x) : "memory");
        // wait (off the critical path, while the mainloop runs) until every CTA of the block zeroed its slice; publish it
        // to the epilogue through a shared flag. (The waits here are bounded with a never-reached trap: the bound
        // keeps the code as tuned.)
        uint32_t c = 0;
        for (int it = 0; it < (1 << 22); it++) {
          asm volatile("ld.acquire.gpu.global.u32 %0, [%1];\n" : "=r"(c) : "l"(p.cnt + blockIdx.x) : "memory");
          if ((int)c >= KS) break;
          __nanosleep(128);
        }
        if ((int)c < KS) asm volatile("trap;");
        if (RQ_STAMP) RQ_ST(7);
        asm volatile("fence.proxy.async.global;\n" ::: "memory");
        asm volatile("st.release.cta.shared.u32 [%0], 1;\n" ::"r"(zflag) : "memory");
        // the last of the block's KS pollers resets both counters for the next launch (the kernel boundary orders it)
        if (atomicAdd(p.cnt + gridDim.x + blockIdx.x, 1) == (KS - 1)) {
          p.cnt[blockIdx.x] = 0;
          p.cnt[gridDim.x + blockIdx.x] = 0;
        }
      }

    } else {
    }
    return;
  }
  // ---------------- math warpgroups: warp w of WG g decodes W rows 64g + 16w .. +15 of the block (= tile 4g + w = warp)
  if constexpr (MREG == 232)
    asm volatile("setmaxnreg.inc.sync.aligned.u32 232;\n");
  else if constexpr (MREG == 128)
    asm volatile("setmaxnreg.inc.sync.aligned.u32 128;\n");
  else
    asm volatile("setmaxnreg.inc.sync.aligned.u32 96;\n");

  const int g = warp >> 2, w = warp & 3;
  int ei = 0;
  uint32_t nx = __shfl_sync(0xffffffffu, ec, 0), nxs = nx >> 17;
  uint32_t Tb = RQ_C7 ? p.Tb & 0x7F7F7F7Fu : p.Tb;
  asm volatile("" : "+r"(Tb));
  const uint32_t wl = wring + warp * 1536 + lane * 16;
  const uint64_t xd0 = rq_desc(xring);
  float
      D[ND];   // never zero-filled: scale-d 0 on the CTA's first k16 (a generic write would serialise the pipe, C7515)
  uint32_t A0[GS][16], A1[GS][16];
#pragma unroll
  for (int j = 0; j < GS; j++)
#pragma unroll
    for (int i = 0; i < 16; i++) A0[j][i] = A1[j][i] = 0u;
  uint4 Q[GS][3];
#pragma unroll
  for (int j = 0; j < GS; j++) Q[j][0] = Q[j][1] = Q[j][2] = make_uint4(0, 0, 0, 0);
  uint32_t sink = 0;

  const int NGF = KCS / GS;
  const int NGT = (KCS + GS - 1) / GS;   // full rounds, all rounds
  // round G = stages GS*G .. GS*G + GS - 1 (the last round may be partial)
  auto ldr = [&](int G) {   // full waits, LDS of this warp's 48 B per lane per stage, W slots released
#pragma unroll
    for (int j = 0; j < GS; j++) {
      const int s = GS * G + j;
      if (s < KCS) {
        const int slot = (s / MS) % ST;
        if ((j % MS) == 0) rq_wait(full + 8 * slot, ((s / MS) / ST) & 1);
        {
          const uint32_t a = wl + slot * WBS + (s % MS) * WB;
          Q[j][0] = rq_lds(a);
          Q[j][1] = rq_lds(a + 512);
          Q[j][2] = rq_lds(a + 1024);
        }
      }
    }
#if RQ_256

#else
    {
      __syncwarp();
      if (lane == 0) {
#pragma unroll
        for (int j = 0; j < GS; j += MS)
          if (GS * G + j < KCS) rq_arrive(wempty + 8 * (((GS * G + j) / MS) % ST));
      }
    }
#endif
  };
  auto xrel = [&](int G) {
    {
      __syncwarp();
      if (lane == 0) {
#pragma unroll
        for (int j = 0; j < GS; j += MS)
          if (GS * G + j < KCS) {
#if RQ_256
            const uint32_t b = (wempty) + 8 * (((GS * G + j) / MS) % ST);
            rq_arrive(b);
#else
            const uint32_t b = (xempty) + 8 * (((GS * G + j) / MS) % ST);
            rq_arrive_r(b);
#endif
          }
      }
    }
  };
  auto dec = [&](int G, int ns, uint32_t (&A)[GS][16]) {

#pragma unroll
    for (int j = 0; j < GS; j++) {
      if (j < ns) {
        {
          {
#if RQ_256
            rq_unit(Q[j][0], Q[j][1], Q[j][2], Tb, A[j]);
            rq_fix(A[j], GS * G + j, nx, nxs, ei, ec, el, lane);
#else
            uint32_t H[8];
            rq_hi8(Q[j][0], Q[j][1], Q[j][2], Tb, H);
            rq_fixp8(H, GS * G + j, E4, hs);
            rq_prmtA(Q[j][0], Q[j][1], H, A[j]);
#endif
          }
        }
      }
    }
  };
  auto mma = [&](int G, auto NSC, uint32_t (&A)[GS][16]) {   // NSC: compile-time stage count of this round
    constexpr int NS = decltype(NSC)::value;
    rq_facc<ND>(D);
    asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
#pragma unroll
    for (int j = 0; j < NS; j++) {
      const int s = GS * G + j;
      const uint64_t d = xd0 + (uint64_t)(((((s / MS) % ST) * XBS + (s % MS) * XB)) >> 4);
#pragma unroll
      for (int q = 0; q < 4; q++) rq_wgmma<NT>(D, A[j] + 4 * q, d + 2 * q, (s | q) != 0);
    }
    asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
    rq_facc<ND>(D);

  };
  auto fa = [&](uint32_t (&A)[GS][16]) {
#pragma unroll
    for (int j = 0; j < GS; j++) rq_fa(A[j]);
  };
  {
    // WM 0: dec(G) -> issue(G) -> LDS(G+1) -> wait_group 1 -> release x(G-1)  (decode on the wait->issue chain)
    // WM 1: issue(G) -> LDS(G+1) + dec(G+1) under wgmma(G) -> wait_group 0 -> release x(G) at once
    auto round = [&](int G, auto NSC, uint32_t (&A)[GS][16], uint32_t (&An)[GS][16], uint32_t (&Ap)[GS][16]) {
      if (WM == 0) {
        dec(G, decltype(NSC)::value, A);
        mma(G, NSC, A);
        if ((G + 1) < NGT) ldr(G + 1);
        asm volatile("wgmma.wait_group.sync.aligned 1;\n" ::: "memory");
        fa(Ap);
        if (G > 0) xrel(G - 1);
      } else {
        mma(G, NSC, A);
        if ((G + 1) < NGT) {
          ldr(G + 1);
          dec(G + 1, KCS - GS * (G + 1), An);
        }
        asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
        fa(A);
        xrel(G);
      }
    };
    using FULLR = std::integral_constant<int, GS>;   // (GS = 1: every round is full)
    ldr(0);
    if (RQ_STAMP && tid == 0) RQ_ST(2);
    if (WM != 0) dec(0, GS, A0);
    round(0, FULLR{}, A0, A1, A1);
    int G = 1;
#pragma unroll 1
    for (; G + 1 < NGF; G += 2) {
      round(G, FULLR{}, A1, A0, A0);
      round(G + 1, FULLR{}, A0, A1, A1);
    }
    if (G < NGF) round(G, FULLR{}, A1, A0, A0);
    asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
    rq_facc<ND>(D);
    fa(A0);
    fa(A1);
    if (WM != 1) xrel(NGT - 1);
    if (RQ_STAMP && tid == 0) RQ_ST(3);
  }
  if constexpr (RQ_C7) {   // 2^-sv, sv = the 4-bit signed scale in the base word's bit 7s
    const int s4 = (int)(((p.Tb >> 7) & 1u) | ((p.Tb >> 14) & 2u) | ((p.Tb >> 21) & 4u) | ((p.Tb >> 28) & 8u));
    const float dsc = __int_as_float((127 - (s4 >= 8 ? s4 - 16 : s4)) << 23);
#pragma unroll
    for (int i = 0; i < ND; i++) D[i] *= dsc;
  }

  {   // ---------------- epilogue: stmatrix.trans -> SW128 slab [NT tok][64 n] -> TMA reduce-add

    {
      if (RQ_REDV != 2)
        asm volatile("bar.sync 1, %0;\n" ::"r"(NW * 32) : "memory");   // both WGs retired every wgmma: the x ring is free
      if (RQ_STAMP && tid == 0) RQ_ST(5);

      // (RQ_REDV 2: each WG's own slab past the barriers, so neither waits for the other WG's last wgmma)
      const uint32_t slab = RQ_REDV == 2 ? ((zflag + 16 + 127) & ~127u) + g * XB : xring + g * XB;
      const int j = lane >> 3, rr = lane & 7;
#if RQ_256
      const bool issuer = w == 0 && lane == 0;
      auto zpoll = [&]() {
        {   // the block's zeroing is complete (flag set by warp 10 long before)
          uint32_t f = 0;
          for (int it = 0; it < (1 << 24) && !f; it++)
            asm volatile("ld.acquire.cta.shared.u32 %0, [%1];\n" : "=r"(f) : "r"(zflag) : "memory");
          if (!f) asm volatile("trap;");
          asm volatile("fence.proxy.async.global;\n" ::: "memory");
        }
      };

      auto stsm = [&](int pp) {
#else
#pragma unroll
        for (int pp = 0; pp < NT / 16; pp++) {
#endif
          const int i0 = 2 * pp;
          const int i = i0 + (j >> 1), ch = 2 * w + (j & 1), tok = 8 * i + rr;
          const uint32_t a = slab + tok * 128 + ((ch ^ rr) << 4);
          const uint32_t r0 = rq_pk(D[4 * i0], D[4 * i0 + 1]);
          const uint32_t r1 = rq_pk(D[4 * i0 + 2], D[4 * i0 + 3]);
          const uint32_t r2 = rq_pk(D[4 * i0 + 4], D[4 * i0 + 5]);
          const uint32_t r3 = rq_pk(D[4 * i0 + 6], D[4 * i0 + 7]);
          asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n" ::"r"(a), "r"(r0),
                       "r"(r1), "r"(r2), "r"(r3)
                       : "memory");
#if RQ_256
        };
        auto issue = [&](int tok0, uint32_t src) {
#else
        }
        if (RQ_REDV != 2) asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");   // (REDV 2: generic readers)
        asm volatile("bar.sync %0, 128;\n" ::"r"(2 + g) : "memory");
        if (RQ_STAMP && tid == 0) RQ_ST(6);
        if (RQ_REDV) {   // posted 16-byte vector reduce-adds straight from the slab: nothing to wait for before exit
          uint32_t f = 0;
          for (int it = 0; it < (1 << 24) && !f; it++)
            asm volatile("ld.acquire.cta.shared.u32 %0, [%1];\n" : "=r"(f) : "r"(zflag) : "memory");
          if (!f) asm volatile("trap;");
          __nv_bfloat16* o = p.out + cb + 64 * g;
#pragma unroll
          for (int i = 0; i < 8; i++) {
            const int e = (tid & 127) + 128 * i, tk = e >> 3, c = e & 7;
            const uint4 v = rq_lds(slab + tk * 128 + ((c ^ (tk & 7)) << 4));
            asm volatile("red.global.add.noftz.v4.bf16x2 [%0], {%1, %2, %3, %4};\n" ::"l"(o + (size_t)tk * p.N + c * 8),
                         "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w)
                         : "memory");
          }
        }

        if (!RQ_REDV && w == 0 && lane == 0) {
          {   // the block's zeroing is complete (flag set by warp 10 long before)
            uint32_t f = 0;
            for (int it = 0; it < (1 << 24) && !f; it++)
              asm volatile("ld.acquire.cta.shared.u32 %0, [%1];\n" : "=r"(f) : "r"(zflag) : "memory");
            if (!f) asm volatile("trap;");
            asm volatile("fence.proxy.async.global;\n" ::: "memory");
          }

#endif
          asm volatile(
#if RQ_PROBE & 4
              "cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%1, %2}], [%3];\n" ::"l"(
#else
              "cp.reduce.async.bulk.tensor.2d.global.shared::cta.add.tile.bulk_group [%0, {%1, %2}], [%3];\n" ::"l"(
#endif
                  (unsigned long long)&to),
#if RQ_256
              "r"(cb + 64 * g), "r"(tok0),
              "r"(src)
#else
                  "r"(cb + 64 * g),
              "r"(0), "r"(slab)
#endif
              : "memory");
#if RQ_256
        };
        {
#pragma unroll
          for (int pp = 0; pp < NT / 32; pp++) stsm(pp);
          asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
          asm volatile("bar.sync %0, 128;\n" ::"r"(2 + g) : "memory");
          if (issuer) {
            zpoll();
            issue(0, slab);
            asm volatile("cp.async.bulk.commit_group;\n" ::: "memory");
          }
#pragma unroll
          for (int pp = NT / 32; pp < NT / 16; pp++) stsm(pp);
          asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
          asm volatile("bar.sync %0, 128;\n" ::"r"(2 + g) : "memory");
#else
          asm volatile("cp.async.bulk.commit_group;\n" ::: "memory");
#endif

#if RQ_256
          if (issuer) {
            issue(NT / 2, slab + (NT / 2) * 128);
            asm volatile("cp.async.bulk.commit_group;\n" ::: "memory");

            asm volatile("cp.async.bulk.wait_group.read 0;\n" ::: "memory");
          }
#else
          asm volatile("cp.async.bulk.wait_group.read 0;\n" ::: "memory");
#endif
        }
      }
    }

    if (RQ_STAMP && tid == 0) RQ_ST(4);   // (tid 0 waited for its reduce copies)
#undef RQ_T0
#undef RQ_T1
#endif
  }

  typedef CUresult (*RqEnc)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*, const cuuint64_t*, const cuuint64_t*,
                            const cuuint32_t*, const cuuint32_t*, CUtensorMapInterleave, CUtensorMapSwizzle,
                            CUtensorMapL2promotion, CUtensorMapFloatOOBfill);
  // 2D bf16 map: dims {inner, outer}, box {64, box_outer}, 128B swizzle; out-of-range rows read as 0 and are not written
  static int rq_map(RqTmap * tm, const void* ptr, int64_t inner, int64_t outer, int box_outer) {
    static RqEnc enc = nullptr;
    if (!enc) {
      void* fn = nullptr;
      cudaDriverEntryPointQueryResult q;
      cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &fn, 12000, cudaEnableDefault, &q);
      enc = (RqEnc)fn;
    }
    cuuint64_t dims[2] = {(cuuint64_t)inner, (cuuint64_t)outer};
    cuuint64_t strides[1] = {(cuuint64_t)inner * 2};
    cuuint32_t box[2] = {64, (cuuint32_t)box_outer};
    cuuint32_t estr[2] = {1, 1};
    enc((CUtensorMap*)tm, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, (void*)ptr, dims, strides, box, estr,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    return 0;
  }
  static void rq_go(RqP & p, TV x, TV out, int KS, bool pdl) {
    using C = RqC;
    auto k = rq;
    p.ks_magic = KS == 1 ? 0u : (uint32_t)(((1ull << 32) + (unsigned)KS - 1u) / (unsigned)KS);
    RqTmap tx, to;
    rq_map(&tx, x.data_ptr(), p.K, p.m, NT);
#if RQ_256
    rq_map(&to, out.data_ptr(), p.N, p.m, (65603 & 65536) ? NT / 2 : NT);
#else
    rq_map(&to, out.data_ptr(), p.N, p.m, (42999939 & 65536) ? NT / 2 : NT);
#endif
    static bool once = false;
    if (!once) {
      cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, C::SMEM);
      once = true;
    }
    cudaLaunchConfig_t lc = {};
    lc.gridDim = dim3(p.NB * (1), KS);
    lc.blockDim = dim3(384);
    lc.dynamicSmemBytes = C::SMEM;
    lc.stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(x.device().device_type, x.device().device_id));
    cudaLaunchAttribute at[1];
    at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    at[0].val.programmaticStreamSerializationAllowed = 1;
    lc.attrs = at;
    lc.numAttrs = pdl ? 1 : 0;
    cudaLaunchKernelEx(&lc, k, p, tx, to);
  }

  // out [m, N] bf16 = x [m, K] bf16 @ W.T, W in the 12-bit lossless layout [NB][KC][8][1536] with its escape lists;
  // KS: the K split; tab: the stream's exponent base
  void run(TV W, TV x, TV out, TV esc, TV esc2, TV cnt, TV sink, int64_t KS, int64_t tab, int64_t pdl) {
    RqP p = {};
    const int64_t nl = KS * W.size(0) * 8;
    p.ncnt = (int)cnt.size(0);
    p.W = (const uint8_t*)W.data_ptr();
    p.esc = (const uint32_t*)esc.data_ptr();
    p.ecap = (int)((esc.size(0) - 32) / nl);
    p.esc2 = (const uint32_t*)esc2.data_ptr();
    p.out = (__nv_bfloat16*)out.data_ptr();
    p.cnt = (int*)cnt.data_ptr();
      p.sink = (uint32_t*)sink.data_ptr();
    p.m = (int)x.size(0);
    p.K = (int)x.size(1);
    p.N = (int)out.size(1);
    p.KC = (int)W.size(1);
    p.NB = (int)W.size(0);
    p.walias = 0;
    p.Tb = (uint32_t)(tab & 0xffffffff);
    rq_go(p, x, out, (int)KS, pdl != 0);
  }
"""

GLUE_CU = r"""
// A3: the base Gluon lane mapping and floating-point operation order, across
// eight 128-thread CTAs. This source is embedded in the single-file candidate.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cooperative_groups.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <stdexcept>
#include <stdint.h>
namespace cg = cooperative_groups;
#ifndef GLUE_RMOM   // router boundary (b1 at 2 <= T <= 8): r1, r2, the router copy's norm and the carry tail from seven
#define GLUE_RMOM 0   // row moments in one DSMEM exchange (the plain b1's / the T = 1 GLE algebra: r2 without the BF16
#endif                // rounding of r1 * P) instead of an r1 exchange, then a second for r2 / tail / router norm
#ifndef GLUE_STAS   // 1: cluster_sums (b1's one exchange after the dependency wait) pushes its float4s with st.async into
#define GLUE_STAS 0   // every rank's slot and that rank's own mbarrier (complete_tx, release at cluster scope), and each
#endif                // CTA waits on its own barrier: no release cluster barrier, so no MEMBAR.ALL.GPU trio on the chain
using BF = __nv_bfloat16;
using TV = tvm::ffi::TensorView;

struct GlueArgs {
  const BF *branch, *post, *pre;
  const float *part, *stream, *alpha, *beta;
  BF* out;
  float *next, *router;
  int rows;
#if GLUE_ROUTER
  const float* alpha2;
  float* tail;
#endif
};

__device__ __forceinline__ float ga(float x, float y) {
  float z;
  asm("add.rn.f32 %0,%1,%2;" : "=f"(z) : "f"(x), "f"(y));
  return z;
}
__device__ __forceinline__ float gm(float x, float y) {
  float z;
  asm("mul.rn.f32 %0,%1,%2;" : "=f"(z) : "f"(x), "f"(y));
  return z;
}
__device__ __forceinline__ float gf(float x, float y, float a) {
  float z;
  asm("fma.rn.f32 %0,%1,%2,%3;" : "=f"(z) : "f"(x), "f"(y), "f"(a));
  return z;
}
__device__ __forceinline__ float gb(float x) {
  BF z = __float2bfloat16_rn(x);
  return __bfloat162float(z);
}
__device__ __forceinline__ float gr(float x, float eps) {
  // start's div.full.f32 by 8192 + eps becomes this FFMA in ptxas SASS.
  float z = gf(x, 0.0001220703125f, eps);
  asm("rsqrt.approx.ftz.f32 %0,%1;" : "=f"(z) : "f"(z));
  return z;
}
template <int V>
__device__ __forceinline__ void glf(const float* p, float* x) {
  if constexpr (V == 4) {
    asm volatile("ld.global.v4.f32 {%0,%1,%2,%3},[%4];"
                 : "=f"(x[0]), "=f"(x[1]), "=f"(x[2]), "=f"(x[3])
                 : "l"(p)
                 : "memory");
  } else {
    asm volatile("ld.global.v2.f32 {%0,%1},[%2];" : "=f"(x[0]), "=f"(x[1]) : "l"(p) : "memory");
  }
}
template <int V>
__device__ __forceinline__ void glb(const BF* p, float* x) {
  unsigned u[V / 2];
  if constexpr (V == 4) {
    asm volatile("ld.global.v2.b32 {%0,%1},[%2];" : "=r"(u[0]), "=r"(u[1]) : "l"(p) : "memory");
  } else {
    asm volatile("ld.global.b32 %0,[%1];" : "=r"(u[0]) : "l"(p) : "memory");
  }
#pragma unroll
  for (int j = 0; j < V / 2; j++) {
    x[2 * j] = __uint_as_float(u[j] << 16);
    x[2 * j + 1] = __uint_as_float(u[j] & 0xffff0000u);
  }
}
template <int V>
__device__ __forceinline__ void gsf(float* p, const float* x) {
  if constexpr (V == 4) {
    asm volatile("st.global.v4.f32 [%0],{%1,%2,%3,%4};" ::"l"(p), "f"(x[0]), "f"(x[1]), "f"(x[2]), "f"(x[3])
                 : "memory");
  } else {
    asm volatile("st.global.v2.f32 [%0],{%1,%2};" ::"l"(p), "f"(x[0]), "f"(x[1]) : "memory");
  }
}
template <int V>
__device__ __forceinline__ void gsb(BF* p, const float* x) {
  unsigned u[V / 2];
#pragma unroll
  for (int j = 0; j < V / 2; j++) {
    asm("cvt.rn.bf16x2.f32 %0,%1,%2;" : "=r"(u[j]) : "f"(x[2 * j + 1]), "f"(x[2 * j]));
  }
  if constexpr (V == 4) {
    asm volatile("st.global.v2.b32 [%0],{%1,%2};" ::"l"(p), "r"(u[0]), "r"(u[1]) : "memory");
  } else {
    asm volatile("st.global.b32 [%0],%1;" ::"l"(p), "r"(u[0]) : "memory");
  }
}

template <int N>
__device__ __forceinline__ float local_squares(const float (&x)[N]) {
  float q[N / 2];
#pragma unroll
  for (int j = 0; j < N / 2; j++) q[j] = gf(x[2 * j], x[2 * j], gm(x[2 * j + 1], x[2 * j + 1]));
// Gluon contracts each even square into its adjacent odd square, then uses
// the balanced in-thread tree. No sequential sum or reassociation is allowed.
#pragma unroll
  for (int width = N / 2; width > 1; width /= 2) {
#pragma unroll
    for (int j = 0; j < width / 2; j++) q[j] = ga(q[2 * j], q[2 * j + 1]);
  }
  return q[0];
}
__device__ __forceinline__ float butterfly(float x, int first) {
#pragma unroll
  for (int d = first; d; d >>= 1) x = ga(x, __shfl_xor_sync(0xffffffffu, x, d));
  return x;
}
template <int NW, int N>
__device__ __forceinline__ float cluster_squares(const float (&x)[N], float* sm, cg::cluster_group cluster, int rank) {
  constexpr int warps_per_cta = NW / 8;
  int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  float sum = butterfly(local_squares(x), 16);
  if (lane == 0 && warp < warps_per_cta) {
#pragma unroll
    for (int dst = 0; dst < 8; dst++) {
      volatile float* p = cluster.map_shared_rank(sm, dst);
      p[rank * warps_per_cta + warp] = sum;
    }
  }
  cluster.sync();
  // Each warp independently reads the original 32 (or 16) warp partials and
  // performs the identical descending XOR butterfly. Separate r1/r2 buffers
  // prevent a fast CTA from overwriting a partial still read by a slow CTA.
  float all = sm[lane & (NW - 1)];
  return butterfly(all, NW / 2);
}

#if GLUE_ROUTER
template <int NW, int N>
__device__ __forceinline__ float cluster_squares_pair(const float (&x)[N], const float (&y)[N], float* sm, float* smy,
                                                      cg::cluster_group cluster, int rank, float& tail) {
  constexpr int warps_per_cta = NW / 8;
  int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  float sx = butterfly(local_squares(x), 16);
  float sy = butterfly(local_squares(y), 16);
  if (lane == 0 && warp < warps_per_cta) {
#pragma unroll
    for (int dst = 0; dst < 8; dst++) {
      volatile float* px = cluster.map_shared_rank(sm, dst);
      volatile float* py = cluster.map_shared_rank(smy, dst);
      px[rank * warps_per_cta + warp] = sx;
      py[rank * warps_per_cta + warp] = sy;
    }
  }
  cluster.sync();
  tail = butterfly(smy[lane & (NW - 1)], NW / 2);
  return butterfly(sm[lane & (NW - 1)], NW / 2);
}
#endif
template <int NW, int K>
#if GLUE_STAS
__device__ __forceinline__ void cluster_sums(float (&q)[K], float4* sm, cg::cluster_group cluster, int rank,
                                             uint32_t xbar) {
#else
__device__ __forceinline__ void cluster_sums(float (&q)[K], float4* sm, cg::cluster_group cluster, int rank) {
#endif
  constexpr int warps_per_cta = NW / 8;
  constexpr int NV = (K + 3) / 4;
  int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
#pragma unroll
  for (int i = 0; i < K; i++) q[i] = butterfly(q[i], 16);
  if (lane == 0 && warp < warps_per_cta) {
#pragma unroll
    for (int v = 0; v < NV; v++) {
      float4 w = make_float4(q[4 * v], 4 * v + 1 < K ? q[4 * v + 1] : 0.f, 4 * v + 2 < K ? q[4 * v + 2] : 0.f,
                             4 * v + 3 < K ? q[4 * v + 3] : 0.f);
#pragma unroll
      for (int dst = 0; dst < 8; dst++) {
#if GLUE_STAS
        uint32_t p, b;
        const uint32_t l = (uint32_t)__cvta_generic_to_shared(sm + v * NW + rank * warps_per_cta + warp);
        asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(p) : "r"(l), "r"(dst));
        asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(b) : "r"(xbar), "r"(dst));
        asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.f32 [%0], {%1, %2, %3, %4}, [%5];"
                     ::"r"(p), "f"(w.x), "f"(w.y), "f"(w.z), "f"(w.w), "r"(b)
                     : "memory");
#else
        float4* p = cluster.map_shared_rank(sm, dst);
        p[v * NW + rank * warps_per_cta + warp] = w;
#endif
      }
    }
  }
#if GLUE_STAS
  // every rank's NV float4s per warp have landed here (the barrier's single arrival and its byte count were set before
  // the kernel's first cluster barrier); acquire at cluster scope pairs with complete_tx's release
  asm volatile("{\n .reg .pred p;\n GSW_%=:\n mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [%0], 0;\n"
               " @!p bra GSW_%=;\n}\n" ::"r"(xbar)
               : "memory");
#else
  cluster.sync();
#endif
#pragma unroll
  for (int v = 0; v < NV; v++) {
    float4 w = sm[v * NW + (lane & (NW - 1))];
    q[4 * v] = butterfly(w.x, NW / 2);
    if (((4 * v) + 1) < K) q[4 * v + 1] = butterfly(w.y, NW / 2);
    if (4 * v + 2 < K) q[4 * v + 2] = butterfly(w.z, NW / 2);
    if (4 * v + 3 < K) q[4 * v + 3] = butterfly(w.w, NW / 2);
  }
}

template <int NW, int N>
#if GLUE_ROUTER
__device__ __forceinline__ float cluster_squares_triple(const float (&x)[N], const float (&y)[N], const float (&z)[N],
                                                        float* sm, float* smy, float* smz, cg::cluster_group cluster,
                                                        int rank, float& tail, float& zsum) {
#else
__device__ __forceinline__ void cluster_sums4(const float (&b)[N], const float (&s)[N], const float (&al)[N],
                                              const float (&be)[N], float* sm, cg::cluster_group cluster, int rank,
                                              float (&q)[4]) {
  // F5: sum b^2, sum (al s)^2, sum (al s)(be b), sum (be b)^2 over the row, one DSMEM exchange of the four warp sums
#endif
  constexpr int warps_per_cta = NW / 8;
  int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
#if GLUE_ROUTER
  float sx = butterfly(local_squares(x), 16);
  float sy = butterfly(local_squares(y), 16);
  float sz = butterfly(local_squares(z), 16);
#else
  float l[4] = {local_squares(b), 0.f, 0.f, 0.f};
#pragma unroll
  for (int k = 0; k < N; k++) {
    float as = gm(al[k], s[k]), bb = gm(be[k], b[k]);
    l[1] = gf(as, as, l[1]);
    l[2] = gf(as, bb, l[2]);
    l[3] = gf(bb, bb, l[3]);
  }
#pragma unroll
  for (int i = 0; i < 4; i++) l[i] = butterfly(l[i], 16);
#endif
  if (lane == 0 && warp < warps_per_cta) {
#pragma unroll
    for (int dst = 0; dst < 8; dst++) {
#if GLUE_ROUTER
      volatile float* px = cluster.map_shared_rank(sm, dst);
      volatile float* py = cluster.map_shared_rank(smy, dst);
      volatile float* pz = cluster.map_shared_rank(smz, dst);
      px[rank * warps_per_cta + warp] = sx;
      py[rank * warps_per_cta + warp] = sy;
      pz[rank * warps_per_cta + warp] = sz;
#else
      volatile float* p = cluster.map_shared_rank(sm, dst);
#pragma unroll
      for (int i = 0; i < 4; i++) p[i * NW + rank * warps_per_cta + warp] = l[i];
#endif
    }
  }
  cluster.sync();
#if GLUE_ROUTER
  tail = butterfly(smy[lane & (NW - 1)], NW / 2);
  zsum = butterfly(smz[lane & (NW - 1)], NW / 2);
  return butterfly(sm[lane & (NW - 1)], NW / 2);
#else
#pragma unroll
  for (int i = 0; i < 4; i++) q[i] = butterfly(sm[i * NW + (lane & (NW - 1))], NW / 2);
#endif
}

template <int V, int NW, bool POST, bool SLICED, bool ROUTER, bool PRELOADED_BASE>
__global__ __launch_bounds__(128) void glue_cluster_boundary(GlueArgs a) {
  constexpr int logical_threads = NW * 32;
  constexpr int per_cta = logical_threads / 8;
  constexpr int N = 8192 / logical_threads;
  constexpr int repeats = N / V;
  cg::cluster_group cluster = cg::this_cluster();
  int rank = cluster.block_rank(), tid = threadIdx.x;
  int logical = rank * per_cta + (tid % per_cta);
  int row = blockIdx.x / 8;
  bool active = tid < per_cta;
#if GLUE_ROUTER
  float s[N], alpha[N], beta[N], pre[N], post[N], b[N], ns[N], a2[N], carry[N], masked[N], nsx[N];
  __shared__ float partials[4][NW];
#else
  float s[N], alpha[N], beta[N], pre[N], post[N], b[N], ns[N];
  __shared__ float partials[2][NW];
  __shared__ float partials4[POST ? 4 * NW : 1];
#endif
#pragma unroll
  for (int r = 0; r < repeats; r++) {
    int col = logical * V + r * logical_threads * V;
    glf<V>(a.stream + row * 8192 + col, s + r * V);
    glf<V>(a.alpha + col, alpha + r * V);
    glf<V>(a.beta + col, beta + r * V);
#if GLUE_ROUTER
    glf<V>(a.alpha2 + col, a2 + r * V);
#endif
    glb<V>(a.pre + col, pre + r * V);
#if GLUE_ROUTER
    if constexpr (POST || ROUTER) glb<V>(a.post + col, post + r * V);   // ROUTER: post = w (pre_ffn_norm)
#else
    if constexpr (POST) glb<V>(a.post + col, post + r * V);
#endif
    if constexpr (SLICED) {
      if (col >= 1024) {
#pragma unroll
        for (int j = 0; j < V; j++) {
          int k = r * V + j;
          // Preserve the signed zero from the post weight as well as the
          // sliced boundary's alpha*stream FMA contraction.
          ns[k] = gf(alpha[k], s[k], gm(beta[k], gm(0.f, post[k])));
        }
#if GLUE_ROUTER
        // carry is stored after both norms are formed.
#else
        if (active) gsf<V>(a.next + row * 8192 + col, ns + r * V);
#endif
      }
    }
  }
#if GLUE_STAS
  // cluster_sums' arrival barrier: one arrival (here) plus every rank's st.async bytes. Initialized and armed before the
  // cluster barrier below, so no remote complete_tx can reach it first
#if GLUE_ROUTER
  constexpr bool XSUMS = GLUE_RMOM;
  constexpr int XK = 7;
#else
  constexpr bool XSUMS = !SLICED && !POST;
  constexpr int XK = 4;
#endif
  __shared__ __align__(8) unsigned long long xbar_s;
  const uint32_t xbar = (uint32_t)__cvta_generic_to_shared(&xbar_s);
  if (XSUMS && tid == 0) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(xbar) : "memory");
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(xbar), "r"((XK + 3) / 4 * NW * 16)
                 : "memory");
  }
#endif
  // DSMEM destinations must exist before the first remote store. All work
  // above this barrier and the dependency wait is independent of the branch.
  cluster.sync();
  asm volatile("griddepcontrol.wait; griddepcontrol.launch_dependents;" ::: "memory");
#pragma unroll
  for (int r = 0; r < repeats; r++) {
    int col = logical * V + r * logical_threads * V;
    if constexpr (SLICED) {
#pragma unroll
      for (int j = 0; j < V; j++) b[r * V + j] = 0.f;
      if (col < 1024) {
#pragma unroll
        for (int sk = 0; sk < 8; sk++) {
          float part[V];
          glf<V>(a.part + (sk * a.rows + row) * 1024 + col, part);
#pragma unroll
          for (int j = 0; j < V; j++) b[r * V + j] = ga(b[r * V + j], part[j]);
        }
      }
#pragma unroll
      for (int j = 0; j < V; j++) b[r * V + j] = gb(b[r * V + j]);
    } else {
      glb<V>(a.branch + row * 8192 + col, b + r * V);
    }
    if constexpr (POST) {
#pragma unroll
      for (int j = 0; j < V; j++) b[r * V + j] = gb(gm(b[r * V + j], post[r * V + j]));
    }
  }
#if GLUE_ROUTER && GLUE_RMOM
  __shared__ float4 sums8[2 * NW];
  float q[7] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#pragma unroll
  for (int r = 0; r < repeats; r++) {
    int col = logical * V + r * logical_threads * V;
#pragma unroll
    for (int j = 0; j < V; j++) {
      int k = r * V + j;
      float as = gm(alpha[k], s[k]), bb = gm(beta[k], b[k]);
      q[0] = gf(b[k], b[k], q[0]);
      q[1] = gf(as, as, q[1]);
      q[2] = gf(as, bb, q[2]);
      q[3] = gf(bb, bb, q[3]);
      if (col >= 1024) {   // the carry's columns: a2 * ns
        float x = gm(a2[k], as), y = gm(a2[k], bb);
        q[4] = gf(x, x, q[4]);
        q[5] = gf(x, y, q[5]);
        q[6] = gf(y, y, q[6]);
      }
    }
  }
#if GLUE_STAS
  cluster_sums<NW, 7>(q, sums8, cluster, rank, xbar);
#else
  cluster_sums<NW, 7>(q, sums8, cluster, rank);
#endif
  float r1 = gr(q[0], 1.e-8f);
#elif GLUE_ROUTER
  float r1 = gr(cluster_squares<NW>(b, partials[0], cluster, rank), 1.e-8f);
#else
  float r1, r2;
  if constexpr (!SLICED && !POST) {
    __shared__ float4 sums4[NW];
    float q[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int k = 0; k < N; k++) {
      const float as = alpha[k] * s[k], bb = beta[k] * b[k];
      q[0] = fmaf(b[k], b[k], q[0]);
      q[1] = fmaf(as, as, q[1]);
      q[2] = fmaf(as, bb, q[2]);
      q[3] = fmaf(bb, bb, q[3]);
    }
#if GLUE_STAS
    cluster_sums<NW, 4>(q, sums4, cluster, rank, xbar);
#else
    cluster_sums<NW, 4>(q, sums4, cluster, rank);
#endif
    r1 = gr(q[0], 1.e-8f);
    r2 = gr(fmaf(r1 * r1, q[3], fmaf(2.f * r1, q[2], q[1])), 1.e-5f);
  } else {
    float q4[4] = {0.f, 0.f, 0.f, 0.f};
    if constexpr (POST) cluster_sums4<NW>(b, s, alpha, beta, partials4, cluster, rank, q4);
    r1 = POST ? gr(q4[0], 1.e-8f) : gr(cluster_squares<NW>(b, partials[0], cluster, rank), 1.e-8f);
    if constexpr (POST) r2 = gr(ga(q4[1], gf(gm(2.f, r1), q4[2], gm(gm(r1, r1), q4[3]))), 1.e-5f);
  }
#endif
#pragma unroll
  for (int r = 0; r < repeats; r++) {
    int col = logical * V + r * logical_threads * V;
    if (!SLICED || col < 1024) {
#pragma unroll
      for (int j = 0; j < V; j++) {
        int k = r * V + j;
        float bn = gb(gm(b[k], r1));
#if GLUE_ROUTER
        if constexpr (ROUTER) {   // the router's FP32 copy: no BF16 rounding of r1 * P
          float bx = gm(b[k], r1);
          nsx[k] = (SLICED || (PRELOADED_BASE && col < 4096)) ? gf(alpha[k], s[k], gm(beta[k], bx))
                                                              : gf(beta[k], bx, gm(alpha[k], s[k]));
        }
#endif
        if (SLICED || (PRELOADED_BASE && col < 4096))
          ns[k] = gf(alpha[k], s[k], gm(beta[k], bn));
        else
          ns[k] = gf(beta[k], bn, gm(alpha[k], s[k]));
      }
#if GLUE_ROUTER
      // carry is stored after both norms are formed.
#else
      if (active) gsf<V>(a.next + row * 8192 + col, ns + r * V);
#endif
    }
  }
#if GLUE_ROUTER
#pragma unroll
  for (int r = 0; r < repeats; r++) {
    int col = logical * V + r * logical_threads * V;
#pragma unroll
    for (int j = 0; j < V; j++) {
      int k = r * V + j;
      carry[k] = gm(a2[k], ns[k]);
      masked[k] = (col + j >= 1024) ? carry[k] : 0.f;
    }
  }
  float tail;
  float r2, r2x = 0.f;
#if GLUE_RMOM
  r2 = r2x = gr(gf(gm(r1, r1), q[3], gf(gm(2.f, r1), q[2], q[1])), 1.e-5f);
  tail = gf(gm(r1, r1), q[6], gf(gm(2.f, r1), q[5], q[4]));
#else
  if constexpr (ROUTER) {   // the router copy's sum of squares rides the same cluster exchange
    float zs;
    r2 = gr(cluster_squares_triple<NW>(ns, masked, nsx, partials[1], partials[2], partials[3], cluster, rank, tail, zs),
            1.e-5f);
    r2x = gr(zs, 1.e-5f);
  } else {
    r2 = gr(cluster_squares_pair<NW>(ns, masked, partials[1], partials[2], cluster, rank, tail), 1.e-5f);
  }
#endif
  if (rank == 0 && tid == 0) a.tail[row] = tail;
#else
  if constexpr (SLICED) r2 = gr(cluster_squares<NW>(ns, partials[1], cluster, rank), 1.e-5f);
#endif
#pragma unroll
  for (int r = 0; r < repeats; r++) {
    int col = logical * V + r * logical_threads * V;
    float h[V];
#pragma unroll
    for (int j = 0; j < V; j++) h[j] = gm(gm(ns[r * V + j], r2), pre[r * V + j]);
    if (active) {
      gsb<V>(a.out + row * 8192 + col, h);
#if GLUE_ROUTER
      gsf<V>(a.next + row * 8192 + col, carry + r * V);
      if constexpr (ROUTER) {
        float hx[V];
#pragma unroll
        for (int j = 0; j < V; j++) hx[j] = gm(gm(nsx[r * V + j], r2x), ga(1.f, post[r * V + j]));
        gsf<V>(a.router + row * 8192 + col, hx);
      }
#else
      if constexpr (ROUTER) gsf<V>(a.router + row * 8192 + col, h);
#endif
    }
  }
}

template <typename F>
void glue_launch(GlueArgs a, F fn, cudaStream_t stream) {
  cudaLaunchConfig_t cfg{};
  cfg.gridDim = dim3(a.rows * 8);
  cfg.blockDim = dim3(128);
  cfg.stream = stream;
  cudaLaunchAttribute attrs[2]{};
  attrs[0].id = cudaLaunchAttributeClusterDimension;
  attrs[0].val.clusterDim = {8, 1, 1};
  attrs[1].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[1].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = attrs;
  cfg.numAttrs = 2;
  auto e = cudaLaunchKernelEx(&cfg, fn, a);
}
// graft: boundary 1 (attention residual), BF16 [rows, 8192] branch, rows <= 16, optional FP32 router copy.
#if GLUE_ROUTER
void b1(TV branch, TV stream, TV pre, TV alpha, TV beta, TV out, TV next, TV alpha2, TV tail, TV router,
        bool write_router, TV w) {
#else
void b1(TV branch, TV stream, TV pre, TV alpha, TV beta, TV out, TV next, TV router, bool write_router) {
#endif
  GlueArgs a{(const BF*)branch.data_ptr(),
#if GLUE_ROUTER
             (const BF*)w.data_ptr(),
#else
             (const BF*)pre.data_ptr(),
#endif
             (const BF*)pre.data_ptr(),
             (const float*)branch.data_ptr(),
             (const float*)stream.data_ptr(),
             (const float*)alpha.data_ptr(),
             (const float*)beta.data_ptr(),
             (BF*)out.data_ptr(),
             (float*)next.data_ptr(),
             (float*)router.data_ptr(),
#if GLUE_ROUTER
             (int)stream.size(0),
             (const float*)alpha2.data_ptr(),
             (float*)tail.data_ptr()};
#else
             (int)stream.size(0)};
#endif
  auto s = (cudaStream_t)TVMFFIEnvGetStream(stream.device().device_type, stream.device().device_id);
  if (write_router) {
    if (a.rows > 1)
      glue_launch(a, glue_cluster_boundary<4, 32, false, false, true, true>, s);
    else
      glue_launch(a, glue_cluster_boundary<4, 32, false, false, true, false>, s);
  } else {
    if (a.rows > 1)
      glue_launch(a, glue_cluster_boundary<4, 32, false, false, false, true>, s);
    else
      glue_launch(a, glue_cluster_boundary<4, 32, false, false, false, false>, s);
  }
}
#if !GLUE_ROUTER
// graft: boundary 2 (FFN residual), full-width BF16 [rows, 8192] unpooler output, rows <= 16 (our kind 1).
void b2c(TV branch, TV stream, TV post, TV pre, TV alpha, TV beta, TV out, TV next) {
  GlueArgs a{(const BF*)branch.data_ptr(),
             (const BF*)post.data_ptr(),
             (const BF*)pre.data_ptr(),
             (const float*)branch.data_ptr(),
             (const float*)stream.data_ptr(),
             (const float*)alpha.data_ptr(),
             (const float*)beta.data_ptr(),
             (BF*)out.data_ptr(),
             (float*)next.data_ptr(),
             (float*)next.data_ptr(),
             (int)stream.size(0)};
  auto s = (cudaStream_t)TVMFFIEnvGetStream(stream.device().device_type, stream.device().device_id);
  if (a.rows > 1)
    glue_launch(a, glue_cluster_boundary<4, 32, true, false, false, true>, s);
  else
    glue_launch(a, glue_cluster_boundary<4, 32, true, false, false, false>, s);
}
#endif
"""

# the experts builds' switches (the rest are 0): MOE_WGS the warpgroup-MMA phases (M > 16); MOE_B4FIN the column-parallel
# final pass (16 < M <= 32); MOE_TINY / TINY2 / MID the skinny builds for M <= 4 / 8 / 16; MOE_DF phase B reading x
# behind per-value data flags; MOE_DQM the DQM code layout; MOE_TUNED_* the tuned scored shapes (see the MoE source)
MOE_SC6S = 1   # 1: the skinny unit copies' scales as three six-bit planes (and -DMOE_SC6S=1 in every skinny build)
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
# the study's switches for this trial
BUILDS['moe_d1s'] = (BUILDS['moe_d1s'][0], BUILDS['moe_d1s'][1], BUILDS['moe_d1s'][2] + ['-DMOE_RAFN=1'])   # raw
BUILDS['moe_d1f'] = (BUILDS['moe_d1f'][0], BUILDS['moe_d1f'][1], BUILDS['moe_d1f'][2] + ['-DMOE_RAFN=1'])   # raw
BUILDS['moe_v1s'] = (BUILDS['moe_v1s'][0], BUILDS['moe_v1s'][1], BUILDS['moe_v1s'][2] + ['-DMOE_RAFN=1'])   # raw
BUILDS['moe_v1f'] = (BUILDS['moe_v1f'][0], BUILDS['moe_v1f'][1], BUILDS['moe_v1f'][2] + ['-DMOE_RAFN=1'])   # raw
BUILDS['moe_d16s'] = (BUILDS['moe_d16s'][0], BUILDS['moe_d16s'][1], BUILDS['moe_d16s'][2] + ['-DMOE_RAFN=1'])   # raw
BUILDS['moe_d16f'] = (BUILDS['moe_d16f'][0], BUILDS['moe_d16f'][1], BUILDS['moe_d16f'][2] + ['-DMOE_RAFN=1'])   # raw
BUILDS['moe_d1s'] = (BUILDS['moe_d1s'][0], BUILDS['moe_d1s'][1], BUILDS['moe_d1s'][2] + ['-DMOE_XSLOT=2'])   # raw
BUILDS['moe_d1f'] = (BUILDS['moe_d1f'][0], BUILDS['moe_d1f'][1], BUILDS['moe_d1f'][2] + ['-DMOE_XSLOT=2'])   # raw
BUILDS['moe_d1s'] = (BUILDS['moe_d1s'][0], BUILDS['moe_d1s'][1], BUILDS['moe_d1s'][2] + ['-DMOE_HCRED=3'])   # raw
BUILDS['moe_d1f'] = (BUILDS['moe_d1f'][0], BUILDS['moe_d1f'][1], BUILDS['moe_d1f'][2] + ['-DMOE_HCRED=3'])   # raw
BUILDS['moe_v1s'] = (BUILDS['moe_v1s'][0], BUILDS['moe_v1s'][1], BUILDS['moe_v1s'][2] + ['-DMOE_HCRED=3'])   # raw
BUILDS['moe_v1f'] = (BUILDS['moe_v1f'][0], BUILDS['moe_v1f'][1], BUILDS['moe_v1f'][2] + ['-DMOE_HCRED=3'])   # raw
BUILDS['moe_d16s'] = (BUILDS['moe_d16s'][0], BUILDS['moe_d16s'][1], BUILDS['moe_d16s'][2] + ['-DMOE_HCRED=3'])   # raw
BUILDS['moe_d16f'] = (BUILDS['moe_d16f'][0], BUILDS['moe_d16f'][1], BUILDS['moe_d16f'][2] + ['-DMOE_HCRED=3'])   # raw
BUILDS['moe_big'] = (BUILDS['moe_big'][0], BUILDS['moe_big'][1], BUILDS['moe_big'][2] + ['-DMOE_PLD=1'])   # raw
BUILDS['moe_d16f'] = (BUILDS['moe_d16f'][0], BUILDS['moe_d16f'][1], BUILDS['moe_d16f'][2] + ['-DMOE_PLD=1'])   # raw
BUILDS['moe_d16s'] = (BUILDS['moe_d16s'][0], BUILDS['moe_d16s'][1], BUILDS['moe_d16s'][2] + ['-DMOE_PLD=1'])   # raw
BUILDS['moe_d1f'] = (BUILDS['moe_d1f'][0], BUILDS['moe_d1f'][1], BUILDS['moe_d1f'][2] + ['-DMOE_PLD=1'])   # raw
BUILDS['moe_d1s'] = (BUILDS['moe_d1s'][0], BUILDS['moe_d1s'][1], BUILDS['moe_d1s'][2] + ['-DMOE_PLD=1'])   # raw
BUILDS['moe_d32f'] = (BUILDS['moe_d32f'][0], BUILDS['moe_d32f'][1], BUILDS['moe_d32f'][2] + ['-DMOE_PLD=1'])   # raw
BUILDS['moe_d32s'] = (BUILDS['moe_d32s'][0], BUILDS['moe_d32s'][1], BUILDS['moe_d32s'][2] + ['-DMOE_PLD=1'])   # raw
BUILDS['moe_v1f'] = (BUILDS['moe_v1f'][0], BUILDS['moe_v1f'][1], BUILDS['moe_v1f'][2] + ['-DMOE_PLD=1'])   # raw
BUILDS['moe_v1s'] = (BUILDS['moe_v1s'][0], BUILDS['moe_v1s'][1], BUILDS['moe_v1s'][2] + ['-DMOE_PLD=1'])   # raw
BUILDS['moe_d1s'] = (BUILDS['moe_d1s'][0], BUILDS['moe_d1s'][1], BUILDS['moe_d1s'][2] + ['-DMOE_IWARP=1'])   # raw
BUILDS['moe_d1f'] = (BUILDS['moe_d1f'][0], BUILDS['moe_d1f'][1], BUILDS['moe_d1f'][2] + ['-DMOE_IWARP=1'])   # raw
BUILDS['moe_v1f'] = (BUILDS['moe_v1f'][0], BUILDS['moe_v1f'][1], BUILDS['moe_v1f'][2] + ['-DMOE_IWARP=1'])   # raw
BUILDS['moe_v1s'] = (BUILDS['moe_v1s'][0], BUILDS['moe_v1s'][1], BUILDS['moe_v1s'][2] + ['-DMOE_IWARP=1'])   # raw
BUILDS['moe_d16s'] = (BUILDS['moe_d16s'][0], BUILDS['moe_d16s'][1], BUILDS['moe_d16s'][2] + ['-DMOE_IWARP=1'])   # raw
BUILDS['moe_d16f'] = (BUILDS['moe_d16f'][0], BUILDS['moe_d16f'][1], BUILDS['moe_d16f'][2] + ['-DMOE_IWARP=1'])   # raw
BUILDS['moe_d1s'] = (BUILDS['moe_d1s'][0], BUILDS['moe_d1s'][1], BUILDS['moe_d1s'][2] + ['-DMOE_ITAIL=1'])   # raw
BUILDS['moe_d1f'] = (BUILDS['moe_d1f'][0], BUILDS['moe_d1f'][1], BUILDS['moe_d1f'][2] + ['-DMOE_ITAIL=1'])   # raw
BUILDS['moe_v1s'] = (BUILDS['moe_v1s'][0], BUILDS['moe_v1s'][1], BUILDS['moe_v1s'][2] + ['-DMOE_ITAIL=1'])   # raw
BUILDS['moe_v1f'] = (BUILDS['moe_v1f'][0], BUILDS['moe_v1f'][1], BUILDS['moe_v1f'][2] + ['-DMOE_ITAIL=1'])   # raw
BUILDS['moe_d16s'] = (BUILDS['moe_d16s'][0], BUILDS['moe_d16s'][1], BUILDS['moe_d16s'][2] + ['-DMOE_ITAIL=1'])   # raw
BUILDS['moe_d16f'] = (BUILDS['moe_d16f'][0], BUILDS['moe_d16f'][1], BUILDS['moe_d16f'][2] + ['-DMOE_ITAIL=1'])   # raw
BUILDS['moe_d16s'] = (BUILDS['moe_d16s'][0], BUILDS['moe_d16s'][1], BUILDS['moe_d16s'][2] + ['-DMOE_ITAIL=0'])   # raw
BUILDS['moe_d16f'] = (BUILDS['moe_d16f'][0], BUILDS['moe_d16f'][1], BUILDS['moe_d16f'][2] + ['-DMOE_ITAIL=0'])   # raw
BUILDS['moe_d32s'] = (BUILDS['moe_d32s'][0], BUILDS['moe_d32s'][1], BUILDS['moe_d32s'][2] + ['-DMOE_CA=9'])   # raw
BUILDS['sk2'] = (BUILDS['sk2'][0], BUILDS['sk2'][1], BUILDS['sk2'][2] + ['-DSK2_RRED=1'])   # raw
BUILDS['moe_d1s'] = (BUILDS['moe_d1s'][0], BUILDS['moe_d1s'][1], BUILDS['moe_d1s'][2] + ['-DMOE_ROUTE1=1'])   # raw
BUILDS['moe_d1f'] = (BUILDS['moe_d1f'][0], BUILDS['moe_d1f'][1], BUILDS['moe_d1f'][2] + ['-DMOE_ROUTE1=1'])   # raw
BUILDS['sk2'] = (BUILDS['sk2'][0], BUILDS['sk2'][1], BUILDS['sk2'][2] + ['-DSK2_ESPL=1'])   # raw
BUILDS['moe_d1s'] = (BUILDS['moe_d1s'][0], BUILDS['moe_d1s'][1], BUILDS['moe_d1s'][2] + ['-DMOE_FNSBAL=1'])   # raw
BUILDS['moe_d1f'] = (BUILDS['moe_d1f'][0], BUILDS['moe_d1f'][1], BUILDS['moe_d1f'][2] + ['-DMOE_FNSBAL=1'])   # raw
BUILDS['moe_v1s'] = (BUILDS['moe_v1s'][0], BUILDS['moe_v1s'][1], BUILDS['moe_v1s'][2] + ['-DMOE_FNSBAL=1'])   # raw
BUILDS['moe_v1f'] = (BUILDS['moe_v1f'][0], BUILDS['moe_v1f'][1], BUILDS['moe_v1f'][2] + ['-DMOE_FNSBAL=1'])   # raw
BUILDS['attention'] = (BUILDS['attention'][0], BUILDS['attention'][1], BUILDS['attention'][2] + ['-DATT_EPOCH_ASYNC=1'])   # raw
BUILDS['moe_d1s'] = (BUILDS['moe_d1s'][0], BUILDS['moe_d1s'][1], BUILDS['moe_d1s'][2] + ['-DMOE_ROLEUNI=1'])   # raw
BUILDS['moe_d1f'] = (BUILDS['moe_d1f'][0], BUILDS['moe_d1f'][1], BUILDS['moe_d1f'][2] + ['-DMOE_ROLEUNI=1'])   # raw
BUILDS['moe_v1s'] = (BUILDS['moe_v1s'][0], BUILDS['moe_v1s'][1], BUILDS['moe_v1s'][2] + ['-DMOE_ROLEUNI=1'])   # raw
BUILDS['moe_v1f'] = (BUILDS['moe_v1f'][0], BUILDS['moe_v1f'][1], BUILDS['moe_v1f'][2] + ['-DMOE_ROLEUNI=1'])   # raw
BUILDS['moe_d16s'] = (BUILDS['moe_d16s'][0], BUILDS['moe_d16s'][1], BUILDS['moe_d16s'][2] + ['-DMOE_ROLEUNI=1'])   # raw
BUILDS['moe_d16f'] = (BUILDS['moe_d16f'][0], BUILDS['moe_d16f'][1], BUILDS['moe_d16f'][2] + ['-DMOE_ROLEUNI=1'])   # raw
BUILDS['glue_router'] = (BUILDS['glue_router'][0], BUILDS['glue_router'][1], BUILDS['glue_router'][2] + ['-DGLUE_RMOM=1'])   # raw
BUILDS['attention'] = (BUILDS['attention'][0], BUILDS['attention'][1], BUILDS['attention'][2] + ['-DATT_V8_QDIR=1'])   # raw
BUILDS['attention'] = (BUILDS['attention'][0], BUILDS['attention'][1], BUILDS['attention'][2] + ['-DATT_V8C_BF16RED=1'])   # raw
BUILDS['attention'] = (BUILDS['attention'][0], BUILDS['attention'][1], BUILDS['attention'][2] + ['-DATT_VERIFY_WST=1'])   # raw
BUILDS['attention'] = (BUILDS['attention'][0], BUILDS['attention'][1], BUILDS['attention'][2] + ['-DATT_V8_SCHEXP=1'])   # raw
BUILDS['glue'] = (BUILDS['glue'][0], BUILDS['glue'][1], BUILDS['glue'][2] + ['-DGLUE_B1_ROWS=32'])   # raw
BUILDS['glue'] = (BUILDS['glue'][0], BUILDS['glue'][1], BUILDS['glue'][2] + ['-DGLUE_B2_ROWS=8'])   # raw
BUILDS['moe_d32s'] = (BUILDS['moe_d32s'][0], BUILDS['moe_d32s'][1], BUILDS['moe_d32s'][2] + ['-DMOE_CA=2'])   # raw
BUILDS['sk2'] = (BUILDS['sk2'][0], BUILDS['sk2'][1], BUILDS['sk2'][2] + ['-DSK2_RSUM=1'])   # raw
BUILDS['sk2_gle'] = (BUILDS['sk2_gle'][0], BUILDS['sk2_gle'][1], BUILDS['sk2_gle'][2] + ['-DSK2_RSUM=1'])   # raw
BUILDS['moe_big'] = (BUILDS['moe_big'][0], BUILDS['moe_big'][1], BUILDS['moe_big'][2] + ['-DMOE_YKEEP=1'])   # raw
BUILDS['attention'] = (BUILDS['attention'][0], BUILDS['attention'][1], BUILDS['attention'][2] + ['-DATT_V1_GATE_EARLY=1'])   # raw
BUILDS['moe_d32f'] = (BUILDS['moe_d32f'][0], BUILDS['moe_d32f'][1], BUILDS['moe_d32f'][2] + ['-DMOE_CA=7'])   # raw
BUILDS['attention'] = (BUILDS['attention'][0], BUILDS['attention'][1], BUILDS['attention'][2] + ['-DATT_V1_GATE_EARLY=2'])   # raw
BUILDS['attention'] = (BUILDS['attention'][0], BUILDS['attention'][1], BUILDS['attention'][2] + ['-DATT_V8F_SCHEXP=1'])   # raw
BUILDS['sk2'] = (BUILDS['sk2'][0], BUILDS['sk2'][1], BUILDS['sk2'][2] + ['-DSK2_XR=3'])   # raw
BUILDS['sk2'] = (BUILDS['sk2'][0], BUILDS['sk2'][1], BUILDS['sk2'][2] + ['-DSK2_XRM=2'])   # raw
BUILDS['attention'] = (BUILDS['attention'][0], BUILDS['attention'][1], BUILDS['attention'][2] + ['-DATT_V1_POLL_NS=0'])   # raw
BUILDS['attention'] = (BUILDS['attention'][0], BUILDS['attention'][1], BUILDS['attention'][2] + ['-DATT_V1_BF16RED=1'])   # raw
BUILDS['glue'] = (BUILDS['glue'][0], BUILDS['glue'][1], BUILDS['glue'][2] + ['-DGLUE_STAS=1'])   # raw
BUILDS['glue_router'] = (BUILDS['glue_router'][0], BUILDS['glue_router'][1], BUILDS['glue_router'][2] + ['-DGLUE_STAS=1'])   # raw
BUILDS['attention'] = (BUILDS['attention'][0], BUILDS['attention'][1], BUILDS['attention'][2] + ['-DATT_V1_CTA_BARRIER=1'])   # raw
BUILDS['attention'] = (BUILDS['attention'][0], BUILDS['attention'][1], BUILDS['attention'][2] + ['-DATT_V8_OVL=1'])   # raw
BUILDS['attention'] = (BUILDS['attention'][0], BUILDS['attention'][1], BUILDS['attention'][2] + ['-DATT_V1_RELAXED_POLL=1'])   # raw
for _c, _sw in MOE_SWITCHES.items():   # dfpoll
    if set(_sw) & set(['MOE_DF']):
        _s, _f, _x = BUILDS["moe_" + _c]
        BUILDS["moe_" + _c] = (_s, _f, _x + ['-DMOE_DFPOLL=1'])
BUILDS['attention'] = (BUILDS['attention'][0], BUILDS['attention'][1], BUILDS['attention'][2] + ['-Xptxas', '--def-load-cache=ca'])   # dlc_att
for _c, _sw in MOE_SWITCHES.items():   # dlc_moes
    if set(_sw) & set(['MOE_MID', 'MOE_TINY', 'MOE_TINY2']):
        _s, _f, _x = BUILDS["moe_" + _c]
        BUILDS["moe_" + _c] = (_s, _f, _x + ['-Xptxas', '--def-load-cache=ca'])
BUILDS['sk2'] = (BUILDS['sk2'][0], BUILDS['sk2'][1], BUILDS['sk2'][2] + ['-Xptxas', '--def-load-cache=ca'])   # dlc_sk2
BUILDS['sk2_narrow'] = (BUILDS['sk2_narrow'][0], BUILDS['sk2_narrow'][1], BUILDS['sk2_narrow'][2] + ['-Xptxas', '--def-load-cache=ca'])   # dlc_sk2
BUILDS['sk2_gle'] = (BUILDS['sk2_gle'][0], BUILDS['sk2_gle'][1], BUILDS['sk2_gle'][2] + ['-Xptxas', '--def-load-cache=ca'])   # dlc_sk2
BUILDS['sk2'] = (BUILDS['sk2'][0], BUILDS['sk2'][1], BUILDS['sk2'][2] + ['-DSK2_P05=1'])   # p05
BUILDS['attention'] = (BUILDS['attention'][0], BUILDS['attention'][1], BUILDS['attention'][2] + ['-Xptxas', '--register-usage-level=7'])   # rul_att
for _c, _sw in MOE_SWITCHES.items():   # rul_moew
    if set(_sw) & set(['MOE_WGS']):
        _s, _f, _x = BUILDS["moe_" + _c]
        BUILDS["moe_" + _c] = (_s, _f, _x + ['-Xptxas', '--register-usage-level=7'])
BUILDS['sk2'] = (BUILDS['sk2'][0], BUILDS['sk2'][1], BUILDS['sk2'][2] + ['-Xptxas', '--register-usage-level=7'])   # rul_sk2
BUILDS['sk2_narrow'] = (BUILDS['sk2_narrow'][0], BUILDS['sk2_narrow'][1], BUILDS['sk2_narrow'][2] + ['-Xptxas', '--register-usage-level=7'])   # rul_sk2
BUILDS['sk2_gle'] = (BUILDS['sk2_gle'][0], BUILDS['sk2_gle'][1], BUILDS['sk2_gle'][2] + ['-Xptxas', '--register-usage-level=7'])   # rul_sk2
BUILDS['sk2'] = (BUILDS['sk2'][0], BUILDS['sk2'][1], BUILDS['sk2'][2] + ['-DSK2_BF16P=1'])   # sk2bf16p
BUILDS['sk2_narrow'] = (BUILDS['sk2_narrow'][0], BUILDS['sk2_narrow'][1], BUILDS['sk2_narrow'][2] + ['-DSK2_BF16P=1'])   # sk2bf16p
BUILDS['sk2_gle'] = (BUILDS['sk2_gle'][0], BUILDS['sk2_gle'][1], BUILDS['sk2_gle'][2] + ['-DSK2_BF16P=1'])   # sk2bf16p
for _c, _sw in MOE_SWITCHES.items():   # ts2
    if set(_sw) & set(['MOE_MID', 'MOE_TINY', 'MOE_TINY2']):
        _s, _f, _x = BUILDS["moe_" + _c]
        BUILDS["moe_" + _c] = (_s, _f, _x + ['-DMOE_TANHSILU=2'])
for _c, _sw in MOE_SWITCHES.items():   # wsblk_m1
    if set(_sw) & set(['MOE_MID']):
        _s, _f, _x = BUILDS["moe_" + _c]
        BUILDS["moe_" + _c] = (_s, _f, _x + ['-DMOE_WSBLK=1'])
for _c, _sw in MOE_SWITCHES.items():   # wsblk_t2
    if set(_sw) & set(['MOE_TINY2']):
        _s, _f, _x = BUILDS["moe_" + _c]
        BUILDS["moe_" + _c] = (_s, _f, _x + ['-DMOE_WSBLK=2'])

_MODULES = {}
_BUILD_ALIASES = {}
CASE_OVERRIDES = {(32, 1, True): {'flags': ((('MOE_WGS',), ('-Xptxas', '--register-usage-level=3')), (('sk2', 'sk2_narrow', 'sk2_gle'), ('-Xptxas', '--register-usage-level=5'))), 'attention_plan': {}, 'switches': {'rul_moew': 3, 'rul_sk2': 5}}, (32, 1, False): {'flags': ((('sk2', 'sk2_narrow', 'sk2_gle'), ('-Xptxas', '--register-usage-level=5')), (('sk2', 'sk2_narrow', 'sk2_gle'), ('-DSK2_BF16P=0',))), 'attention_plan': {}, 'switches': {'rul_sk2': 5, 'sk2bf16p': 0}}, (256, 8, True): {'flags': ((('rq128', 'rq256'), ('-Xptxas', '--register-usage-level=3')),), 'attention_plan': {}, 'switches': {'rul_rq': 3}}, (256, 8, False): {'flags': ((('attention',), ('-Xptxas', '--register-usage-level=7')),), 'attention_plan': {}, 'switches': {'rul_att': 7}}, (1, 1, False): {'flags': ((('sk2', 'sk2_narrow', 'sk2_gle'), ('-Xptxas', '--register-usage-level=5')), (('MOE_TINY', 'MOE_TINY2', 'MOE_MID'), ('-DMOE_TANHSILU=0',))), 'attention_plan': {}, 'switches': {'rul_sk2': 5, 'ts2': 0}}, (16, 1, True): {'flags': ((('MOE_TINY', 'MOE_TINY2', 'MOE_MID'), ('-DMOE_BREL=1',)), (('attention',), ('-Xptxas', '--def-load-cache=cg')), (('sk2', 'sk2_narrow', 'sk2_gle'), ('-Xptxas', '--def-load-cache=cg')), (('MOE_DF',), ('-DMOE_SPOLL2=1',))), 'attention_plan': {}, 'switches': {'dlc_sk2': 'cg', 'dlc_att': 'cg', 'spoll2': 1, 'brel': 1}}, (8, 8, False): {'flags': ((('MOE_TINY', 'MOE_TINY2', 'MOE_MID'), ('-Xptxas', '--def-load-cache=ca')), (('sk2', 'sk2_narrow', 'sk2_gle'), ('-Xptxas', '--register-usage-level=3')), (('sk2', 'sk2_narrow', 'sk2_gle'), ('-DSK2_BF16P=0',))), 'attention_plan': {}, 'switches': {'rul_sk2': 3, 'dlc_moes': 'ca', 'sk2bf16p': 0}}, (16, 1, False): {'flags': ((('MOE_TINY', 'MOE_TINY2', 'MOE_MID'), ('-DMOE_BREL=1',)), (('sk2',), ('-DSK2_P05=0',)), (('sk2', 'sk2_narrow', 'sk2_gle'), ('-DSK2_C7=1',))), 'attention_plan': {}, 'switches': {'sk2c7': 1, 'p05': 0, 'brel': 1}}, (1, 1, True): {'flags': ((('moe_d1s', 'moe_d1f'), ('-DMOE_XSLOT=0',)),), 'attention_plan': {}, 'switches': {'xslot_t1': 0}}, (128, 8, False): {'flags': ((('rq128', 'rq256'), ('-DRQ_WPRE=1',)), (('rq128',), ('-DRQ_STAGES=6',)), (('rq128', 'rq256'), ('-Xptxas', '--register-usage-level=3'))), 'attention_plan': {}, 'switches': {'rul_rq': 3, 'rq_wpre': 1, 'rqst128': 6}}, (8, 8, True): {'flags': ((('sk2', 'sk2_narrow', 'sk2_gle'), ('-Xptxas', '--def-load-cache=none')),), 'attention_plan': {}, 'switches': {'dlc_sk2': 'none'}}}   # assembly-time per-case flags and attention-plan overrides

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
RQ_O = 1   # 1: at T = 128 the o projection streams 12-bit through rq128 (KS 2) instead of the BF16 Lt GEMM


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
