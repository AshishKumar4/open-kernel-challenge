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
