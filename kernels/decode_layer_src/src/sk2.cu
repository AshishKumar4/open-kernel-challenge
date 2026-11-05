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
@@SK2_CASES@@
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
