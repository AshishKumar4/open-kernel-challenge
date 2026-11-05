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
