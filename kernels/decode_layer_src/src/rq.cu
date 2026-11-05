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
