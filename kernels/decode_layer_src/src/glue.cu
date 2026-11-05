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
