// Lossless NVFP4 -> BF16 decode layouts with one right-moving source: r_i = M & (x ^ y), x, y in {q, q >> c, q << k_j}
// (q >> c: SHF.R on the ALU pipe, or IMAD.HI.U32 q * 2^(32-c) on the FMA pipe; q << k: IMAD.SHL). Exhaustive over c in
// 1..31 and up to 3 left shifts in 1..31, every source used, registers a multiset of singles / XOR pairs; invertibility
// over GF(2). Left shifts only can never work: the 28 output bits below 25 would depend on the 25 stored bits 0..24
// alone (multiplication moves information only upward), so at least one right-moving source is required.
//   cc -O2 -o fp4_layout2 algebra/fp4_layout2.c && ./fp4_layout2
#include <stdint.h>
#include <stdio.h>
static const uint32_t M = 0x81C081C0u;
static int ob[8];
static int rank32(uint32_t* rows) {
  int r = 0;
  for (int bit = 31; bit >= 0 && r < 32; --bit) {
    int p = -1;
    for (int i = r; i < 32; ++i) if ((rows[i] >> bit) & 1) { p = i; break; }
    if (p < 0) continue;
    uint32_t t = rows[p]; rows[p] = rows[r]; rows[r] = t;
    for (int i = 0; i < 32; ++i) if (i != r && ((rows[i] >> bit) & 1)) rows[i] ^= rows[r];
    ++r;
  }
  return r;
}
// source s: shift amount (positive = left, negative = right), as the dependency of output bit t
static uint32_t dep(int sh, int t) { int j = t - sh; return (j >= 0 && j < 32) ? (1u << j) : 0u; }
int main(void) {
  int n = 0;
  for (int t = 0; t < 32; ++t) if ((M >> t) & 1) ob[n++] = t;
  for (int nl = 0; nl <= 3; ++nl) {
    long found = 0;
    int c[3];
    for (int i = 0; i < nl; ++i) c[i] = i + 1;
    while (1) {
      for (int rs = 1; rs <= 31; ++rs) {
        int src[5], ns = 0;
        src[ns++] = 0; src[ns++] = -rs;
        for (int i = 0; i < nl; ++i) src[ns++] = c[i];
        int ch[16][2], nc = 0;
        for (int u = 0; u < ns; ++u) { ch[nc][0] = u; ch[nc][1] = u; ++nc; }
        for (int u = 0; u < ns; ++u) for (int v = u + 1; v < ns; ++v) { ch[nc][0] = u; ch[nc][1] = v; ++nc; }
        int id[4];
        for (id[0] = 0; id[0] < nc; ++id[0]) for (id[1] = id[0]; id[1] < nc; ++id[1])
        for (id[2] = id[1]; id[2] < nc; ++id[2]) for (id[3] = id[2]; id[3] < nc; ++id[3]) {
          uint32_t used = 0;
          for (int i = 0; i < 4; ++i) used |= (1u << ch[id[i]][0]) | (1u << ch[id[i]][1]);
          if ((used | 1u) != (1u << ns) - 1) continue;
          uint32_t rows[32];
          for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 8; ++j) {
              int a = src[ch[id[i]][0]], b = src[ch[id[i]][1]];
              rows[8 * i + j] = dep(a, ob[j]) ^ (a != b ? dep(b, ob[j]) : 0u);
            }
          if (rank32(rows) == 32) {
            if (found < 12) {
              printf("right %d, left {", rs);
              for (int i = 0; i < nl; ++i) printf("%s%d", i ? "," : "", c[i]);
              printf("}:");
              for (int i = 0; i < 4; ++i) {
                int a = src[ch[id[i]][0]], b = src[ch[id[i]][1]];
                if (a == b) printf("  r%d = M & s(%d)", i, a); else printf("  r%d = M & (s(%d) ^ s(%d))", i, a, b);
              }
              printf("\n");
            }
            ++found;
          }
        }
      }
      if (nl == 0) break;
      int i = nl - 1;
      while (i >= 0 && c[i] == 31 - (nl - 1 - i)) --i;
      if (i < 0) break;
      ++c[i];
      for (int j = i + 1; j < nl; ++j) c[j] = c[j - 1] + 1;
    }
    printf("== 1 right + %d left shifts, 4 LOP3: %ld solutions (s(k): q << k, s(-k): q >> k)\n", nl, found);
  }
  return 0;
}
