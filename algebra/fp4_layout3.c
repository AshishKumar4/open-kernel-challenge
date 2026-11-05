// Lossless NVFP4 -> BF16 decode layouts, 4 LOP3 with chained registers: r_i = M & (u ^ v), u, v in the sources
// {q, R(q), q << k_1..k_L} or an earlier register r_j (masked already, so M & (x ^ r_j) is one LOP3 with M immediate).
// Left shifts may apply to q or to R (IMAD.SHL of the SHF result). R is the one right-moving source: SHF.R (logical) or SHF.W (rotate) by c, on the ALU pipe; IMAD.HI.U32 q * 2^(32-c)
// = q >> c on the FMA pipe. Left shifts are IMAD.SHL (FMA pipe). Every source used; registers built in order, each
// required to add 8 independent rows (prunes); invertibility over GF(2).
//   cc -O2 -o fp4_layout3 algebra/fp4_layout3.c && ./fp4_layout3 [max left shift]
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
static const uint32_t M = 0x81C081C0u;
static int ob[8];
static long found;
static int maxprint = 6;
static int rank_n(const uint32_t* in, int n) {
  uint32_t rows[32];
  for (int i = 0; i < n; ++i) rows[i] = in[i];
  int r = 0;
  for (int bit = 31; bit >= 0 && r < n; --bit) {
    int p = -1;
    for (int i = r; i < n; ++i) if ((rows[i] >> bit) & 1) { p = i; break; }
    if (p < 0) continue;
    uint32_t t = rows[p]; rows[p] = rows[r]; rows[r] = t;
    for (int i = 0; i < n; ++i) if (i != r && ((rows[i] >> bit) & 1)) rows[i] ^= rows[r];
    ++r;
  }
  return r;
}
// source rows: for source s, its dependency at each of the 8 output bits
static uint32_t srow[8][8];
static int nsrc;
static const char* sname[8];
static char names[8][32];
static uint32_t regrows[4][8];
static int pick[4][2];   // source index, or 100 + register index
static void report(void) {
  if (found++ < maxprint) {
    for (int i = 0; i < 4; ++i) {
      printf("  r%d = M & (", i);
      for (int k = 0; k < 2; ++k) {
        if (k == 1 && pick[i][1] == pick[i][0]) break;
        if (k) printf(" ^ ");
        if (pick[i][k] >= 100) printf("r%d", pick[i][k] - 100); else printf("%s", sname[pick[i][k]]);
      }
      printf(")");
    }
    printf("\n");
  }
}
static void rec(int i, uint32_t used, uint32_t* acc) {
  if (i == 4) {
    if (used == (1u << nsrc) - 1) report();
    return;
  }
  int nopt = nsrc + i;
  for (int a = 0; a < nopt; ++a)
    for (int b = a; b < nopt; ++b) {
      int ia = a < nsrc ? a : 100 + (a - nsrc), ib = b < nsrc ? b : 100 + (b - nsrc);
      if (a == b && a >= nsrc) continue;   // a copy of an earlier register
      uint32_t u2 = used;
      for (int j = 0; j < 8; ++j) {
        uint32_t ra = a < nsrc ? srow[a][j] : regrows[a - nsrc][j];
        uint32_t rb = (b == a) ? 0u : (b < nsrc ? srow[b][j] : regrows[b - nsrc][j]);
        regrows[i][j] = ra ^ rb;
        acc[8 * i + j] = ra ^ rb;
      }
      if (rank_n(acc, 8 * (i + 1)) != 8 * (i + 1)) continue;
      if (a < nsrc) u2 |= 1u << a;
      if (b < nsrc) u2 |= 1u << b;
      pick[i][0] = ia; pick[i][1] = ib;
      rec(i + 1, u2, acc);
    }
}
int main(int argc, char** argv) {
  int kmax = argc > 1 ? atoi(argv[1]) : 24;
  int n = 0;
  for (int t = 0; t < 32; ++t) if ((M >> t) & 1) ob[n++] = t;
  for (int nl = 0; nl <= 3; ++nl) {
    found = 0;
    int rot_lo = argc > 2 ? atoi(argv[2]) : 0, rot_hi = argc > 3 ? atoi(argv[3]) : 1;
    for (int rot = rot_lo; rot <= rot_hi; ++rot)
      for (int c = 1; c <= 31; ++c) {
        int k[3];
        for (int i = 0; i < nl; ++i) k[i] = i + 1;
        while (1) {
         for (int base = 0; base < (1 << nl); ++base) {
          nsrc = 0;
          int shifts[5];
          shifts[nsrc] = 0; snprintf(names[nsrc], 32, "q"); ++nsrc;
          shifts[nsrc] = -c; snprintf(names[nsrc], 32, rot ? "rotr(q,%d)" : "q>>%d", c); ++nsrc;
          for (int i = 0; i < nl; ++i) {
            shifts[nsrc] = k[i];
            snprintf(names[nsrc], 32, ((base >> i) & 1) ? (rot ? "rotr(q,%d)<<%d" : "(q>>%d)<<%d") : "q<<%d", ((base >> i) & 1) ? c : k[i], k[i]);
            ++nsrc;
          }
          for (int s = 0; s < nsrc; ++s) {
            sname[s] = names[s];
            int onR = (s == 1) || (s >= 2 && ((base >> (s - 2)) & 1));
            int ls = s >= 2 ? shifts[s] : 0;
            for (int j = 0; j < 8; ++j) {
              int t = ob[j] - ls;          // bit of the shifted source's input
              int src;
              if (t < 0) src = -1;
              else if (!onR) src = (s == 0) ? ob[j] : t;
              else src = rot ? ((t + c) & 31) : (t + c);
              srow[s][j] = (src >= 0 && src < 32) ? (1u << src) : 0u;
            }
          }
          uint32_t acc[32];
          rec(0, 0, acc);
         }
          if (nl == 0) break;
          int i = nl - 1;
          while (i >= 0 && k[i] == kmax - (nl - 1 - i)) --i;
          if (i < 0) break;
          ++k[i];
          for (int j = i + 1; j < nl; ++j) k[j] = k[j - 1] + 1;
        }
      }
    printf("== 1 right-moving source + %d IMAD.SHL (<= %d), 4 LOP3 (chained allowed): %ld solutions\n", nl, kmax, found);
    fflush(stdout);
    if (found) break;
  }
  return 0;
}
