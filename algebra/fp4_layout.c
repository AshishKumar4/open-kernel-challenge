// Exhaustive search: lossless NVFP4 (E2M1) -> BF16 decode with 4 masking LOP3 and IMAD.SHL shifts only (no SHF / PRMT).
// Output register i (i = 0..3) = M & (q << a_i ^ q << b_i) (a_i = b_i: a single source; shift 0 is q itself, free),
// M = 0x81C081C0: each 16-bit half holds one code's mantissa / e0 / e1 / sign at bits 6 / 7 / 8 / 15. The stored word q
// is a prepare-time GF(2)-linear image of the 32 code bits; the layout is lossless iff the 32 x 32 map q -> outputs is
// invertible over GF(2), and the stored word is then q = Phi^-1(codes). The cost is the number of distinct nonzero
// shifts (one IMAD.SHL each; shared across registers). A lower bound of 4 ALU ops holds for any sequence: each output
// register needs its own masking op. Prints every minimum-IMAD solution class.
//   cc -O2 -o fp4_layout algebra/fp4_layout.c && ./fp4_layout
#include <stdint.h>
#include <stdio.h>

static const uint32_t M = 0x81C081C0u;
static int outbits[8];

static int rank32(uint32_t* rows) {   // GF(2) rank of 32 rows (destroys rows)
  int r = 0;
  for (int bit = 31; bit >= 0 && r < 32; --bit) {
    int p = -1;
    for (int i = r; i < 32; ++i)
      if ((rows[i] >> bit) & 1) { p = i; break; }
    if (p < 0) continue;
    uint32_t t = rows[p]; rows[p] = rows[r]; rows[r] = t;
    for (int i = 0; i < 32; ++i)
      if (i != r && ((rows[i] >> bit) & 1)) rows[i] ^= rows[r];
    ++r;
  }
  return r;
}

// the 8 rows (dependencies on q's bits) of an output register using shifts a, b
static void reg_rows(int a, int b, uint32_t* rows) {
  for (int j = 0; j < 8; ++j) {
    int t = outbits[j];
    uint32_t r = 0;
    if (t - a >= 0) r ^= 1u << (t - a);
    if (b != a && t - b >= 0) r ^= 1u << (t - b);
    rows[j] = r;
  }
}

int main(void) {
  int n = 0;
  for (int t = 0; t < 32; ++t)
    if ((M >> t) & 1) outbits[n++] = t;
  for (int nimad = 0; nimad <= 4; ++nimad) {
    long found = 0;
    // shift sets: {0} plus nimad distinct values in 1..31 (ascending)
    int ks[5] = {0};
    int idx[4] = {0};
    // iterate combinations of nimad values from 1..31
    int c[4];
    for (int i = 0; i < nimad; ++i) c[i] = i + 1;
    while (1) {
      ks[0] = 0;
      for (int i = 0; i < nimad; ++i) ks[i + 1] = c[i];
      int ns = nimad + 1;
      // the source choices: single (u) or pair (u < v)
      int ch[16][2], nc = 0;
      for (int u = 0; u < ns; ++u) { ch[nc][0] = ks[u]; ch[nc][1] = ks[u]; ++nc; }
      for (int u = 0; u < ns; ++u)
        for (int v = u + 1; v < ns; ++v) { ch[nc][0] = ks[u]; ch[nc][1] = ks[v]; ++nc; }
      // multisets of 4 choices (registers are interchangeable)
      for (idx[0] = 0; idx[0] < nc; ++idx[0])
        for (idx[1] = idx[0]; idx[1] < nc; ++idx[1])
          for (idx[2] = idx[1]; idx[2] < nc; ++idx[2])
            for (idx[3] = idx[2]; idx[3] < nc; ++idx[3]) {
              // every nonzero shift of the set must be used (else a smaller set covers it)
              uint32_t used = 0;
              for (int i = 0; i < 4; ++i) {
                for (int s = 0; s < ns; ++s)
                  if (ks[s] == ch[idx[i]][0] || ks[s] == ch[idx[i]][1]) used |= 1u << s;
              }
              if ((used | 1u) != (1u << ns) - 1) continue;
              uint32_t rows[32];
              for (int i = 0; i < 4; ++i) reg_rows(ch[idx[i]][0], ch[idx[i]][1], rows + 8 * i);
              if (rank32(rows) == 32) {
                if (found < 20) {
                  printf("IMAD %d:", nimad);
                  for (int i = 0; i < 4; ++i) printf(" r%d = M & (q<<%d ^ q<<%d)", i, ch[idx[i]][0], ch[idx[i]][1]);
                  printf("\n");
                }
                ++found;
              }
            }
      if (nimad == 0) break;
      int i = nimad - 1;
      while (i >= 0 && c[i] == 31 - (nimad - 1 - i)) --i;
      if (i < 0) break;
      ++c[i];
      for (int j = i + 1; j < nimad; ++j) c[j] = c[j - 1] + 1;
    }
    printf("== %d IMAD.SHL + 4 LOP3, no SHF: %ld solutions\n", nimad, found);
    if (found) break;
  }
  return 0;
}
