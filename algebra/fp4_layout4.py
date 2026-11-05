"""Lossless NVFP4 -> BF16 decode as a bit permutation with one shared right-moving op composed with left shifts:
r_i = M & (S_i << k_i), S_i in {q, R(q)}, R = rotr(q, c) or q >> c (one SHF, or IMAD.HI for the logical shift), k_i in
0..31 (IMAD.SHL; 0 free). Lossless iff the four registers' source bits partition the 32 stored bits. Exhaustive.
    python3 algebra/fp4_layout4.py"""
import itertools

M = 0x81C081C0
T = [t for t in range(32) if (M >> t) & 1]


def src_bits(kind, c, k):
    """stored-bit positions feeding register outputs T, for source kind ('q' or 'rot' / 'shr' by c) shifted left by k"""
    out = []
    for t in T:
        j = t - k                      # bit of S that lands on t
        if j < 0:
            return None                # an output bit with no source: that register is not lossless on its own
        if kind == "q":
            p = j
        elif kind == "rot":
            p = (j + c) % 32
        else:
            p = j + c
            if p > 31:
                return None
        out.append(p)
    return frozenset(out) if len(set(out)) == 8 else None


sols = []
for kind in ("rot", "shr"):
    for c in range(1, 32):
        cands = []
        for k in range(32):
            for s in ("q", kind):
                b = src_bits(s, c, k)
                if b is not None:
                    cands.append((s, k, b))
        for combo in itertools.combinations(cands, 4):
            u = frozenset().union(*[x[2] for x in combo])
            if len(u) == 32:
                imads = len({x[1] for x in combo if x[1] != 0})
                uses_r = any(x[0] != "q" for x in combo)
                if uses_r:
                    sols.append((imads, kind, c, [(x[0], x[1]) for x in combo]))
sols.sort(key=lambda s: s[0])
print(len(sols), "permutation layouts with one shared right-moving op")
for s in sols[:20]:
    print(s)
