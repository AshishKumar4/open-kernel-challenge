"""Permutation decode layouts with up to two SHF rotations: r_i = M & (S_i << k_i), S_i in {q, rotr(q, c1), rotr(q, c2)}
(4 LOP3 + #SHF + #distinct (source, k != 0) IMAD.SHL). Lossless iff the source bits of the four registers partition the
32 stored bits. Also the FP16 target (code bits at 9 / 10 / 11 / 15) for comparison.
    python3 algebra/fp4_layout6.py"""
import itertools


def run(M, label):
    T = [t for t in range(32) if (M >> t) & 1]

    def bits(c, k):   # rotr(q, c) << k, masked to M: stored bits it reads (None if some M bit has no source)
        out = set()
        for t in T:
            j = t - k
            if j < 0:
                return None
            out.add((j + c) % 32)
        return frozenset(out)

    cands = {}
    for c in range(32):
        for k in range(32):
            b = bits(c, k)
            if b is not None:
                cands.setdefault(b, []).append((c, k))
    sets = list(cands)
    sols = []
    for combo in itertools.combinations(sets, 4):
        if len(frozenset().union(*combo)) != 32:
            continue
        # cheapest realization: choose (c, k) per register minimizing SHF (distinct nonzero c) then IMAD (nonzero k)
        best = None
        for pick in itertools.product(*[cands[s] for s in combo]):
            shf = len({c for c, k in pick if c})
            imad = len({(c, k) for c, k in pick if k})
            key = (shf, imad)
            if best is None or key < best[0]:
                best = (key, pick)
        sols.append(best)
    sols.sort()
    print(f"{label}: {len(sols)} permutation layouts")
    for (shf, imad), pick in sols[:10]:
        print(f"  4 LOP3 + {shf} SHF + {imad} IMAD.SHL:", "  ".join(f"M & (rotr(q,{c}) << {k})" for c, k in pick))


run(0x81C081C0, "BF16 (6, 7, 8, 15)")
run(0x8E008E00, "FP16 (9, 10, 11, 15)")
