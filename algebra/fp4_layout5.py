"""Lossless NVFP4 -> BF16 decode where shifts also discard bits: one masked intermediate can feed several outputs.

An intermediate is v = S & m (one LOP3; S = q, or R = one SHF of q: rotr(q, c)). An output register is v shifted by s
(s > 0: IMAD.SHL, s < 0: IMAD.HI.U32 v * 2^(32+s) = v >> -s on the FMA pipe, s = 0: v itself), and it is clean iff
the shifted copy of m, truncated to 32 bits, is exactly M (the bits meant for other outputs fall off the ends). The
layout is lossless iff the outputs' source bits partition the 32 stored bits (a permutation, applied in prepare).
Exhaustive over the intermediates' shift sets; minimizes ALU ops (LOP3 + SHF), then FMA-pipe IMADs.

The earlier bound "each output needs its own LOP3" assumed shifted copies keep every bit; truncating shifts break it.

    python3 algebra/fp4_layout5.py
"""
import itertools

M = 0x81C081C0
FULL = 0xFFFFFFFF


def shift(x, s):
    return ((x << s) & FULL) if s >= 0 else (x >> -s)


def unshift_exact(s):
    """the stored-bit mask whose shift by s lands exactly on M (None if some M bit has no source in range)"""
    m = shift(M, -s)
    return m if shift(m, s) == M else None


# intermediates: a mask m and the shifts it serves (2..4 outputs from one LOP3), each shift landing exactly on M
inters = []
for n in range(1, 5):
    for ss in itertools.combinations(range(-31, 32), n):
        parts = [unshift_exact(s) for s in ss]
        if any(p is None for p in parts):
            continue
        m = 0
        ok = True
        for p in parts:
            if m & p:
                ok = False
                break
            m |= p
        if not ok:
            continue
        if all(shift(m, s) == M for s in ss):
            inters.append((m, ss))
print(len(inters), "clean intermediates (mask, output shifts)")


def rotr_bits(m, c):
    """stored-bit set feeding v = rotr(q, c) & m"""
    out = 0
    for b in range(32):
        if (m >> b) & 1:
            out |= 1 << ((b + c) % 32)
    return out


best = []
# no SHF: intermediates of q only
for k in range(1, 5):
    for combo in itertools.combinations(inters, k):
        if sum(len(ss) for _, ss in combo) != 4:
            continue
        u = 0
        bad = False
        for m, _ in combo:
            if u & m:
                bad = True
                break
            u |= m
        if not bad and u == FULL:
            imad = sum(1 for _, ss in combo for s in ss if s != 0)
            best.append((k, 0, imad, "q", [(hex(m), ss) for m, ss in combo]))
# one SHF (rotr by c) feeding some intermediates
for c in range(1, 32):
    for k in range(1, 5):
        for combo in itertools.combinations(inters, k):
            if sum(len(ss) for _, ss in combo) != 4:
                continue
            for srcs in itertools.product((0, 1), repeat=k):
                if not any(srcs):
                    continue
                u = 0
                bad = False
                for (m, _), r in zip(combo, srcs):
                    b = rotr_bits(m, c) if r else m
                    if u & b:
                        bad = True
                        break
                    u |= b
                if not bad and u == FULL:
                    imad = sum(1 for _, ss in combo for s in ss if s != 0)
                    best.append((k, 1, imad, f"rotr {c}", [(hex(m), ss, "R" if r else "q") for (m, ss), r in zip(combo, srcs)]))
best.sort(key=lambda b: (b[0] + b[1], b[2]))
print(len(best), "lossless layouts; the cheapest (LOP3, SHF, IMAD):")
for b in best[:25]:
    print(f"  LOP3 {b[0]} SHF {b[1]} IMAD {b[2]}  {b[3]}  {b[4]}")
