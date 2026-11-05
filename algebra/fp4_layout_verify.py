"""Bit-exactness of the 4 LOP3 + 1 SHF + 3 IMAD.SHL NVFP4 -> BF16 decode layout found by fp4_layout3.c:

    R = rotr(q, 6)                         SHF.W (ALU)
    a = R << 2, b = q << 4, c = q << 6     IMAD.SHL x3 (FMA pipe)
    r0 = M & q                             LOP3
    r1 = M & (q ^ a)                       LOP3
    r2 = M & (R ^ b)                       LOP3
    r3 = M & (R ^ c)                       LOP3        M = 0x81C081C0

The decode is GF(2)-linear in q; prepare stores q = Phi^-1(target), where target is the 32-bit image of the 8 codes in
production's register layout (r0..r3: codes at bits 6 / 7 / 8 / 15 = m / e0 / e1 / s, +16 for the upper weight). Checks:
Phi invertible; decode(encode(x)) == x for every basis vector (linearity then gives every word) and, as asked, for all
256 values of each byte position and 10^6 random words; and the decoded BF16 values equal production's dqm decode.
    python3 algebra/fp4_layout_verify.py"""
import random

M = 0x81C081C0
FULL = 0xFFFFFFFF


def rotr(x, c):
    return ((x >> c) | (x << (32 - c))) & FULL


def decode(q):
    R = rotr(q, 6)
    a, b, c = (R << 2) & FULL, (q << 4) & FULL, (q << 6) & FULL
    return (M & q, M & (q ^ a), M & (R ^ b), M & (R ^ c))


def pack(regs):
    """the 32 decoded code bits as one word: register i's M bits, in order (bit 8 i + j = M bit j of register i)"""
    out = 0
    pos = [t for t in range(32) if (M >> t) & 1]
    for i, r in enumerate(regs):
        for j, t in enumerate(pos):
            out |= ((r >> t) & 1) << (8 * i + j)
    return out


def phi_rows():
    """column view: phi(e_k) for every stored bit k"""
    return [pack(decode(1 << k)) for k in range(32)]


def invert(cols):
    """GF(2) inverse of the linear map given by its images of the basis: returns enc with phi(enc(y)) = y"""
    # solve for each target basis vector by Gaussian elimination on [cols | I]
    rows = [(cols[k], 1 << k) for k in range(32)]   # (image, preimage)
    basis = {}
    for img, pre in rows:
        for bit in range(31, -1, -1):
            if not (img >> bit) & 1:
                continue
            if bit in basis:
                bi, bp = basis[bit]
                img ^= bi
                pre ^= bp
            else:
                basis[bit] = (img, pre)
                break
    assert len(basis) == 32, "not invertible"
    # back-substitute to unit images
    for bit in sorted(basis):
        img, pre = basis[bit]
        for b2 in range(bit - 1, -1, -1):
            if (img >> b2) & 1:
                bi, bp = basis[b2]
                img ^= bi
                pre ^= bp
        basis[bit] = (img, pre)
    unit = {}
    for bit, (img, pre) in basis.items():
        assert img == 1 << bit
        unit[bit] = pre
    return lambda y: _apply(unit, y)


def _apply(unit, y):
    q = 0
    for bit in range(32):
        if (y >> bit) & 1:
            q ^= unit[bit]
    return q


def bf16(code):
    """production's decoded BF16 bits of one E2M1 code (s e1 e0 m): m at bit 6, e0 at 7, e1 at 8, s at 15"""
    m, e0, e1, s = code & 1, (code >> 1) & 1, (code >> 2) & 1, (code >> 3) & 1
    return (m << 6) | (e0 << 7) | (e1 << 8) | (s << 15)


def target(codes):
    """8 codes (register i holds codes 2 i, 2 i + 1) -> the packed 32-bit decoded image"""
    regs = []
    for i in range(4):
        regs.append(bf16(codes[2 * i]) | (bf16(codes[2 * i + 1]) << 16))
    return pack(regs), regs


cols = phi_rows()
enc = invert(cols)
for k in range(32):
    assert pack(decode(enc(1 << k))) == 1 << k
n = 0
for pos in range(4):
    for v in range(256):
        y = v << (8 * pos)
        assert pack(decode(enc(y))) == y
        n += 1
rng = random.Random(1)
for _ in range(1_000_000 // 10):
    codes = [rng.randrange(16) for _ in range(8)]
    y, regs = target(codes)
    assert decode(enc(y)) == tuple(regs)
    n += 1
print(f"invertible; {n} checks bit-exact (basis, every byte value at every position, random code words)")
print("encode basis (stored word for each decoded bit):", [hex(enc(1 << k)) for k in range(32)])


# ---- the prepare-time transform from production's stored words (dqm layout) to this layout
def decode_dqm(q):
    """production's decode (_v_dqm.py): r0 = q & M, r1 = rotr(q, 3) & M, r2 = rotr(q, 6) & M,
    r3 = ((q & 0x001b001b) * 4228) & M"""
    return (q & M, rotr(q, 3) & M, rotr(q, 6) & M, ((q & 0x001B001B) * 4228 & FULL) & M)


# L: new stored word = L(dqm stored word), GF(2)-linear (dqm's decode is linear: its multiply has no carries)
L = [enc(pack(decode_dqm(1 << k))) for k in range(32)]
# as masks per output bit: new bit j = parity(q_dqm & LMASK[j])
LMASK = [sum(((L[k] >> j) & 1) << k for k in range(32)) for j in range(32)]


def to_new(q):
    out = 0
    for j in range(32):
        out |= (bin(q & LMASK[j]).count("1") & 1) << j
    return out


for pos in range(4):
    for v in range(256):
        q = v << (8 * pos)
        assert decode(to_new(q)) == decode_dqm(q)
for _ in range(200000):
    q = rng.getrandbits(32)
    assert decode(to_new(q)) == decode_dqm(q)
print("dqm -> new layout: decode(L(q)) == decode_dqm(q) for every byte value at every position and 2e5 random words")
print("LMASK (new bit j = parity(q_dqm & LMASK[j])):", [hex(m) for m in LMASK])
