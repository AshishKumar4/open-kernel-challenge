"""Error budget of the hoisted pooler + router (HPR) with its stored precisions, numpy, against float64 math.
Pooler output and router logits per token, with what production's T = 1 path rounds (BF16 a in the v term).
    python3 tools/algebra/hpr_accuracy.py"""
import numpy as np


def bf16(x):
    x = np.ascontiguousarray(x, np.float32)
    b = x.view(np.uint32).astype(np.uint64)
    b = ((b + 0x7FFF + ((b >> 16) & 1)) >> 16) << 16
    return b.astype(np.uint32).view(np.float32).astype(np.float64)


def rms(v, eps):
    return np.sqrt((v * v).mean(-1, keepdims=True) + eps)


rng = np.random.default_rng(7)
H, DQ, DE, NE, T = 8192, 1024, 4096, 256, 8
O, P = bf16(rng.standard_normal((H, DQ)) / 32), bf16(rng.standard_normal((DE, H)) / np.sqrt(H))
R = 0.045 * rng.standard_normal((NE, H)).astype(np.float32).astype(np.float64)
w = bf16(0.3 * rng.standard_normal(H)) / 0.3
beta = 1 / (1 + np.exp(-w))
alpha = np.sqrt(np.maximum(1 / (1 + np.exp(w)) * (1 + beta), 1e-3))
pre = 1 + bf16(0.1 * rng.standard_normal(H))
s = rng.standard_normal((T, H)) + 0.7
z = bf16(rng.standard_normal((T, DQ)))
x = bf16(s / rms(s, 1e-5))
# float64 truth
a = z @ O.T
s1 = alpha * s + beta * a / rms(a, 1e-8)
h = s1 / rms(s1, 1e-5) * pre
pool, logit = h @ P.T, h @ R.T
p_true = pool / rms(pool, 1e-5)
# HPR as built: E = bf16(P cs) and R cs hi + lo over bf16 x, L = [O; bf16(Q); M hi + lo] over z, FP32 out
cs, ca = alpha * pre, beta * pre
rs = (R * cs).astype(np.float32).astype(np.float64)
hi = bf16(rs)
lo = bf16(rs - hi)
Pp, Q = bf16(P * cs), bf16(P @ (ca[:, None] * O))
Mf = (R @ (ca[:, None] * O)).astype(np.float32).astype(np.float64)
M = bf16(Mf) + bf16(Mf - bf16(Mf))   # BF16 hi + lo pair
ab = bf16(z @ O.T)   # the kernel's a (BF16) for ra and rs1, as production
r1 = 1 / rms(ab, 1e-8)
rs0 = rms(s, 1e-5)
sum_ = lambda v: v.sum(-1, keepdims=True)  # noqa: E731
r2 = 1 / np.sqrt((sum_((alpha * s) ** 2) + 2 * r1 * sum_(alpha * s * beta * ab) + r1 ** 2 * sum_((beta * ab) ** 2)) / H + 1e-5)
pooled = (rs0 * (x @ Pp.T) + r1 * (z @ Q.T)) * r2
logits = (rs0 * (x @ (hi + lo).T) + r1 * (z @ M.T)) * r2
p = pooled / rms(pooled, 1e-5)
rel = lambda u, v: np.linalg.norm(u - v) / np.linalg.norm(v)  # noqa: E731
print(f"pooler (normalized) rel L2 {rel(p, p_true):.2e}   logits max abs {np.abs(logits - logit).max():.2e}"
      f"   score max abs {np.abs(1 / (1 + np.exp(-0.17678 * logits)) - 1 / (1 + np.exp(-0.17678 * logit))).max():.2e}")
# production's T = 1 path for comparison: u, v rows in BF16 for the pooler, FP32 for the router
u, v = alpha * s * pre, beta * ab * pre
pooled_p = (bf16(u) @ P.T + r1 * (bf16(v) @ P.T)) * r2
logits_p = (u @ R.T + r1 * (v @ R.T)) * r2
print(f"production  rel L2 {rel(pooled_p / rms(pooled_p, 1e-5), p_true):.2e}   logits max abs {np.abs(logits_p - logit).max():.2e}")
