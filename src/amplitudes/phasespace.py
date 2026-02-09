from __future__ import annotations
import math
import numpy as np

def _rambo_weight_massless(n: int, s: float) -> float:
    num = (math.pi / 2.0) ** (n - 1) * (s ** (n - 2))
    den = ((2.0 * math.pi) ** (3 * n - 4)) * math.factorial(n - 1) * math.factorial(n - 2)
    return num / den

def rambo_massless(n: int, Ecm: float, rng: np.random.Generator) -> tuple[np.ndarray, float]:
    r = rng.random((n, 4))
    c = 2.0 * r[:, 0] - 1.0
    phi = 2.0 * np.pi * r[:, 1]
    s = np.sqrt(1.0 - c * c)
    E = -np.log(r[:, 2] * r[:, 3])
    q = np.zeros((n, 4), dtype=np.float64)
    q[:, 0] = E
    q[:, 1] = E * s * np.cos(phi)
    q[:, 2] = E * s * np.sin(phi)
    q[:, 3] = E * c

    Q = q.sum(axis=0)
    bx, by, bz = -Q[1] / Q[0], -Q[2] / Q[0], -Q[3] / Q[0]
    b2 = bx * bx + by * by + bz * bz
    g = 1.0 / np.sqrt(1.0 - b2)
    gp = (g - 1.0) / b2 if b2 > 0 else 0.0

    p = q.copy()
    for i in range(n):
        qi0, qix, qiy, qiz = p[i]
        bdotq = bx * qix + by * qiy + bz * qiz
        p[i, 0] = g * (qi0 + bdotq)
        p[i, 1] = qix + gp * bdotq * bx + g * bx * qi0
        p[i, 2] = qiy + gp * bdotq * by + g * by * qi0
        p[i, 3] = qiz + gp * bdotq * bz + g * bz * qi0

    scale = Ecm / p[:, 0].sum()
    p *= scale

    w = _rambo_weight_massless(n, s=Ecm * Ecm)
    return p, w
