from __future__ import annotations
import numpy as np
import math
from .phasespace import rambo_massless

def rambo_massive_equal_m(n: int, Ecm: float, m: float, rng: np.random.Generator):
    """
    Generate n-body phase space for identical masses m using RAMBO-massless then energy rescaling.
    Returns (momenta, weight) in the same convention as rambo_massless.

    This is a standard production technique for equal masses:
      - generate massless momenta {k_i} with sum K=(Ecm,0,0,0)
      - find scale x such that sum sqrt((x k_i)^2 + m^2) = Ecm
      - set p_i = (sqrt((x k_i)^2 + m^2), x vec{k_i})
    Jacobian included via 1D root + known factor.
    """
    k, w0 = rambo_massless(n, Ecm, rng)  # massless, already in COM
    # Solve for x by monotonic root (bisection)
    kv = np.linalg.norm(k[:,1:].real, axis=1)
    def f(x):
        return np.sum(np.sqrt((x*kv)**2 + m*m)) - Ecm
    lo, hi = 0.0, 1.0
    while f(hi) < 0:
        hi *= 2.0
        if hi > 1e6:
            raise RuntimeError("Failed to bracket root for massive RAMBO scaling")
    for _ in range(120):
        mid = 0.5*(lo+hi)
        if f(mid) > 0:
            hi = mid
        else:
            lo = mid
    x = 0.5*(lo+hi)
    p = np.zeros_like(k)
    p[:,1:] = (x*k[:,1:]).astype(np.complex128)
    pE = np.sqrt((x*kv)**2 + m*m)
    p[:,0] = pE.astype(np.complex128)
    # weight: w0 times Jacobian of transformation (product E_i / |k_i|) times dx factor
    # For equal-m rescaling in COM, the Jacobian is:
    #   J = x^{3n-4} * (prod (|k_i|/E_i)) * (Ecm / sum (x^2 |k_i|^2 / E_i))
    # derived from the delta constraint. We implement it explicitly.
    denom = np.sum((x*x*kv*kv) / pE)
    if denom <= 0:
        raise RuntimeError("Invalid Jacobian denom")
    J = (x ** (3*n - 4)) * np.prod(kv / pE) * (Ecm / denom)
    return p, float(w0 * J)
