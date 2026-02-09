from __future__ import annotations
import numpy as np
import math

def kallen(a: float, b: float, c: float) -> float:
    return a*a + b*b + c*c - 2*a*b - 2*a*c - 2*b*c

def com_incoming_momenta(Ecm: float, m1: float = 0.0, m2: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Return p1, p2 in COM with p1 along +z, p2 along -z."""
    s = Ecm*Ecm
    lam = kallen(s, m1*m1, m2*m2)
    if lam < 0:
        raise ValueError("Ecm below threshold for given masses")
    p = math.sqrt(lam) / (2.0*Ecm)
    E1 = math.sqrt(p*p + m1*m1)
    E2 = math.sqrt(p*p + m2*m2)
    p1 = np.array([E1, 0.0, 0.0, +p], dtype=np.complex128)
    p2 = np.array([E2, 0.0, 0.0, -p], dtype=np.complex128)
    return p1, p2

def flux_factor(Ecm: float, m1: float = 0.0, m2: float = 0.0) -> float:
    """Flux factor for 2->n in COM: 4 |p| sqrt(s)"""
    s = Ecm*Ecm
    lam = kallen(s, m1*m1, m2*m2)
    if lam < 0:
        raise ValueError("Ecm below threshold for given masses")
    p = math.sqrt(lam) / (2.0*Ecm)
    return 4.0 * p * math.sqrt(s)
