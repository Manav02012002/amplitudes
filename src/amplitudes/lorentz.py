from __future__ import annotations
import numpy as np

def minkowski_dot(p: np.ndarray, q: np.ndarray) -> complex:
    p = np.asarray(p, dtype=np.complex128)
    q = np.asarray(q, dtype=np.complex128)
    return p[0]*q[0] - p[1]*q[1] - p[2]*q[2] - p[3]*q[3]

def mass2(p: np.ndarray) -> complex:
    return minkowski_dot(p, p)

def boost_matrix(beta: np.ndarray) -> np.ndarray:
    beta = np.asarray(beta, dtype=np.float64)
    b2 = float(np.dot(beta, beta))
    if b2 <= 0.0:
        return np.eye(4, dtype=np.float64)
    if b2 >= 1.0:
        raise ValueError("Superluminal beta")
    g = 1.0 / np.sqrt(1.0 - b2)
    bp = (g - 1.0) / b2
    bx, by, bz = beta
    L = np.eye(4, dtype=np.float64)
    L[0,0] = g
    L[0,1:] = -g*beta
    L[1:,0] = -g*beta
    L[1,1] += bp*bx*bx
    L[1,2] += bp*bx*by
    L[1,3] += bp*bx*bz
    L[2,1] += bp*by*bx
    L[2,2] += bp*by*by
    L[2,3] += bp*by*bz
    L[3,1] += bp*bz*bx
    L[3,2] += bp*bz*by
    L[3,3] += bp*bz*bz
    return L

def apply_lorentz(L: np.ndarray, v: np.ndarray) -> np.ndarray:
    L = np.asarray(L, dtype=np.float64)
    v = np.asarray(v, dtype=np.complex128)
    return (L @ v).astype(np.complex128)
