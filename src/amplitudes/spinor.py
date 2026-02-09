from __future__ import annotations
from dataclasses import dataclass
import numpy as np

Complex = np.complex128

def _c(x) -> Complex:
    return np.asarray(x, dtype=np.complex128)

def minkowski_dot(p: np.ndarray, q: np.ndarray) -> Complex:
    return _c(p[0] * q[0] - p[1] * q[1] - p[2] * q[2] - p[3] * q[3])

def mass2(p: np.ndarray) -> Complex:
    return minkowski_dot(p, p)

def sigma_dot(p: np.ndarray) -> np.ndarray:
    p0, p1, p2, p3 = map(_c, p)
    return np.array(
        [[p0 + p3, p1 - 1j * p2],
         [p1 + 1j * p2, p0 - p3]],
        dtype=np.complex128,
    )

def _spinors_from_massless_momentum(p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Crossing-safe handling for negative-energy massless momenta.
    # For p0<0, define spinors via p'=-p (positive energy) and the relation
    #   | -p >  =  i | p ]
    #   | -p ]  =  i | p >
    # (overall phases are irrelevant for |M|^2 but this avoids numerical blow-ups.)

    p = np.asarray(p, dtype=np.complex128)
    if np.real(p[0]) < 0:
        lam_p, lt_p = _spinors_from_massless_momentum(-p)
        # swap with phase i
        lam = 1j * lt_p
        lt = 1j * lam_p
        return lam.astype(np.complex128), lt.astype(np.complex128)

    m2 = mass2(p)
    if np.abs(m2) > 1e-8:
        raise ValueError(f"Momentum not massless within tolerance: p^2={m2}")

    M = sigma_dot(p)
    row0, row1 = M[0, :], M[1, :]

    if np.linalg.norm(row0) >= np.linalg.norm(row1):
        lt = row0.copy()
        if np.linalg.norm(lt) < 1e-14:
            lt = row1.copy()
    else:
        lt = row1.copy()
        if np.linalg.norm(lt) < 1e-14:
            lt = row0.copy()

    j = 0 if np.abs(lt[0]) >= np.abs(lt[1]) else 1
    lam = M[:, j] / lt[j]

    lam = lam.astype(np.complex128)
    lt = lt.astype(np.complex128)

    s = np.sqrt(np.abs(lam[0]) ** 2 + np.abs(lam[1]) ** 2)
    if s > 0:
        lam = lam / s
        lt = lt * s

    return lam, lt

def angle(lam_i: np.ndarray, lam_j: np.ndarray) -> Complex:
    return _c(lam_i[0] * lam_j[1] - lam_i[1] * lam_j[0])

def square(lti: np.ndarray, ltj: np.ndarray) -> Complex:
    return _c(lti[0] * ltj[1] - lti[1] * ltj[0])

def sandwich(lam_i: np.ndarray, P: np.ndarray, lt_j: np.ndarray) -> Complex:
    M = sigma_dot(P)
    return _c(lam_i @ (M @ lt_j))

@dataclass(frozen=True)
class SpinorPoint:
    lam: np.ndarray   # (n,2)
    lamt: np.ndarray  # (n,2)

    @staticmethod
    def from_momenta(p: np.ndarray) -> "SpinorPoint":
        p = np.asarray(p, dtype=np.complex128)
        if p.ndim != 2 or p.shape[1] != 4:
            raise ValueError("momenta must have shape (n,4)")
        n = p.shape[0]
        lam = np.zeros((n, 2), dtype=np.complex128)
        lamt = np.zeros((n, 2), dtype=np.complex128)
        for i in range(n):
            lam[i], lamt[i] = _spinors_from_massless_momentum(p[i])
        return SpinorPoint(lam=lam, lamt=lamt)

    def momenta(self) -> np.ndarray:
        n = self.lam.shape[0]
        p = np.zeros((n, 4), dtype=np.complex128)
        for i in range(n):
            M = np.outer(self.lam[i], self.lamt[i])
            p0 = 0.5 * (M[0, 0] + M[1, 1])
            p3 = 0.5 * (M[0, 0] - M[1, 1])
            p1 = 0.5 * (M[0, 1] + M[1, 0])
            p2 = (0.5 / (1j)) * (M[1, 0] - M[0, 1])
            p[i] = np.array([p0, p1, p2, p3], dtype=np.complex128)
        return p

    def ang(self, i: int, j: int) -> Complex:
        return angle(self.lam[i], self.lam[j])

    def sqr(self, i: int, j: int) -> Complex:
        return square(self.lamt[i], self.lamt[j])

    def sand(self, i: int, P: np.ndarray, j: int) -> Complex:
        return sandwich(self.lam[i], P, self.lamt[j])
