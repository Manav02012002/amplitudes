from __future__ import annotations
import numpy as np
from .lorentz import mass2

_eta = np.diag([1,-1,-1,-1]).astype(np.complex128)

def _V3_tensor(p: np.ndarray, q: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Color-ordered 3-gluon vertex V^{μνρ} with all momenta flowing in."""
    V = np.zeros((4,4,4), dtype=np.complex128)
    for mu in range(4):
        for nu in range(4):
            for rho in range(4):
                V[mu,nu,rho] = (
                    _eta[nu,rho]*(q-r)[mu] +
                    _eta[rho,mu]*(r-p)[nu] +
                    _eta[mu,nu]*(p-q)[rho]
                )
    return V

def _contract_V3(Jq: np.ndarray, Jr: np.ndarray, p: np.ndarray, q: np.ndarray, r: np.ndarray) -> np.ndarray:
    V = _V3_tensor(p,q,r)
    out = np.zeros(4, dtype=np.complex128)
    for mu in range(4):
        out[mu] = np.sum(V[mu,:,:] * Jq[None,:] * Jr[:,None])
    return out

def _V4_contact(Ja: np.ndarray, Jb: np.ndarray, Jc: np.ndarray) -> np.ndarray:
    """Color-ordered 4-gluon contact term contribution."""
    out = np.zeros(4, dtype=np.complex128)
    # V^{μνρσ} = 2 η^{μρ}η^{νσ} - η^{μν}η^{ρσ} - η^{μσ}η^{νρ}
    for mu in range(4):
        s = 0.0 + 0j
        for nu in range(4):
            for rho in range(4):
                for sig in range(4):
                    V = 2*_eta[mu,rho]*_eta[nu,sig] - _eta[mu,nu]*_eta[rho,sig] - _eta[mu,sig]*_eta[nu,rho]
                    s += V * Jb[nu]*Jc[rho]*Ja[sig]
        out[mu] = s
    return out

def gluon_current_color_ordered(mom: np.ndarray, hels: tuple[int, ...], g_s: float) -> np.ndarray:
    """
    Berends–Giele color-ordered off-shell gluon current J^μ(1..n) in Feynman gauge.

    Inputs:
      - mom: (n,4) complex array of on-shell momenta for ordered gluons (all outgoing)
      - hels: tuple of helicities (+1/-1) length n
      - g_s: strong coupling

    Output:
      - 4-vector current J^μ for the ordered set, including the off-shell propagator 1/P^2.
    """
    from .polarization import massless_vector_polarizations
    mom = np.asarray(mom, dtype=np.complex128)
    n = len(hels)
    if mom.shape != (n,4):
        raise ValueError("mom shape mismatch")
    # DP tables for segments [i:j)
    J = [[None]*(n+1) for _ in range(n)]
    P = [[None]*(n+1) for _ in range(n)]

    for i in range(n):
        P[i][i+1] = mom[i]
        epsp, epsm = massless_vector_polarizations(mom[i])
        J[i][i+1] = epsp if hels[i] == +1 else epsm

    for length in range(2, n+1):
        for i in range(0, n-length+1):
            j = i+length
            Pij = np.sum(mom[i:j], axis=0)
            P[i][j] = Pij
            cur = np.zeros(4, dtype=np.complex128)

            # 3-vertex partitions
            for k in range(i+1, j):
                J1 = J[i][k]
                J2 = J[k][j]
                p = -Pij
                q = np.sum(mom[i:k], axis=0)
                r = np.sum(mom[k:j], axis=0)
                cur += _contract_V3(J1, J2, p, q, r)

            # 4-vertex partitions
            for k in range(i+1, j-1):
                for l in range(k+1, j):
                    Ja = J[i][k]; Jb = J[k][l]; Jc = J[l][j]
                    cur += _V4_contact(Ja, Jb, Jc)

            den = mass2(Pij)
            J[i][j] = (g_s * cur) / (den + 1e-30)

    return J[0][n]
