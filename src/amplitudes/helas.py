from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from .lorentz import minkowski_dot, mass2
from .polarization import massless_vector_polarizations, massive_vector_polarizations, propagator_den

# Gamma matrices in Weyl basis acting on Dirac spinors (4 components).
# We'll use chiral representation with gamma^0 = [[0,I],[I,0]], gamma^i = [[0,sigma^i],[-sigma^i,0]]
_sigma = [
    np.array([[1,0],[0,1]], dtype=np.complex128),
    np.array([[0,1],[1,0]], dtype=np.complex128),
    np.array([[0,-1j],[1j,0]], dtype=np.complex128),
    np.array([[1,0],[0,-1]], dtype=np.complex128),
]

def gamma_mu(mu: int) -> np.ndarray:
    I2 = np.eye(2, dtype=np.complex128)
    if mu == 0:
        top = np.hstack([np.zeros((2,2),dtype=np.complex128), I2])
        bot = np.hstack([I2, np.zeros((2,2),dtype=np.complex128)])
        return np.vstack([top, bot])
    # spatial
    sig = _sigma[mu]
    top = np.hstack([np.zeros((2,2),dtype=np.complex128), sig])
    bot = np.hstack([-sig, np.zeros((2,2),dtype=np.complex128)])
    return np.vstack([top, bot])

_GAMMA = [gamma_mu(i) for i in range(4)]
_GAMMA5 = np.block([
    [ -np.eye(2, dtype=np.complex128), np.zeros((2,2),dtype=np.complex128)],
    [ np.zeros((2,2),dtype=np.complex128), np.eye(2, dtype=np.complex128)]
])

def proj_L() -> np.ndarray:
    return 0.5*(np.eye(4, dtype=np.complex128) - _GAMMA5)

def proj_R() -> np.ndarray:
    return 0.5*(np.eye(4, dtype=np.complex128) + _GAMMA5)

@dataclass
class VectorWF:
    p: np.ndarray  # (4,)
    eps: np.ndarray  # (4,)
    m: float = 0.0
    width: float = 0.0
    kind: str = "V"  # A/Z/W/g

@dataclass
class FermionWF:
    p: np.ndarray  # (4,)
    u: np.ndarray  # (4,) Dirac spinor
    kind: str = "f" # f or fbar
    hel: int = +1

def massless_dirac_spinors_from_spinor_helicity(p: np.ndarray, hel: int) -> np.ndarray:
    """
    Build a Dirac spinor u(p,hel) for massless p using a numerically stable construction from 2-component spinors.

    We reuse the existing spinor-helicity extraction in SpinorPoint.from_momenta by mapping to a 2-spinor basis:
      p_{a dot a} = |p>_a [p|_{dot a}
    Then u = ( |p> , |p] ) in chiral basis, with helicity selecting which component is non-zero in amplitudes.
    For practical HELAS-like contractions, we populate both components.
    """
    from .spinor import SpinorPoint
    sp = SpinorPoint.from_momenta(np.asarray([p], dtype=np.complex128))
    lam = sp.lam[0]   # |p>
    lamt = sp.lamt[0] # |p]
    # Dirac spinor in Weyl basis: (chi_a, phi^{dot a})
    # We'll set chi=lam and phi=lamt.
    u = np.zeros(4, dtype=np.complex128)
    u[0:2] = lam
    u[2:4] = lamt
    # helicity is used via projectors in vertices; keep full spinor here.
    return u

def vector_wf(p: np.ndarray, hel: int, kind: str, m: float = 0.0, width: float = 0.0) -> VectorWF:
    p = np.asarray(p, dtype=np.complex128)
    if abs(m) < 1e-14:
        epsp, epsm = massless_vector_polarizations(p)
        eps = epsp if hel == +1 else epsm
    else:
        epsp, epsm, eps0 = massive_vector_polarizations(p, m)
        if hel == +1:
            eps = epsp
        elif hel == -1:
            eps = epsm
        else:
            # treat 0 as longitudinal if user uses 0 (not in Particle API, but internal use)
            eps = eps0
    return VectorWF(p=p, eps=eps, m=float(m), width=float(width), kind=kind)

def fermion_wf(p: np.ndarray, hel: int, kind: str) -> FermionWF:
    p = np.asarray(p, dtype=np.complex128)
    u = massless_dirac_spinors_from_spinor_helicity(p, hel)
    return FermionWF(p=p, u=u, kind=kind, hel=hel)

def ffv_amplitude(psi_out: FermionWF, psi_in: FermionWF, V: VectorWF, gL: complex, gR: complex) -> complex:
    """Compute ar{u}(out) gamma^mu (gL PL + gR PR) u(in) eps_mu."""
    ubar = psi_out.u.conjugate().T @ _GAMMA[0]  # Dirac adjoint: u^\dagger gamma^0
    PL = proj_L()
    PR = proj_R()
    J = np.zeros(4, dtype=np.complex128)
    for mu in range(4):
        J[mu] = ubar @ _GAMMA[mu] @ (gL*PL + gR*PR) @ psi_in.u
    return np.dot(J, V.eps)

def vvv_amplitude(V1: VectorWF, V2: VectorWF, V3: VectorWF, g: complex) -> complex:
    """Triple gauge vertex contraction (colorless version) with Feynman rule structure."""
    p1,p2,p3 = V1.p, V2.p, V3.p
    e1,e2,e3 = V1.eps, V2.eps, V3.eps
    eta = np.diag([1,-1,-1,-1]).astype(np.complex128)
    # V^{mu nu rho} e1_mu e2_nu e3_rho
    out = 0.0 + 0j
    for mu in range(4):
        for nu in range(4):
            for rho in range(4):
                Vmunurho = (
                    eta[nu,rho]*(p2-p3)[mu] +
                    eta[rho,mu]*(p3-p1)[nu] +
                    eta[mu,nu]*(p1-p2)[rho]
                )
                out += Vmunurho * e1[mu]*e2[nu]*e3[rho]
    return g*out

def vector_propagator_factor(p: np.ndarray, m: float, width: float, scheme: str) -> complex:
    p2 = mass2(p)
    den = propagator_den(p2, m, width, scheme=scheme)
    return 1.0/den


def slash(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=np.complex128)
    return p[0]*_GAMMA[0] + p[1]*_GAMMA[1] + p[2]*_GAMMA[2] + p[3]*_GAMMA[3]

def antifermion_wf(p: np.ndarray, hel: int, kind: str) -> FermionWF:
    """Outgoing antifermion v(p,hel). For massless spinors, v(p,h) can be represented by u(p,-h) up to a phase."""
    p = np.asarray(p, dtype=np.complex128)
    u = massless_dirac_spinors_from_spinor_helicity(p, -hel)
    return FermionWF(p=p, u=u, kind=kind, hel=hel)
