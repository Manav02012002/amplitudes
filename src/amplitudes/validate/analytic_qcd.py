from __future__ import annotations
import numpy as np

from ..color_quark import color_matrix_qqbar_ng_exact
from ..me_quark import all_partials_qqbar_ng
from ..parke_taylor import parke_taylor_mhv
from ..spinor import SpinorPoint

def mandelstam_s_t_u(p1, p2, p3, p4):
    """Return (s,t,u) with all-incoming convention:
      p1+p2 -> p3+p4  (p3,p4 outgoing physically, but here treated as outgoing in momentum array)
    We assume momenta are (E,px,py,pz) with metric (+,-,-,-).
    """
    p1 = np.asarray(p1, dtype=float); p2=np.asarray(p2, dtype=float)
    p3 = np.asarray(p3, dtype=float); p4=np.asarray(p4, dtype=float)

    def dot(a,b):
        return a[0]*b[0] - a[1]*b[1] - a[2]*b[2] - a[3]*b[3]
    s = dot(p1+p2, p1+p2)
    t = dot(p1-p3, p1-p3)
    u = dot(p1-p4, p1-p4)
    return float(s), float(t), float(u)

def me2_avg_qqp_to_qqp(s: float, t: float, u: float, gs: float, Nc: int = 3) -> float:
    """Spin+color averaged |M|^2 for q q' -> q q' (different flavours), massless, tree-level QCD.

    Standard result (for Nc=3):
      |M|^2_avg = (4/9) g_s^4 (s^2 + u^2)/t^2

    For general Nc:
      color factor is (Nc^2-1)/(4 Nc^2) * 4 = (Nc^2-1)/(Nc^2)
      but the conventional prefactors depend on averaging conventions.
    We provide Nc=3 exact factor commonly used in phenomenology.
    """
    if abs(t) < 1e-18:
        return float("inf")
    if Nc != 3:
        # provide a reasonable generalization using C_F = (Nc^2-1)/(2Nc)
        # averaged over colors: 1/Nc^2, summed: C_F^2*4? We keep Nc=3 as the tested case.
        raise NotImplementedError("Analytic general-Nc qq'->qq' reference implemented for Nc=3 only.")
    return (4.0/9.0) * (gs**4) * ((s*s + u*u) / (t*t))


def mhv_gluon_reference(sp: SpinorPoint, negative_legs: tuple[int, int]) -> complex:
    """Exact Parke-Taylor reference for an n-gluon MHV tree amplitude."""
    return parke_taylor_mhv(sp, negative_legs[0], negative_legs[1])


def me2_qqbar_gg_from_partials(sp: SpinorPoint, hels: tuple[int, int, int, int], Nc: int = 3) -> float:
    """Exact q qbar + 2 gluons color-summed |M|^2 from partial amplitudes at g_s=1."""
    basis, partials = all_partials_qqbar_ng(sp, hels)
    if len(basis) != 2:
        raise ValueError("Expected exactly two color orderings for q qbar + 2 gluons.")
    color = color_matrix_qqbar_ng_exact(2, Nc)
    return float((partials.conjugate() @ (color @ partials)).real)
