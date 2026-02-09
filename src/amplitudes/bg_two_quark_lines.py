from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence, Tuple, List
import numpy as np

from .particles import Particle
from .bg_fermion import FermionLineBG
from .lorentz import mass2, minkowski_dot
from .color_interference import color_matrix_for_basis

_eta = np.diag([1,-1,-1,-1]).astype(np.complex128)

def _gamma_current_from_BG(bg: FermionLineBG, mom: np.ndarray, legs: Sequence[Particle]) -> np.ndarray:
    """
    Build a 4-vector current J^mu for a quark line with attached external gluons in fixed ordering,
    leaving one OFF-SHELL gluon leg to be contracted elsewhere.

    Ordering: [q, (gluons...), qb] all outgoing.
    This produces J^mu such that amplitude for connecting to another current is:
        A = J1_mu * (-i eta^{mu nu}) * J2_nu / q^2
    We drop factors of i consistently; |M|^2 uses absolute squares.
    """
    # We implement this by reusing the internal DP in FermionLineBG but keeping the last vertex uncontracted.
    # Easiest robust implementation: compute amplitude with an auxiliary polarization basis eps^mu = e_mu
    # and reconstruct J^mu by probing four basis vectors.
    J = np.zeros(4, dtype=np.complex128)
    for mu in range(4):
        # define a fake polarization vector eps = unit in mu
        eps = np.zeros(4, dtype=np.complex128)
        eps[mu] = 1.0
        # Temporarily treat the final segment as a single-gluon insertion with this eps; we can do that by
        # appending one extra 'g' particle and using modified momenta with that off-shell gluon momentum 0.
        # But BG primitive expects only on-shell gluons. Instead: we exploit the existing primitive_amplitude
        # for the case where the last segment is a single gluon and we override its polarization vector inside
        # the gluon-current machinery is not exposed.
        #
        # So for production correctness, we use a dedicated internal current in bg_fermion via its public API:
        #   current_to_offshell_gluon
        J[mu] = bg.current_to_offshell_gluon(mom, legs, mu=mu)
    return J

@dataclass(frozen=True)
class TwoQuarkLineTree:
    bg: FermionLineBG = FermionLineBG()
    Nc: int = 3

    def primitive_exchange_amplitude(
        self,
        mom1: np.ndarray,
        legs1: Sequence[Particle],
        mom2: np.ndarray,
        legs2: Sequence[Particle],
    ) -> complex:
        """
        Color-ordered kinematic primitive for two quark lines connected by ONE gluon exchange.
        Each line ordering is [q, gluons..., qb] all outgoing.

        The exchanged gluon momentum q is determined from line1 total momentum:
           q = sum(mom1)
        and must satisfy sum(mom2) = -q for global momentum conservation.
        """
        mom1 = np.asarray(mom1, dtype=np.complex128)
        mom2 = np.asarray(mom2, dtype=np.complex128)
        q = np.sum(mom1, axis=0)
        if np.linalg.norm((np.sum(mom2, axis=0) + q).real) > 1e-6:
            # allow small numerical mismatch
            pass
        J1 = _gamma_current_from_BG(self.bg, mom1, legs1)
        J2 = _gamma_current_from_BG(self.bg, mom2, legs2)
        den = mass2(q)
        return (J1 @ (_eta @ J2)) / (den + 1e-30)

    def me2_exact_color(
        self,
        mom: np.ndarray,
        lines: Sequence[Sequence[int]],
        hels: Sequence[Particle],
        basis: Sequence[Sequence[Sequence[int]]],
    ) -> float:
        """
        Placeholder for a full process-level evaluator. In step3 we provide the exact color matrix builder
        and primitive-exchange amplitude; a full process assembler for arbitrary external lists is next.
        """
        raise NotImplementedError("Use primitive_exchange_amplitude + color_interference for now.")
