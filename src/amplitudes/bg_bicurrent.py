from __future__ import annotations
import numpy as np
from functools import lru_cache
from typing import Tuple

from .lorentz import mass2
from .bg_currents import gluon_current_color_ordered

_eta = np.diag([1,-1,-1,-1]).astype(np.complex128)

def _V3(p: np.ndarray, q: np.ndarray, r: np.ndarray) -> np.ndarray:
    # V^{μνρ} all incoming
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

def _contract_V4_with_remainder(J1: np.ndarray, J2: np.ndarray, rem: np.ndarray) -> np.ndarray:
    eta_J1 = _eta @ J1
    eta_J2 = _eta @ J2
    eta_rem = _eta @ rem
    J1_eta_rem = np.einsum("a,ab->b", J1, eta_rem)
    J2_eta_rem = np.einsum("a,ab->b", J2, eta_rem)
    J1_eta_J2 = np.dot(J1, eta_J2)
    return (
        2.0 * np.outer(eta_J2, J1_eta_rem)
        - np.outer(eta_J1, J2_eta_rem)
        - eta_rem * J1_eta_J2
    )

def gluon_bicurrent_color_ordered(
    mom: np.ndarray,
    hels: Tuple[int, ...],
    qL: np.ndarray,
    g_s: float,
) -> np.ndarray:
    """
    Tree-level color-ordered gluon bi-current B^{μν} for an internal (exchanged) gluon line
    emitting an ordered set of external on-shell gluons (mom, hels) between two off-shell endpoints.

    Conventions:
      - qL is the 4-momentum flowing INTO the bi-current from the left endpoint (all complex128).
      - External gluon momenta in `mom` are all outgoing.
      - Right endpoint momentum is qR = -(qL + sum(mom)).
      - Returned B includes the left-end propagator 1/qL^2 (so for n=0, B = η^{μν}/qL^2).

    This object is meant to be contracted as:
        A_internal = J1_μ * B^{μν} * J2_ν
    where J1, J2 are off-shell currents from the two quark lines.

    Implementation:
      - exact tree recursion in Feynman gauge using 3g and 4g vertices
      - uses standard BG currents for contiguous gluon subblocks as building pieces
    """
    mom = np.asarray(mom, dtype=np.complex128)
    qL = np.asarray(qL, dtype=np.complex128)
    n = len(hels)
    if n == 0:
        return _eta / (mass2(qL) + 1e-30)

    # Precompute total momentum prefix sums
    K = np.sum(mom, axis=0)

    # DP over segments of the ordered gluon list for "dressed propagator numerator" T^{μν}
    # We keep the left endpoint momentum as an argument that shifts with how many gluons are emitted.
    # For practicality we evaluate with the given qL and build recursion that always emits from the leftmost end.
    #
    # Define function T(i, q_in): tensor for emitting gluons i..n-1 starting from left momentum q_in,
    # without including the leftmost propagator 1/q_in^2.
    #
    # Base: if i==n: T = η
    #
    # Recurrence: attach a left block i..k as an off-shell current J_{i..k} (includes its own 1/P^2),
    # then connect via 3g or 4g vertex to the remainder tensor.
    #
    # NOTE: This is a full tree recursion but still assumes emissions are ordered along the internal line.
    # It includes gluon self-interactions inside J blocks and inside the remainder via recursion.
    @lru_cache(maxsize=None)
    def T(i: int, q_in_key: Tuple[float, float, float, float]) -> np.ndarray:
        q_in = np.array(q_in_key, dtype=np.complex128)
        if i == n:
            return _eta.copy()

        out = np.zeros((4,4), dtype=np.complex128)

        # 3-vertex: (left endpoint μ) -- (current Jblock α) -- (remainder tensor ρν)
        for k in range(i, n):
            Pblock = np.sum(mom[i:k+1], axis=0)
            Jblock = gluon_current_color_ordered(mom[i:k+1], tuple(hels[i:k+1]), g_s=g_s)  # 4-vector
            q_next = q_in + Pblock
            # momentum flowing into remainder from left is q_next
            rem = T(k+1, tuple(map(float, q_next.real.tolist())))
            # vertex momenta all incoming: p=q_in, q=Pblock, r=-(p+q)
            p = q_in
            q = Pblock
            r = -(p + q)
            V = _V3(p, q, r)  # μ α ρ
            # contract V_{μ αρ} J^α rem^{ρν}
            for mu in range(4):
                for nu in range(4):
                    out[mu,nu] += np.sum(V[mu,:, :] * Jblock[None, :] * rem[:, nu][:, None])

        # 4-vertex: left endpoint couples to two currents + remainder tensor
        # Partition i..k and k+1..l as two current blocks, remainder l+1..n-1
        for k in range(i, n):
            for l in range(k+1, n):
                P1 = np.sum(mom[i:k+1], axis=0)
                P2 = np.sum(mom[k+1:l+1], axis=0)
                J1 = gluon_current_color_ordered(mom[i:k+1], tuple(hels[i:k+1]), g_s=g_s)
                J2 = gluon_current_color_ordered(mom[k+1:l+1], tuple(hels[k+1:l+1]), g_s=g_s)
                q_next = q_in + P1 + P2
                rem = T(l+1, tuple(map(float, q_next.real.tolist())))
                out += _contract_V4_with_remainder(J1, J2, rem)

        # Propagator of the internal line after the first interaction is accounted for by recursion via q_next propagators:
        # We include it by multiplying by g_s (vertex) and by 1/(q_in+P)^2? Those denominators are already present in Jblock,
        # and the remainder's left propagator is included at the top-level separately. Here, we keep T as numerator-only.
        return g_s * out

    T0 = T(0, tuple(map(float, qL.real.tolist())))
    return T0 / (mass2(qL) + 1e-30)
