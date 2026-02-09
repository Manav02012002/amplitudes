from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence, List, Tuple
import itertools
import numpy as np

from .particles import Particle
from .sm import SMParams
from .bg_fermion import FermionLineBG
from .lorentz import mass2
from .color_interference import color_matrix_for_basis

_eta = np.diag([1,-1,-1,-1]).astype(np.complex128)

def _splits_with_exchange_positions(ng: int) -> list[tuple[list[int], int, list[int], int]]:
    """Enumerate basis with gluon assignment to two lines and exchange insertion positions.

    Returns tuples (w1, s1, w2, s2) where:
      - w1: labels on line1 in order; s1 in [0,len(w1)] is exchange split position on that line
      - w2: labels on line2 in order; s2 similarly
    """
    out=[]
    for mask in range(1<<ng):
        w1=[]
        w2=[]
        for i in range(ng):
            lab=i+1
            if (mask>>i)&1:
                w2.append(lab)
            else:
                w1.append(lab)
        for s1 in range(len(w1)+1):
            for s2 in range(len(w2)+1):
                out.append((w1, s1, w2, s2))
    return out

def _color_basis_words(basis: list[tuple[list[int],int,list[int],int]]) -> list[list[list[int]]]:
    """Convert basis with splits to color words per line (exchange split doesn't affect color word)."""
    words=[]
    for w1,_,w2,_ in basis:
        words.append([w1, w2])
    return words

@dataclass(frozen=True)
class TwoLineQQbarToQQbarJetsExchange:
    """Exact color-summed |M|^2 for two quark lines connected by one exchanged gluon,
    including exchange-point insertion (gluons on both sides of the exchange on each line).

    This includes all diagrams where:
      - external gluons attach to either quark line,
      - and, on a given line, some gluons are emitted before the exchanged gluon and some after,
      - gluon self-interactions are included separately within the 'before' block and within the 'after' block via BG currents.

    Still not included:
      - gluon radiation emitted from the exchanged gluon itself (internal-gluon radiation), which couples left/right blocks.
    """
    params: SMParams = SMParams()
    Nc: int = 3

    def _J_split(self, mom_line: np.ndarray, legs_line: Sequence[Particle], split: int) -> np.ndarray:
        bg = FermionLineBG(params=self.params)
        J = np.zeros(4, dtype=np.complex128)
        for mu in range(4):
            J[mu] = bg.current_to_offshell_gluon_split(mom_line, legs_line, split=split, mu=mu)
        return J

    def primitive_amplitude(
        self,
        mom: np.ndarray,
        legs: Sequence[Particle],
        w1: list[int], s1: int,
        w2: list[int], s2: int,
    ) -> complex:
        mom = np.asarray(mom, dtype=np.complex128)
        idx_q = [i for i,p in enumerate(legs) if p.kind=="q"]
        idx_qb = [i for i,p in enumerate(legs) if p.kind=="qb"]
        idx_g = [i for i,p in enumerate(legs) if p.kind=="g"]
        if len(idx_q)!=2 or len(idx_qb)!=2:
            raise ValueError("Need 2 quarks + 2 antiquarks (all-outgoing).")
        q0,q1 = idx_q
        qb0,qb1 = idx_qb
        if legs[q0].flavor and legs[qb0].flavor and legs[q0].flavor != legs[qb0].flavor:
            qb0,qb1 = qb1,qb0

        ng = len(idx_g)
        label_to_idx = {lab: idx_g[lab-1] for lab in range(1, ng+1)}

        line1_glu_idx = [label_to_idx[lab] for lab in w1]
        line2_glu_idx = [label_to_idx[lab] for lab in w2]

        line1_idx = [q0] + line1_glu_idx + [qb0]
        line2_idx = [q1] + line2_glu_idx + [qb1]
        mom1 = mom[line1_idx]; legs1 = [legs[i] for i in line1_idx]
        mom2 = mom[line2_idx]; legs2 = [legs[i] for i in line2_idx]

        J1 = self._J_split(mom1, legs1, split=s1)
        J2 = self._J_split(mom2, legs2, split=s2)

        qex = np.sum(mom1, axis=0)  # exchanged momentum leaving line1
        den = mass2(qex)
        return (J1 @ (_eta @ J2)) / (den + 1e-30)

    def me2_exact_color_sum(self, mom: np.ndarray, legs: Sequence[Particle]) -> float:
        idx_g = [i for i,p in enumerate(legs) if p.kind=="g"]
        ng = len(idx_g)

        basis = _splits_with_exchange_positions(ng)
        color_words = _color_basis_words(basis)
        C = color_matrix_for_basis(color_words, Nc=self.Nc)

        A = np.zeros(len(basis), dtype=np.complex128)
        for i,(w1,s1,w2,s2) in enumerate(basis):
            A[i] = self.primitive_amplitude(mom, legs, w1, s1, w2, s2)

        me2 = 0.0 + 0j
        for i in range(len(basis)):
            ai = np.conjugate(A[i])
            for j in range(len(basis)):
                me2 += ai * C[i][j] * A[j]
        return float(me2.real)
