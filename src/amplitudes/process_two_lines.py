from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence, List
import itertools
import numpy as np

from .particles import Particle
from .sm import SMParams
from .bg_fermion import FermionLineBG
from .lorentz import mass2
from .color_interference import color_matrix_for_basis

_eta = np.diag([1,-1,-1,-1]).astype(np.complex128)

def _basis_splits(ng: int) -> list[list[list[int]]]:
    """Basis elements: assign each gluon label 1..ng to either line1 or line2 preserving label order."""
    basis = []
    for mask in range(1 << ng):
        w1: list[int] = []
        w2: list[int] = []
        for i in range(ng):
            lab = i + 1
            if (mask >> i) & 1:
                w2.append(lab)
            else:
                w1.append(lab)
        basis.append([w1, w2])
    return basis

@dataclass(frozen=True)
class TwoLineQQbarToQQbarJets:
    """
    Exact color-summed |M|^2 for a two-fermion-line process with a single exchanged gluon topology.

    Included:
      - Two fermion lines connected by ONE internal gluon propagator.
      - External gluons assigned to either fermion line with a fixed ordering per line.
      - On each line, the exchanged gluon is emitted AFTER all external gluons in that line ordering.

    Not yet included (requires richer multi-line current algebra):
      - emissions on both sides of the exchange on a given line
      - gluons emitted from the exchanged gluon via 3g/4g
      - multi-exchange topologies
    """
    params: SMParams = SMParams()
    Nc: int = 3

    def _current(self, mom_line: np.ndarray, legs_line: Sequence[Particle]) -> np.ndarray:
        bg = FermionLineBG(params=self.params)
        J = np.zeros(4, dtype=np.complex128)
        for mu in range(4):
            J[mu] = bg.current_to_offshell_gluon(mom_line, legs_line, mu=mu)
        return J

    def primitive_amplitude_for_split(
        self,
        mom: np.ndarray,
        legs: Sequence[Particle],
        split: Sequence[Sequence[int]],
    ) -> complex:
        """
        Compute the kinematic amplitude for a given basis split.

        Conventions:
          legs are all-outgoing and must contain:
            q1, qb1, q2, qb2, and ng gluons (kind='g') with implicit labels 1..ng
            given by the order of appearance of gluons in `legs`.
          split is [w1, w2] where w1/w2 are ordered lists of gluon labels assigned to line1/line2.

        We build two line arrays [q, gluons..., qb] for each line and compute:
            A = (J1·J2) / q^2
        where q is the exchanged gluon momentum leaving line1.
        """
        mom = np.asarray(mom, dtype=np.complex128)
        if len(mom) != len(legs):
            raise ValueError("mom/legs length mismatch")

        idx_q = [i for i, p in enumerate(legs) if p.kind == "q"]
        idx_qb = [i for i, p in enumerate(legs) if p.kind == "qb"]
        idx_g = [i for i, p in enumerate(legs) if p.kind == "g"]
        if len(idx_q) != 2 or len(idx_qb) != 2:
            raise ValueError("Need exactly two quarks and two antiquarks (all-outgoing).")

        q0, q1 = idx_q
        qb0, qb1 = idx_qb
        # Pair by flavor if provided
        if legs[q0].flavor and legs[qb0].flavor and legs[q0].flavor != legs[qb0].flavor:
            qb0, qb1 = qb1, qb0

        ng = len(idx_g)
        label_to_idx = {lab: idx_g[lab - 1] for lab in range(1, ng + 1)}

        w1, w2 = split
        line1_glu_idx = [label_to_idx[lab] for lab in w1]
        line2_glu_idx = [label_to_idx[lab] for lab in w2]

        line1_idx = [q0] + line1_glu_idx + [qb0]
        line2_idx = [q1] + line2_glu_idx + [qb1]

        mom1 = mom[line1_idx]
        mom2 = mom[line2_idx]
        legs1 = [legs[i] for i in line1_idx]
        legs2 = [legs[i] for i in line2_idx]

        J1 = self._current(mom1, legs1)
        J2 = self._current(mom2, legs2)

        q = np.sum(mom1, axis=0)
        den = mass2(q)
        return (J1 @ (_eta @ J2)) / (den + 1e-30)

    def me2_exact_color_sum(
        self,
        mom: np.ndarray,
        legs: Sequence[Particle],
    ) -> float:
        """Exact color-summed |M|^2 for the implemented topology (see class docstring)."""
        idx_g = [i for i, p in enumerate(legs) if p.kind == "g"]
        ng = len(idx_g)

        basis = _basis_splits(ng)
        C = color_matrix_for_basis(basis, Nc=self.Nc)

        A = np.zeros(len(basis), dtype=np.complex128)
        for i, split in enumerate(basis):
            A[i] = self.primitive_amplitude_for_split(mom, legs, split)

        me2 = 0.0 + 0j
        for i in range(len(basis)):
            ai = np.conjugate(A[i])
            for j in range(len(basis)):
                me2 += ai * C[i][j] * A[j]
        return float(me2.real)
