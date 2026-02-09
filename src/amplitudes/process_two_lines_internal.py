from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence, List, Tuple
import itertools
import numpy as np

from .particles import Particle
from .sm import SMParams
from .bg_fermion import FermionLineBG
from .bg_bicurrent import gluon_bicurrent_color_ordered
from .lorentz import mass2
from .color_interference import color_matrix_for_basis

_eta = np.diag([1,-1,-1,-1]).astype(np.complex128)

def _basis_assignments(ng: int):
    """Assign each gluon label to one of: line1, internal, line2."""
    # 0=line1, 1=internal, 2=line2
    for assign in itertools.product((0,1,2), repeat=ng):
        w1=[]; wint=[]; w2=[]
        for i,a in enumerate(assign):
            lab=i+1
            if a==0: w1.append(lab)
            elif a==1: wint.append(lab)
            else: w2.append(lab)
        yield (w1, wint, w2)

def _all_exchange_splits(w1, w2):
    for s1 in range(len(w1)+1):
        for s2 in range(len(w2)+1):
            yield (s1, s2)

def _color_words_only_lines(w1, w2):
    return [w1, w2]

@dataclass(frozen=True)
class TwoLineWithInternalGluonRadiation:
    """Two-line qqbar->qqbar+ng with internal-gluon radiation kinematics (bi-current).

    IMPORTANT: Exact SU(Nc) color sum for internal-gluon radiation requires explicit f^{abc} color factors
    (and their trace expansion). That extension is nontrivial and not implemented here.
    Current implementation:
      - exact kinematics for internal-line radiation via bi-current B^{μν}
      - exact color interference for the FUNDAMENTAL-LINE words only (ignores extra structure-constant factors for internal emissions)

    This is provided as a kinematic engine + scaffolding for the upcoming exact-color completion.
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
        wint: list[int],
        w2: list[int], s2: int,
    ) -> complex:
        mom = np.asarray(mom, dtype=np.complex128)
        idx_q = [i for i,p in enumerate(legs) if p.kind=="q"]
        idx_qb = [i for i,p in enumerate(legs) if p.kind=="qb"]
        idx_g = [i for i,p in enumerate(legs) if p.kind=="g"]
        if len(idx_q)!=2 or len(idx_qb)!=2:
            raise ValueError("Need 2 quarks + 2 antiquarks.")
        q0,q1 = idx_q
        qb0,qb1 = idx_qb
        if legs[q0].flavor and legs[qb0].flavor and legs[q0].flavor != legs[qb0].flavor:
            qb0,qb1 = qb1,qb0

        ng = len(idx_g)
        label_to_idx = {lab: idx_g[lab-1] for lab in range(1, ng+1)}

        line1_glu_idx = [label_to_idx[lab] for lab in w1]
        line2_glu_idx = [label_to_idx[lab] for lab in w2]
        int_glu_idx   = [label_to_idx[lab] for lab in wint]

        line1_idx = [q0] + line1_glu_idx + [qb0]
        line2_idx = [q1] + line2_glu_idx + [qb1]
        mom1 = mom[line1_idx]; legs1 = [legs[i] for i in line1_idx]
        mom2 = mom[line2_idx]; legs2 = [legs[i] for i in line2_idx]

        J1 = self._J_split(mom1, legs1, split=s1)
        J2 = self._J_split(mom2, legs2, split=s2)

        qL = np.sum(mom1, axis=0)
        # internal gluons:
        momI = mom[int_glu_idx]
        helI = tuple(legs[i].hel for i in int_glu_idx)
        B = gluon_bicurrent_color_ordered(momI, helI, qL=qL, g_s=self.params.gs())
        # contract:
        return complex(J1 @ (B @ J2))

    def me2_color_approx(
        self,
        mom: np.ndarray,
        legs: Sequence[Particle],
    ) -> float:
        """Approximate color sum for internal-radiation basis.

        Internal-gluon radiation introduces adjoint self-couplings (structure constants) on the exchanged line.
        The current trace-only interference engine does not yet include those exact color factors and their
        interference between different internal/line assignments.

        Safe intermediate: diagonal-only sum over basis elements with exact line-word weights:
            |M|^2 ≈ Σ_i C_ii |A_i|^2,
        where C_ii is computed exactly in SU(Nc) for the fundamental-line trace words of that element.
        """
        idx_g = [i for i,p in enumerate(legs) if p.kind=="g"]
        ng = len(idx_g)

        from .color_interference import color_gram_multi_quark_lines

        me2 = 0.0
        for w1, wint, w2 in _basis_assignments(ng):
            # exact trace weight for this assignment (line words only)
            Cii = color_gram_multi_quark_lines([w1, w2], [w1, w2], Nc=self.Nc)
            for s1, s2 in _all_exchange_splits(w1, w2):
                Ai = self.primitive_amplitude(mom, legs, w1, s1, wint, w2, s2)
                me2 += Cii * (abs(Ai)**2)
        return float(me2)

    def me2_exact_color_sum(
        self,
        mom: np.ndarray,
        legs: Sequence[Particle],
    ) -> float:
        """Exact SU(Nc) color-summed |M|^2 including internal-gluon radiation on the exchanged line.

        This computes the full Gram matrix in a trace basis after expanding the adjoint-chain color
        factors (products of structure constants) into linear combinations of fundamental traces.
        """
        idx_g = [i for i,p in enumerate(legs) if p.kind=="g"]
        ng = len(idx_g)

        # Build basis elements: (w1,wint,w2,s1,s2)
        basis=[]
        triplets=[]
        for w1, wint, w2 in _basis_assignments(ng):
            triplets.append((tuple(w1), tuple(wint), tuple(w2)))
            for s1, s2 in _all_exchange_splits(w1, w2):
                basis.append((tuple(w1), tuple(wint), tuple(w2), s1, s2))

        # Unique triplets for color matrix
        uniq = list(dict.fromkeys(triplets).keys())
        from .color_internal_radiation import color_matrix_two_lines_internal
        Ctrip = color_matrix_two_lines_internal(uniq, Nc=self.Nc)

        # map triplet -> index
        tindex = {t:i for i,t in enumerate(uniq)}

        # Compute amplitudes for each basis element, grouped by triplet
        A = np.zeros(len(basis), dtype=np.complex128)
        t_of = np.zeros(len(basis), dtype=int)
        for i,(w1,wint,w2,s1,s2) in enumerate(basis):
            A[i] = self.primitive_amplitude(mom, legs, list(w1), s1, list(wint), list(w2), s2)
            t_of[i] = tindex[(w1,wint,w2)]

        # Full sum: A_i^* C_{t(i),t(j)} A_j
        me2 = 0.0 + 0j
        for i in range(len(basis)):
            ai = np.conjugate(A[i])
            ti = t_of[i]
            for j in range(len(basis)):
                tj = t_of[j]
                me2 += ai * Ctrip[ti][tj] * A[j]
        return float(me2.real)
