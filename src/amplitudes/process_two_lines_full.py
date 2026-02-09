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
from .color_internal_radiation import color_matrix_two_lines_internal

_eta = np.diag([1,-1,-1,-1]).astype(np.complex128)

def _assignments(ng: int, include_internal: bool) -> list[tuple[tuple[int,...], tuple[int,...], tuple[int,...]]]:
    """Return list of (w1,wint,w2) triplets as label tuples.

    If include_internal=False, wint is always empty.
    """
    out=[]
    if not include_internal:
        for mask in range(1<<ng):
            w1=[]; w2=[]
            for i in range(ng):
                lab=i+1
                if (mask>>i)&1:
                    w2.append(lab)
                else:
                    w1.append(lab)
            out.append((tuple(w1), tuple(), tuple(w2)))
        return out

    for assign in itertools.product((0,1,2), repeat=ng):  # 0=line1,1=internal,2=line2
        w1=[]; wint=[]; w2=[]
        for i,a in enumerate(assign):
            lab=i+1
            if a==0: w1.append(lab)
            elif a==1: wint.append(lab)
            else: w2.append(lab)
        out.append((tuple(w1), tuple(wint), tuple(w2)))
    return out

def _exchange_splits(w1: tuple[int,...], w2: tuple[int,...]) -> list[tuple[int,int]]:
    return [(s1,s2) for s1 in range(len(w1)+1) for s2 in range(len(w2)+1)]

@dataclass(frozen=True)
class TwoLineFullTreeEngine:
    """Unified two-line exact-color process engine.

    Includes:
      - gluons attached to either quark line, with exchange-point insertion on each line
      - gluons emitted from the exchanged internal gluon via the bi-current B_{μν}
      - exact SU(Nc) color sum including structure-constant chains on the exchanged line
        via f→trace expansion + linear-combo Gram matrix.

    External state convention:
      legs are all-outgoing and must contain exactly:
        q, qb, q, qb  (two flavours allowed) and ng gluons.
      Gluons are labeled 1..ng in the order they appear in legs.
    """
    params: SMParams = SMParams()
    Nc: int = 3
    include_internal: bool = True

    def _J_split(self, mom_line: np.ndarray, legs_line: Sequence[Particle], split: int) -> np.ndarray:
        bg = FermionLineBG(params=self.params)
        J = np.zeros(4, dtype=np.complex128)
        for mu in range(4):
            J[mu] = bg.current_to_offshell_gluon_split(mom_line, legs_line, split=split, mu=mu)
        return J

    def _extract_lines(self, mom: np.ndarray, legs: Sequence[Particle]):
        idx_q = [i for i,p in enumerate(legs) if p.kind=="q"]
        idx_qb = [i for i,p in enumerate(legs) if p.kind=="qb"]
        idx_g = [i for i,p in enumerate(legs) if p.kind=="g"]
        if len(idx_q)!=2 or len(idx_qb)!=2:
            raise ValueError("Need 2 quarks + 2 antiquarks (all-outgoing).")
        q0,q1 = idx_q
        qb0,qb1 = idx_qb
        if legs[q0].flavor and legs[qb0].flavor and legs[q0].flavor != legs[qb0].flavor:
            qb0,qb1 = qb1,qb0
        label_to_idx = {lab: idx_g[lab-1] for lab in range(1, len(idx_g)+1)}
        return q0,qb0,q1,qb1,label_to_idx,len(idx_g)

    def primitive_amplitude(
        self,
        mom: np.ndarray,
        legs: Sequence[Particle],
        w1: tuple[int,...], wint: tuple[int,...], w2: tuple[int,...],
        s1: int, s2: int,
    ) -> complex:
        mom = np.asarray(mom, dtype=np.complex128)
        q0,qb0,q1,qb1,label_to_idx,ng = self._extract_lines(mom, legs)

        line1_idx = [q0] + [label_to_idx[x] for x in w1] + [qb0]
        line2_idx = [q1] + [label_to_idx[x] for x in w2] + [qb1]
        int_idx   = [label_to_idx[x] for x in wint]

        mom1 = mom[line1_idx]; legs1=[legs[i] for i in line1_idx]
        mom2 = mom[line2_idx]; legs2=[legs[i] for i in line2_idx]

        J1 = self._J_split(mom1, legs1, split=s1)
        J2 = self._J_split(mom2, legs2, split=s2)

        qL = np.sum(mom1, axis=0)
        if len(int_idx)==0:
            # no internal emissions: B = η / q^2
            B = _eta / (mass2(qL) + 1e-30)
        else:
            momI = mom[int_idx]
            helI = tuple(legs[i].hel for i in int_idx)
            B = gluon_bicurrent_color_ordered(momI, helI, qL=qL, g_s=self.params.gs())

        return complex(J1 @ (B @ J2))

    def me2_exact_color_sum(self, mom: np.ndarray, legs: Sequence[Particle]) -> float:
        mom = np.asarray(mom, dtype=np.complex128)
        _,_,_,_,_,ng = self._extract_lines(mom, legs)

        triplets = _assignments(ng, include_internal=self.include_internal)
        # Build exact color matrix on unique triplets
        C = color_matrix_two_lines_internal(triplets, Nc=self.Nc)

        # Expand with exchange split positions
        amps: List[complex] = []
        tids: List[int] = []
        for ti,(w1,wint,w2) in enumerate(triplets):
            for s1,s2 in _exchange_splits(w1,w2):
                amps.append(self.primitive_amplitude(mom, legs, w1,wint,w2, s1,s2))
                tids.append(ti)

        A = np.asarray(amps, dtype=np.complex128)
        t = np.asarray(tids, dtype=int)

        me2 = 0.0 + 0j
        for i in range(len(A)):
            ai = np.conjugate(A[i])
            ti = t[i]
            for j in range(len(A)):
                me2 += ai * C[ti][t[j]] * A[j]
        return float(me2.real)


    def me2_helicity_sum(
        self,
        mom: np.ndarray,
        legs: Sequence[Particle],
        average_initial: bool = False,
        average_initial_colors: bool = False,
    ) -> float:
        """Sum |M|^2 over external helicities (fermions+gluons) by brute force.

        This is intended for validation at low multiplicity.

        Parameters
        ----------
        average_initial:
            If True, divides by 4 for two initial spin-1/2 particles (2 helicities each).

        average_initial_colors:
            If True, divides by Nc^2 for two initial colored fermions.
            (Color averaging is NOT applied here; color sum is exact by construction.)
        """
        # identify helicity-carrying legs
        idx = list(range(len(legs)))
        # build per-leg helicity options
        opts=[]
        for i,p in enumerate(legs):
            if p.kind in ("q","qb","g"):
                opts.append((i, (-1, +1)))
            else:
                opts.append((i, (p.hel,)))
        total = 0.0
        for assn in itertools.product(*[o[1] for o in opts]):
            legs2 = list(legs)
            for (i,_),h in zip(opts, assn):
                legs2[i] = Particle(legs2[i].kind, int(h), flavor=getattr(legs2[i], "flavor", None))
            total += self.me2_exact_color_sum(mom, legs2)
        if average_initial:
            total /= 4.0
        if average_initial_colors:
            total /= float(self.Nc * self.Nc)
        return float(total)
