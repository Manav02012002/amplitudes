from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence
import numpy as np

from .particles import Particle
from .sm import SMParams
from .bg_fermion import FermionLineBG
from .lorentz import mass2
from .color_two_exchange import color_matrix_two_lines_two_exchange
from .color_two_exchange_numeric import color_matrix_two_exchange_su3

_eta = np.diag([1,-1,-1,-1]).astype(np.complex128)

def _line_assignments(ng: int):
    for mask in range(1<<ng):
        w1=[]; w2=[]
        for i in range(ng):
            lab=i+1
            (w2 if (mask>>i)&1 else w1).append(lab)
        yield (tuple(w1), tuple(w2))

@dataclass(frozen=True)
class TwoLineTwoExchangeEngine:
    """Two-line tree engine with TWO exchanged gluons between fermion lines (no internal radiation).

    Uses rank-2 quark-line current J^{mu nu} with split positions:
      q -- A -- G1 -- B -- G2 -- C -- qb

    Includes exact SU(Nc) color weights for ladder + crossed ordering of the two exchanges.
    """
    params: SMParams = SMParams()
    Nc: int = 3

    def _J2(self, mom_line: np.ndarray, legs_line: Sequence[Particle], s1: int, s2: int) -> np.ndarray:
        bg = FermionLineBG(params=self.params)
        J = np.zeros((4,4), dtype=np.complex128)
        for mu in range(4):
            for nu in range(4):
                J[mu,nu] = bg.current_to_two_offshell_gluons_split(mom_line, legs_line, split1=s1, split2=s2, mu=mu, nu=nu)
        return J

    def primitive(self, mom: np.ndarray, legs: Sequence[Particle], w1, w2, s1a,s1b,s2a,s2b, ordering: str) -> complex:
        mom = np.asarray(mom, dtype=np.complex128)
        idx_q = [i for i,p in enumerate(legs) if p.kind=="q"]
        idx_qb = [i for i,p in enumerate(legs) if p.kind=="qb"]
        idx_g = [i for i,p in enumerate(legs) if p.kind=="g"]
        q0,q1 = idx_q; qb0,qb1 = idx_qb
        if legs[q0].flavor and legs[qb0].flavor and legs[q0].flavor != legs[qb0].flavor:
            qb0,qb1 = qb1,qb0
        ng=len(idx_g)
        lab2idx = {lab: idx_g[lab-1] for lab in range(1,ng+1)}

        l1 = [q0] + [lab2idx[x] for x in w1] + [qb0]
        l2 = [q1] + [lab2idx[x] for x in w2] + [qb1]
        mom1=mom[l1]; legs1=[legs[i] for i in l1]
        mom2=mom[l2]; legs2=[legs[i] for i in l2]

        J1=self._J2(mom1, legs1, s1a, s1b)
        J2=self._J2(mom2, legs2, s2a, s2b)

        qex = np.sum(mom1, axis=0)
        den = (mass2(qex)+1e-30)**2

        amp=0.0+0j
        if ordering=="ladder":
            for mu in range(4):
                for nu in range(4):
                    for a in range(4):
                        for b in range(4):
                            amp += J1[mu,nu]*_eta[mu,a]*_eta[nu,b]*J2[a,b]
        elif ordering=="crossed":
            for mu in range(4):
                for nu in range(4):
                    for a in range(4):
                        for b in range(4):
                            amp += J1[mu,nu]*_eta[mu,a]*_eta[nu,b]*J2[b,a]
        else:
            raise ValueError("ordering must be ladder/crossed")
        return complex(amp/den)

    def me2_exact_color_sum(self, mom: np.ndarray, legs: Sequence[Particle]) -> float:
        idx_g=[i for i,p in enumerate(legs) if p.kind=="g"]
        ng=len(idx_g)
        basis=[]
        for w1,w2 in _line_assignments(ng):
            for ordering in ("ladder","crossed"):
                basis.append((w1,w2,ordering))
        if self.Nc == 3 and ng <= 2:
            C = color_matrix_two_exchange_su3(basis, ng=ng)
        else:
            raise NotImplementedError("Two-exchange exact color currently implemented for SU(3) with ng<=2 (explicit sum).")

        A=np.zeros(len(basis), dtype=np.complex128)
        for i,(w1,w2,ordering) in enumerate(basis):
            acc=0.0+0j
            for s1a in range(len(w1)+1):
                for s1b in range(s1a, len(w1)+1):
                    for s2a in range(len(w2)+1):
                        for s2b in range(s2a, len(w2)+1):
                            acc += self.primitive(mom, legs, w1,w2, s1a,s1b,s2a,s2b, ordering)
            A[i]=acc

        me2=0.0+0j
        for i in range(len(basis)):
            ai=np.conjugate(A[i])
            for j in range(len(basis)):
                me2 += ai*C[i][j]*A[j]
        return float(me2.real)
