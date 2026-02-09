from __future__ import annotations
from dataclasses import dataclass
from itertools import permutations, product
import numpy as np

class _DSU:
    def __init__(self, n: int):
        self.p = list(range(n))
        self.r = [0]*n
    def find(self, a: int) -> int:
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a
    def union(self, a: int, b: int):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.r[ra] < self.r[rb]:
            ra, rb = rb, ra
        self.p[rb] = ra
        if self.r[ra] == self.r[rb]:
            self.r[ra] += 1
    def n_components(self) -> int:
        return len({self.find(i) for i in range(len(self.p))})

@dataclass(frozen=True)
class QuarkLineBasis:
    """Color basis for A(q, g... , qb): permutations of gluon labels (1..ng)."""
    ng: int

    def orderings(self):
        legs = list(range(1, self.ng+1))
        return list(permutations(legs))

def _pos_map(seq):
    return {lab: i for i, lab in enumerate(seq)}

def color_matrix_qqbar_ng_exact(ng: int, Nc: int) -> np.ndarray:
    """
    Exact SU(Nc) interference matrix for single quark line with ng external gluons:
      C_sigma = (T^{a_{sigma1}} ... T^{a_{sigmang}})_i^j
    S_{σρ} = sum_{colors} Cσ Cρ^* with Tr(T^a T^b)=1/2 δ^{ab}.
    Implemented via Fierz per gluon by embedding into a single trace of length 2*ng:
      Tr( T^{a_{σ1}}...T^{a_{σng}} T^{a_{ρng}}...T^{a_{ρ1}} )
    where each external gluon color index appears twice.
    Complexity: O((ng!)^2 * 2^ng). Fine for ng <= 6-7.
    """
    basis = QuarkLineBasis(ng).orderings()
    m = len(basis)
    S = np.zeros((m, m), dtype=np.float64)

    for a, sigma in enumerate(basis):
        ps = _pos_map(sigma)  # positions in 0..ng-1
        for b, rho in enumerate(basis):
            pr = _pos_map(rho)  # positions in 0..ng-1

            # Combined cyclic trace positions 0..2ng-1:
            # first segment: sigma in order
            # second segment: rho in reverse
            total = 0.0
            for choices in product([0, 1], repeat=ng):
                dsu = _DSU(2*ng)
                coeff = (0.5) ** ng
                sing = 0

                for g in range(1, ng+1):
                    ch = choices[g-1]
                    if ch == 1:
                        sing += 1

                    p1 = ps[g]  # in [0,ng-1]
                    p2 = ng + (ng - 1 - pr[g])  # position in reversed rho segment

                    aL, aU = p1, (p1 + 1) % (2*ng)
                    bL, bU = p2, (p2 + 1) % (2*ng)

                    if ch == 0:
                        # δ_{aL}^{bU} δ_{bL}^{aU}
                        dsu.union(aL, bU)
                        dsu.union(bL, aU)
                    else:
                        # δ_{aL}^{aU} δ_{bL}^{bU}
                        dsu.union(aL, aU)
                        dsu.union(bL, bU)

                coeff *= ((-1.0 / Nc) ** sing)
                loops = dsu.n_components()
                total += coeff * (Nc ** loops)

            S[a, b] = total
    return S

@dataclass(frozen=True)
class TwoQuarkLinesNonCrossingBasis:
    """
    A restricted but fully functional basis for two quark lines in a non-crossing ordering:
      q1 (gluons on line1 in some order) qb1 q2 (gluons on line2 in some order) qb2
    with ng1 gluons on line1 labeled 1..ng1 and ng2 gluons on line2 labeled 1..ng2 (local labels).
    """
    ng1: int
    ng2: int

    def orderings(self):
        return list(permutations(range(1, self.ng1+1))), list(permutations(range(1, self.ng2+1)))

def color_matrix_two_lines_non_crossing_exact(ng1: int, ng2: int, Nc: int) -> np.ndarray:
    """
    Exact SU(Nc) interference matrix for the restricted two-line basis above.
    Color factor:
      C = (T^{a_{σ}})_i^j (T^{b_{τ}})_k^l
    S computed by applying Fierz independently on each line and summing fundamental indices.
    We implement this by building two separate trace-closures (like Tr(A A'^\dagger)) and multiplying.
    This is exact for the non-crossing basis where gluons are assigned to a line.
    """
    B1, B2 = TwoQuarkLinesNonCrossingBasis(ng1, ng2).orderings()
    m = len(B1)*len(B2)
    S = np.zeros((m, m), dtype=np.float64)

    S1 = color_matrix_qqbar_ng_exact(ng1, Nc) if ng1>0 else np.array([[Nc]], dtype=np.float64)
    S2 = color_matrix_qqbar_ng_exact(ng2, Nc) if ng2>0 else np.array([[Nc]], dtype=np.float64)

    # Kronecker product because lines are independent in this restricted basis
    S = np.kron(S1, S2)
    return S
