from __future__ import annotations
from dataclasses import dataclass
from itertools import product
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
        roots = {self.find(i) for i in range(len(self.p))}
        return len(roots)

def _pos_map(order: tuple[int, ...]) -> dict[int, int]:
    return {leg: i for i, leg in enumerate(order)}

@dataclass(frozen=True)
class TraceBasis:
    """
    Single-trace color basis for n gluons, fixing leg 0 at the start:
      orderings are permutations of (1..n-1) appended after 0.
    """
    n: int

    def orderings(self) -> list[tuple[int, ...]]:
        from itertools import permutations
        legs = list(range(1, self.n))
        out = []
        for perm in permutations(legs):
            out.append((0,) + perm)
        return out

def color_matrix_single_trace_exact(n: int, Nc: int) -> np.ndarray:
    """
    Exact SU(Nc) color matrix S_{σρ} for tree-level gluon single-trace basis:
      S_{σρ} = sum_{colors} Tr(T^{a_{σ1}}...T^{a_{σn}}) Tr(T^{a_{ρ1}}...T^{a_{ρn}})^*
    with Tr(T^a T^b)=1/2 δ^{ab}, using Fierz:
      (T^a)_i^j (T^a)_k^l = 1/2(δ_i^l δ_k^j - 1/Nc δ_i^j δ_k^l)
    evaluated exactly by expanding the 2-term identity for each external gluon (2^n terms).
    """
    basis = TraceBasis(n).orderings()
    m = len(basis)
    S = np.zeros((m, m), dtype=np.float64)

    for a, sigma in enumerate(basis):
        ps = _pos_map(sigma)
        for b, rho in enumerate(basis):
            pr = _pos_map(rho)

            total = 0.0
            for choices in product([0, 1], repeat=n):
                dsu = _DSU(2 * n)
                coeff = (0.5) ** n
                sing = 0
                for g, ch in enumerate(choices):
                    if ch == 1:
                        sing += 1

                    ts = ps[g]
                    tr = pr[g]

                    xs_low = ts
                    xs_up  = (ts + 1) % n
                    yr_low = n + tr
                    yr_up  = n + ((tr + 1) % n)

                    if ch == 0:
                        dsu.union(xs_low, yr_up)
                        dsu.union(yr_low, xs_up)
                    else:
                        dsu.union(xs_low, xs_up)
                        dsu.union(yr_low, yr_up)

                coeff *= ((-1.0 / Nc) ** sing)
                loops = dsu.n_components()
                total += coeff * (Nc ** loops)

            S[a, b] = total

    return S
