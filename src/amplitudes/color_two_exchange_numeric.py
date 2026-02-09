from __future__ import annotations
from typing import Sequence
import numpy as np
import itertools
from .color_numeric_su3 import su3_generators

def _string_matrix(T: list[np.ndarray], word: Sequence[int], cache: dict[tuple[int,...], np.ndarray]) -> np.ndarray:
    key = tuple(word)
    if key in cache:
        return cache[key]
    M = np.eye(3, dtype=np.complex128)
    for a in word:
        M = M @ T[a-1]   # external gluon labels are 1..ng, but here we pass actual color indices as 1..8
    cache[key]=M
    return M

def color_gram_two_exchange_su3(
    w1: Sequence[int], w2: Sequence[int],
    w1p: Sequence[int], w2p: Sequence[int],
    ordering: str,
    ordering_p: str,
    ng: int,
) -> float:
    """Exact SU(3) color Gram element for two-exchange topology by explicit summation.

    Supports ng<=2 external gluons total (distributed on lines). External gluon colors are summed explicitly.
    Internal exchange colors (b,c) are summed explicitly (8^2).

    w1,w2 are words of external *labels* (1..ng) on each line. We sum over their color indices.
    The two exchanged gluons are inserted at the end of each word as (b,c) on line1; line2 sees (b,c) or (c,b).
    Conjugate uses primed words and ordering_p.

    This is expensive but exact for low multiplicity and removes the limitation of pairwise-label color recursion.
    """
    if ng > 2:
        raise NotImplementedError("Explicit SU(3) color sum for two-exchange implemented for ng<=2.")
    if ordering not in ("ladder","crossed") or ordering_p not in ("ladder","crossed"):
        raise ValueError("ordering must be ladder/crossed")
    T = su3_generators()
    cache={}
    # map external label -> color index (1..8)
    # sum over all assignments for external gluons present
    ext_labels = list(range(1, ng+1))
    total = 0.0 + 0j
    for colors in itertools.product(range(1,9), repeat=ng):
        col = {lab: colors[lab-1] for lab in ext_labels}
        # build actual color-index words for each basis
        W1  = [col[x] for x in w1]
        W2  = [col[x] for x in w2]
        W1p = [col[x] for x in w1p]
        W2p = [col[x] for x in w2p]
        M1  = _string_matrix(T, W1, cache)
        M2  = _string_matrix(T, W2, cache)
        M1p = _string_matrix(T, W1p, cache)
        M2p = _string_matrix(T, W2p, cache)
        # sum over internal exchange colors b,c
        for b in range(1,9):
            Tb = T[b-1]
            for c in range(1,9):
                Tc = T[c-1]
                # line insertions
                L1  = M1 @ Tb @ Tc
                L1p = M1p @ Tb @ Tc  # conjugate shares same internal labels under full sum
                if ordering == "ladder":
                    L2  = M2 @ Tb @ Tc
                else:
                    L2  = M2 @ Tc @ Tb
                if ordering_p == "ladder":
                    L2p = M2p @ Tb @ Tc
                else:
                    L2p = M2p @ Tc @ Tb
                # contract open indices with color sums:
                # sum_{i,j,k,l} (L1_{ij} L2_{kl}) (L1p_{ij} L2p_{kl})^*
                total += np.vdot(L1p, L1) * np.vdot(L2p, L2)
    return float(total.real)

def color_matrix_two_exchange_su3(basis: Sequence[tuple[Sequence[int],Sequence[int],str]], ng: int) -> list[list[float]]:
    m=len(basis)
    C=[[0.0]*m for _ in range(m)]
    for i,(w1,w2,ord1) in enumerate(basis):
        for j in range(i,m):
            w1p,w2p,ord2 = basis[j]
            cij = color_gram_two_exchange_su3(w1,w2,w1p,w2p,ord1,ord2,ng=ng)
            C[i][j]=cij; C[j][i]=cij
    return C
