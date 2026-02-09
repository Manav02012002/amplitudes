from __future__ import annotations
import numpy as np
from .spinor import SpinorPoint
from .bcfw_quark import primitive_q_qb_gluons
from .color_quark import color_matrix_qqbar_ng_exact, QuarkLineBasis

def all_partials_qqbar_ng(sp: SpinorPoint, hels: tuple[int, ...]) -> tuple[list[tuple[int,...]], np.ndarray]:
    """
    External ordering expected: [q, g1, g2, ..., g_ng, qb] with gluons labeled 1..ng in this order.
    Partials are permutations of the gluon legs between q and qb.
    """
    n = len(hels)
    if n < 3:
        raise ValueError("need at least q,g...,qb")
    ng = n - 2
    basis = QuarkLineBasis(ng).orderings()
    A = np.zeros(len(basis), dtype=np.complex128)

    for k, perm in enumerate(basis):
        # build order indices: q=0, then gluons in perm order (labels 1..ng map to indices 0+label)
        order = (0,) + tuple(int(x) for x in perm) + (ng+1,)
        lam = sp.lam[list(order)]
        lamt = sp.lamt[list(order)]
        sp_ord = SpinorPoint(lam=lam, lamt=lamt)
        hel_ord = tuple(hels[i] for i in order)
        A[k] = primitive_q_qb_gluons(sp_ord, hel_ord)  # works for q...qb ordering
    return list(basis), A

def matrix_element_squared_qqbar_ng_exact_SU_N(sp: SpinorPoint, hels: tuple[int, ...], Nc: int = 3, g_s: float = 1.0) -> float:
    n = len(hels)
    ng = n - 2
    _, A = all_partials_qqbar_ng(sp, hels)
    S = color_matrix_qqbar_ng_exact(ng, Nc)
    me2 = (A.conjugate() @ (S @ A)).real
    return float((g_s ** (2*ng)) * me2)  # qqbar + ng: vertices scale as g_s^ng
