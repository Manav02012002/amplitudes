from __future__ import annotations
import numpy as np
from .spinor import SpinorPoint
from .bcfw import bcfw_color_ordered_tree
from .parke_taylor import parke_taylor_mhv
from .color import color_matrix_single_trace_exact, TraceBasis

def partial_amplitude_gluons(sp: SpinorPoint, hels: tuple[int, ...], order: tuple[int, ...]) -> complex:
    lam = sp.lam[list(order)]
    lamt = sp.lamt[list(order)]
    sp_ord = SpinorPoint(lam=lam, lamt=lamt)
    hel_ord = tuple(hels[i] for i in order)

    neg = [i for i,h in enumerate(hel_ord) if h == -1]
    if len(hel_ord) >= 4 and len(neg) == 2:
        return parke_taylor_mhv(sp_ord, neg[0], neg[1])

    return bcfw_color_ordered_tree(sp_ord, hel_ord, i=0, j=1)

def all_partials_single_trace(sp: SpinorPoint, hels: tuple[int, ...]) -> tuple[list[tuple[int,...]], np.ndarray]:
    n = len(hels)
    basis = TraceBasis(n).orderings()
    A = np.zeros(len(basis), dtype=np.complex128)
    for k, order in enumerate(basis):
        A[k] = partial_amplitude_gluons(sp, hels, order)
    return basis, A

def matrix_element_squared_gluons_exact_SU_N(sp: SpinorPoint, hels: tuple[int, ...], Nc: int = 3, g_s: float = 1.0) -> float:
    n = len(hels)
    _, A = all_partials_single_trace(sp, hels)
    S = color_matrix_single_trace_exact(n, Nc)
    me2 = (A.conjugate() @ (S @ A)).real
    return float((g_s ** (2*n - 4)) * me2)
