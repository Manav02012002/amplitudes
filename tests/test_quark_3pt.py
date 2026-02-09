import numpy as np
from amplitudes.spinor import SpinorPoint
from amplitudes.bcfw_quark import primitive_q_qb_gluons

def test_quark_3pt_nonzero_configs():
    rng = np.random.default_rng(7)
    # build a massless spinor point directly (no need for real 3pt momentum conservation here)
    lam = rng.normal(size=(3,2)) + 1j*rng.normal(size=(3,2))
    lamt = rng.normal(size=(3,2)) + 1j*rng.normal(size=(3,2))
    sp = SpinorPoint(lam=lam.astype(np.complex128), lamt=lamt.astype(np.complex128))

    # ordering (q, g, qb)
    A1 = primitive_q_qb_gluons(sp, (-1, -1, +1), i_shift=0, j_shift=1)
    A2 = primitive_q_qb_gluons(sp, (+1, +1, -1), i_shift=0, j_shift=1)
    A0 = primitive_q_qb_gluons(sp, (-1, +1, +1), i_shift=0, j_shift=1)
    assert abs(A1) > 0
    assert abs(A2) > 0
    assert abs(A0) == 0
