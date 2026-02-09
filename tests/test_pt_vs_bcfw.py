import numpy as np
from amplitudes.phasespace import rambo_massless
from amplitudes.spinor import SpinorPoint
from amplitudes.parke_taylor import parke_taylor_mhv
from amplitudes.bcfw import bcfw_color_ordered_tree

def test_parke_matches_bcfw_4pt_mhv():
    rng = np.random.default_rng(123)
    p, _ = rambo_massless(4, 1000.0, rng)
    sp = SpinorPoint.from_momenta(p)
    hel = (-1, -1, +1, +1)
    A_b = bcfw_color_ordered_tree(sp, hel, i=0, j=1)
    A_pt = parke_taylor_mhv(sp, 0, 1)
    denom = max(1.0, abs(A_pt))
    assert abs(A_b - A_pt) / denom < 1e-6
