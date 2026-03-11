import math
import numpy as np
import pytest
from amplitudes.phasespace import rambo_massless
from amplitudes.spinor import SpinorPoint
from amplitudes.bcfw import bcfw_color_ordered_tree
from amplitudes.validate.analytic_qcd import mhv_gluon_reference

@pytest.mark.parametrize(
    ("nlegs", "negative_legs", "seed"),
    [
        (4, (0, 1), 123),
        (5, (0, 1), 456),
    ],
)
def test_parke_matches_bcfw_mhv(nlegs: int, negative_legs: tuple[int, int], seed: int):
    rng = np.random.default_rng(seed)
    p, _ = rambo_massless(nlegs, 1000.0, rng)
    sp = SpinorPoint.from_momenta(p)
    hel = tuple(-1 if i in negative_legs else +1 for i in range(nlegs))
    A_b = bcfw_color_ordered_tree(sp, hel, i=0, j=1)
    A_pt = mhv_gluon_reference(sp, negative_legs)
    assert math.isclose(A_b.real, A_pt.real, rel_tol=1e-12, abs_tol=1e-12)
    assert math.isclose(A_b.imag, A_pt.imag, rel_tol=1e-12, abs_tol=1e-12)
