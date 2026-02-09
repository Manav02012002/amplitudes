import numpy as np
from amplitudes.process_two_lines_internal import TwoLineWithInternalGluonRadiation
from amplitudes.particles import Particle
from amplitudes.phasespace import rambo_massless

def test_internal_bicurrent_runs():
    rng = np.random.default_rng(3)
    eng = TwoLineWithInternalGluonRadiation()
    ng = 1
    mom, _ = rambo_massless(4+ng, 700.0, rng)
    legs = [
        Particle("q", -1, flavor="u"),
        Particle("qb", +1, flavor="u"),
        Particle("q", -1, flavor="d"),
        Particle("qb", +1, flavor="d"),
        Particle("g", +1),
    ]
    me2 = eng.me2_color_approx(mom, legs)
    assert np.isfinite(me2)
    assert me2 >= 0.0
