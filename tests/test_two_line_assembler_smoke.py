import numpy as np
from amplitudes.process_two_lines import TwoLineQQbarToQQbarJets
from amplitudes.particles import Particle
from amplitudes.phasespace import rambo_massless

def test_two_line_me2_finite():
    rng = np.random.default_rng(0)
    eng = TwoLineQQbarToQQbarJets()
    for ng in (0, 1, 2):
        mom, _ = rambo_massless(4 + ng, 1000.0, rng)
        legs = [
            Particle("q", -1, flavor="u"),
            Particle("qb", +1, flavor="u"),
            Particle("q", -1, flavor="d"),
            Particle("qb", +1, flavor="d"),
        ] + [Particle("g", +1) for _ in range(ng)]
        me2 = eng.me2_exact_color_sum(mom, legs)
        assert np.isfinite(me2)
        assert me2 >= 0.0
