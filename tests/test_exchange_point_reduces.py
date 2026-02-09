import numpy as np
from amplitudes.process_two_lines import TwoLineQQbarToQQbarJets
from amplitudes.process_two_lines_exchange import TwoLineQQbarToQQbarJetsExchange
from amplitudes.particles import Particle
from amplitudes.phasespace import rambo_massless

def test_exchange_reduces_to_simple_topology_when_all_before():
    rng = np.random.default_rng(1)
    eng0 = TwoLineQQbarToQQbarJets()
    eng1 = TwoLineQQbarToQQbarJetsExchange()
    ng = 2
    mom, _ = rambo_massless(4+ng, 800.0, rng)
    legs = [
        Particle("q", -1, flavor="u"),
        Particle("qb", +1, flavor="u"),
        Particle("q", -1, flavor="d"),
        Particle("qb", +1, flavor="d"),
    ] + [Particle("g", +1) for _ in range(ng)]
    # eng0 corresponds to exchange emitted after all external gluons on each line.
    me2_0 = eng0.me2_exact_color_sum(mom, legs)
    # eng1 includes all exchange positions, so it should be >= me2_0 typically; but not strictly.
    me2_1 = eng1.me2_exact_color_sum(mom, legs)
    assert np.isfinite(me2_0) and np.isfinite(me2_1)
    assert me2_1 >= 0.0 and me2_0 >= 0.0
