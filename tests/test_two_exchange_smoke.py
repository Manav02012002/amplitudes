import numpy as np
from amplitudes.process_two_lines_two_exchange import TwoLineTwoExchangeEngine
from amplitudes.particles import Particle
from amplitudes.phasespace import rambo_massless

def test_two_exchange_runs_ng0():
    rng = np.random.default_rng(21)
    mom, _ = rambo_massless(4, 600.0, rng)
    legs = [
        Particle("q", -1, flavor="u"),
        Particle("qb", +1, flavor="u"),
        Particle("q", -1, flavor="d"),
        Particle("qb", +1, flavor="d"),
    ]
    eng = TwoLineTwoExchangeEngine()
    me2 = eng.me2_exact_color_sum(mom, legs)
    assert np.isfinite(me2)
