import numpy as np
from amplitudes.process_two_lines_exchange import TwoLineQQbarToQQbarJetsExchange
from amplitudes.process_two_lines_full import TwoLineFullTreeEngine
from amplitudes.particles import Particle
from amplitudes.phasespace import rambo_massless

def test_full_engine_matches_exchange_when_no_internal():
    rng = np.random.default_rng(7)
    mom, _ = rambo_massless(6, 900.0, rng)  # ng=2
    legs = [
        Particle("q", -1, flavor="u"),
        Particle("qb", +1, flavor="u"),
        Particle("q", -1, flavor="d"),
        Particle("qb", +1, flavor="d"),
        Particle("g", +1),
        Particle("g", -1),
    ]
    eng_ex = TwoLineQQbarToQQbarJetsExchange()
    eng_full = TwoLineFullTreeEngine(include_internal=False)
    me2_ex = eng_ex.me2_exact_color_sum(mom, legs)
    me2_full = eng_full.me2_exact_color_sum(mom, legs)
    # Numerical agreement (same basis: only line assignments + exchange splits)
    assert abs(me2_ex - me2_full) < 1e-10
