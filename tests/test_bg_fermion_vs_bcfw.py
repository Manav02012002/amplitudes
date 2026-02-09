import numpy as np
import math
from amplitudes.bg_fermion import FermionLineBG
from amplitudes.particles import Particle
from amplitudes.bcfw_quark import primitive_q_qb_gluons
from amplitudes.spinor import SpinorPoint
from amplitudes.phasespace import rambo_massless
from amplitudes.sm import SMParams

def test_bg_matches_bcfw_modsq_small_ng():
    rng = np.random.default_rng(2)
    Ecm = 1000.0
    # Choose alpha_s so that g_s = 1 (since bcfw_quark primitives are normalized with g_s=1)
    alpha_s = 1.0/(4.0*math.pi)
    bg = FermionLineBG(params=SMParams(alpha_s=alpha_s))
    for ng in (1,2,3):
        mom, _ = rambo_massless(ng+2, Ecm, rng)
        legs = [Particle("q", -1, flavor="u")]
        for _ in range(ng):
            legs.append(Particle("g", +1))
        legs.append(Particle("qb", +1, flavor="u"))
        hels = tuple(p.hel for p in legs)
        sp = SpinorPoint.from_momenta(mom)
        A_bcfw = primitive_q_qb_gluons(sp, hels)
        A_bg = bg.primitive_amplitude(mom, legs)
        assert np.isfinite(abs(A_bg))
        assert abs(abs(A_bg)**2 - abs(A_bcfw)**2) / (abs(A_bcfw)**2 + 1e-30) < 1e-6
