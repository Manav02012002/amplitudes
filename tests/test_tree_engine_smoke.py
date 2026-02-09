import numpy as np
from amplitudes.tree_engine import TreeEngine
from amplitudes.crossing import External
from amplitudes.particles import gluon
from amplitudes.kinematics import com_incoming_momenta
from amplitudes.phasespace import rambo_massless

def test_tree_engine_gg_to_4g_runs():
    eng = TreeEngine()
    Ecm = 1000.0
    p1, p2 = com_incoming_momenta(Ecm)
    rng = np.random.default_rng(1)
    pf, _ = rambo_massless(4, Ecm, rng)
    init = [External(gluon(+1), p1, incoming=True), External(gluon(-1), p2, incoming=True)]
    final = [External(gluon(+1), pf[i], incoming=False) for i in range(4)]
    me2 = eng.me2(init, final, sum_helicities=True, average_initial=True)
    assert me2 >= 0.0
