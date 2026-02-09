import numpy as np
from amplitudes.crossing import External, cross_external, to_all_outgoing
from amplitudes.particles import quark, antiquark, gluon

def test_cross_quark_swaps_kind_and_flips_momentum():
    p = np.array([10.0, 1.0, 2.0, 3.0], dtype=np.complex128)
    e = External(quark(+1), p, incoming=True)
    c = cross_external(e)
    assert c.particle.kind == "qb"
    assert np.allclose(c.momentum, -p)

def test_to_all_outgoing_shapes():
    p1 = np.array([1.0,0,0,1.0], dtype=np.complex128)
    p2 = np.array([1.0,0,0,-1.0], dtype=np.complex128)
    pf = np.array([[2.0,0,0,0]], dtype=np.complex128)
    proc, mom = to_all_outgoing(
        [External(gluon(+1), p1, incoming=True), External(gluon(-1), p2, incoming=True)],
        [External(gluon(+1), pf[0], incoming=False)]
    )
    assert len(proc) == 3
    assert mom.shape == (3,4)
