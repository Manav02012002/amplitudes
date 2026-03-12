import numpy as np
from amplitudes.crossing import External, cross_external, to_all_outgoing
from amplitudes.particles import quark, antiquark, gluon
from amplitudes.process import ProcessSpec

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


def test_process_spec_uses_crossing_for_all_outgoing_views():
    p1 = np.array([5.0, 0.0, 0.0, 5.0], dtype=np.complex128)
    p2 = np.array([5.0, 0.0, 0.0, -5.0], dtype=np.complex128)
    p3 = np.array([3.0, 1.0, 0.0, 2.0], dtype=np.complex128)
    p4 = -(p1 + p2 + p3)
    spec = ProcessSpec(
        initial=(External(quark(-1), p1, incoming=True), External(gluon(+1), p2, incoming=True)),
        final=(External(gluon(-1), p3, incoming=False), External(antiquark(+1), p4, incoming=False)),
    )
    particles = spec.all_outgoing_particles()
    assert tuple(p.kind for p in particles) == ("qb", "g", "g", "qb")
    assert spec.all_outgoing_momenta().shape == (4, 4)
    assert spec.all_outgoing_kinds() == ("qb", "g", "g", "qb")
    assert spec.helicity_tuple_all_outgoing() == (-1, +1, -1, +1)


def test_process_spec_validates_supported_tree_process():
    p1 = np.array([6.0, 0.0, 0.0, 6.0], dtype=np.complex128)
    p2 = np.array([6.0, 0.0, 0.0, -6.0], dtype=np.complex128)
    p3 = np.array([4.0, 1.0, 2.0, 3.0], dtype=np.complex128)
    p4 = -(p1 + p2 + p3)
    spec = ProcessSpec(
        initial=(External(quark(-1), p1, incoming=True), External(antiquark(+1), p2, incoming=True)),
        final=(External(gluon(+1), p3, incoming=False), External(gluon(-1), p4, incoming=False)),
    )
    assert spec.validate_supported_tree_process() == "qqbar_ng"
