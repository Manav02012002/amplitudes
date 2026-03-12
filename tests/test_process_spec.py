import numpy as np
import pytest

from amplitudes.crossing import External
from amplitudes.particles import gluon, quark, antiquark, vector
from amplitudes.process import Process, ProcessSpec, matrix_element_squared


def test_process_spec_all_outgoing_normalization_gg_to_gg():
    p1 = np.array([5.0, 0.0, 0.0, 5.0], dtype=np.complex128)
    p2 = np.array([5.0, 0.0, 0.0, -5.0], dtype=np.complex128)
    p3 = np.array([4.0, 1.0, 0.0, 3.0], dtype=np.complex128)
    p4 = -(p1 + p2 + p3)
    spec = ProcessSpec(
        initial=(External(gluon(+1), p1, incoming=True), External(gluon(-1), p2, incoming=True)),
        final=(External(gluon(+1), p3, incoming=False), External(gluon(-1), p4, incoming=False)),
    )

    assert spec.all_outgoing_kinds() == ("g", "g", "g", "g")
    assert spec.helicity_tuple_all_outgoing() == (+1, -1, +1, -1)
    assert np.allclose(spec.all_outgoing_momenta()[0], -p1)
    assert np.allclose(spec.all_outgoing_momenta()[1], -p2)
    assert np.allclose(spec.all_outgoing_momenta()[2], p3)
    assert np.allclose(spec.all_outgoing_momenta()[3], p4)
    assert spec.validate_supported_tree_process() == "gluons"


def test_process_spec_all_outgoing_normalization_qqbar_to_gg():
    p1 = np.array([6.0, 0.0, 0.0, 6.0], dtype=np.complex128)
    p2 = np.array([6.0, 0.0, 0.0, -6.0], dtype=np.complex128)
    p3 = np.array([5.0, 2.0, 0.0, 3.0], dtype=np.complex128)
    p4 = -(p1 + p2 + p3)
    spec = ProcessSpec(
        initial=(External(quark(-1), p1, incoming=True), External(antiquark(+1), p2, incoming=True)),
        final=(External(gluon(+1), p3, incoming=False), External(gluon(-1), p4, incoming=False)),
    )

    assert spec.all_outgoing_kinds() == ("qb", "q", "g", "g")
    assert spec.helicity_tuple_all_outgoing() == (-1, +1, +1, -1)
    assert np.allclose(spec.all_outgoing_momenta()[0], -p1)
    assert np.allclose(spec.all_outgoing_momenta()[1], -p2)
    assert spec.validate_supported_tree_process() == "qqbar_ng"


def test_process_spec_rejects_unsupported_process_content():
    p1 = np.array([4.0, 0.0, 0.0, 4.0], dtype=np.complex128)
    p2 = np.array([4.0, 0.0, 0.0, -4.0], dtype=np.complex128)
    p3 = np.array([3.0, 1.0, 0.0, 2.0], dtype=np.complex128)
    p4 = -(p1 + p2 + p3)
    spec = ProcessSpec(
        initial=(External(gluon(+1), p1, incoming=True), External(gluon(-1), p2, incoming=True)),
        final=(External(vector(+1), p3, incoming=False), External(gluon(-1), p4, incoming=False)),
    )

    with pytest.raises(ValueError, match=r"Unsupported process kinds: \('g', 'g', 'v', 'g'\)"):
        spec.validate_supported_tree_process()


def test_legacy_process_matches_process_spec():
    p1 = np.array([7.0, 0.0, 0.0, 7.0], dtype=np.complex128)
    p2 = np.array([7.0, 0.0, 0.0, -7.0], dtype=np.complex128)
    p3 = np.array([7.0, 3.0, 2.0, 6.0], dtype=np.complex128)
    p4 = p1 + p2 - p3
    initial = [External(quark(-1), p1, incoming=True), External(antiquark(+1), p2, incoming=True)]
    final = [External(gluon(+1), p3, incoming=False), External(gluon(-1), p4, incoming=False)]

    legacy = Process(initial=initial, final=final)
    canonical = ProcessSpec(initial=tuple(initial), final=tuple(final))

    assert legacy.initial == canonical.initial
    assert legacy.final == canonical.final
    assert legacy.all_outgoing_kinds() == canonical.all_outgoing_kinds()
    assert legacy.helicity_tuple_all_outgoing() == canonical.helicity_tuple_all_outgoing()
    assert legacy.validate_supported_tree_process() == canonical.validate_supported_tree_process()
    assert matrix_element_squared(legacy, sum_helicities=False, average_initial=False) == pytest.approx(
        matrix_element_squared(canonical, sum_helicities=False, average_initial=False),
        rel=1e-12,
        abs=1e-12,
    )
