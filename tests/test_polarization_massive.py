import numpy as np
import math
from amplitudes.polarization import massive_vector_polarizations
from amplitudes.lorentz import minkowski_dot

def _assert_massive_polarization_properties(p: np.ndarray, m: float) -> None:
    epsp, epsm, eps0 = massive_vector_polarizations(p, m)
    for e in (epsp, epsm, eps0):
        assert abs(minkowski_dot(p, e)) < 1e-6
        n = minkowski_dot(e.conjugate(), e).real
        assert abs(n + 1.0) < 1e-6
    assert abs(minkowski_dot(epsp.conjugate(), epsm)) < 1e-6
    assert abs(minkowski_dot(epsp.conjugate(), eps0)) < 1e-6
    assert abs(minkowski_dot(epsm.conjugate(), eps0)) < 1e-6


def test_massive_polarizations_generic_boosted_momentum():
    m = 80.379
    pvec = np.array([30.0, 40.0, 180.0], dtype=float)
    E = math.sqrt(float(np.dot(pvec, pvec)) + m * m)
    p = np.array([E, *pvec], dtype=np.complex128)
    _assert_massive_polarization_properties(p, m)


def test_massive_polarizations_rest_limit_branch():
    m = 80.379
    p = np.array([m, 0.0, 0.0, 0.0], dtype=np.complex128)
    epsp, epsm, eps0 = massive_vector_polarizations(p, m)

    _assert_massive_polarization_properties(p, m)
    np.testing.assert_allclose(eps0, np.array([0.0, 0.0, 0.0, 1.0], dtype=np.complex128), atol=0.0, rtol=0.0)


def test_massive_polarizations_near_rest_are_finite_and_stable():
    m = 80.379
    pvec = np.array([1e-16, -2e-16, 3e-16], dtype=float)
    E = math.sqrt(float(np.dot(pvec, pvec)) + m * m)
    p = np.array([E, *pvec], dtype=np.complex128)
    epsp, epsm, eps0 = massive_vector_polarizations(p, m)

    _assert_massive_polarization_properties(p, m)
    assert np.all(np.isfinite(epsp))
    assert np.all(np.isfinite(epsm))
    assert np.all(np.isfinite(eps0))
    np.testing.assert_allclose(eps0, np.array([0.0, 0.0, 0.0, 1.0], dtype=np.complex128), atol=0.0, rtol=0.0)
