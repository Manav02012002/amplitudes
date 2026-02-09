import numpy as np
import math
from amplitudes.polarization import massive_vector_polarizations
from amplitudes.lorentz import minkowski_dot

def test_massive_polarizations_transverse_and_norm():
    m = 80.379
    pvec = np.array([30.0, 40.0, 180.0], dtype=float)
    E = math.sqrt(float(np.dot(pvec,pvec)) + m*m)
    p = np.array([E, *pvec], dtype=np.complex128)
    epsp, epsm, eps0 = massive_vector_polarizations(p, m)
    for e in (epsp, epsm, eps0):
        assert abs(minkowski_dot(p, e)) < 1e-6
        n = minkowski_dot(e.conjugate(), e).real
        assert abs(n + 1.0) < 1e-6
