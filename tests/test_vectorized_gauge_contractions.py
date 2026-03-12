import numpy as np

from amplitudes.bg_bicurrent import _contract_V4_with_remainder
from amplitudes.bg_currents import _V4_contact
from amplitudes.helas import VectorWF, vvv_amplitude

_ETA = np.diag([1, -1, -1, -1]).astype(np.complex128)


def _slow_v4_contact(Ja: np.ndarray, Jb: np.ndarray, Jc: np.ndarray) -> np.ndarray:
    out = np.zeros(4, dtype=np.complex128)
    for mu in range(4):
        total = 0.0 + 0.0j
        for nu in range(4):
            for rho in range(4):
                for sig in range(4):
                    vertex = 2 * _ETA[mu, rho] * _ETA[nu, sig]
                    vertex -= _ETA[mu, nu] * _ETA[rho, sig]
                    vertex -= _ETA[mu, sig] * _ETA[nu, rho]
                    total += vertex * Jb[nu] * Jc[rho] * Ja[sig]
        out[mu] = total
    return out


def _slow_vvv_amplitude(V1: VectorWF, V2: VectorWF, V3: VectorWF, g: complex) -> complex:
    out = 0.0 + 0.0j
    for mu in range(4):
        for nu in range(4):
            for rho in range(4):
                vertex = _ETA[nu, rho] * (V2.p - V3.p)[mu]
                vertex += _ETA[rho, mu] * (V3.p - V1.p)[nu]
                vertex += _ETA[mu, nu] * (V1.p - V2.p)[rho]
                out += vertex * V1.eps[mu] * V2.eps[nu] * V3.eps[rho]
    return g * out


def _slow_contract_v4_with_remainder(J1: np.ndarray, J2: np.ndarray, rem: np.ndarray) -> np.ndarray:
    out = np.zeros((4, 4), dtype=np.complex128)
    for mu in range(4):
        for nu in range(4):
            total = 0.0 + 0.0j
            for a in range(4):
                for b in range(4):
                    for rmu in range(4):
                        vertex = 2 * _ETA[mu, b] * _ETA[a, rmu]
                        vertex -= _ETA[mu, a] * _ETA[b, rmu]
                        vertex -= _ETA[mu, rmu] * _ETA[a, b]
                        total += vertex * J1[a] * J2[b] * rem[rmu, nu]
            out[mu, nu] = total
    return out


def test_v4_contact_matches_explicit_loop_formula():
    Ja = np.array([1.0 + 0.2j, -0.5 + 0.3j, 0.7 - 0.1j, 0.4 + 0.6j], dtype=np.complex128)
    Jb = np.array([-0.3 + 0.5j, 0.2 - 0.8j, 1.1 + 0.4j, -0.9 + 0.7j], dtype=np.complex128)
    Jc = np.array([0.8 - 0.6j, -1.2 + 0.1j, 0.3 + 0.9j, 0.5 - 0.4j], dtype=np.complex128)

    np.testing.assert_allclose(_V4_contact(Ja, Jb, Jc), _slow_v4_contact(Ja, Jb, Jc), rtol=1e-14, atol=1e-14)


def test_vvv_amplitude_matches_explicit_loop_formula():
    V1 = VectorWF(
        p=np.array([10.0, 1.0, -2.0, 3.0], dtype=np.complex128),
        eps=np.array([0.2 + 0.1j, -0.4 + 0.3j, 0.5 - 0.2j, 0.1 + 0.6j], dtype=np.complex128),
    )
    V2 = VectorWF(
        p=np.array([11.0, -1.5, 0.5, -2.5], dtype=np.complex128),
        eps=np.array([-0.3 + 0.2j, 0.7 + 0.1j, -0.6 + 0.5j, 0.4 - 0.3j], dtype=np.complex128),
    )
    V3 = VectorWF(
        p=np.array([9.0, 0.5, 1.5, -0.5], dtype=np.complex128),
        eps=np.array([0.6 - 0.4j, -0.2 + 0.7j, 0.8 + 0.2j, -0.5 + 0.1j], dtype=np.complex128),
    )
    g = 0.73 - 0.11j

    np.testing.assert_allclose(vvv_amplitude(V1, V2, V3, g), _slow_vvv_amplitude(V1, V2, V3, g), rtol=1e-14, atol=1e-14)


def test_bicurrent_v4_remainder_contraction_matches_explicit_loop_formula():
    J1 = np.array([0.4 + 0.8j, -0.7 + 0.2j, 0.1 - 0.5j, 1.2 + 0.3j], dtype=np.complex128)
    J2 = np.array([-0.2 + 0.6j, 0.9 - 0.4j, -1.1 + 0.7j, 0.3 + 0.2j], dtype=np.complex128)
    rem = np.array(
        [
            [0.5 + 0.1j, -0.3 + 0.7j, 0.2 - 0.4j, 0.9 + 0.6j],
            [-0.8 + 0.2j, 1.1 - 0.5j, -0.6 + 0.3j, 0.4 - 0.9j],
            [0.7 - 0.8j, -0.1 + 0.2j, 0.3 + 0.5j, -0.4 + 0.6j],
            [0.2 + 0.9j, -0.5 - 0.3j, 0.8 - 0.7j, 0.6 + 0.4j],
        ],
        dtype=np.complex128,
    )

    np.testing.assert_allclose(
        _contract_V4_with_remainder(J1, J2, rem),
        _slow_contract_v4_with_remainder(J1, J2, rem),
        rtol=1e-14,
        atol=1e-14,
    )
