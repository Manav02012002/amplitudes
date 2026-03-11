import numpy as np
from amplitudes.helas import build_massless_dirac_spinor, fermion_wf, vector_wf, ffv_amplitude
from amplitudes.sm import SMParams, gamma_coupling
from amplitudes.spinor import SpinorPoint


def test_massless_dirac_spinor_helper_uses_one_q_qb_embedding_convention():
    p = np.array([100.0, 30.0, 40.0, 86.60254037844386], dtype=np.complex128)
    sp = SpinorPoint.from_momenta(np.asarray([p], dtype=np.complex128))
    lam = sp.lam[0]
    lamt = sp.lamt[0]

    q_minus = build_massless_dirac_spinor("q", p, hel=-1)
    qb_minus = build_massless_dirac_spinor("qb", p, hel=-1)
    q_plus = build_massless_dirac_spinor("q", p, hel=+1)
    qb_plus = build_massless_dirac_spinor("qb", p, hel=+1)

    np.testing.assert_allclose(q_minus[:2], lam, atol=0.0, rtol=0.0)
    np.testing.assert_allclose(qb_minus[:2], lam, atol=0.0, rtol=0.0)
    np.testing.assert_allclose(q_plus[2:], lamt, atol=0.0, rtol=0.0)
    np.testing.assert_allclose(qb_plus[2:], lamt, atol=0.0, rtol=0.0)
    np.testing.assert_allclose(q_minus[2:], np.zeros(2, dtype=np.complex128), atol=0.0, rtol=0.0)
    np.testing.assert_allclose(qb_minus[2:], np.zeros(2, dtype=np.complex128), atol=0.0, rtol=0.0)
    np.testing.assert_allclose(q_plus[:2], np.zeros(2, dtype=np.complex128), atol=0.0, rtol=0.0)
    np.testing.assert_allclose(qb_plus[:2], np.zeros(2, dtype=np.complex128), atol=0.0, rtol=0.0)

def test_qed_ward_identity_ffv():
    params = SMParams()
    # Two distinct lightlike momenta => nonzero q
    p_in  = np.array([100.0, 0.0, 0.0, 100.0], dtype=np.complex128)
    p_out = np.array([100.0, 10.0, 0.0, np.sqrt(100.0**2 - 10.0**2)], dtype=np.complex128)
    q = p_in - p_out

    u_in = fermion_wf(p_in, hel=-1, kind="l")
    u_out = fermion_wf(p_out, hel=-1, kind="l")
    g = gamma_coupling(params, "e")

    V = vector_wf(q, hel=+1, kind="A", m=0.0, width=0.0)
    V.eps = q  # Ward identity check: eps -> q
    amp = ffv_amplitude(u_out, u_in, V, gL=g, gR=g)
    assert abs(amp) < 1e-6
