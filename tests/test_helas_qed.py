import numpy as np
from amplitudes.helas import fermion_wf, vector_wf, ffv_amplitude
from amplitudes.sm import SMParams, gamma_coupling

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
