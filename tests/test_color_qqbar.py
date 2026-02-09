import numpy as np
from amplitudes.color_quark import color_matrix_qqbar_ng_exact

def test_color_matrix_qqbar_symmetry():
    S = color_matrix_qqbar_ng_exact(3, Nc=3)
    assert np.allclose(S, S.T, atol=1e-12)

def test_color_matrix_qqbar_psd_small():
    S = color_matrix_qqbar_ng_exact(2, Nc=3)
    w = np.linalg.eigvalsh(S)
    assert np.min(w) > -1e-10
