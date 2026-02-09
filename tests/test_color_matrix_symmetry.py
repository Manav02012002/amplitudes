import numpy as np
from amplitudes.color import color_matrix_single_trace_exact

def test_color_matrix_is_symmetric_real():
    S = color_matrix_single_trace_exact(4, Nc=3)
    assert np.allclose(S, S.T, atol=1e-12)
