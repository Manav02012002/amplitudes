import numpy as np
from amplitudes.color_internal_radiation import color_gram_two_lines_internal

def test_internal_color_known_value_nc3():
    # Configuration:
    #  - no gluons on quark lines
    #  - one gluon label '1' emitted on the exchanged adjoint line in both amp and conj
    # With our conventions (Tr(T^a T^b)=1/2 δ^{ab}, F^a=-i f^{a..}),
    # the exact contraction evaluates to -6 for Nc=3 (see f^{abc}f^{abc} identity and 1/2 factors from traces).
    val = color_gram_two_lines_internal([], [], [1], [], [], [1], Nc=3)
    assert abs(val + 6.0) < 1e-12
